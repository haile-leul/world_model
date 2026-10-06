"""Real-environment policy evaluation, plus offline multi-step latent evaluation."""

import json
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from .envs import ActionCodec, frame, make_env, step
from .planner import MPCPolicy
from .train import check_dataset, load_checkpoint, resolve_device
from .data import split_episodes


def evaluate(
    checkpoint,
    episodes=10,
    seed=100000,
    device="auto",
    random_policy=False,
    video=None,
    planner_overrides=None,
):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    device = resolve_device(device)
    policy = MPCPolicy.from_checkpoint(checkpoint, device, seed, **(planner_overrides or {}))
    env = make_env(policy.cfg)
    writer = None
    rows = []
    try:
        if ActionCodec.from_space(env.action_space).spec != policy.codec.spec:
            raise ValueError("Environment action space differs from checkpoint")
        if video:
            import imageio.v2 as imageio

            writer = imageio.get_writer(video, fps=env.metadata.get("render_fps", 30))
        for episode in range(episodes):
            env.reset(seed=seed + episode)
            env.action_space.seed(seed + episode)
            policy.reset()
            total, latencies = 0.0, []
            for t in range(policy.cfg.max_steps):
                pixels = frame(env, policy.cfg.image_size)
                if writer and episode == 0:
                    writer.append_data(pixels)
                start = time.perf_counter()
                action = env.action_space.sample() if random_policy else policy.act(pixels)
                latencies.append(time.perf_counter() - start)
                reward, terminated, truncated = step(env, action, policy.cfg.action_repeat)
                total += reward
                if terminated or truncated:
                    break
            rows.append(
                {
                    "seed": seed + episode,
                    "return": total,
                    "steps": t + 1,
                    "terminated": terminated,
                    "truncated": truncated or (t + 1 == policy.cfg.max_steps and not terminated),
                    "mean_action_seconds": float(np.mean(latencies)),
                }
            )
    finally:
        env.close()
        if writer:
            writer.close()
    returns = np.array([row["return"] for row in rows])
    return {
        "policy": "random" if random_policy else "cem",
        "checkpoint": str(checkpoint),
        "planner": {k: getattr(policy.cfg, k) for k in ("horizon", "population", "iterations")},
        "episodes": rows,
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "standard_error": float(returns.std(ddof=1) / np.sqrt(episodes)) if episodes > 1 else None,
    }


@torch.inference_mode()
def evaluate_model(checkpoint, data, device="auto", horizon=10, max_windows=100):
    """Roll out recorded actions on held-out episodes, without teacher forcing."""
    if horizon < 1 or max_windows < 1:
        raise ValueError("horizon and max_windows must be positive")
    device = resolve_device(device)
    model, cfg, saved = load_checkpoint(checkpoint, device)
    manifest_text = (Path(data) / "manifest.json").read_text()
    manifest = json.loads(manifest_text)
    check_dataset(cfg, manifest)
    if hashlib.sha256(manifest_text.encode()).hexdigest() != saved["dataset_fingerprint"]:
        raise ValueError("Use the training dataset to reproduce its held-out episode split")
    if manifest["action_spec"] != saved["action_spec"]:
        raise ValueError("Dataset action specification differs from checkpoint")
    _, episodes = split_episodes(manifest, cfg.seed, cfg.validation_fraction)
    latent, reward, termination = [], [], []
    for episode in episodes:
        root = Path(data) / episode["name"]
        pixels = np.load(root / "pixels.npy", mmap_mode="r")
        actions = np.load(root / "actions.npy", mmap_mode="r")
        rewards = np.load(root / "rewards.npy", mmap_mode="r")
        terms = np.load(root / "terminated.npy", mmap_mode="r")
        for start in range(0, len(actions) - cfg.history - horizon + 2, horizon):
            offset = start + cfg.history - 1
            p = torch.tensor(np.array(pixels[start : offset + horizon + 1]), device=device)[None]
            truth = model.encode(p)
            z = truth[:, : cfg.history]
            a = torch.tensor(np.array(actions[start:offset]), device=device)[None]
            le, re, te = [], [], []
            for t in range(horizon):
                current = torch.tensor(np.array(actions[offset + t]), device=device)[None, None]
                a = torch.cat([a, current], 1)[:, -cfg.history :]
                prediction = model.predict(z[:, -cfg.history :], a)[:, -1:]
                r, done = model.outcomes(z[:, -1:], current, prediction)
                le.append(
                    (prediction - truth[:, cfg.history + t : cfg.history + t + 1])
                    .square()
                    .mean()
                    .item()
                )
                re.append((r.item() - float(rewards[offset + t])) ** 2)
                te.append((done.sigmoid().item() - float(terms[offset + t])) ** 2)
                z = torch.cat([z, prediction], 1)[:, -cfg.history :]
                a = a[:, -(cfg.history - 1) :] if cfg.history > 1 else a[:, :0]
            latent.append(le)
            reward.append(re)
            termination.append(te)
            if len(latent) >= max_windows:
                break
        if len(latent) >= max_windows:
            break
    if not latent:
        raise ValueError("No validation episodes long enough for requested rollout horizon")
    return {
        "windows": len(latent),
        "horizon": horizon,
        "latent_mse_by_step": np.mean(latent, 0).tolist(),
        "reward_rmse_by_step": np.sqrt(np.mean(reward, 0)).tolist(),
        "termination_brier_by_step": np.mean(termination, 0).tolist(),
    }
