import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train LeWorldModel from Gymnasium RGB frames")
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect", help="Collect episode-separated image data")
    collect.add_argument("--config", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--checkpoint", help="Collect MPC trajectories instead of random actions")
    collect.add_argument("--device", default="auto")
    train = sub.add_parser("train", help="Train and save resumable checkpoints")
    train.add_argument("--config", required=True)
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--resume")
    train.add_argument("--device", default="auto")
    for command in ("eval", "eval-model"):
        p = sub.add_parser(command)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--device", default="auto")
        p.add_argument("--output", required=True)
        if command == "eval":
            p.add_argument("--episodes", type=int, default=10)
            p.add_argument("--seed", type=int, default=100000)
            p.add_argument("--random", action="store_true")
            p.add_argument("--video")
            for name in ("horizon", "population", "elites", "iterations", "candidate-chunk"):
                p.add_argument(f"--{name}", type=int)
        else:
            p.add_argument("--data", required=True)
            p.add_argument("--horizon", type=int, default=10)
            p.add_argument("--max-windows", type=int, default=100)
    args = parser.parse_args()
    if args.command == "collect":
        from .config import Config
        from .data import collect

        cfg = Config.load(args.config)
        policy = None
        if args.checkpoint:
            from .planner import MPCPolicy
            from .train import resolve_device

            policy = MPCPolicy.from_checkpoint(
                args.checkpoint, resolve_device(args.device), cfg.seed
            )
            for key in ("env_id", "factory", "env_kwargs", "image_size", "action_repeat"):
                if getattr(cfg, key) != getattr(policy.cfg, key):
                    raise ValueError(f"Collection config/checkpoint mismatch: {key}")
        result = collect(cfg, args.output, policy)
        print(f"Collected {len(result['episodes'])} episodes into {args.output}")
    elif args.command == "train":
        from .config import Config
        from .train import train

        train(Config.load(args.config), args.data, args.output, args.device, args.resume)
    else:
        from .evaluate import evaluate, evaluate_model

        output = Path(args.output)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        if args.command == "eval":
            overrides = {
                key: getattr(args, key)
                for key in ("horizon", "population", "elites", "iterations", "candidate_chunk")
                if getattr(args, key) is not None
            }
            result = evaluate(
                args.checkpoint,
                args.episodes,
                args.seed,
                args.device,
                args.random,
                args.video,
                overrides,
            )
        else:
            result = evaluate_model(
                args.checkpoint, args.data, args.device, args.horizon, args.max_windows
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
