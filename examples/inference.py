"""python -m examples.inference --checkpoint runs/pendulum/best.pt"""

import argparse
from world_model.envs import frame, make_env, step
from world_model.planner import MPCPolicy
from world_model.train import resolve_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    policy = MPCPolicy.from_checkpoint(args.checkpoint, resolve_device(args.device))
    env = make_env(policy.cfg)
    try:
        env.reset(seed=12345)
        policy.reset()
        total = 0.0
        for _ in range(policy.cfg.max_steps):
            action = policy.act(frame(env, policy.cfg.image_size))
            reward, terminated, truncated = step(env, action, policy.cfg.action_repeat)
            total += reward
            if terminated or truncated:
                break
        print(f"Episode return: {total:.3f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
