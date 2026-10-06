"""Run real CPU collection/training/evaluation in all three bundled environments.

These tiny runs check plumbing, not control quality. Outputs stay out of git.
"""

import argparse
import json
from pathlib import Path
import torch
from world_model.config import Config
from world_model.data import collect
from world_model.evaluate import evaluate, evaluate_model
from world_model.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/smoke")
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    reports = {}
    for name in ("cartpole", "pendulum", "mountain_car"):
        cfg = Config.load(Path(__file__).resolve().parents[1] / "configs" / f"smoke_{name}.json")
        data, run = root / name / "data", root / name / "train"
        collect(cfg, data)
        checkpoint = train(cfg, data, run, "cpu")
        reports[name] = {
            "config": cfg.to_dict(),
            "cem": evaluate(checkpoint, episodes=2, device="cpu"),
            "random": evaluate(checkpoint, episodes=2, device="cpu", random_policy=True),
            "model": evaluate_model(checkpoint, data, horizon=3, max_windows=10),
        }
        (root / "report.json").write_text(json.dumps(reports, indent=2))
    print(f"Smoke results: {root / 'report.json'}")


if __name__ == "__main__":
    main()
