# Upstream integration and attribution

Official repository: https://github.com/lucas-maes/le-wm

Pinned commit: `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`.

`src/world_model/_vendor/lewm/jepa.py` and `module.py` are unmodified upstream Python sources. Their MIT license is retained alongside them. `UPSTREAM.json` records the origin and revision. They are imported directly by this project's `model.py`; this is not merely a README link to an unused dependency.

## What is reused

- JEPA encoding and prediction implementation.
- Action-conditioned autoregressive transformer (`ARPredictor`).
- Action embedder and projection MLPs.
- Single-device SIGReg implementation and end-to-end prediction objective.

## What this project adds or changes

- A direct Hugging Face ViT construction instead of `stable_pretraining`'s factory.
- Small RGB Gymnasium image defaults (64 pixels; smoke 32), patch size 8, and explicit `[-1,1]` preprocessing instead of upstream's 224-pixel experiment pipeline.
- Configurable predictor sizes and a reduced default SIGReg projection count for accessible experiments. Upstream's paper configs can be inspected separately; these presets are not exact reproductions.
- Plain PyTorch training, NumPy episode data, JSON config, and portable state-dict checkpoints instead of Hydra/Lightning/stable-worldmodel storage conventions.
- Reward and termination heads for generic reward-based RL, with detached features.
- Continuous and discrete CEM over predicted rewards; upstream uses goal-embedding distance for its goal-conditioned benchmarks.

The pretrained checkpoints linked upstream are for different environments/configurations and are **not directly loadable** with this adapter. This pipeline learns from scratch. Upstream's `get_cost`/`rollout` goal API remains in the vendor snapshot, but the reward planner uses our explicitly aligned `WorldModel.imagine` method.

For an upgrade, review upstream API/loss changes, replace the two source files and license at a chosen immutable commit, update provenance and `UPSTREAM_COMMIT`, then rerun tests and smoke training. Do not silently follow `main`: checkpoint compatibility depends on these definitions.

## Citation

```bibtex
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

[Paper](https://arxiv.org/abs/2603.19312) · [Project page](https://le-wm.github.io/)
