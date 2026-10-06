# Validation performed

Executed on 2026-10-06 using Python 3.12, PyTorch 2.6.0+cpu, Gymnasium 1.1.1, and Transformers 4.49.0. No GPU was available. See `requirements-tested-cpu.txt` for the full dependency snapshot.

## Tests

`pytest -q`: **10 passed**. Tests exercise bounded/multidimensional Box and nonzero-start Discrete action round trips, action repeat at truncation, real Pendulum image collection, nonoverlapping episode splits, continuous and categorical CEM optimization, action/latent rollout timing, termination survival weighting, detached outcome heads, a real optimizer run, checkpoint reload/resume, and closed-loop inference.

`ruff check .` and `ruff format --check .` are also required by CI. Vendored upstream files are intentionally excluded from formatting to preserve their exact source bytes.

## Actual smoke training and evaluation

Command: `python scripts/smoke.py --output runs/verified_smoke`.

Each environment used 8 random collection episodes, at most 40 transitions per episode, two epochs, 32×32 RGB, a one-layer 48-dimensional ViT and one-layer predictor. CEM used horizon 3, 16 candidates, four elites, and two iterations. Evaluation used the two fresh seeds 100000 and 100001 with the same 40-step cap. These runs validate execution, **not task solving or statistically meaningful policy improvement**.

| Environment | Mean MPC return | Mean random return |
|---|---:|---:|
| CartPole-v1 | 9.50 | 24.50 |
| Pendulum-v1 | -211.01 | -299.99 |
| MountainCarContinuous-v0 | -3.88 | -1.57 |

Full per-episode returns, rollout errors, planner settings, and measured latencies are in [`smoke-results.json`](smoke-results.json). The environment horizon caps differ from full benchmarks. MPC was worse than random on CartPole and MountainCar in these tiny runs; the Pendulum difference across two seeds is not sufficient evidence of reliable improvement. Low held-out latent variation in these short runs also warrants longer training and representation diagnostics.

The first smoke attempt exposed an incomplete manifest during final model evaluation. Collection was changed to finalize the manifest atomically, and the complete three-environment smoke run was repeated successfully. The environment emitted CPU-info/NNPACK fallback warnings; computation completed. Pytest also reports a pygame/pkg_resources deprecation warning.

## Not yet established

- Long-run performance of the default model recipes or solved-task scores.
- GPU throughput, CUDA resume determinism, or multi-device training (not implemented).
- Generalization to an arbitrary custom simulator or unseen visual distribution.
- Reliability of long imagined horizons outside the collection distribution.
- Cross-platform execution beyond the tested Linux CPU environment.

Datasets and checkpoints were generated locally and are excluded from source control. Reproduce them with the smoke script, then use the default recipes and larger datasets for actual experiments. GitHub Actions is configured to run the same smoke pipeline; its remote status is independent of the local results above.
