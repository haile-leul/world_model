# world_model

**Train [LeWorldModel](https://github.com/lucas-maes/le-wm) from Gymnasium-rendered images and control environments with cross-entropy method (CEM) model predictive control (MPC).**

The pipeline is explicit: collect RGB trajectories → learn latent dynamics → fit reward/termination heads → evaluate imagined rollouts → run closed-loop MPC. Simulator state observations are never model inputs. Train a separate checkpoint per environment.

This repository includes the **actual upstream `JEPA`, `ARPredictor`, action embedder, projection MLPs, and `SIGReg`**, pinned at commit [`8edfeb3`](https://github.com/lucas-maes/le-wm/tree/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac). A small unmodified source snapshot is included so a normal clone and package install work without submodule setup. See [upstream integration](docs/upstream.md) for provenance and the differences from the paper's experimental setup.

> **Validation status:** CPU end-to-end smoke runs exercise collection, two training epochs, checkpoint loading, CEM evaluation, and latent rollout evaluation on the bundled NumPy-rendered DotReach environment. These tiny models are not trained benchmark policies. Default training recipes require longer runs and task-specific tuning; solved-task performance is not claimed. See [validation](docs/validation.md).

## Quick start

**Recommended: Python 3.11.** GitHub CI runs the tests and DotReach smoke pipeline on Python 3.11. Python 3.12 was also tested locally; the package declares support for Python 3.10–3.12, but Python 3.10 has not been validated here. Python 3.13 and newer are outside the supported range.

Install Python 3.11 and Git before starting. Authenticate with GitHub to clone this private repository.

```bash
git clone https://github.com/haile-leul/world_model.git
cd world_model
python3.11 -m venv .venv                 # Windows: py -3.11 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python --version                        # should report Python 3.11.x
python -m pip install --upgrade pip
python -m pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
python scripts/check_no_pygame.py

# Actual small training/evaluation runs in the NumPy-rendered DotReach environment:
python scripts/smoke.py
```

For an NVIDIA GPU, install the appropriate PyTorch 2.6.0 build for your driver instead of the CPU wheel, then install this package. Commands accept `--device auto` (CUDA when available), `cpu`, or `cuda:0`. macOS CPU works in principle; only Linux CPU has been exercised here. Custom environments need their own simulator dependencies; choose renderers that do not require pygame/SDL if you want to retain this dependency policy.

No model downloads, WandB account, or upstream training framework are needed. The ViT is initialized from scratch. A clean venv avoids unrelated torchvision/transformers version conflicts.

## No pygame / SDL dependency

Installation uses core `gymnasium`, **without** the `classic-control` extra. The pygame dependency, its pinned requirement, and the CartPole, Pendulum, and MountainCarContinuous presets have been removed. Default configuration, tests, and smoke training now use DotReach, which renders RGB images directly with NumPy. No pygame or SDL renderer is needed by the bundled pipeline.

**Existing virtual environments:** pulling this change or reinstalling the project does not remove packages already installed. Create a fresh Python 3.11 venv using the quick-start instructions (choose a new name, such as `.venv-no-pygame`, if `.venv` already exists). If maintaining an existing venv instead, activate that specific venv and run:

```bash
python -m pip uninstall pygame pygame-ce
python scripts/check_no_pygame.py
```

The check fails if either pygame distribution, its importable module, or an SDL2 library remains in that venv. CI runs it after a fresh install. Custom environment packages and optional dependencies installed separately may reintroduce pygame/SDL; inspect their requirements before adding them. This change avoids the reported dependency; it does not determine whether the previous malware alert was a true or false positive.

## Train a useful-sized model

```bash
world-model collect --config configs/custom.json --output data/custom
world-model train --config configs/custom.json --data data/custom --output runs/custom

# Held-out multi-step dynamics/reward prediction:
world-model eval-model --checkpoint runs/custom/best.pt --data data/custom \
  --horizon 10 --output runs/custom/model_eval.json

# Closed-loop MPC and a random baseline on identical fresh seeds:
world-model eval --checkpoint runs/custom/best.pt --episodes 10 --seed 100000 \
  --output runs/custom/mpc_eval.json
world-model eval --checkpoint runs/custom/best.pt --episodes 10 --seed 100000 \
  --random --output runs/custom/random_eval.json
```

These are starting recipes, not tuned hyperparameters. CPU smoke configs are intentionally much smaller. Collection/run/result paths must be new, except when explicitly resuming. Data, checkpoints, and videos are gitignored.

| Config | Environment | Action space | Main caveat |
|---|---|---|---|
| `custom.json` | Included `DotReach` example | Continuous Box | RGB rendering uses NumPy only |
| `smoke_custom.json` | Same `DotReach` example, tiny model | Continuous Box | Short execution check, not a trained benchmark policy |

Change the JSON files to set collection size, image size, model dimensions, batch size, optimizer, training epochs, and CEM settings. Unknown fields fail early. Defaults are defined in [`config.py`](src/world_model/config.py); the fully resolved config is saved in every run and checkpoint.

To resume, increase `epochs` in the same config (it means **total** epochs) and use:

```bash
world-model train --config configs/custom.json --data data/custom \
  --output runs/custom --resume runs/custom/last.pt
```

Resume restores model, optimizer, epoch, RNG, and loader state. Use `last.pt` for uninterrupted training continuity. `best.pt` minimizes held-out prediction + normalized reward + termination loss, **not policy return**. It is not a substitute for evaluating the controller. Keep the original dataset immutable; its manifest fingerprint protects the split and resume path.

## Inference

```bash
python -m examples.inference --checkpoint runs/custom/best.pt
```

The public API is small:

```python
from world_model.planner import MPCPolicy
from world_model.envs import frame, make_env, step

policy = MPCPolicy.from_checkpoint("runs/custom/best.pt", device="cpu")
env = make_env(policy.cfg)
try:
    env.reset(seed=123)
    policy.reset()                         # required at every episode boundary
    for _ in range(policy.cfg.max_steps):
        action = policy.act(frame(env, policy.cfg.image_size))
        reward, terminated, truncated = step(env, action, policy.cfg.action_repeat)
        if terminated or truncated:
            break
finally:
    env.close()
```

`act()` consumes one resized `uint8` RGB frame and returns an action in the environment's original units. For a non-simulator caller, provide `(image_size, image_size, 3)` pixels with the same viewpoint and resizing as training. A policy instance owns one environment's history and must not be shared between parallel episodes.

LeWM predicts **embeddings, not decoded images**. For open-loop inference, use `WorldModel.encode`, `predict`, and `imagine`; [`evaluate_model`](src/world_model/evaluate.py) shows a complete recorded-action rollout. There is no pixel decoder or value function in this implementation.

## CEM MPC

At each environment step, CEM samples action sequences, rolls out latent dynamics, ranks predicted discounted returns, and refits to the best sequences. Only the first action is executed; the next observed frame corrects the state estimate before planning again.

- Continuous actions: bounded Gaussian sampling in normalized `[-1, 1]` coordinates.
- Discrete actions: categorical sampling and elite-frequency updates, without continuous rounding.
- Candidate rollouts are vectorized and chunked; the current frame is encoded once per control step.
- Previous CEM means/probabilities warm-start the next plan; history and plans reset per episode.
- Rewards are weighted by predicted survival. Time-limit truncation never becomes a terminal training target.

Override planning cost/quality tradeoffs without retraining:

```bash
world-model eval --checkpoint runs/custom/best.pt --horizon 20 \
  --population 512 --elites 64 --iterations 6 --candidate-chunk 64 \
  --output runs/custom/larger_planner.json
```

Longer horizons can amplify model error. Increase population for search quality and reduce `candidate_chunk` for memory. Changing `action_repeat` or image preprocessing requires compatible data and retraining.

## Your own environments

Any environment meeting the documented contract can use the same pipeline. It must expose Gymnasium's reset/step API, RGB rendering, and a `Discrete` or finite floating-point `Box` action space. Other spaces need an explicit wrapper. Older Gym's four-return API needs an adapter; it is not silently guessed.

See the complete [custom environment guide](docs/custom-environments.md) and runnable [`DotReach`](examples/custom_env.py):

```bash
world-model collect --config configs/custom.json --output data/custom
world-model train --config configs/custom.json --data data/custom --output runs/custom
world-model eval --checkpoint runs/custom/best.pt --output runs/custom/eval.json
```

## Understand the implementation

| File | Responsibility |
|---|---|
| `envs.py` | Factory loading, RGB resizing, action normalization, action repeat |
| `data.py` | Collection, episode split, lazy memory-mapped sequence windows |
| `model.py` | Upstream LeWM construction, loss, reward/termination heads, imagined returns |
| `train.py` | Optimizer, held-out metrics, checkpoints, resume |
| `planner.py` | CEM search and stateful inference policy |
| `evaluate.py` | Offline model evaluation and real environment evaluation |
| `_vendor/lewm/` | Unmodified pinned upstream source and MIT license |

Read [design and data format](docs/design.md) for tensor alignment, losses, limitations, and how to supply expert trajectories. See [upstream attribution](docs/upstream.md) for the original paper.

## Development

```bash
pytest -q
ruff check .
ruff format --check .
```

Tests cover action round trips, episode boundaries, termination/truncation handling, CEM optimization, rollout alignment, detached outcome heads, training, reload, resume, and closed-loop inference. GitHub Actions runs tests and DotReach smoke training on CPU. Optional videos: `pip install -e '.[video]'`, then add `--video runs/custom/episode.mp4` to `eval`.
