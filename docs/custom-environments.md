# Bring your own Gym-style environment

The bundled DotReach renderer uses NumPy only. Keep your custom environment's renderer
free of pygame/SDL to preserve this project's dependency policy; installing another
environment package can add dependencies independently of this repository. After
installing one, run `python scripts/check_no_pygame.py` in the same venv.

## Required contract

```python
env = factory(render_mode="rgb_array", **env_kwargs)
observation, info = env.reset(seed=123)
observation, reward, terminated, truncated, info = env.step(action)
image = env.render()  # NumPy uint8 array [height, width, 3]
env.close()
```

Use Gymnasium `spaces.Discrete` (including nonzero starts) or finite floating-point `spaces.Box` (including multidimensional shapes). Dict/Tuple/MultiDiscrete and unbounded Box actions require a wrapper. Make rendering deterministic with respect to state where practical. The camera should expose task-relevant objects, goals, and motion; never change image geometry between collection and deployment.

The returned observation may be a vector, dict, or image: this pipeline ignores it and trains exclusively on `render()`. Rewards must be finite scalars. Episodes must end explicitly, or collection ends at the configured `max_steps` as a truncation.

An older Gym environment returning `(obs, reward, done, info)` needs an adapter that distinguishes true termination from `info['TimeLimit.truncated']`. Implement or use a compatibility wrapper; do not label every time limit as terminal. Convert its spaces to Gymnasium spaces too.

## Registered environment

Install your environment package in the venv and ensure it registers its ID. A small factory module can import the registration package and call `gymnasium.make`. This is preferable to relying on registration side effects from an unrelated notebook.

## Direct factory (recommended)

Create an importable module, for example `my_robot/factory.py`:

```python
from my_robot.sim import RobotEnv

def make_env(render_mode="rgb_array", **kwargs):
    return RobotEnv(render_mode=render_mode, **kwargs)
```

Use a config:

```json
{
  "env_id": "MyRobot-v0",
  "factory": "my_robot.factory:make_env",
  "env_kwargs": {"task": "reach", "camera": "front"},
  "image_size": 64,
  "history": 4,
  "action_repeat": 1,
  "max_steps": 300,
  "episodes": 500,
  "epochs": 100
}
```

`env_id` remains a descriptive identity when a factory is supplied. `env_kwargs` is passed unchanged. Your package must be importable in **every** collection/training/evaluation machine. Use `pip install -e /path/to/my_robot` during development or publish/install a versioned package. Pin simulator/environment versions for repeatability.

The included `world_model.example_env:make_env` factory uses a rendered blue dot moving toward a green goal; it has no optional physics dependencies. The example is installed with this package and works from any working directory. Install your own environment as a package for the same portability.

## Checklist before a long run

1. Use a smoke-sized config and collect a handful of episodes.
2. Inspect RGB arrays and verify that action `t` moves frame `t` to `t+1`.
3. Check true terminal flags separately from timeouts.
4. Run one short training job, reload, and evaluate both MPC and random actions.
5. Increase data coverage and model capacity; inspect held-out errors and real returns.

Changing camera, action scaling, frame history, or environment behavior after training can invalidate a checkpoint even when tensor shapes still match. Environment kwargs, image size, action repeat, and action specs are checked automatically; custom code semantics remain your responsibility.
