# Validation performed

Updated on 2026-10-06 after removing the pygame dependency and classic-control environment presets. Local execution used Python 3.12, PyTorch 2.6.0+cpu, Gymnasium 1.1.1, and Transformers 4.49.0. GitHub CI is configured for Python 3.11. No GPU was available locally.

## Dependency policy

- Installation uses core Gymnasium, without environment extras.
- `python -m pip check`: no broken requirements.
- `python scripts/check_no_pygame.py`: passed after pygame was uninstalled. No pygame/pygame-ce distribution, importable pygame module, or SDL2 shared library was found in the active environment's site-packages.
- CI runs the same absence check immediately after installation in a fresh environment.
- DotReach renders RGB arrays directly using NumPy. Rendering requires no pygame, SDL, display server, or audio library.

This verification covers the repository's configured Python environment, not arbitrary packages installed later or libraries elsewhere on a machine. The reported Windows malware alert was not investigated or classified as a true/false positive.

## Tests

`pytest -q`: **12 passed**, with pygame uninstalled. Coverage includes the default NumPy renderer, action encoding, episode-separated image collection, termination/truncation boundaries, continuous and categorical CEM optimization, rollout timing, detached outcome heads, training, reload, resume, and custom-factory CLI collection from another working directory.

`ruff check .` and `ruff format --check .` passed. Vendored upstream source is excluded from formatting to preserve exact bytes.

## Actual smoke training and evaluation

Command: `python scripts/smoke.py --output runs/no_pygame_smoke`.

The bundled DotReach environment completed collection, two training epochs, checkpoint loading, two-episode CEM and random-policy evaluations, and held-out autoregressive model evaluation. It used 8 random collection episodes capped at 40 transitions, 32×32 RGB, a one-layer 48-dimensional ViT and one-layer predictor. CEM used horizon 3, 16 candidates, four elites, and two iterations.

Full per-episode results, config, and model errors are in [smoke-results.json](smoke-results.json). These short runs verify execution, not task solving or statistically meaningful policy improvement. Low latent variation and prediction errors require further training and evaluation before practical use.

The runtime emitted CPU-info/NNPACK fallback warnings; execution completed successfully. No pygame import/deprecation warning occurred in the updated tests.

## Scope

Long-run performance, GPU throughput, and Windows/macOS execution have not been validated here. Custom simulators may introduce their own dependencies; rerun the absence check after installing one. Prior classic-control smoke results have been replaced with current DotReach results because those environment presets are no longer included. Datasets/checkpoints remain gitignored and can be reproduced with the smoke script.
