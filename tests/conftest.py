import torch
import pytest
from world_model.config import Config


@pytest.fixture(autouse=True)
def small_cpu_threads():
    torch.set_num_threads(1)


@pytest.fixture
def tiny_config():
    return Config(
        env_id="Pendulum-v1",
        image_size=16,
        patch_size=8,
        dim=24,
        encoder_depth=1,
        encoder_heads=3,
        predictor_depth=1,
        predictor_heads=2,
        predictor_head_dim=12,
        mlp_dim=48,
        sigreg_projections=8,
        history=3,
        batch_size=4,
        episodes=4,
        max_steps=8,
        epochs=1,
        horizon=3,
        population=16,
        elites=4,
        iterations=3,
        candidate_chunk=8,
    )
