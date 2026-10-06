import torch
from world_model.config import Config
from world_model.envs import ActionCodec
from world_model.model import WorldModel
from world_model.planner import CEM


def test_continuous_cem_finds_optimum():
    cfg = Config(horizon=4, population=256, elites=32, iterations=8)
    codec = ActionCodec({"kind": "box", "low": [-2], "high": [2], "shape": [1], "dtype": "float32"})
    cem = CEM(cfg, codec, seed=1)
    action = cem.plan(lambda candidates: -(candidates - 0.65).square().sum((1, 2)))
    assert abs(action.item() - 0.65) < 0.1
    assert -1 <= action.item() <= 1


def test_discrete_cem_uses_categories():
    cfg = Config(horizon=3, population=128, elites=16, iterations=5)
    codec = ActionCodec({"kind": "discrete", "n": 3, "start": 0})
    action = CEM(cfg, codec, seed=2).plan(lambda x: x[..., 2].sum(1))
    assert torch.equal(action, torch.tensor([0.0, 0.0, 1.0]))


def test_rollout_action_alignment_and_survival():
    class FakeModel:
        history = 3

        def predict(self, z, actions):
            assert z.shape[1] == actions.shape[1]
            return z + actions

        def outcomes(self, z, actions, next_z):
            # Reward on transition before termination; then 50% survival.
            return next_z[..., 0], torch.zeros_like(next_z[..., 0])

    # Already executed actions 1,2; current state 13. Candidate actions 3,4,5.
    score = WorldModel.imagine(
        FakeModel(),
        torch.tensor([[[10.0], [11.0], [13.0]]]),
        torch.tensor([[[1.0], [2.0]]]),
        torch.tensor([[[3.0], [4.0], [5.0]]]),
        discount=1.0,
    )
    assert torch.allclose(score, torch.tensor([16 + 0.5 * 20 + 0.25 * 25]))
