"""The upstream JEPA/ARPredictor/SIGReg, with Gym-specific outcome heads."""

import torch
from torch import nn
from transformers import ViTConfig, ViTModel
from ._vendor.lewm.jepa import JEPA
from ._vendor.lewm.module import ARPredictor, Embedder, MLP, SIGReg


def preprocess(pixels):
    """uint8 (..., H, W, C) -> float (..., C, H, W), consistently in [-1, 1]."""
    return pixels.movedim(-1, -3).float().div(127.5).sub(1)


class WorldModel(nn.Module):
    def __init__(self, cfg, action_dim):
        super().__init__()
        d = cfg.dim
        self.history = cfg.history
        encoder = ViTModel(
            ViTConfig(
                image_size=cfg.image_size,
                patch_size=cfg.patch_size,
                hidden_size=d,
                num_hidden_layers=cfg.encoder_depth,
                num_attention_heads=cfg.encoder_heads,
                intermediate_size=4 * d,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
            ),
            add_pooling_layer=False,
        )
        self.jepa = JEPA(
            encoder=encoder,
            predictor=ARPredictor(
                num_frames=cfg.history,
                depth=cfg.predictor_depth,
                heads=cfg.predictor_heads,
                dim_head=cfg.predictor_head_dim,
                mlp_dim=cfg.mlp_dim,
                input_dim=d,
                hidden_dim=d,
            ),
            action_encoder=Embedder(input_dim=action_dim, emb_dim=d),
            projector=MLP(d, cfg.mlp_dim, d, norm_fn=nn.BatchNorm1d),
            pred_proj=MLP(d, cfg.mlp_dim, d, norm_fn=nn.BatchNorm1d),
        )
        # Heads receive (z_t, a_t, predicted z_{t+1}); reward belongs to this transition.
        self.outcome = nn.Sequential(nn.Linear(2 * d + action_dim, d), nn.GELU(), nn.Linear(d, 2))
        self.sigreg = SIGReg(num_proj=cfg.sigreg_projections)
        self.register_buffer("reward_mean", torch.tensor(0.0))
        self.register_buffer("reward_std", torch.tensor(1.0))

    def encode(self, pixels):
        return self.jepa.encode({"pixels": preprocess(pixels)})["emb"]

    def predict(self, z, actions):
        return self.jepa.predict(z, self.jepa.action_encoder(actions))

    def outcomes(self, z, actions, next_z):
        out = self.outcome(torch.cat([z, actions, next_z], dim=-1))
        return out[..., 0] * self.reward_std + self.reward_mean, out[..., 1]

    def losses(self, batch, sigreg_weight):
        z = self.encode(batch["pixels"])
        pred = self.predict(z[:, :-1], batch["actions"])
        prediction = (pred - z[:, 1:]).square().mean()  # upstream: no stop gradient
        regularizer = self.sigreg(z.transpose(0, 1))
        # Detached features keep the representation objective exactly JEPA + SIGReg.
        features = torch.cat([z[:, :-1].detach(), batch["actions"], pred.detach()], -1)
        outcome = self.outcome(features)
        target = (batch["rewards"] - self.reward_mean) / self.reward_std
        reward = (outcome[..., 0] - target).square().mean()
        termination = nn.functional.binary_cross_entropy_with_logits(
            outcome[..., 1], batch["terminated"].float()
        )
        total = prediction + sigreg_weight * regularizer + reward + termination
        return {
            "loss": total,
            "prediction": prediction,
            "sigreg": regularizer,
            "reward": reward,
            "termination": termination,
            "latent_std": z.detach().std(dim=(0, 1)).mean(),
        }

    @torch.no_grad()
    def imagine(self, z, past_actions, candidates, discount=0.99):
        """z: (1,L,D); past_actions: (1,L-1,A); candidates: (N,H,A).

        Each candidate is aligned with the most recent observation. History contains
        only actions that have already been executed. Returns discounted rewards,
        weighting future rewards by survival probability after each transition.
        """
        n, horizon, _ = candidates.shape
        z = z.expand(n, -1, -1)
        actions = past_actions.expand(n, -1, -1)
        score = torch.zeros(n, device=z.device)
        survival = torch.ones_like(score)
        for t in range(horizon):
            action = candidates[:, t : t + 1]
            actions = torch.cat([actions, action], dim=1)[:, -self.history :]
            next_z = self.predict(z[:, -self.history :], actions)[:, -1:]
            reward, done_logit = self.outcomes(z[:, -1:], action, next_z)
            score += discount**t * survival * reward[:, 0]
            survival *= 1 - done_logit[:, 0].sigmoid()
            z = torch.cat([z, next_z], dim=1)[:, -self.history :]
            actions = actions[:, -(self.history - 1) :] if self.history > 1 else actions[:, :0]
        return score
