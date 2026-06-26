"""Per-frame joint-torque predictor for stage-2 (NEW, isolated).

A small deterministic temporal head that maps a sequence of latent maps + actions
(``q_des``) to per-frame joint torques. It is intentionally kept OUT of the
diffusion U-Net: ``CMLatentDynamics.forward`` stays tensor-only (its output is a
``v``-parameterization consumed by ``CTM_calc_out`` arithmetic), so torque -- a
deterministic physical quantity -- is predicted separately from the predicted /
clean latents.

Input
-----
latents : (B, T, C_latent, H, W)   clean context latents + predicted terminal
actions : (B, T, A)                right-arm q_des (8 incl. gripper)

Output
------
torque  : (B, T, torque_dim)       per-frame joint torque (normalized space)
"""
import torch
from torch import nn


class TorquePredictor(nn.Module):
    """Latent+action -> per-frame torque via a causal GRU."""

    def __init__(
        self,
        latent_dim: int = 4,
        action_dim: int = 8,
        torque_dim: int = 8,
        latent_feat_dim: int = 128,
        action_feat_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.torque_dim = torque_dim
        # per-frame latent encoder: [B, C, 32, 32] -> [B, latent_feat_dim]
        self.latent_enc = nn.Sequential(
            nn.Conv2d(latent_dim, 32, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),          # 16 -> 8
            nn.SiLU(),
            nn.Conv2d(64, latent_feat_dim, kernel_size=3, stride=2, padding=1),  # 8 -> 4
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),  # -> [B, latent_feat_dim, 1, 1]
        )
        # per-frame action encoder
        self.action_enc = nn.Sequential(
            nn.Linear(action_dim, action_feat_dim),
            nn.SiLU(),
            nn.Linear(action_feat_dim, action_feat_dim),
        )
        # causal temporal model over the window
        self.gru = nn.GRU(
            input_size=latent_feat_dim + action_feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, torque_dim)

    def forward(self, latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """latents: (B,T,C,H,W); actions: (B,T,A) -> torque: (B,T,torque_dim)."""
        b, t = latents.shape[:2]
        z = self.latent_enc(latents.flatten(0, 1))  # (B*T, F, 1, 1)
        z = z.reshape(b, t, -1)                       # (B, T, latent_feat_dim)
        a = self.action_enc(actions)                  # (B, T, action_feat_dim)
        x = torch.cat([z, a], dim=-1)                 # (B, T, F+A)
        h, _ = self.gru(x)                            # (B, T, hidden) -- causal
        return self.head(h)                           # (B, T, torque_dim)
