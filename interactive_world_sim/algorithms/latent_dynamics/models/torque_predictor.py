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
torque  : (B, T, torque_dim)       per-frame joint torque (raw physical N·m)
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


class TrunkTorqueHead(nn.Module):
    """Per-frame torque head that branches off the dynamics U-Net trunk.

    Instead of re-encoding the latent (as ``TorquePredictor`` does), this head
    consumes the spatially-pooled features produced *inside* ``CMLatentDynamics``
    after its spatio-temporal attention (the mid-block bottleneck). Those features
    are already causally temporal-attended, so a per-frame MLP is enough (no GRU).

    Because the trunk runs on *noisy* latents (terminal frame ~ pure noise under
    ``terminal_only`` sampling, context frames near-clean via
    ``prev_frame_noise_scale``), the head is optionally conditioned on the
    per-frame diffusion noise level so it can normalize across the noise regime it
    sees in training (and again at inference, where the same sampler is reused).

    Input
    -----
    feat        : (B, T, feat_dim)   spatially-pooled mid-block features
    noise_level : (B, T) long        per-frame diffusion noise level (optional)

    Output
    ------
    torque      : (B, T, torque_dim) per-frame joint torque (raw physical N·m)
    """

    def __init__(
        self,
        feat_dim: int,
        torque_dim: int = 8,
        hidden_dim: int = 256,
        noise_emb_dim: int = 64,
        timesteps: int = 1000,
        use_noise_cond: bool = True,
    ) -> None:
        super().__init__()
        self.torque_dim = torque_dim
        self.use_noise_cond = use_noise_cond
        in_dim = feat_dim
        if use_noise_cond:
            self.noise_emb = nn.Embedding(timesteps, noise_emb_dim)
            in_dim += noise_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, torque_dim),
        )

    def forward(
        self, feat: torch.Tensor, noise_level: torch.Tensor = None
    ) -> torch.Tensor:
        """feat: (B,T,feat_dim); noise_level: (B,T) -> torque: (B,T,torque_dim)."""
        if self.use_noise_cond:
            assert noise_level is not None, "TrunkTorqueHead needs noise_level"
            ne = self.noise_emb(noise_level.long()).to(feat.dtype)  # (B,T,noise_emb_dim)
            feat = torch.cat([feat, ne], dim=-1)
        return self.mlp(feat)
