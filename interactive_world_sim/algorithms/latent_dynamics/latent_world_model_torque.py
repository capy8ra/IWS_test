"""Stage-2 latent dynamics + joint-torque head (NEW, isolated).

Subclass of ``LatentWorldModel`` that adds a per-frame torque head on top of the
existing CTM latent dynamics. Only the stage-2 path is changed; stages 1/3
delegate to the parent unchanged.

Two interchangeable torque heads are supported, selected by
``cfg.torque_head_source`` (see docs/stage2_dynamics_plan_updated.md):

  * ``"separate"`` (default, original): a standalone ``TorquePredictor`` that
    re-encodes the latents (CNN) + actions (MLP) with its own causal GRU. The
    torque latent sequence is ``[z_gt[0..T-2], z_pred[-1]]`` (clean context +
    predicted terminal), so the terminal-frame gradient flows into the dynamics
    U-Net while the encoder/decoder stay frozen.

  * ``"trunk_midblock"``: an MLP ``TrunkTorqueHead`` that branches off the shared
    U-Net representation -- the mid-block bottleneck features, captured via a
    forward hook *during the existing dynamics forward* (no change to
    ``CMLatentDynamics.forward``'s tensor-only return type, so ``EinopsWrapper``
    and ``CTM_calc_out`` arithmetic are untouched). The torque loss then backprops
    through the whole trunk, shaping a genuinely shared representation. Features
    come from noisy latents, so the head is conditioned on the per-frame noise
    level.

In both cases torque stays OUT of diffusion (it is a deterministic physical
prediction, not a denoised spatial variable) and is supervised densely over all
T frames.
"""
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau

from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import (
    LatentWorldModel,
)
from interactive_world_sim.algorithms.latent_dynamics.models.torque_predictor import (
    TorquePredictor,
    TrunkTorqueHead,
)


class LatentWorldModelTorque(LatentWorldModel):
    """LatentWorldModel + stage-2 right-arm joint-torque prediction."""

    def _build_model(self) -> None:
        super()._build_model()
        self.torque_loss_weight = float(self.cfg.get("torque_loss_weight", 1.0))
        self.torque_head_source = str(self.cfg.get("torque_head_source", "separate"))
        torque_dim = int(self.cfg.torque_dim)
        action_dim = int(self.cfg.action_dim)

        # keep both attributes defined (one stays None) so optimizer / steps can
        # branch on a single flag.
        self.torque_predictor = None
        self.torque_head = None
        self._trunk_feat = None  # set by the mid-block forward hook

        if self.torque_head_source == "separate":
            self.torque_predictor = TorquePredictor(
                latent_dim=self.num_latent_channel,
                action_dim=action_dim,
                torque_dim=torque_dim,
            )
        elif self.torque_head_source == "trunk_midblock":
            # mid-block feature width = dynamics.dim * dynamics.dim_mults[-1]
            dyn_dim = int(self.cfg.dynamics.dim)
            dyn_mults = list(self.cfg.dynamics.dim_mults)
            feat_dim = dyn_dim * int(dyn_mults[-1])
            self.torque_head = TrunkTorqueHead(
                feat_dim=feat_dim,
                torque_dim=torque_dim,
                timesteps=int(self.cfg.diffusion.timesteps),
                use_noise_cond=bool(self.cfg.get("torque_noise_cond", True)),
            )
            # capture the bottleneck output during every dynamics forward.
            self.dynamics.module.mid_block.register_forward_hook(
                self._capture_trunk_feat
            )
        else:
            raise ValueError(
                f"Unknown torque_head_source={self.torque_head_source!r}; "
                "expected 'separate' or 'trunk_midblock'."
            )

    # ----- mid-block hook: stash features in EinopsWrapper "b c f h w" layout ----
    def _capture_trunk_feat(self, _module: Any, _inp: Any, out: torch.Tensor) -> None:
        self._trunk_feat = out  # (B, C, T, H, W)

    def _pool_trunk_feat(self, t_levels: torch.Tensor) -> torch.Tensor:
        """(B,C,T,H,W) trunk feat + (T,B) noise levels -> torque (B,T,8)."""
        feat = self._trunk_feat.mean(dim=(-1, -2))  # (B, C, T) spatial GAP
        feat = feat.permute(0, 2, 1)                # (B, T, C)
        return self.torque_head(feat, noise_level=t_levels.transpose(0, 1))

    @property
    def _torque_module(self) -> torch.nn.Module:
        return (
            self.torque_head
            if self.torque_head_source == "trunk_midblock"
            else self.torque_predictor
        )

    # ----- optimizer: stage-2 optimizes dynamics + torque head -----
    def configure_optimizers(self) -> Any:
        if self.training_stage != 2:
            return super().configure_optimizers()
        param_groups = [
            {"params": self.dynamics.parameters(), "lr": self.cfg.lr},
            {"params": self._torque_module.parameters(), "lr": self.cfg.lr},
        ]
        optimizer = torch.optim.AdamW(
            params=param_groups,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.optimizer_beta,
        )
        if self.lr_scheduler == "linear":
            lr_scheduler = LinearLR(
                optimizer, start_factor=1e-4, end_factor=1.0,
                total_iters=self.cfg.warmup_steps,
            )
        elif self.lr_scheduler == "plateau":
            lr_scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=50000,
                threshold=1e-3, threshold_mode="rel",
            )
        else:
            raise NotImplementedError(f"LR scheduler {self.lr_scheduler} not included")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler, "interval": "step", "frequency": 1,
                "monitor": "training/loss", "strict": True, "name": "lr_scheduler",
            },
        }

    def _torque_target(self, batch: dict) -> torch.Tensor:
        # raw torque in N·m -- intentionally NOT normalized, so the head's output
        # is directly usable as actual torque (no normalizer stats needed later).
        return batch["joint_torque"].float()

    # ----- training: stage-2 latent loss + dense torque loss -----
    def training_step(self, batch: dict, batch_idx: int) -> STEP_OUTPUT:
        if self.training_stage != 2:
            return super().training_step(batch, batch_idx)
        if batch["obs"][self.obs_keys[0]].shape[0] == 0:
            return None

        obs = torch.cat(
            [self.normalizer[k].normalize(batch["obs"][k]) for k in self.obs_keys], dim=2
        ).float()
        action_bt = self.normalizer["action"].normalize(batch["action"]).float()  # (B,T,A)
        b = obs.shape[0]
        xs = rearrange(obs, "b t c h w -> (b t) c h w")

        with torch.no_grad():
            z = self.encoder_forward(xs)  # frozen encoder
        z = rearrange(z, "(b t) c h w -> t b c h w", b=b)
        action = rearrange(action_bt, "b t a -> t b a")

        t, s = self._generate_noise_levels(z, self.dyn_infer_steps)
        weights_t = self.noise_scheduler.get_weights(t)
        weights_s = self.noise_scheduler.get_weights(s)
        noisy_z_t, noisy_z_s = self.noise_scheduler.add_noise_to_t_s(z, t, s)
        u = torch.zeros_like(t).to(self.device)
        if self.mask_prev_action:
            action = action.clone()
            action[:-1] = 0
        pred_s = self._forward(self.dynamics, noisy_z_t, t, s, external_cond=action)
        # capture trunk features from THIS (pred_s) forward before any other call.
        trunk_feat_seen = self._trunk_feat
        pred_u = None
        if self.dyn_infer_steps > 1:
            pred_u = self._forward(self.dynamics, noisy_z_s, s, u, external_cond=action)

        # ---- latent dynamics loss (faithful to parent stage-2) ----
        if self.last_frame_loss_only:
            loss_s = F.mse_loss(pred_s[-1:], noisy_z_s[-1:].detach(), reduction="none")
            wt = weights_t.view(*weights_t.shape, *((1,) * (loss_s.ndim - 2)))[-1:]
            loss = loss_s * wt
            if pred_u is not None:
                loss_u = F.mse_loss(pred_u[-1:], z[-1:].detach(), reduction="none")
                ws = weights_s.view(*weights_s.shape, *((1,) * (loss_u.ndim - 2)))[-1:]
                loss = loss + loss_u * ws
        else:
            loss_s = F.mse_loss(pred_s, noisy_z_s.detach(), reduction="none")
            wt = weights_t.view(*weights_t.shape, *((1,) * (loss_s.ndim - 2)))
            loss = loss_s * wt
            if pred_u is not None:
                loss_u = F.mse_loss(pred_u, z.detach(), reduction="none")
                ws = weights_s.view(*weights_s.shape, *((1,) * (loss_s.ndim - 2)))
                loss = loss + loss_u * ws
        dyn_loss = loss.mean()

        # ---- dense torque loss ----
        if self.torque_head_source == "trunk_midblock":
            # branch off the shared trunk features from the pred_s forward.
            self._trunk_feat = trunk_feat_seen
            torque_pred = self._pool_trunk_feat(t)  # (B,T,8)
            self._trunk_feat = None  # release graph ref
        else:
            # context = clean latents z[0..T-2]; terminal = predicted latent z_pred[-1]
            torque_latents = torch.cat([z[:-1], pred_s[-1:]], dim=0)  # (T,B,C,H,W)
            torque_latents = rearrange(torque_latents, "t b c h w -> b t c h w")
            torque_pred = self.torque_predictor(torque_latents, action_bt)  # (B,T,8)
        torque_target = self._torque_target(batch)  # raw N·m
        torque_loss = F.mse_loss(torque_pred, torque_target)  # in N·m^2

        total_loss = dyn_loss + self.torque_loss_weight * torque_loss
        self.log("training/loss", total_loss)
        self.log("training/dyn_loss", dyn_loss)
        self.log("training/torque_mse", torque_loss)
        # predictions are already in N·m -> RMSE is just sqrt(MSE).
        self.log("training/torque_rmse_nm", torch.sqrt(torque_loss.detach()))
        return {"loss": total_loss}

    # ----- validation: parent latent metrics + torque metric -----
    def _val_torque_pred(
        self, z_btchw: torch.Tensor, action_bt: torch.Tensor
    ) -> torch.Tensor:
        """Teacher-forced torque over the (possibly long) val sequence.

        For the separate head this runs once on clean latents. For the trunk head
        we must run the dynamics to populate the mid-block features, so we replay
        the training-style (terminal_only) noise+forward in non-overlapping
        windows of ``n_tokens`` (matching the trained context length) and stitch
        the per-frame torque predictions back together. Full autoregressive-rollout
        torque (using predicted latents as context) remains TODO.
        """
        if self.torque_head_source != "trunk_midblock":
            return self.torque_predictor(z_btchw, action_bt)

        z_tb = rearrange(z_btchw, "b t c h w -> t b c h w")
        action_tb = rearrange(action_bt, "b t a -> t b a")
        total = z_tb.shape[0]
        outs = []
        for start in range(0, total, self.n_tokens):
            end = min(start + self.n_tokens, total)
            zc, ac = z_tb[start:end], action_tb[start:end]
            t, s = self._generate_noise_levels(zc, self.dyn_infer_steps)
            noisy_zt, _ = self.noise_scheduler.add_noise_to_t_s(zc, t, s)
            _ = self._forward(self.dynamics, noisy_zt, t, s, external_cond=ac)
            outs.append(self._pool_trunk_feat(t))  # (B, t_chunk, 8)
        self._trunk_feat = None
        return torch.cat(outs, dim=1)  # (B, T, 8)

    def validation_step(
        self, batch: dict, batch_idx: int, namespace: str = "validation"
    ) -> STEP_OUTPUT:
        super().validation_step(batch, batch_idx, namespace)
        if self.training_stage != 2:
            return None
        with torch.no_grad():
            obs = torch.cat(
                [self.normalizer[k].normalize(batch["obs"][k]) for k in self.obs_keys],
                dim=2,
            ).float()
            action_bt = self.normalizer["action"].normalize(batch["action"]).float()
            b = obs.shape[0]
            xs = rearrange(obs, "b t c h w -> (b t) c h w")
            z = self.encoder_forward(xs)
            z = rearrange(z, "(b t) c h w -> b t c h w", b=b)
            torque_pred = self._val_torque_pred(z, action_bt)  # raw N·m
            torque_target = self._torque_target(batch)
            tmse = F.mse_loss(torque_pred, torque_target)  # in N·m^2
            trmse = torch.sqrt(tmse)  # predictions already in N·m
        self.log(f"{namespace}/torque_mse", tmse)
        self.log(f"{namespace}/torque_rmse_nm", trmse)
        return None
