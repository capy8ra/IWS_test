"""Stage-2 latent dynamics + joint-torque head (NEW, isolated).

Subclass of ``LatentWorldModel`` that adds a deterministic per-frame torque head
on top of the existing CTM latent dynamics. Only the stage-2 path is changed;
stages 1/3 delegate to the parent unchanged.

Design (see docs/stage2_dynamics_plan_updated.md):
  * ``CMLatentDynamics.forward`` stays tensor-only -> torque is a SEPARATE module.
  * dense torque supervision: predict torque for all T frames; context frames use
    the clean encoder latents ``z_gt[0..T-2]`` and the terminal frame uses the
    dynamics-predicted latent ``z_pred[-1]`` (so the torque gradient also flows
    into the dynamics U-Net), encoder/decoder stay frozen.
"""
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau

from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import (
    LatentWorldModel,
)
from interactive_world_sim.algorithms.latent_dynamics.models.torque_predictor import (
    TorquePredictor,
)


class LatentWorldModelTorque(LatentWorldModel):
    """LatentWorldModel + stage-2 right-arm joint-torque prediction."""

    def _build_model(self) -> None:
        super()._build_model()
        self.torque_loss_weight = float(self.cfg.get("torque_loss_weight", 1.0))
        self.torque_predictor = TorquePredictor(
            latent_dim=self.num_latent_channel,
            action_dim=int(self.cfg.action_dim),
            torque_dim=int(self.cfg.torque_dim),
        )

    # ----- optimizer: stage-2 optimizes dynamics + torque head -----
    def configure_optimizers(self) -> Any:
        if self.training_stage != 2:
            return super().configure_optimizers()
        param_groups = [
            {"params": self.dynamics.parameters(), "lr": self.cfg.lr},
            {"params": self.torque_predictor.parameters(), "lr": self.cfg.lr},
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
        return self.normalizer["joint_torque"].normalize(batch["joint_torque"]).float()

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
        # context = clean latents z[0..T-2]; terminal = predicted latent z_pred[-1]
        torque_latents = torch.cat([z[:-1], pred_s[-1:]], dim=0)  # (T,B,C,H,W)
        torque_latents = rearrange(torque_latents, "t b c h w -> b t c h w")
        torque_pred = self.torque_predictor(torque_latents, action_bt)  # (B,T,8)
        torque_target = self._torque_target(batch)
        torque_loss = F.mse_loss(torque_pred, torque_target)

        total_loss = dyn_loss + self.torque_loss_weight * torque_loss
        self.log("training/loss", total_loss)
        self.log("training/dyn_loss", dyn_loss)
        self.log("training/torque_mse", torque_loss)
        with torch.no_grad():
            tp_nm = self.normalizer["joint_torque"].unnormalize(torque_pred)
            rmse = torch.sqrt(F.mse_loss(tp_nm, batch["joint_torque"].float()))
            self.log("training/torque_rmse_nm", rmse)
        return {"loss": total_loss}

    # ----- validation: parent latent metrics + torque metric -----
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
            # teacher-forced on clean latents (autoregressive-rollout torque is TODO)
            torque_pred = self.torque_predictor(z, action_bt)
            torque_target = self._torque_target(batch)
            tmse = F.mse_loss(torque_pred, torque_target)
            tp_nm = self.normalizer["joint_torque"].unnormalize(torque_pred)
            trmse = torch.sqrt(F.mse_loss(tp_nm, batch["joint_torque"].float()))
        self.log(f"{namespace}/torque_mse", tmse)
        self.log(f"{namespace}/torque_rmse_nm", trmse)
        return None
