# Stage-2 implementation — files to commit

Tracking list so we commit/push only the stage-2 work after testing.

## Torque is predicted in RAW physical units (N·m) — NOT normalized
The joint-torque target/loss/prediction are all kept in raw N·m end to end (no
`normalizer["joint_torque"]`), so the head's output is directly usable as actual
torque with no normalizer statistics required downstream. Images and actions are
still normalized by the base class. Consequence: `torque_mse` is now in N·m^2 and
`torque_rmse_nm = sqrt(torque_mse)` directly; the torque loss magnitude is in
physical units, so `torque_loss_weight` likely needs tuning to balance it against
the (normalized-latent) dynamics loss, and per-joint scale differences are no
longer equalized.

## New files
- `interactive_world_sim/datasets/latent_dynamics/world_ft_dyn_dataset.py` — `WorldFtDynDataset` (q_des action + raw joint_torque target, no torque normalizer). **[done, tested]**
- `interactive_world_sim/algorithms/latent_dynamics/models/torque_predictor.py` — `TorquePredictor` (causal GRU latent+action -> torque) **and** `TrunkTorqueHead` (MLP off the dynamics U-Net mid-block features + noise-level cond). **[done, tested]**
- `interactive_world_sim/algorithms/latent_dynamics/latent_world_model_torque.py` — `LatentWorldModelTorque` (stage-2 dense torque loss; config-switchable head). **[done, smoke-tested]**
- `interactive_world_sim/experiments/exp_latent_dyn_torque.py` — experiment subclass registering the new dataset + algorithm. **[done]**
- `configurations/algorithm/latent_world_model_torque.yaml` — incl. `torque_head_source` / `torque_noise_cond`. **[done]**
- `configurations/dataset/world_ft_dyn_dataset.yaml` — **[done]**
- `configurations/experiment/exp_latent_dyn_torque.yaml` — **[done]**
- `scripts/train/train_world_ft_stage2.sh` — separate-head launcher. **[done, smoke-tested]**
- `scripts/train/train_world_ft_stage2_trunk.sh` — trunk-head launcher (`torque_head_source=trunk_midblock`; WANDB key from env). **[done, smoke-tested]**

### Config-switchable torque head (`algorithm.torque_head_source`)
- `separate` (default): standalone `TorquePredictor` CNN+GRU re-encoder on latents
  (`[z_gt[0..T-2], z_pred[-1]]`); torque gradient reaches the U-Net only through the
  predicted terminal latent.
- `trunk_midblock`: `TrunkTorqueHead` MLP branches off the shared U-Net mid-block
  features (`dim*dim_mults[-1]` = 128 ch, spatially pooled), captured via a forward
  hook on `dynamics.module.mid_block` during the *existing* dynamics forward — so
  `CMLatentDynamics.forward` stays tensor-only and `CTM_calc_out`/`EinopsWrapper`
  are untouched. Torque loss backprops through the whole trunk (verified: torque-only
  loss yields nonzero grad in mid_block + down_blocks). Conditioned on the per-frame
  diffusion noise level because trunk features come from noisy latents. Validation
  replays the training-style terminal_only forward in `n_tokens` windows to populate
  the hook over the long val horizon (autoregressive-rollout torque still TODO, as
  for the separate head).

### A100 note (baked into the stage-2 script)
The latent-dynamics attention forces **flash-only** SDPA on A100 (sm_80), which has
no fp32 kernel; the experiment defaults to `precision: 32-true` (fine for stage 1,
whose decoder doesn't use this attention). The stage-2 script therefore sets
`experiment.training.precision=16-mixed` so the dynamics attention runs in fp16.

## Modified existing files (small, guarded, backward-compatible)
- `scripts/data_collection/convert_world_ft.py` — rewritten as the unified raw->hdf5 converter (images + q_des + joint_torque + wrench/vel/tau). **[done, tested]**
- `interactive_world_sim/datasets/latent_dynamics/real_aloha_dataset.py` — replay-buffer builder: `right_qpos` raw-action branch + load `obs/joint_torque` when present (guarded; existing modes unchanged). **[done, tested]**
- `interactive_world_sim/experiments/__init__.py` — register `exp_latent_dyn_torque` in `exp_registry` (2 lines). **[todo §6]**

## Docs (optional to commit)
- `docs/stage2_dynamics_plan_updated.md` — the agreed plan (dense torque supervision folded in).
- `docs/stage2_dynamics_plan.md` — earlier draft.
- `docs/stage2_changed_files.md` — this tracking file.

## Generated data (NOT committed — gitignored under data/)
- `data/world_ft_v2/{train,val}/episode_*.hdf5` + cache.
