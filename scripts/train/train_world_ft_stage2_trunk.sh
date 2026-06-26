#!/usr/bin/env bash
# Stage 2 (latent dynamics + right-arm joint-torque head) -- TRUNK variant.
# Same as train_world_ft_stage2.sh, but the torque head branches off the shared
# dynamics U-Net mid-block features (algorithm.torque_head_source=trunk_midblock)
# instead of the standalone CNN+GRU re-encoder. Use this to A/B the two heads.
set -euo pipefail

# --- W&B credentials (never hardcode the key in this file) ---
# Provide WANDB_API_KEY via either:
#   * your environment:  export WANDB_API_KEY=...          (https://wandb.ai/authorize)
#   * a gitignored file: scripts/train/wandb_key.local.sh  (export WANDB_API_KEY=...)
_KEYFILE="$(dirname "$0")/wandb_key.local.sh"
# shellcheck disable=SC1090
[ -f "$_KEYFILE" ] && source "$_KEYFILE"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env or create scripts/train/wandb_key.local.sh}"

cd "$(dirname "$0")/../.."   # repo root

# --- frozen Stage-1 autoencoder: newest world_ft_stage_1 checkpoint ------------
STAGE1_CKPT="$(ls -t outputs/world_ft_stage_1/*/checkpoints/last.ckpt 2>/dev/null | head -n1)"
if [[ -z "${STAGE1_CKPT}" ]]; then
  echo "[train] ERROR: no Stage-1 checkpoint under outputs/world_ft_stage_1/*/checkpoints/last.ckpt" >&2
  exit 1
fi
echo "[train] load_ae (frozen encoder/decoder) <- ${STAGE1_CKPT}"

# --- auto-resume ---------------------------------------------------------------
# A single rolling last.ckpt is written at the end of every epoch
# (save_top_k=1 + filename=last). On startup we resume full training state from
# the most recent trunk Stage-2 checkpoint of this experiment.
RESUME_ARG=()
LATEST_CKPT="$(ls -t outputs/world_ft_stage_2_trunk/*/checkpoints/last.ckpt 2>/dev/null | head -n1 || true)"
if [[ -n "${LATEST_CKPT}" ]]; then
  echo "[train] Resuming from latest checkpoint: ${LATEST_CKPT}"
  RESUME_ARG=("load=${LATEST_CKPT}")
else
  echo "[train] No previous trunk Stage-2 checkpoint found -- starting from scratch."
fi

python main.py +name=world_ft_stage_2_trunk \
  'hydra.run.dir=outputs/world_ft_stage_2_trunk/${now:%Y-%m-%d}_${now:%H-%M-%S}' \
  "${RESUME_ARG[@]}" \
  experiment=exp_latent_dyn_torque \
  algorithm=latent_world_model_torque \
  dataset=world_ft_dyn_dataset \
  dataset.dataset_dir=data/world_ft_v2 \
  dataset.obs_keys=[camera_1_color] \
  '~dataset.shape_meta.obs.camera_0_color' \
  dataset.horizon=10 dataset.val_horizon=200 \
  experiment.training.precision=16-mixed \
  experiment.training.batch_size=16 \
  experiment.training.max_steps=1000005 \
  experiment.training.log_every_n_steps=100 \
  experiment.training.checkpointing.every_n_train_steps=0 \
  experiment.training.checkpointing.every_n_epochs=1 \
  +experiment.training.checkpointing.save_on_train_epoch_end=true \
  +experiment.training.checkpointing.save_top_k=1 \
  +experiment.training.checkpointing.filename=last \
  experiment.validation.limit_batch=1.0 \
  experiment.validation.batch_size=2 \
  experiment.validation.val_every_n_step=6000 \
  algorithm.training_stage=2 \
  "algorithm.load_ae=${STAGE1_CKPT}" \
  algorithm.latent_dim=512 algorithm.action_dim=8 algorithm.torque_dim=8 \
  algorithm.torque_head_source=trunk_midblock \
  algorithm.sampling_strategy=terminal_only \
  algorithm.noise_scheduler.loss_weighting=uniform \
  "$@"   # extra hydra overrides, e.g. algorithm.torque_loss_weight=0.1
