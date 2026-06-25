#!/usr/bin/env bash
# Stage 1 (autoencoder: image encoder + diffusion decoder) on the world_ft hammer data.
# Self-supervised reconstruction -- uses only the camera frames; actions are ignored in stage 1.
# Set WANDB_API_KEY below before running. Do NOT commit your key back into this file.
set -euo pipefail

# --- secret ---
# Paste your Weights & Biases API key here before running (https://wandb.ai/authorize).
export WANDB_API_KEY=

cd "$(dirname "$0")/../.."   # repo root

# --- auto-resume ---------------------------------------------------------------
# Every run of this script writes checkpoints under
#   outputs/world_ft_stage_1/<timestamp>/checkpoints/
# A single rolling `last.ckpt` is written at the end of every training epoch
# (save_top_k=1 + filename=last). On startup we look for the most recent
# `last.ckpt` from a previous run of THIS experiment and resume full training
# state from it (weights + optimizer + lr-scheduler + global step count), so an
# interrupted run picks up exactly where it left off.
RESUME_ARG=()
LATEST_CKPT="$(ls -t outputs/world_ft_stage_1/*/checkpoints/last.ckpt 2>/dev/null | head -n1 || true)"
if [[ -n "${LATEST_CKPT}" ]]; then
  echo "[train] Resuming from latest checkpoint: ${LATEST_CKPT}"
  RESUME_ARG=("load=${LATEST_CKPT}")
else
  echo "[train] No previous checkpoint found under outputs/world_ft_stage_1/*/checkpoints/ -- starting from scratch."
fi

python main.py +name=world_ft_stage_1 \
  'hydra.run.dir=outputs/world_ft_stage_1/${now:%Y-%m-%d}_${now:%H-%M-%S}' \
  "${RESUME_ARG[@]}" \
  algorithm=latent_world_model \
  experiment=exp_latent_dyn \
  dataset=real_aloha_dataset \
  dataset.dataset_dir=data/world_ft \
  dataset.horizon=1 dataset.val_horizon=1 \
  dataset.obs_keys=[camera_1_color] \
  dataset.action_mode=bimanual_push \
  '~dataset.shape_meta.obs.camera_0_color' \
  experiment.training.batch_size=16 \
  experiment.training.max_steps=1000005 \
  experiment.training.log_every_n_steps=100 \
  experiment.training.checkpointing.every_n_train_steps=0 \
  experiment.training.checkpointing.every_n_epochs=1 \
  +experiment.training.checkpointing.save_on_train_epoch_end=true \
  +experiment.training.checkpointing.save_top_k=1 \
  +experiment.training.checkpointing.filename=last \
  experiment.validation.limit_batch=1.0 \
  experiment.validation.batch_size=10 \
  experiment.validation.val_every_n_step=6000 \
  algorithm.latent_dim=512 algorithm.action_dim=4 \
  algorithm.training_stage=1
