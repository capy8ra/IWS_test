#!/usr/bin/env bash
# Stage 2（latent dynamics + torque head），自动加载最新 Stage-1 ckpt 作冻结 AE。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[[ -f "$ROOT/env.local.sh" ]] || { echo "缺 $ROOT/env.local.sh"; exit 1; }
set +u; source "$ROOT/env.local.sh"; set -u
cd "$ROOT"

NGPU="${NGPU:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NGPU-1)))}"
NGPU=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
BATCH="${BATCH:-16}"
METRICS="${METRICS:-[fvd]}"
HEAD="${HEAD:-separate}"          # separate | trunk_midblock
TAG=""; if [ "$HEAD" = trunk_midblock ]; then TAG="_trunk"; fi

STAGE1_CKPT="$(ls -t outputs/world_ft_stage_1/*/checkpoints/last.ckpt 2>/dev/null | head -n1 || true)"
[[ -n "$STAGE1_CKPT" ]] || { echo "[stage2] 没找到 Stage-1 ckpt，先训 stage1"; exit 1; }
RESUME="$(ls -t outputs/world_ft_stage_2${TAG}/*/checkpoints/last.ckpt 2>/dev/null | head -n1 || true)"
echo "[stage2] GPUs=$CUDA_VISIBLE_DEVICES NGPU=$NGPU batch/gpu=$BATCH head=$HEAD metrics=$METRICS"
echo "[stage2] load_ae <- $STAGE1_CKPT"

python main.py +name=world_ft_stage_2 \
  "hydra.run.dir=outputs/world_ft_stage_2${TAG}/\${now:%Y-%m-%d}_\${now:%H-%M-%S}" \
  ${RESUME:+load=$RESUME} \
  experiment=exp_latent_dyn_torque algorithm=latent_world_model_torque dataset=world_ft_dyn_dataset \
  dataset.dataset_dir=data/world_ft_v3 \
  dataset.obs_keys=[camera_1_color] \
  '~dataset.shape_meta.obs.camera_0_color' \
  dataset.horizon=10 dataset.val_horizon=200 \
  experiment.training.precision=bf16-mixed \
  experiment.training.batch_size="$BATCH" \
  experiment.training.max_steps=1000005 \
  experiment.training.log_every_n_steps=100 \
  experiment.training.checkpointing.every_n_train_steps=0 \
  experiment.training.checkpointing.every_n_epochs=1 \
  +experiment.training.checkpointing.save_on_train_epoch_end=true \
  +experiment.training.checkpointing.monitor=step \
  +experiment.training.checkpointing.mode=max \
  +experiment.training.checkpointing.save_top_k=3 \
  +experiment.training.checkpointing.save_last=true \
  experiment.validation.limit_batch=1.0 experiment.validation.batch_size=2 \
  experiment.validation.val_every_n_step=6000 \
  algorithm.training_stage=2 \
  "algorithm.load_ae=$STAGE1_CKPT" \
  algorithm.latent_dim=512 algorithm.action_dim=8 algorithm.torque_dim=8 \
  algorithm.sampling_strategy=terminal_only \
  algorithm.noise_scheduler.loss_weighting=uniform \
  algorithm.torque_head_source="$HEAD" \
  experiment.num_devices="$NGPU" \
  wandb.mode=online \
  "algorithm.metrics=$METRICS" \
  "$@"
