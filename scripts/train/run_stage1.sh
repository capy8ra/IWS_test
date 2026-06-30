#!/usr/bin/env bash
# Stage 1（autoencoder）。卡数用 NGPU 控制（默认 2）。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[[ -f "$ROOT/env.local.sh" ]] || { echo "缺 $ROOT/env.local.sh"; exit 1; }
set +u; source "$ROOT/env.local.sh"; set -u
cd "$ROOT"

NGPU="${NGPU:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NGPU-1)))}"
NGPU=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
BATCH="${BATCH:-16}"
METRICS="${METRICS:-[fvd]}"     # i3d_torchscript.pt 已就位用 [fvd]；关掉用 METRICS='[]'
echo "[stage1] GPUs=$CUDA_VISIBLE_DEVICES NGPU=$NGPU batch/gpu=$BATCH metrics=$METRICS"

RESUME="$(ls -t outputs/world_ft_stage_1/*/checkpoints/last.ckpt 2>/dev/null | head -n1 || true)"
python main.py +name=world_ft_stage_1 \
  'hydra.run.dir=outputs/world_ft_stage_1/${now:%Y-%m-%d}_${now:%H-%M-%S}' \
  ${RESUME:+load=$RESUME} \
  algorithm=latent_world_model experiment=exp_latent_dyn dataset=real_aloha_dataset \
  dataset.dataset_dir=data/world_ft_v3 \
  dataset.horizon=1 dataset.val_horizon=1 \
  dataset.obs_keys=[camera_1_color] dataset.action_mode=bimanual_push \
  '~dataset.shape_meta.obs.camera_0_color' \
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
  experiment.validation.limit_batch=1.0 experiment.validation.batch_size=10 \
  experiment.validation.val_every_n_step=5000 \
  algorithm.latent_dim=512 algorithm.action_dim=4 algorithm.training_stage=1 \
  experiment.num_devices="$NGPU" \
  wandb.mode=online \
  "algorithm.metrics=$METRICS" \
  "$@"
