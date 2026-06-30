#!/usr/bin/env bash
# Stage 3（autoencoder finetuning）：冻结 encoder，只用"latent 加 0.02 噪声 -> 重建干净图"
# 的目标、以 0.1x 学习率微调 decoder，让它对 dynamics 预测 latent 的误差更鲁棒。
#
# load_ae 取最新的 STAGE-1 ckpt（而非 stage-2）：stage-3 只需要冻结的 encoder/decoder，
# 而 stage-2 并未改动它们（stage-2 冻结 enc/dec、只训 dynamics），所以 stage-1 的 enc/dec
# 与 stage-2 完全一致；并且 world_ft 的 stage-2 是 torque 子类，直接 load 进基础模型会有
# 多余的 torque 权重 + action_dim(8 vs 4) 形状不匹配。用 stage-1 ckpt 最干净。
# 卡数用 NGPU 控制（默认 2）。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[[ -f "$ROOT/env.local.sh" ]] || { echo "缺 $ROOT/env.local.sh"; exit 1; }
set +u; source "$ROOT/env.local.sh"; set -u
cd "$ROOT"

NGPU="${NGPU:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NGPU-1)))}"
NGPU=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
BATCH="${BATCH:-16}"          # 每卡 batch；L20(44G) 建议 BATCH=8
METRICS="${METRICS:-[fvd]}"   # i3d_torchscript.pt 已就位用 [fvd]；关掉用 METRICS='[]'

# 冻结的 AE 来源：最新的 stage-1 ckpt（可用 AE_CKPT=... 覆盖）
AE_CKPT="${AE_CKPT:-$(ls -t outputs/world_ft_stage_1/*/checkpoints/last.ckpt 2>/dev/null | head -n1 || true)}"
[[ -n "$AE_CKPT" ]] || { echo "[stage3] 没找到 Stage-1 ckpt，先训 stage1"; exit 1; }
# stage-3 自身的 auto-resume（FRESH=1 可跳过，从头开始）
RESUME=""
if [ "${FRESH:-0}" != "1" ]; then
  RESUME="$(ls -t outputs/world_ft_stage_3/*/checkpoints/last.ckpt 2>/dev/null | head -n1 || true)"
fi
echo "[stage3] GPUs=$CUDA_VISIBLE_DEVICES NGPU=$NGPU batch/gpu=$BATCH metrics=$METRICS"
echo "[stage3] load_ae (frozen enc/dec) <- $AE_CKPT"

python main.py +name=world_ft_stage_3 \
  'hydra.run.dir=outputs/world_ft_stage_3/${now:%Y-%m-%d}_${now:%H-%M-%S}' \
  ${RESUME:+load=$RESUME} \
  algorithm=latent_world_model experiment=exp_latent_dyn dataset=real_aloha_dataset \
  dataset.dataset_dir=data/world_ft_v3 \
  dataset.horizon=1 dataset.val_horizon=200 \
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
  experiment.validation.limit_batch=1.0 experiment.validation.batch_size=2 \
  experiment.validation.val_every_n_step=30000 \
  algorithm.latent_dim=512 algorithm.action_dim=4 algorithm.training_stage=3 \
  "algorithm.load_ae=$AE_CKPT" \
  algorithm.sampling_strategy=terminal_only \
  algorithm.noise_scheduler.loss_weighting=uniform \
  experiment.num_devices="$NGPU" \
  wandb.mode=online \
  "algorithm.metrics=$METRICS" \
  "$@"
