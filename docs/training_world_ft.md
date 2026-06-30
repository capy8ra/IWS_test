# Training runbook — world_ft (Stage 1 → 2 → 3)

End-to-end recipe for training the two-stage latent world model on the `world_ft`
robot data. The active dataset is **`data/world_ft_v3`** (26 train / 5 val
episodes, converted from the `2026_06_26_1` batch).

```
raw *_PDT/  --convert_world_ft.py-->  data/world_ft_v3/{train,val}/episode_*.hdf5
                                               │
   Stage 1: train encoder + diffusion decoder (self-supervised, actions ignored)
   Stage 2: freeze enc/dec, train latent dynamics + right-arm joint-torque head
   Stage 3 (optional): freeze encoder, finetune decoder on noised latents (rollout robustness)
```

Everything runs through `main.py` + Hydra. The same unified HDF5 serves both
stages; only `dataset` / `action_mode` / `training_stage` differ.

---

## 0. Prerequisites

W&B credentials (never hardcode the key). Provide **either**:

```bash
export WANDB_API_KEY=...                       # https://wandb.ai/authorize
# or create the gitignored file:
#   scripts/train/wandb_key.local.sh   ->   export WANDB_API_KEY=...
```

All commands below are run from the repo root (`/home/peng33/interactive_world_sim`).

---

## 1. Data conversion (already done for `world_ft_v3`)

Unified raw→HDF5 converter. Point `--src` at the **batch subdir** (not the parent,
which mixes recording sessions). `--n_val` episodes (the last N by name) go to `val/`.

```bash
python scripts/data_collection/convert_world_ft.py \
  --src /home/peng33/supernova/projects/world_ft/2026_06_26_1 \
  --out data/world_ft_v3 \
  --n_val 5 \
  --target_hz 10
```

Per episode it writes `action` (T,8 right `q_des`), `obs/joint_torque` (T,8 right
N·m — the Stage-2 target), images, and the FK-compat `joint_pos`/`full_joint_pos`/
`world_t_robot_base`. The controller wrench streams are optional and were absent
in this batch (skipped cleanly; Stage 1/2 never read them).

> First training run builds a `cache.zarr.zip` (images → JPEG2k zarr) inside each
> `train/` and `val/` dir. Delete it to force a rebuild after re-converting.

---

## 2. Stage 1 — autoencoder (encoder + diffusion decoder)

Self-supervised image reconstruction; **actions are ignored** in the loss. Trains
`encoder` + `decoder`. A single rolling `last.ckpt` is written every epoch and the
script auto-resumes from the newest one.

### Option A — existing launcher (edit one line)

`scripts/train/train_world_ft_stage1.sh` currently points at `data/world_ft`.
Change its dataset line to the new dir:

```bash
dataset.dataset_dir=data/world_ft_v3 \
```

then:

```bash
bash scripts/train/train_world_ft_stage1.sh
```

### Option B — explicit command

```bash
STAGE1_RESUME="$(ls -t outputs/world_ft_stage_1/*/checkpoints/last.ckpt 2>/dev/null | head -n1)"

python main.py +name=world_ft_stage_1 \
  'hydra.run.dir=outputs/world_ft_stage_1/${now:%Y-%m-%d}_${now:%H-%M-%S}' \
  ${STAGE1_RESUME:+load=$STAGE1_RESUME} \
  algorithm=latent_world_model \
  experiment=exp_latent_dyn \
  dataset=real_aloha_dataset \
  dataset.dataset_dir=data/world_ft_v3 \
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
```

Notes:
- `action_mode=bimanual_push` runs the FK action path on the compat `joint_pos`;
  the result is unused by the Stage-1 loss, so it's harmless. (You could switch to
  `right_qpos` to skip FK entirely — Stage 1 ignores actions either way.)
- Output: `outputs/world_ft_stage_1/<timestamp>/checkpoints/last.ckpt`.

---

## 3. Stage 2 — latent dynamics + joint-torque head

Loads the frozen Stage-1 autoencoder via `load_ae`, then trains the dynamics U-Net
+ a torque head. Predicts the next image latent and the aligned 8-DoF right-arm
joint torque (raw N·m) from image history + `q_des`. Recipe: `horizon=10`,
`val_horizon=200`, `terminal_only` sampling, `uniform` noise loss weighting,
`16-mixed` precision (A100 flash-only SDPA has no fp32 kernel).

Two interchangeable torque heads, selected by `algorithm.torque_head_source`:
- `separate` (default) — standalone CNN+GRU re-encoder on latents.
- `trunk_midblock` — MLP branching off the dynamics U-Net mid-block features.

### Separate head

`scripts/train/train_world_ft_stage2.sh` hardcodes `data/world_ft_v2` and does
**not** forward extra overrides. Edit its dataset line to:

```bash
dataset.dataset_dir=data/world_ft_v3 \
```

then:

```bash
bash scripts/train/train_world_ft_stage2.sh
```

### Trunk head

`train_world_ft_stage2_trunk.sh` already forwards extra Hydra overrides (`"$@"`),
so no edit is needed:

```bash
bash scripts/train/train_world_ft_stage2_trunk.sh dataset.dataset_dir=data/world_ft_v3
# tune the torque term, e.g.:
bash scripts/train/train_world_ft_stage2_trunk.sh \
  dataset.dataset_dir=data/world_ft_v3 algorithm.torque_loss_weight=0.1
```

### Explicit command (separate head)

```bash
STAGE1_CKPT="$(ls -t outputs/world_ft_stage_1/*/checkpoints/last.ckpt 2>/dev/null | head -n1)"
STAGE2_RESUME="$(ls -t outputs/world_ft_stage_2/*/checkpoints/last.ckpt 2>/dev/null | head -n1)"

python main.py +name=world_ft_stage_2 \
  'hydra.run.dir=outputs/world_ft_stage_2/${now:%Y-%m-%d}_${now:%H-%M-%S}' \
  ${STAGE2_RESUME:+load=$STAGE2_RESUME} \
  experiment=exp_latent_dyn_torque \
  algorithm=latent_world_model_torque \
  dataset=world_ft_dyn_dataset \
  dataset.dataset_dir=data/world_ft_v3 \
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
  algorithm.sampling_strategy=terminal_only \
  algorithm.noise_scheduler.loss_weighting=uniform
```

For the **trunk** head add: `algorithm.torque_head_source=trunk_midblock` (and
optionally `algorithm.torque_noise_cond=true`).

Outputs:
- separate → `outputs/world_ft_stage_2/<timestamp>/checkpoints/last.ckpt`
- trunk    → `outputs/world_ft_stage_2_trunk/<timestamp>/checkpoints/last.ckpt`

Logged metrics: `training/{loss,dyn_loss,torque_mse,torque_rmse_nm}` and
`validation/{dyn_loss,torque_mse,torque_rmse_nm}`. Torque is in raw N·m, so
`torque_rmse_nm = sqrt(torque_mse)` directly.

---

## 4. Stage 3 — autoencoder finetuning (decoder robustness)

Finetune **only the decoder** so it stays sharp when fed *imperfect* latents — the
kind the Stage-2 dynamics produces at rollout. The encoder is **frozen** (the latent
space must not move, or Stage-2's dynamics would be invalidated); the decoder is
retrained at `0.1×` LR with Gaussian noise (`σ=0.02`) injected into the latent before
decoding. Same single-frame reconstruction objective as Stage 1 — the only differences
are: encoder frozen, latent noise always on, decoder-only at low LR, initialized from a
trained AE. Mechanically, **Stage 3 ≈ "Stage 1 with a frozen encoder + forced latent
noise + a decoder-only low-LR finetune."** It does **not** load any Stage-2 weights.

### `load_ae` points at the Stage-1 ckpt (not Stage-2)

Stage 3 only needs the frozen encoder/decoder, and Stage 2 never changed them (it
freezes enc/dec and trains dynamics), so the Stage-1 enc/dec are identical to the ones
inside the Stage-2 ckpt. Moreover the world_ft Stage-2 ckpt is the
`latent_world_model_torque` subclass (`action_dim=8` + extra torque-head weights):
loading it into the base `latent_world_model` used here would hit unexpected-key and
`action_dim` (8 vs 4) shape errors. So `run_stage3.sh` loads the newest **Stage-1**
`last.ckpt` — clean and equivalent.

### Launcher

```bash
NGPU=4 bash scripts/train/run_stage3.sh                 # A100/A800, per-GPU batch 16
NGPU=8 BATCH=8 bash scripts/train/run_stage3.sh         # L20 (44 GB): per-GPU batch 8
FRESH=1 NGPU=4 bash scripts/train/run_stage3.sh         # start fresh (don't resume stage-3)
AE_CKPT=outputs/world_ft_stage_1/<ts>/checkpoints/last.ckpt bash scripts/train/run_stage3.sh
```

### Explicit command

```bash
STAGE1_CKPT="$(ls -t outputs/world_ft_stage_1/*/checkpoints/last.ckpt 2>/dev/null | head -n1)"
STAGE3_RESUME="$(ls -t outputs/world_ft_stage_3/*/checkpoints/last.ckpt 2>/dev/null | head -n1)"

python main.py +name=world_ft_stage_3 \
  'hydra.run.dir=outputs/world_ft_stage_3/${now:%Y-%m-%d}_${now:%H-%M-%S}' \
  ${STAGE3_RESUME:+load=$STAGE3_RESUME} \
  algorithm=latent_world_model experiment=exp_latent_dyn dataset=real_aloha_dataset \
  dataset.dataset_dir=data/world_ft_v3 \
  dataset.horizon=1 dataset.val_horizon=200 \
  dataset.obs_keys=[camera_1_color] dataset.action_mode=bimanual_push \
  '~dataset.shape_meta.obs.camera_0_color' \
  experiment.training.batch_size=16 \
  experiment.training.max_steps=1000005 \
  experiment.validation.val_every_n_step=30000 \
  algorithm.latent_dim=512 algorithm.action_dim=4 algorithm.training_stage=3 \
  "algorithm.load_ae=${STAGE1_CKPT}" \
  algorithm.sampling_strategy=terminal_only \
  algorithm.noise_scheduler.loss_weighting=uniform
```

Notes:
- **Decoder-only / encoder frozen.** `configure_optimizers` (stage 3) optimizes just the
  decoder at `lr*0.1`; the encoder runs under `torch.no_grad()`.
- **Latent noise σ=0.02** is hardcoded in `latent_world_model.py` (`z += randn*0.02`);
  tune it toward your actual dynamics prediction error if needed (code change).
- Output: `outputs/world_ft_stage_3/<timestamp>/checkpoints/last.ckpt`.
- **Inference pipeline:** encoder (Stage 1) → dynamics (Stage 2) predicts latents →
  **Stage-3 decoder** renders them. Stage 3 improves rollout image quality; it does not
  touch torque (that stays the Stage-2 head).
- **Caveat:** the dynamics loaded here is the (untrained) Stage-1 one — harmless, since
  the Stage-3 loss never uses dynamics. To make Stage-3 validation rollouts use the
  *trained* dynamics, a small loader tweak is needed (non-strict load + `action_dim=8`).

---

## 5. Notes & gotchas

- **Stage-1 checkpoint reuse.** Stage 2 loads the newest
  `outputs/world_ft_stage_1/.../last.ckpt`. The current one was trained on the old
  data; reusing it is fine (Stage 1 is self-supervised), but for best results
  retrain Stage 1 on `world_ft_v3` first, then run Stage 2.
- **Auto-resume.** Each launcher resumes full training state (weights + optimizer
  + LR scheduler + global step) from its own newest `last.ckpt`. To start fresh,
  move/rename the prior `outputs/world_ft_stage_*` run dir.
- **Cache.** After re-converting data, delete `data/world_ft_v3/{train,val}/cache.zarr.zip`
  so the loader rebuilds it.
- **A/B the torque heads** by running the separate and trunk launchers; they write
  to separate output trees so they don't collide.
```