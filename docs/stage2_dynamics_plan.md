# Stage-2 Dynamics + Joint-Torque Training — Implementation Plan

Status: **PLAN ONLY — no code written yet.**
Author: drafted for review before implementation.

## 1. Goal

Train the **stage-2 latent dynamics** model so that, given past observation
latents + action, it predicts **future latent states *and* the right arm's 8
joint torques**.

- **Action input**: right-arm `q_des` (8 DoF incl. gripper) =
  `follower_right_mit_command:q`.
- **Torque output (new target)**: right-arm measured joint torque (8 DoF) =
  `follower_right_state:actual_torque_nm`.
- **Rollout horizon**: ~8 future frames; torque predicted per future frame
  (aligned with each predicted latent).
- **Encoder/decoder**: reuse the **existing stage-1 autoencoder** (frozen),
  loaded via `load_ae`.

Scope decisions already confirmed: **right arm only**; q_des = follower
`mit_command:q`; horizon ≈ 8; data regenerated from the raw collection folder
(not augmented in place).

## 2. Background — how stage 2 works today

- `LatentWorldModel` (`algorithms/latent_dynamics/latent_world_model.py`):
  - stage 1 trains `encoder` + `decoder`; stage 2 trains only `self.dynamics`
    with encoder/decoder loaded from a stage-1 ckpt (`load_ae`); stage 3
    fine-tunes the decoder.
  - stage-2 training step: encode frames → latents `z`; add CTM/diffusion noise;
    `self.dynamics` denoises future latents conditioned on `action`
    (`external_cond`); loss = weighted MSE on latents.
- `CMLatentDynamics` (`algorithms/latent_dynamics/models/cm_latent_dynamics.py`):
  3D-UNet over latent maps; `action` is embedded (`action_emd` MLP) and injected
  as FiLM scale/shift in each `ResnetBlock`; output head `self.out` →
  `Conv3d(dim, latent_dim)`.
- Dataset (`datasets/latent_dynamics/real_aloha_dataset.py`): currently builds
  `action` by FK (`joint_pos_to_action_primitive`, e.g. `bimanual_push` → 4-D),
  **ignoring** the raw `action` field; returns `obs` (images) + `action`.

## 3. Input data format (raw collection)

One folder per episode, e.g.
`/home/peng33/supernova/projects/world_ft/<TS>_PDT/`:

- `camera_head.mp4` — head camera; frame **PTS encode epoch microseconds**
  (~30 fps, ~56 s, ~1680 frames).
- `low_dim_npys/<stream>:<field>.npy` — multi-rate streams (colon-separated
  names). Relevant right-arm streams (left-arm equivalents also exist):

| stream:field | shape | rate | use |
|---|---|---|---|
| `follower_right_mit_command:q` | (M,8) | ~2 kHz | **action (q_des)** |
| `follower_right_state:actual_torque_nm` | (N,8) | ~627 Hz | **torque target** |
| `follower_right_state:actual_angle_rad` | (N,8) | ~627 Hz | joint_pos |
| `follower_right_state:actual_velocity_radps` | (N,8) | ~627 Hz | joint_vel (opt) |
| `follower_right_state:timestamp` | (N,8) | — | per-joint state times |
| `follower_right_mit_command:timestamp` | (M,) | — | command times |
| `follower_controller_state:wrench_hand_tcp_R` | (K,6) | ~509 Hz | **wrench F/T (save only)** |
| `follower_controller_state:wrench_hand_tcp_inertiacomp_R` | (K,6) | ~509 Hz | wrench, inertia-comp (save only) |
| `follower_controller_state:tau_interaction_R` | (K,8) | ~509 Hz | external/contact torque (save only) |
| `follower_controller_state:timestamp` | (K,) | — | controller times |

Notes:
- The 8th DoF of `q`/torque is the **gripper**.
- The hand wrench is **measured/computed in the logs** → no MJCF inverse
  dynamics needed (the old `convert_episode_to_hdf5.py` residual-torque method is
  obsolete for this).
- There are 6 episodes in the example folder; they correspond 1:1 to the current
  `data/world_ft` episodes (verified by exact epoch-timestamp overlap).

## 4. Output HDF5 schema (unified for stage 1 + 2 + wrench)

One `episode_<i>.hdf5` per episode, all per-step arrays length `T` (frames kept
after subsampling), **right arm = the modeled arm**:

| key | shape | source |
|---|---|---|
| `action` | (T, 8) | right `mit_command:q` (q_des) — stage-2 action; ignored by stage 1 |
| `timestamp` | (T,) | kept video-frame epoch times |
| `obs/images/camera_1_color` | (T, H, W, 3) uint8 | `camera_head.mp4` frames |
| `obs/joint_pos` | (T, 14) | `arm7(right)` + `arm7(left)` Trossen-7 map (loader/FK + stage-1 compat) |
| `obs/full_joint_pos` | (T, 16) | right8 + left8 measured angle |
| `obs/world_t_robot_base` | (T, 2, 4, 4) | base pose (identity stub unless calib supplied) |
| `obs/joint_torque` | (T, 8) | right `actual_torque_nm` — **stage-2 target** |
| `obs/joint_vel` | (T, 8) | right `actual_velocity_radps` (optional) |
| `obs/wrench_hand_tcp` | (T, 6) | right `wrench_hand_tcp_R` (Fx,Fy,Fz,Mx,My,Mz) — **save only** |
| `obs/wrench_hand_tcp_inertiacomp` | (T, 6) | right inertia-comp wrench — **save only** |
| `obs/tau_interaction` | (T, 8) | right `tau_interaction_R` — **save only** |

- `obs/images/camera_1_*intrinsics/extrinsics` stubs kept for loader compat (as
  in the current simple converter).
- Keeping `obs/joint_pos`, `obs/full_joint_pos`, `obs/world_t_robot_base`,
  `obs/images` means the **stage-1 loader path keeps working unchanged**.

## 5. Converter design (`scripts/data_collection/convert_world_ft.py`, rewritten)

Single script: **input = raw data folder, output = hdf5 dataset for both stages.**

CLI: `python scripts/data_collection/convert_world_ft.py --src <RAW_PARENT_DIR>
--out data/world_ft --n_val 1 [--target_hz 10]`

Per run:
1. Discover episode dirs: all `*_PDT/` subfolders of `--src`, sorted by name.
2. For each episode:
   a. **Frames + times**: decode `camera_head.mp4` with `av`; frame epoch time =
      `pts * time_base`. Subsample to `--target_hz` (≈10 Hz, matching stage-1)
      → keep frame indices, store frames at native resolution, `timestamp`.
   b. **Resample low-dim onto kept frame times** via per-DoF `np.interp`, each
      stream using its own timestamps (state ts = mean over joints):
      - `action`     ← right `mit_command:q`
      - `joint_torque` ← right `actual_torque_nm`
      - `joint_pos`  ← `arm7(right angle)` + `arm7(left angle)`
      - `full_joint_pos` ← right8 + left8 angle
      - `joint_vel`, `wrench_hand_tcp`, `wrench_hand_tcp_inertiacomp`,
        `tau_interaction` likewise
   c. Write `episode_<i>.hdf5` with the §4 schema.
3. Split: last `--n_val` episodes → `out/val/`, rest → `out/train/`.
4. Force cache rebuild (delete stale `cache.zarr.zip`).

Reused helpers from the current converter: `arm7()` (8→7 Trossen map),
center-crop/resize is still done in the **loader** (not the converter), so stored
frames stay native-res.

### 5.1 Image consistency (important)

The current stage-1 encoder was trained on `data/world_ft` produced by the old
simple converter. To reuse that encoder, regenerated images must match. Plan:
- Reproduce the stage-1 frame selection (same camera, ~10 Hz subsample, native
  resolution) so the loader's center-crop+resize yields identical 128×128 inputs.
- **Validation gate**: encode a few frames from old vs regenerated episodes and
  compare latents. If they match (within tolerance) → reuse stage-1 ckpt as-is.
  If not → quick **stage-1 retrain** on the unified dataset (cheap, gives a clean
  single source of truth). Decision recorded after this check.

## 6. Dataset loader changes (`real_aloha_dataset.py`)

Add a new control mode `action_mode = "right_qpos"`:
- **action** = raw `file["action"]` (q_des, 8-D) directly — **bypass** the
  `joint_pos_to_action_primitive` FK path.
- Load **`obs/joint_torque`** (8-D) into the replay buffer and return it per
  sample as `batch["joint_torque"]` (shape (B, T, 8)).
- Normalizer: add `normalizer["joint_torque"]` (per-joint standardize / range);
  keep `normalizer["action"]` (now 8-D q_des).
- `shape_meta`: `action.shape = [8]`; add `joint_torque.shape = [8]`.

Other action modes untouched (back-compat).

## 7. Model changes (`cm_latent_dynamics.py` + `models/utils.py`)

Add a **torque head** to `CMLatentDynamics`:
- Recommended (decoupled) design: a small head `torque_head(z_future, action)` →
  per-frame torque. Input = predicted/denoised future latent (B,C,T,H,W) global
  -avg-pooled over (H,W) → (B,C,T), concatenated with the action embedding, →
  temporal MLP → (B, T, 8). This keeps torque **out of the CTM diffusion sampler**
  (stable, deterministic) while still being a function of (predicted state, action).
- `forward` returns `(latent_pred, torque_pred)`.
- `EinopsWrapper` (`algorithms/models/utils.py`) updated to rearrange the latent
  tensor and **pass the torque tensor through** unchanged (tuple-aware).
- Alternative (noted, fallback): regress torque from internal UNet features inside
  the denoise forward — rejected as default because it entangles with the
  multi-step CTM sampler.

## 8. Stage-2 training/inference changes (`latent_world_model.py`)

- `_build_model`: build the torque head (only active for stage 2; `action_dim=8`,
  `torque_dim=8`).
- **training_step (stage 2)**: after the latent denoise/prediction, run the torque
  head on the predicted future latents + action → `torque_pred`; add
  `torque_loss = w * MSE(torque_pred, normalizer.normalize(batch["joint_torque"]))`;
  total loss = `latent_loss + torque_loss`. Log `training/dyn_loss`,
  `training/torque_mse`.
- **dynamics_forward / validation_step**: return torque alongside latents; log
  `validation/torque_mse` (denormalized RMS in N·m for interpretability).
- `configure_optimizers` (stage 2): include `torque_head` params with the
  dynamics params.

## 9. Config + training script

- `configurations/algorithm/latent_world_model.yaml`: `action_dim: 8`,
  `torque_dim: 8`, `torque_loss_weight: 1.0` (tune), dynamics `action_dim: 8`.
- `configurations/dataset/real_aloha_dataset.yaml`: `action_mode: right_qpos`,
  `shape_meta.action.shape:[8]`, add `shape_meta.obs... ` / `joint_torque[8]`.
- New `scripts/train/train_world_ft_stage2.sh` (mirrors stage-1 script):
  `algorithm.training_stage=2`,
  `algorithm.load_ae=<stage-1 last.ckpt>`,
  `dataset.action_mode=right_qpos`,
  `dataset.horizon=8 dataset.val_horizon=8`,
  `algorithm.action_dim=8` + torque config,
  same epoch-checkpoint + auto-resume block (under `outputs/world_ft_stage_2/`),
  WANDB key blank (skip-worktree pattern).

## 10. File-by-file change list

1. `scripts/data_collection/convert_world_ft.py` — **rewrite** to unified converter (§5).
2. `interactive_world_sim/datasets/latent_dynamics/real_aloha_dataset.py` — `right_qpos` mode, `joint_torque` target, normalizer, shape_meta (§6).
3. `interactive_world_sim/algorithms/latent_dynamics/models/cm_latent_dynamics.py` — torque head, tuple return (§7).
4. `interactive_world_sim/algorithms/models/utils.py` — `EinopsWrapper` tuple pass-through (§7).
5. `interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py` — torque loss/logging, dynamics_forward return, optimizer group (§8).
6. `configurations/algorithm/latent_world_model.yaml`, `configurations/dataset/real_aloha_dataset.yaml` — config (§9).
7. `scripts/train/train_world_ft_stage2.sh` — new launch script (§9).

## 11. Open decisions / assumptions (please confirm)

1. **Image consistency**: reuse stage-1 encoder if latents match, else retrain
   stage 1 on the unified data. (Recommended: validate first.)
2. **Torque head**: deterministic head on predicted latent + action (recommended)
   vs. inside the diffusion UNet. 
3. **Torque normalization**: per-joint standardization (recommended).
4. **Loss weight** `torque_loss_weight`: start 1.0 on normalized torque, then tune
   so neither loss dominates.
5. **Left arm / wrench / vel**: saved in hdf5 but unused in training now
   (per "just in case"). OK?
6. **Output dir**: regenerate into `data/world_ft` (overwrites current) vs a new
   `data/world_ft_v2`. (Recommend new dir to preserve the stage-1 dataset until
   the consistency check passes.)

## 12. Validation / test plan

- **Converter**: run on the 6 episodes; assert per-episode shapes; sanity-check
  alignment (q_des vs measured angle tracking, torque magnitude ranges, wrench Fz
  during contact); confirm frame count/timestamps vs current `data/world_ft`.
- **Image check**: latent diff old vs new frames (§5.1 gate).
- **Dataset**: pull one batch; verify `action (B,T,8)`, `joint_torque (B,T,8)`,
  normalizers populated.
- **Model unit test**: `CMLatentDynamics` returns latent `(B,C,T,H,W)` + torque
  `(B,T,8)`; `EinopsWrapper` round-trips both.
- **Stage-2 smoke run**: a few hundred steps; both losses decrease; single rolling
  `last.ckpt` saved per epoch; resume works.

## 13. Suggested order of work

1. Rewrite converter → regenerate dataset → run validation/image checks (§5, §11.1).
2. Dataset loader `right_qpos` + torque target (§6).
3. Model torque head + `EinopsWrapper` (§7).
4. Stage-2 loss/logging (§8).
5. Config + `train_world_ft_stage2.sh` (§9).
6. Smoke train, then full run.
