# Stage-2 Latent Dynamics + Right-Arm Joint-Torque Training

Status: **UPDATED IMPLEMENTATION PLAN — no Stage-2 code implemented yet.**

This document supersedes `docs/stage2_dynamics_plan.md`. It incorporates review
of the current implementation, the upstream Stage-2 training recipe, and the
raw `world_ft` recordings.

## 1. Goal

Train Stage 2 so that the model uses image history and the right arm's desired
joint positions to:

1. predict the next image latent with the existing CTM latent-dynamics model;
2. predict the measured right-arm joint torque aligned with that next latent.

The Stage-1 image encoder and decoder remain frozen and are loaded from the
existing Stage-1 checkpoint.

### Confirmed scope

- Modeled arm: **right arm only**.
- Action: right-arm desired joint position `q_des`, all 8 DoF including the
  gripper:
  `follower_right_mit_command:q`.
- Torque target: measured right-arm joint torque, all 8 DoF:
  `follower_right_state:actual_torque_nm`.
- Stage-2 training sequence length: **10 frames**, matching the upstream
  documented Stage-2 recipe.
- Stage-2 validation rollout: **200 frames**, matching upstream.
- Wrench, joint velocity, and interaction torque are saved in HDF5 but are not
  Stage-2 targets in this iteration.
- Regenerate a unified dataset from the raw collection folders rather than
  modifying the existing HDF5 files in place.

## 2. Upstream Stage-2 semantics

The upstream documented Stage-2 command uses:

```text
dataset.horizon=10
dataset.val_horizon=200
algorithm.noise_scheduler.loss_weighting=uniform
algorithm.sampling_strategy=terminal_only
```

Although the dataset YAML defaults to a horizon of 16, the documented Stage-2
experiment explicitly overrides it to 10. This project will use the documented
experiment settings to minimize behavioral changes.

### 2.1 Meaning of `horizon=10`

A training sample contains 10 consecutive latent/action pairs:

```text
latents: z0 z1 z2 z3 z4 z5 z6 z7 z8 z9
actions: a0 a1 a2 a3 a4 a5 a6 a7 a8 a9
                                          ^
                                  terminal target
```

With `sampling_strategy=terminal_only`, the first nine latent frames are
context and the final latent is the meaningful denoising target. Stage 2 learns
approximately:

```text
(z0 ... z8, a0 ... a9) -> z9
```

Inference remains one-step autoregressive. Each predicted latent is appended to
the history, and the dynamics model is called again using a sliding context
window of at most 10 frames.

Torque prediction will follow the same semantics:

```text
(z0 ... z8, z9_pred, a0 ... a9) -> torque9
```

The training target is the terminal torque in each 10-frame window. Because
training windows slide over the episodes, every eligible frame appears as a
terminal target in some sample. During inference, one torque is produced for
each autoregressively generated latent.

`val_horizon=200` is intentionally longer than the training context and tests
accumulated autoregressive error.

## 3. Existing architecture

### 3.1 Stage-1 autoencoder

For the single head-camera view, the frozen encoder maps:

```text
[B, 3, 128, 128] -> [B, 4, 32, 32]
```

Stage 1 trains the encoder and diffusion decoder. Stage 2 loads both from
`load_ae`, freezes them, and trains the latent dynamics components.

Current preferred Stage-1 artifact:

```text
outputs/world_ft_stage_1/2026-06-25_19-54-07/checkpoints/last.ckpt
```

At inspection time this checkpoint was at epoch 48, global step 7448.

### 3.2 Existing latent dynamics

`CMLatentDynamics` is a spatial-temporal U-Net operating on:

```text
[B, C_latent, T, H_latent, W_latent]
```

Its main path is:

```text
noisy latent sequence
  -> Conv3D
  -> temporal attention
  -> down ResNet blocks with spatial/temporal attention
  -> middle ResNet/attention block
  -> up ResNet blocks with skip connections
  -> Conv3D to latent channels
  -> diffusion-v tensor
```

Actions are embedded by:

```text
action_dim -> 64 -> 128 -> 128 -> action_emb_dim
```

The per-frame action embedding is injected into each conditioned ResNet block
using FiLM scale and shift:

```text
h = norm(h) * (1 + action_scale) + action_shift
```

Changing the action from the original 4-D task primitive to 8-D `q_des` only
changes the first action-embedding layer from `Linear(4, 64)` to
`Linear(8, 64)`. The dynamics network is initialized for Stage 2, so this does
not conflict with the Stage-1 checkpoint.

### 3.3 Why torque cannot be appended to the U-Net output

The existing U-Net output is not a clean latent. It is a diffusion
`v`-parameterization with exactly the same spatial shape as the noisy latent:

```text
[B, C_latent, T, H_latent, W_latent]
```

`DDPMScheduler.CTM_calc_out()` performs tensor arithmetic on this output to
recover the denoised transition. An 8-D torque vector:

```text
[B, T, 8]
```

is not compatible with that spatial diffusion tensor.

The low-level `CMLatentDynamics.forward()` must therefore remain tensor-only.
Returning `(latent_output, torque_output)` from it would break CTM scheduler
arithmetic. Torque must also remain outside diffusion because it is a
deterministic physical prediction, not a spatial variable that should be
corrupted and denoised at every diffusion noise level.

## 4. Proposed torque predictor

Add a separate `TorquePredictor` owned by `LatentWorldModel` or by the dynamics
module but invoked through a separate method. Do not change the return type of
`CMLatentDynamics.forward()` and do not make `EinopsWrapper` tuple-aware.

### 4.1 Per-frame latent encoder

For each clean or predicted latent map:

```text
[B, 4, 32, 32]
  -> Conv2D(4, 32, stride=2)       # [B, 32, 16, 16]
  -> SiLU
  -> Conv2D(32, 64, stride=2)      # [B, 64, 8, 8]
  -> SiLU
  -> Conv2D(64, 128, stride=2)     # [B, 128, 4, 4]
  -> SiLU
  -> global average pooling         # [B, 128]
```

This provides more capacity than directly averaging the four Stage-1 latent
channels.

### 4.2 Per-frame action encoder

```text
q_des [B, 8]
  -> Linear(8, 64)
  -> SiLU
  -> Linear(64, 64)
```

Concatenate the latent and action features:

```text
[B, T, 128] + [B, T, 64] -> [B, T, 192]
```

### 4.3 Temporal predictor

Use a small causal temporal model over the 10-frame sequence:

```text
[B, T, 192]
  -> two-layer causal temporal Conv1D or GRU, hidden size 128
  -> Linear(128, 8)
  -> [B, T, 8]
```

Supervise torque **densely** over all 10 frames (not only the terminal frame).
Torque is a deterministic per-frame quantity, so it does not need the diffusion's
terminal-only constraint, and the clean context latents are already available:

```python
# torque_pred: [B, T, 8] from the causal temporal head
torque_loss = mse(torque_pred, torque_target_norm)   # averaged over all T frames
```

Frames 0..8 are supervised against their clean encoder latents `z_gt`; frame 9 is
supervised against the dynamics-predicted latent `z9_pred` so the torque gradient
still couples into the dynamics U-Net. This is ~10x the torque signal per sample
at negligible cost.

A temporal model is required because torque depends on motion, velocity,
tracking error, inertia, contact state, and recent history. One latent image and
one command are generally insufficient to infer these quantities.

The initial implementation should use a two-layer GRU unless profiling shows a
reason to prefer causal Conv1D. A GRU gives explicit causal state processing,
handles the fixed 10-frame window directly, and keeps the implementation small.

### 4.4 Latents presented to the torque head

During training:

1. encode all 10 RGB frames with the frozen Stage-1 encoder -> clean `z_gt[0..9]`;
2. run the existing CTM dynamics transition -> predicted terminal latent `z9_pred`;
3. form the torque-head latent sequence as `[z_gt[0..8], z9_pred]` (clean context,
   predicted terminal);
4. combine it with actions `a0...a9`;
5. predict the full torque sequence `torque[0..9]` and supervise every frame.

The terminal frame's torque-loss gradient flows through `z9_pred` into the
dynamics U-Net (not into the frozen Stage-1 encoder), encouraging the predicted
latent to preserve torque-relevant information; frames 0..8 train the head on
clean latents. (At inference the context is the previously predicted latents -- a
mild, expected teacher-forcing gap, consistent with how the dynamics context is
itself trained near-clean.)

If this coupled training is initially unstable, the fallback is a short
teacher-forced warm-up using clean `z9`, followed by training with `z9_pred`.
Teacher forcing is a fallback, not the default.

## 5. Raw input data

Each raw episode is stored under:

```text
/home/peng33/supernova/projects/world_ft/<TIMESTAMP>_PDT/
```

There are six current episodes:

```text
2026_06_24_14_51_42_PDT
2026_06_24_14_52_32_PDT
2026_06_24_14_53_12_PDT
2026_06_24_14_54_32_PDT
2026_06_24_14_55_47_PDT
2026_06_24_14_57_04_PDT
```

The first five are training episodes and the final episode is validation.

Relevant files and streams:

| Source | Typical shape/rate | Use |
|---|---:|---|
| `camera_head.mp4` | ~30 FPS | RGB observation |
| `follower_right_mit_command:q` | `(M,8)`, ~2 kHz | action `q_des` |
| `follower_right_mit_command:timestamp` | `(M,)` | action time |
| `follower_right_state:actual_torque_nm` | `(N,8)`, ~627 Hz | torque target |
| `follower_right_state:actual_angle_rad` | `(N,8)` | measured position |
| `follower_right_state:actual_velocity_radps` | `(N,8)` | saved auxiliary data |
| `follower_right_state:timestamp` | `(N,8)` | per-joint state times |
| `follower_left_state:actual_angle_rad` | `(N,8)` | compatibility state |
| `follower_left_state:timestamp` | `(N,8)` | per-joint state times |
| `follower_controller_state:wrench_hand_tcp_R` | `(K,6)`, ~509 Hz | saved only |
| `follower_controller_state:wrench_hand_tcp_inertiacomp_R` | `(K,6)` | saved only |
| `follower_controller_state:tau_interaction_R` | `(K,8)` | saved only |
| `follower_controller_state:timestamp` | `(K,)` | controller time |

Video PTS are epoch-based. For example, the first inspected frame of the final
episode has:

```text
pts_time = 1782338224.505
```

The raw command and controller timestamp arrays contain duplicate and backward
steps. They must be sanitized before interpolation.

The eighth DoF is believed to be the gripper, but the exact source-to-Trossen
mapping must be verified against robot semantics before conversion. The current
converter's `arm7()` uses source index 6 as the gripper, which conflicts with
the stated eighth-DoF convention and must not be copied blindly.

## 6. Unified HDF5 schema

One `episode_<i>.hdf5` is written per episode. Every time-series field has
length `T`, aligned to retained video frames.

| Key | Shape | Source/use |
|---|---:|---|
| `action` | `(T,8)` | right `mit_command:q`; Stage-2 action |
| `timestamp` | `(T,)` | retained video-frame epoch PTS |
| `obs/images/camera_1_color` | `(T,H,W,3)` uint8 | native RGB frames |
| `obs/joint_pos` | `(T,14)` | mapped right7 + left7 for compatibility |
| `obs/full_joint_pos` | `(T,16)` | measured right8 + left8 |
| `obs/world_t_robot_base` | `(T,2,4,4)` | identity stub unless calibrated |
| `obs/joint_torque` | `(T,8)` | right measured torque; Stage-2 target |
| `obs/joint_vel` | `(T,8)` | right measured velocity; saved only |
| `obs/wrench_hand_tcp` | `(T,6)` | right raw wrench; saved only |
| `obs/wrench_hand_tcp_inertiacomp` | `(T,6)` | right compensated wrench; saved only |
| `obs/tau_interaction` | `(T,8)` | right interaction torque; saved only |

Keep camera intrinsics/extrinsics stubs required by existing loader conventions.
Images remain at native resolution; center-crop and resize to 128x128 remain in
the dataset loader.

## 7. Converter changes

Rewrite `scripts/data_collection/convert_world_ft.py` as the single raw-to-HDF5
converter.

Proposed CLI:

```bash
python scripts/data_collection/convert_world_ft.py \
  --src /home/peng33/supernova/projects/world_ft \
  --out data/world_ft_v2 \
  --n_val 1 \
  --target_hz 10
```

Use `data/world_ft_v2` initially so the current Stage-1 dataset remains intact
until image/latent compatibility has been verified.

### 7.1 Episode discovery and split

- Discover all `*_PDT` episode directories and sort them by name.
- Place the final `--n_val` episodes in `val/`; place the rest in `train/`.
- Remove stale cache files under the new output after conversion.

### 7.2 Frame selection and timing

- Decode with PyAV and use `frame.pts * stream.time_base` as the epoch timestamp.
- Preserve the old Stage-1 frame-index selection exactly where possible. The
  current dataset used every third decoded frame for approximately 10 Hz.
- Store selected frames at native resolution.
- Do not synthesize frame times from a robot-state start time.

### 7.3 Timestamp sanitization

Before using `np.interp`, every source stream must be made strictly monotonic:

1. reject non-finite timestamp/value rows;
2. stable-sort by timestamp;
3. collapse duplicate timestamps deterministically, preferably keeping the last
   sample at each duplicate timestamp;
4. assert strict monotonicity after cleanup;
5. record cleanup counts for diagnostics.

For state arrays with per-joint timestamps `(N,8)`, interpolate each DoF using
its own timestamp column. Do not replace per-joint times with their mean.

For controller fields sharing one `(K,)` timestamp stream, sanitize once and
apply the same retained indices to all controller values.

### 7.4 Resampling

Interpolate each low-dimensional stream onto selected video PTS:

- `action`: right command `q`;
- `joint_torque`: right measured torque;
- `joint_vel`: right measured velocity;
- `joint_pos` and `full_joint_pos`: measured right/left angles;
- right raw and compensated wrench;
- right interaction torque.

Assert that selected frame times lie within the usable source interval, or
explicitly report any endpoint clamping performed by `np.interp`.

### 7.5 Image/latent compatibility gate

The Stage-1 checkpoint can be reused only if regenerated image inputs match the
old Stage-1 dataset.

Validation:

1. pair old and regenerated episodes by epoch range;
2. compare selected native RGB frames exactly;
3. compare loader-produced 128x128 tensors;
4. encode representative frames with the Stage-1 encoder;
5. compare latent tensors within a documented tolerance.

If image tensors/latents match, reuse the current Stage-1 checkpoint. If not,
retrain Stage 1 on the unified dataset before Stage 2.

## 8. Dataset loader changes

Modify
`interactive_world_sim/datasets/latent_dynamics/real_aloha_dataset.py`.

### 8.1 New action mode

Add:

```text
action_mode: right_qpos
```

For this mode:

- read `file["action"]` directly;
- do not call `joint_pos_to_action_primitive`;
- do not apply FK;
- preserve shape `(T,8)`.

All existing control modes must remain unchanged.

### 8.2 Explicit torque loading

The current replay-buffer conversion does not automatically copy arbitrary
low-dimensional `obs` fields. Explicitly load:

```text
file["obs"]["joint_torque"]
```

into replay-buffer key:

```text
joint_torque
```

Ensure `SequenceSampler` returns the torque sequence aligned with images and
actions, and `_sample_to_data()` returns:

```python
batch["joint_torque"]  # [B, T, 8]
```

The same mechanism may load the other saved auxiliary fields if configured,
but they should not be included in batches by default.

### 8.3 Normalization

- Keep 8-D action range normalization for compatibility with existing action
  conditioning.
- Add per-joint Gaussian standardization for torque:

```text
(torque - training_mean) / training_std
```

Use only training-split statistics. Add a helper based on
`SingleFieldLinearNormalizer.fit(..., mode="gaussian")` or an equivalent
manual normalizer from `array_to_stats`.

Do not rely on the current suffix-based low-dimensional normalizer dispatch;
`joint_torque` is not supported by that code and needs an explicit branch.

Configuration:

```yaml
shape_meta:
  action:
    shape: [8]
  obs:
    joint_torque:
      shape: [8]
      type: low_dim
```

Images remain the only entries in `dataset.obs_keys`; `joint_torque` is a
supervision target, not an encoder observation channel.

## 9. Model and training changes

### 9.1 Model construction

In `LatentWorldModel._build_model()`:

- instantiate the existing `CMLatentDynamics` with `action_dim=8`;
- instantiate `TorquePredictor` with `latent_dim=4`, `action_dim=8`,
  `torque_dim=8`;
- keep encoder and decoder loading unchanged;
- freeze encoder and decoder for Stage 2.

`CMLatentDynamics.forward()` continues to return only the latent diffusion
tensor. `EinopsWrapper` requires no tuple-related change.

### 9.2 Stage-2 training step

For each 10-frame batch:

1. normalize RGB and action;
2. encode all images using the frozen encoder;
3. execute the existing Stage-2 CTM latent loss;
4. obtain the predicted clean terminal latent from the CTM transition;
5. construct the torque-head latent sequence `[z_gt[0..8], z9_pred]`;
6. predict the full normalized torque sequence `[B, 10, 8]`;
7. compute normalized per-joint MSE over all 10 frames.

Loss:

```text
total_loss = dynamics_loss
           + torque_loss_weight * torque_mse_normalized
```

Start with:

```text
torque_loss_weight = 1.0
```

Log at least:

```text
training/loss
training/dyn_loss
training/torque_mse
training/torque_rmse_nm
```

`torque_rmse_nm` is computed after denormalizing and is for interpretation, not
optimization.

### 9.3 Optimizer

For Stage 2, optimize:

- all `self.dynamics` parameters;
- all `self.torque_predictor` parameters.

Do not optimize the encoder or decoder.

### 9.4 Inference API

Keep the low-level diffusion call unchanged.

At the high-level rollout API, support returning:

```python
latent_future, torque_future
```

For backward compatibility, either add a flag such as `return_torque=False` or
add a dedicated method such as `dynamics_and_torque_forward()`. Existing
inference callers that expect only a latent tensor must continue to work.

At each autoregressive step:

1. predict the next latent using the existing dynamics sampler;
2. append it to the latent context;
3. run the torque predictor on the current latent/action window;
4. append the terminal torque prediction.

### 9.5 Validation

Use `val_horizon=200` and autoregressively predict:

- future latents/images;
- one right-arm torque vector per generated frame.

Log:

```text
validation/dyn_loss
validation/torque_mse
validation/torque_rmse_nm
```

Also report per-joint RMSE in N m so gripper or individual arm-joint failures
are not hidden by one aggregate metric.

## 10. Configuration and launch script

Update or override:

```yaml
algorithm:
  action_dim: 8
  torque_dim: 8
  torque_loss_weight: 1.0
  training_stage: 2
  sampling_strategy: terminal_only
  noise_scheduler:
    loss_weighting: uniform

dataset:
  action_mode: right_qpos
  horizon: 10
  val_horizon: 200
```

Create:

```text
scripts/train/train_world_ft_stage2.sh
```

It should:

- use the unified dataset path;
- load the validated Stage-1 checkpoint;
- use the upstream Stage-2 horizon/noise settings;
- use single-camera `camera_1_color`;
- use an 8-D action;
- write under `outputs/world_ft_stage_2/`;
- preserve the existing rolling `last.ckpt` and auto-resume behavior;
- obtain W&B credentials from the environment only.

No API key or other credential may be stored in the script.

## 11. File-by-file implementation list

1. `scripts/data_collection/convert_world_ft.py`
   - unified raw converter;
   - exact video PTS;
   - timestamp sorting/deduplication;
   - per-DoF interpolation;
   - corrected joint/gripper mapping;
   - unified HDF5 schema.
2. `interactive_world_sim/datasets/latent_dynamics/real_aloha_dataset.py`
   - `right_qpos` mode;
   - explicit torque loading;
   - aligned torque samples;
   - Gaussian torque normalizer.
3. `interactive_world_sim/algorithms/latent_dynamics/models/cm_latent_dynamics.py`
   - retain tensor-only dynamics `forward`;
   - optionally define the separate `TorquePredictor` here, or place it in a
     dedicated model file.
4. `interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py`
   - build torque predictor;
   - terminal torque loss;
   - optimizer parameters;
   - metrics;
   - backward-compatible inference output.
5. `configurations/algorithm/latent_world_model.yaml`
   - torque dimensions/weight and 8-D action support.
6. `configurations/dataset/real_aloha_dataset.yaml`
   - `right_qpos`, 8-D action, torque metadata.
7. `scripts/train/train_world_ft_stage2.sh`
   - upstream Stage-2 horizon/noise settings and safe checkpoint resume.

No tuple-related modification is required in
`interactive_world_sim/algorithms/models/utils.py`.

## 12. Validation and test plan

### 12.1 Converter

- Convert all six episodes to a new output directory.
- Assert identical length `T` for every per-frame dataset.
- Assert finite values and expected dimensions.
- Assert timestamps are strictly increasing after selection.
- Report duplicate/backward timestamp cleanup counts per raw stream.
- Check `q_des` versus measured angle tracking.
- Check torque ranges against raw arrays.
- Check wrench/contact events against the synchronized video.
- Confirm five training episodes and one validation episode.

### 12.2 Stage-1 compatibility

- Compare old/new selected frames.
- Compare loader outputs.
- Compare Stage-1 latents.
- Record whether the existing checkpoint is reusable.

### 12.3 Dataset

Pull a training batch and assert:

```text
RGB:          [B, 10, 3, 128, 128]
action:       [B, 10, 8]
joint_torque: [B, 10, 8]
```

Confirm action normalization and per-joint Gaussian torque normalization use
training statistics.

Pull a validation batch and confirm a 200-frame sequence.

### 12.4 Model

- Verify the original dynamics output shape is unchanged.
- Verify `TorquePredictor` maps latent/action sequences to `[B,T,8]`.
- Verify terminal selection gives `[B,8]`.
- Verify torque loss backpropagates into the torque predictor and dynamics but
  not the frozen encoder/decoder.
- Verify existing latent-only inference callers still work.

### 12.5 Training

- Run a short overfit test on one small batch.
- Run a several-hundred-step Stage-2 smoke test.
- Confirm both dynamics and torque losses are finite and decrease.
- Confirm normalized torque MSE and denormalized N m metrics are sensible.
- Confirm validation performs a 200-frame autoregressive rollout.
- Confirm one rolling `last.ckpt` is written and resume restores optimizer,
  scheduler, and global step.

## 13. Implementation order

1. Rewrite the converter and generate `data/world_ft_v2`.
2. Validate timestamp alignment, joint mapping, and HDF5 contents.
3. Run the Stage-1 image/latent compatibility gate.
4. Add `right_qpos`, torque loading, and normalization to the dataset.
5. Add the separate temporal `TorquePredictor`.
6. Add Stage-2 torque loss, metrics, optimizer parameters, and inference output.
7. Add configuration and `train_world_ft_stage2.sh`.
8. Run unit tests, one-batch overfit, Stage-2 smoke training, and then full
   training.

## 14. Resolved decisions

- Use upstream documented Stage-2 `horizon=10`, not 8.
- Use upstream `val_horizon=200`.
- Use upstream `terminal_only` sampling and uniform noise-loss weighting.
- Supervise torque densely over all 10 frames per training window (clean context
  latents + predicted terminal); produce one torque per autoregressive inference
  step.
- Use a separate deterministic temporal torque predictor after latent
  denoising.
- Keep `CMLatentDynamics.forward()` tensor-only.
- Do not modify `EinopsWrapper` for tuple outputs.
- Use per-joint Gaussian torque standardization.
- Initially generate `data/world_ft_v2` to preserve the existing Stage-1 data.
- Save wrench, velocity, and interaction torque without training on them.

## 15. Remaining verification gates

These are implementation-time checks, not unresolved architecture choices:

1. Verify/fix the 8-DoF-to-Trossen-7 gripper mapping in `arm7()` (currently uses
   src idx 6, dropping idx 7). NOTE: this only affects `obs/joint_pos` (the FK
   path that `right_qpos` bypasses, and Stage 1 ignores actions) -- it does **not**
   affect the Stage-2 action (full 8-D `q_des`) or torque target (full 8-D). Fix
   for cleanliness; not a Stage-2 correctness blocker.
2. Verify regenerated images and Stage-1 latents against the existing dataset.
3. Tune `torque_loss_weight` only if measured gradient/loss scales show that
   the initial value of 1.0 is unbalanced.
4. Use teacher-forced terminal latents only if direct coupled torque training is
   demonstrably unstable.
