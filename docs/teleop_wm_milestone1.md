# World-model teleop — Milestone 1 (visual only)

Drive the trained world model as a **virtual follower**: the human moves the real
OpenArm **right leader**, and the world simulator's follower moves on screen. The
real follower, its camera, and the left arm are **not** needed — the world model
generates the image. **No force feedback yet** (that is Milestone 2).

```
[human] moves real right leader (CAN)
        │  leader_right_state:actual_angle_rad  (redis, ~kHz)
        ▼
[WM follower worker]  (world_ft / iws env, GPU, ~10 Hz)
   action = [leader 7 arm joints , seed gripper]  → normalize
   z ← dynamics_forward(z, action)     # one sim step, from a fixed seed latent
   decode(z) → on-screen image         # the virtual follower
        ▲
        │ (no torque path in milestone 1)
[leader home + float]  (supernova env, ~kHz)
   homes leader to the seed pose, then gravity-comp float (kp=kd=0, tau=gravity)
```

The sim and the leader both start from **one fixed pose** = frame 0 of a chosen
validation episode. The leader is first homed to that pose; the sim latent is seeded
from that episode's first frame. Use the **same** seed episode on both sides.

---

## Files

| File | Repo / env | Role |
|------|------------|------|
| `projects/world_ft/leader_home_and_float.py` | **supernova** / nova env (CAN + pinocchio + h5py) | Homes the right leader to the seed pose (smooth, speed-limited, gains ramped), then holds it in gravity-compensation float. 7 arm joints only; gripper left limp; per-joint torque clamped ±20 N·m. |
| `projects/world_ft/teleop_wm.yaml` | **supernova** (process-compose) | Minimal orchestration: `control_redis` + `leader_mit_controller_right` (can3) + `leader_home_and_float`. |
| `scripts/inference/wm_follower_worker.py` | **world_ft** / iws env (torch + model) | Seeds the sim latent from the episode frame 0, then at ~10 Hz reads the leader angle from redis, steps the world model one frame, decodes, and shows the predicted view. Press `q` to quit. |

The two envs talk over **localhost redis (6379)**; `nova.redis_client` only needs
`redis`+`numpy`, so the worker imports it via `PYTHONPATH=<supernova>`.

---

## Prerequisites

- A trained **Stage-2 (torque) checkpoint** with its Hydra config, e.g.
  `outputs/world_ft_stage_2/<ts>/checkpoints/last.ckpt` and the sibling
  `.hydra/config.yaml`. (Milestone 1 does not use the torque head, but the worker
  loads the torque subclass so the same command works for Milestone 2.)
- A **seed episode** hdf5, e.g. `data/world_ft_v3/val/episode_0.hdf5`. Pick one whose
  frame-0 scene is clean and whose `action[0]` pose the leader can reach comfortably.
- The **right leader** arm powered and on CAN. Per
  `projects/world_ft/collect_data.yaml` the mapping is:
  `can0=follower_left, can1=follower_right, can2=leader_left, can3=leader_right`.
  > ⚠️ The mapping is **not** consistent across every yaml in supernova
  > (`teleop_and_record.yaml` / `teleop_leader_tuning.yaml` differ). Confirm the
  > physical wiring: power `can3` and verify the **right leader** actually moves
  > before trusting it.
- `h5py` in the supernova env (present — `nova/workers/examine_h5.py` uses it).

---

## Pre-flight checks (do these once, before the full run)

1. **torch sees the GPU** (iws env):
   ```bash
   python -c "import torch; print(torch.cuda.get_device_name(0)); \
     print((torch.randn(4,4,device='cuda')@torch.randn(4,4,device='cuda')).sum())"
   ```
   On a 5080 (Blackwell/sm_120) this needs a CUDA 12.8+ torch build; a `sm_120 not
   supported` error means the torch build must be upgraded.
2. **Leader angle readable + sane** (iws env, with redis + leader controller up):
   ```bash
   PYTHONPATH=/path/to/supernova python -c "
   import numpy as np; from nova.redis_client import RedisClient
   c=RedisClient('localhost',6379)
   print(c.stream_get_batch({'leader_right_state:actual_angle_rad':np.ndarray}))"
   ```
   Expect an 8-vector in radians; sanity-check magnitude/sign vs the training action.
3. **Per-step latency**: watch the worker's printed Hz — it should hold ~10 Hz. If
   not, we make the decode async (decode is the slow part; the step itself is fast).

---

## Running Milestone 1

Pull both repos on the robot PC first.

**1) Set the seed-episode path** in `projects/world_ft/teleop_wm.yaml`
(`--seed-episode CHANGE_ME/episode_0.hdf5`) to the real path on that machine. It
**must** match the `--seed-episode` you pass to the worker.

**2) Start the nova side** (supernova env). Roughly place the leader near the seed
pose first, and keep a hand near the e-stop:
```bash
# roughly position the leader, then:
process-compose -f projects/world_ft/teleop_wm.yaml
# wait for "leader float ready"  → homing done, leader is now floating
```

**3) Start the WM worker** in a second terminal (world_ft / iws env):
```bash
cd /path/to/world_ft
PYTHONPATH=/path/to/supernova python scripts/inference/wm_follower_worker.py \
  --ckpt "$(ls -t outputs/world_ft_stage_2/*/checkpoints/last.ckpt | head -1)" \
  --seed-episode data/world_ft_v3/val/episode_0.hdf5
# a window shows the seed frame; move the leader → the virtual follower moves.
# press q to quit.
```

---

## Parameters / knobs

**`leader_home_and_float.py`** (typer, hyphenated flags)

| Flag | Default | Meaning |
|------|---------|---------|
| `--seed-episode` | (required) | hdf5 whose `action[0][:7]` is the homing target |
| `--leader-right` | `leader_right` | redis name of the right leader |
| `--homing-time-s` | `3.0` | nominal homing duration (extended if speed-limited) |
| `--max-vel-deg-s` | `30.0` | per-joint speed cap during homing |
| `--eps-deg` | `1.5` | "homed" tolerance (max joint error) |
| `--gain-ramp-s` | `0.5` | ramp kp/kd 0→full over this time |
| `--freq-hz` | `500` | control loop rate |

Arm gains and the ±20 N·m torque clamp are constants near the top of the file.

**`wm_follower_worker.py`** (argparse)

| Flag | Default | Meaning |
|------|---------|---------|
| `--ckpt` | (required) | Stage-2 (torque) checkpoint |
| `--seed-episode` | (required) | same hdf5 as the leader side |
| `--hz` | `10.0` | sim step rate (training cadence = 10 Hz) |
| `--n-context` | model `n_tokens` | latent context window (sliding) |
| `--leader-right` | `leader_right` | redis name to read the action from |
| `--display-size` | `512` | window size |
| `--device` | `cuda:0` | inference device |

---

## Safety & known limits

- **Homing**: keep a hand near the e-stop; place the leader close to the seed pose
  so the auto move is small. The gripper column is limp (kp=kd=tau=0) and may sag —
  expected, it is unused.
- **Move slowly**: the model was trained at 10 Hz at training-speed motions. Fast
  motion aliases the 10 Hz sampling and goes out-of-distribution → the prediction
  degrades. Ease in.
- **Open-loop drift**: there is no real follower image to re-anchor the latent, so
  the sim drifts over a long session. Keep sessions short; restart to reset.
- **Seed consistency**: the leader pose and the sim scene must start from the same
  episode, or they are misaligned from frame 0 and it compounds.
- **No force feedback**: Milestone 1 is visual only.

---

## Next: Milestone 2 (force feedback)

Add the torque path:
1. In the worker, also compute
   `tau_interaction = torque_head(z, action) − gravity_comp(q_des)` (via
   `OpenArm1Solver`, same as the follower solver's `mit_command:tau`) and publish it
   to redis, e.g. `virtual_follower_right_state:tau_interaction`.
2. Adapt `nova/robot/openarm1/teleop_leader_solver.py` to read that redis value
   **instead of** the real follower term
   (`follower_actual_torque − follower_mit_command:tau`), and command the leader
   `tau = gravity_comp(leader_q) − k·LPF(virtual tau_interaction) + friction/stiction comp`.
3. Because the world model updates torque only at ~10 Hz while the leader loop runs
   at ~2 kHz, low-pass filter + rate-limit + clamp the reflected torque, and add a
   **watchdog** that decays the reflected term to zero if the WM value goes stale.
   Ramp the reflection gain up from 0; keep the e-stop reachable.
