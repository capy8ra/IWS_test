"""Convert raw world_ft recordings -> interactive_world_sim episode_N.hdf5.

Unified converter for BOTH training stages.

Source layout (per episode dir ``<ts>_PDT/``)::

    camera_head.mp4
    low_dim_npys/<stream>:<field>.npy   (multi-rate; right arm is the active one)

Output (per episode), all per-step arrays length ``T`` aligned to retained video
frames (subsampled to ~``TARGET_HZ``)::

    action                              (T, 8)   right q_des  (mit_command:q)   [stage-2 action]
    timestamp                           (T,)     retained video-frame epoch PTS
    obs/images/camera_1_color           (T,H,W,3) uint8 RGB
    obs/images/camera_1_{intrinsics,extrinsics}  stubs (loader convention)
    obs/joint_pos                       (T, 14)  arm7(right) + arm7(left)        [stage-1/FK compat]
    obs/full_joint_pos                  (T, 16)  right8 + left8 measured angle
    obs/ee_pos                          (T, 14)  stub (= joint_pos), not read by loader
    obs/world_t_robot_base              (T, 2, 4, 4)  identity stub
    obs/joint_torque                    (T, 8)   right measured torque           [stage-2 target]
    obs/joint_vel                       (T, 8)   right measured velocity         [saved only]
    obs/wrench_hand_tcp                 (T, 6)   right hand wrench               [saved only]
    obs/wrench_hand_tcp_inertiacomp     (T, 6)   right inertia-comp wrench       [saved only]
    obs/tau_interaction                 (T, 8)   right interaction torque        [saved only]

Notes
-----
* Frame epoch times come from the real video PTS (``pts * time_base``), NOT a
  synthesized robot-clock, so low-dim resampling is correctly aligned.
* Frame selection mirrors the original stage-1 pipeline: every ``stride``-th
  decoded frame (``stride = round(fps / TARGET_HZ)``), stored at native
  resolution; the dataset loader still center-crops + resizes to 128.
* Raw command/controller timestamps contain duplicate and backward steps and are
  sanitized (finite -> stable-sort -> dedup-keep-last -> strict monotonic) before
  ``np.interp``. State streams carry per-joint timestamps and are interpolated
  per DoF using each joint's own timestamp column.
"""
import argparse
import glob
import os

import av
import h5py
import numpy as np

TARGET_HZ = 10.0
IMG_KEY = "camera_1_color"  # reuse existing shape_meta entry (single view)


def L(ep, stream, field):
    """Load low_dim_npys/<stream>:<field>.npy."""
    return np.load(os.path.join(ep, "low_dim_npys", f"{stream}:{field}.npy"))


def _monotonic(t, v):
    """Sanitize a (timestamp, value) series into strictly-increasing form.

    finite-only -> stable sort by t -> keep the LAST sample at duplicate t.
    Returns (t_clean, v_clean, n_dropped).
    """
    t = np.asarray(t, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    finite = np.isfinite(t) & (np.isfinite(v) if v.ndim == 1 else np.isfinite(v).all(1))
    t, v = t[finite], v[finite]
    order = np.argsort(t, kind="stable")
    t, v = t[order], v[order]
    keep = np.ones(len(t), dtype=bool)
    keep[:-1] = t[1:] != t[:-1]  # last of each equal-timestamp run
    n_dropped = int((~finite).sum() + (~keep).sum())
    return t[keep], v[keep], n_dropped


def interp_field(ref_t, src_t, src_v, diag=None, name=""):
    """Interpolate ``src_v`` onto ``ref_t``.

    ``src_t`` may be 1-D (shared clock) or 2-D (N, D) per-DoF timestamps; in the
    latter case each column is interpolated against its own timestamp column.
    """
    ref_t = np.asarray(ref_t, dtype=np.float64)
    if src_v.ndim == 1:
        t, v, nd = _monotonic(src_t if src_t.ndim == 1 else src_t[:, 0], src_v)
        if diag is not None:
            diag[name] = nd
        return np.interp(ref_t, t, v).astype(np.float32)
    D = src_v.shape[1]
    out = np.empty((len(ref_t), D), dtype=np.float32)
    dropped = 0
    for d in range(D):
        td = src_t[:, d] if src_t.ndim == 2 else src_t
        t, v, nd = _monotonic(td, src_v[:, d])
        dropped += nd
        out[:, d] = np.interp(ref_t, t, v)
    if diag is not None:
        diag[name] = dropped
    return out


def read_frames_pts(video_path, stride):
    """Decode video; return (frames uint8 [T,H,W,3] RGB, pts_epoch [T], n_total).

    Keeps every ``stride``-th decoded frame, matching the original stage-1
    selection. Frame time = ``pts * time_base`` (epoch seconds).
    """
    frames, ts, i = [], [], 0
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        tb = stream.time_base
        for frame in container.decode(stream):
            if i % stride == 0:
                if frame.pts is None:
                    raise ValueError(f"frame {i} has no PTS in {video_path}")
                frames.append(frame.to_ndarray(format="rgb24"))
                ts.append(float(frame.pts * tb))
            i += 1
    return np.stack(frames, 0), np.asarray(ts, dtype=np.float64), i


def video_fps(video_path):
    with av.open(video_path) as container:
        return float(container.streams.video[0].average_rate)


def arm7(ang8):
    """8-DoF source angles -> 7-DoF Trossen layout (6 arm joints + 1 gripper).

    OpenArm exposes 7 arm joints + gripper (gripper at source index 7). Trossen
    has only 6 arm joints, so this is a lossy compatibility mapping used ONLY for
    ``obs/joint_pos`` (the FK path that ``right_qpos`` bypasses, and stage-1
    ignores actions). It does not affect the stage-2 action or torque target,
    which use the full 8-DoF arrays. Source index 6 (the 7th arm joint) is dropped;
    source index 7 (gripper) maps to the Trossen gripper slot.
    """
    out = np.zeros((ang8.shape[0], 7), np.float32)
    out[:, :6] = ang8[:, :6]
    out[:, 6] = ang8[:, 7]  # gripper = source idx 7 (incl-gripper convention)
    return out


def convert_episode(ep, out_path):
    video = os.path.join(ep, "camera_head.mp4")
    fps = video_fps(video)
    stride = max(1, round(fps / TARGET_HZ))
    frames, frame_t, n_total = read_frames_pts(video, stride)
    T, H, W, _ = frames.shape

    diag = {}
    F = lambda stream, field: interp_field(  # noqa: E731
        frame_t, L(ep, stream, "timestamp"), L(ep, stream, field),
        diag=diag, name=f"{stream}:{field}",
    )

    # right arm (active) -- full 8 DoF
    r_qdes = F("follower_right_mit_command", "q")             # (T,8) action
    r_torque = F("follower_right_state", "actual_torque_nm")  # (T,8) target
    r_vel = F("follower_right_state", "actual_velocity_radps")  # (T,8)
    r_ang = F("follower_right_state", "actual_angle_rad")     # (T,8)
    l_ang = F("follower_left_state", "actual_angle_rad")      # (T,8)

    # controller wrench / interaction torque (saved only)
    r_wrench = F("follower_controller_state", "wrench_hand_tcp_R")              # (T,6)
    r_wrench_ic = F("follower_controller_state", "wrench_hand_tcp_inertiacomp_R")  # (T,6)
    r_tau_int = F("follower_controller_state", "tau_interaction_R")             # (T,8)

    joint_pos = np.concatenate([arm7(r_ang), arm7(l_ang)], axis=1).astype(np.float32)  # (T,14)
    full_joint_pos = np.concatenate([r_ang, l_ang], axis=1).astype(np.float32)         # (T,16)
    base = np.tile(np.eye(4, dtype=np.float32), (T, 2, 1, 1))                          # (T,2,4,4)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("action", data=r_qdes.astype(np.float32))   # right q_des (8) [stage-2]
        f.create_dataset("timestamp", data=frame_t.astype(np.float64))
        obs = f.create_group("obs")
        obs.create_dataset("joint_pos", data=joint_pos)
        obs.create_dataset("full_joint_pos", data=full_joint_pos)
        obs.create_dataset("ee_pos", data=joint_pos)  # stub, not read by loader
        obs.create_dataset("world_t_robot_base", data=base)
        obs.create_dataset("joint_torque", data=r_torque.astype(np.float32))   # [stage-2 target]
        obs.create_dataset("joint_vel", data=r_vel.astype(np.float32))
        obs.create_dataset("wrench_hand_tcp", data=r_wrench.astype(np.float32))
        obs.create_dataset("wrench_hand_tcp_inertiacomp", data=r_wrench_ic.astype(np.float32))
        obs.create_dataset("tau_interaction", data=r_tau_int.astype(np.float32))
        imgs = obs.create_group("images")
        imgs.create_dataset(
            IMG_KEY, data=frames, dtype="uint8",
            chunks=(1, H, W, 3), compression="gzip", compression_opts=4,
        )
        imgs.create_dataset(f"{IMG_KEY[:-6]}_intrinsics", data=np.zeros((T, 3, 3), np.float32))
        imgs.create_dataset(
            f"{IMG_KEY[:-6]}_extrinsics", data=np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
        )
    return T, n_total, diag


def main():
    global TARGET_HZ
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/peng33/supernova/projects/world_ft")
    ap.add_argument("--out", default="data/world_ft_v2")
    ap.add_argument("--n_val", type=int, default=1, help="episodes held out for val")
    ap.add_argument("--target_hz", type=float, default=TARGET_HZ)
    args = ap.parse_args()
    TARGET_HZ = args.target_hz

    eps = sorted(glob.glob(os.path.join(args.src, "*_PDT")))
    assert eps, f"no *_PDT episodes under {args.src}"
    val_eps = eps[-args.n_val:] if args.n_val else []
    train_eps = eps[: len(eps) - args.n_val]
    print(f"found {len(eps)} episodes; {len(train_eps)} train / {len(val_eps)} val")

    for split, split_eps in [("train", train_eps), ("val", val_eps)]:
        d = os.path.join(args.out, split)
        os.makedirs(d, exist_ok=True)
        # drop stale cache so the loader rebuilds it
        for stale in ("cache.zarr.zip",):
            p = os.path.join(d, stale)
            if os.path.exists(p):
                os.remove(p)
        for i, ep in enumerate(split_eps):
            out = os.path.join(d, f"episode_{i}.hdf5")
            T, n, diag = convert_episode(ep, out)
            sz = os.path.getsize(out) / 1e6
            worst = max(diag.items(), key=lambda kv: kv[1]) if diag else ("-", 0)
            print(
                f"[{split}] {os.path.basename(ep)} -> episode_{i}.hdf5  "
                f"T={T} (from {n} frames)  {sz:.0f} MB  "
                f"max_ts_cleanup={worst[1]} ({worst[0]})"
            )


if __name__ == "__main__":
    main()
