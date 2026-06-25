"""Convert world_ft hammer recordings -> interactive_world_sim episode_N.hdf5.

Source layout (per episode dir <ts>_PDT/):
  camera_head.mp4
  low_dim_npys/<stream>:<field>.npy   (multi-rate; right arm is the active one)

Target schema (matches data/mini/pusht/train/episode_*.hdf5), only the fields the
RealAlohaDataset loader actually reads are populated with real/aligned data; the
rest are valid-shaped stubs. NOTE: the loader recomputes `action` from joint_pos
via Trossen FK, and stage-1 (encoder/decoder) training ignores actions entirely --
so the stub joints only need to be finite and time-varying (so the action
range-normalizer doesn't divide by zero).

Video is subsampled to ~TARGET_HZ to match the repo's 10 Hz collection convention
and keep file sizes reasonable. All per-step arrays share the same length T.
"""
import argparse
import glob
import os

import cv2
import h5py
import numpy as np

TARGET_HZ = 10.0
IMG_KEY = "camera_1_color"  # reuse existing shape_meta entry (single view)


def L(ep, stream, field):
    return np.load(os.path.join(ep, "low_dim_npys", f"{stream}:{field}.npy"))


def interp_to(ref_t, src_t, src_v):
    src_t = np.asarray(src_t, float)
    if src_v.ndim == 1:
        return np.interp(ref_t, src_t, src_v)
    out = np.empty((len(ref_t), src_v.shape[1]), np.float32)
    for d in range(src_v.shape[1]):
        out[:, d] = np.interp(ref_t, src_t, src_v[:, d])
    return out


def read_frames(video_path, stride):
    """Return (frames uint8 [N,H,W,3] RGB, count)."""
    cap = cv2.VideoCapture(video_path)
    frames, i = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return np.stack(frames, 0), i


def arm7(ang8):
    """8-dim arm angles -> 7-dim Trossen layout (6 arm joints + 1 gripper)."""
    out = np.zeros((ang8.shape[0], 7), np.float32)
    out[:, :6] = ang8[:, :6]
    out[:, 6] = ang8[:, 6]  # use joint idx 6 as gripper
    return out


def convert_episode(ep, out_path):
    fps = cv2.VideoCapture(os.path.join(ep, "camera_head.mp4")).get(cv2.CAP_PROP_FPS)
    stride = max(1, round(fps / TARGET_HZ))
    frames, n_total = read_frames(os.path.join(ep, "camera_head.mp4"), stride)
    T, H, W, _ = frames.shape

    # reference clock: right-arm state stream, frame i at t0 + i*stride/fps
    st = L(ep, "follower_right_state", "timestamp")
    st = st[:, 0] if st.ndim == 2 else st
    t0 = float(st.min())
    frame_t = t0 + (np.arange(T) * stride) / fps

    r_ang = interp_to(frame_t, st, L(ep, "follower_right_state", "actual_angle_rad"))
    lt = L(ep, "follower_left_state", "timestamp")
    lt = lt[:, 0] if lt.ndim == 2 else lt
    l_ang = interp_to(frame_t, lt, L(ep, "follower_left_state", "actual_angle_rad"))

    # joint_pos (T,14)=2x7 ; full_joint_pos (T,16)=2x8 ; robot0=right, robot1=left
    joint_pos = np.concatenate([arm7(r_ang), arm7(l_ang)], axis=1).astype(np.float32)
    full_joint_pos = np.concatenate([r_ang[:, :8], l_ang[:, :8]], 1).astype(np.float32)

    base = np.tile(np.eye(4, dtype=np.float32), (T, 2, 1, 1))  # (T,2,4,4)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("action", data=np.zeros((T, 4), np.float32))  # recomputed by loader
        f.create_dataset("joint_action", data=joint_pos)
        f.create_dataset("timestamp", data=frame_t.astype(np.float64))
        obs = f.create_group("obs")
        obs.create_dataset("joint_pos", data=joint_pos)
        obs.create_dataset("full_joint_pos", data=full_joint_pos)
        obs.create_dataset("ee_pos", data=joint_pos)  # stub, not read by loader
        obs.create_dataset("world_t_robot_base", data=base)
        imgs = obs.create_group("images")
        imgs.create_dataset(
            IMG_KEY, data=frames, dtype="uint8",
            chunks=(1, H, W, 3), compression="gzip", compression_opts=4,
        )
        imgs.create_dataset(f"{IMG_KEY[:-6]}_intrinsics", data=np.zeros((T, 3, 3), np.float32))
        imgs.create_dataset(f"{IMG_KEY[:-6]}_extrinsics", data=np.tile(np.eye(4, dtype=np.float32), (T, 1, 1)))
    return T, n_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/peng33/supernova/projects/world_ft")
    ap.add_argument("--out", default="/home/peng33/interactive_world_sim/data/world_ft")
    ap.add_argument("--n_val", type=int, default=1, help="episodes held out for val")
    args = ap.parse_args()

    eps = sorted(glob.glob(os.path.join(args.src, "2026_06_24_14_*_PDT")))
    assert eps, f"no episodes under {args.src}"
    val_eps = eps[-args.n_val:] if args.n_val else []
    train_eps = eps[: len(eps) - args.n_val]

    for split, split_eps in [("train", train_eps), ("val", val_eps)]:
        d = os.path.join(args.out, split)
        os.makedirs(d, exist_ok=True)
        for i, ep in enumerate(split_eps):
            out = os.path.join(d, f"episode_{i}.hdf5")
            T, n = convert_episode(ep, out)
            sz = os.path.getsize(out) / 1e6
            print(f"[{split}] {os.path.basename(ep)} -> episode_{i}.hdf5  "
                  f"T={T} (from {n} frames)  {sz:.0f} MB")


if __name__ == "__main__":
    main()
