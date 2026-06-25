"""Convert one OpenArm `world_ft` episode folder into an IWS-format episode HDF5
(plus wrist force/torque).

Input  : an episode dir containing
           low_dim_npys/   (follower/leader *_state_* and follower_*_mit_command_* .npy)
           camera_head.mp4
Output : episode_<id>.hdf5 with the same layout the IWS dataset loaders read,
         augmented with wrench/FT, all streams resampled onto the video frames:

   action                       (T, 16)          follower commanded q  [left8, right8]
   robot_bases                  (T, 2, 4, 4)     arm base pose in world [left, right]
   obs/joint_pos                (T, 16)          measured angle         [left8, right8]
   obs/joint_vel                (T, 16)          measured velocity
   obs/ee_pos                   (T, 2, 4, 4)     world_T_hand           [left, right]
   obs/wrist_force              (T, 2, 3)        EE force  Fx,Fy,Fz (world axes) [L, R]
   obs/wrist_torque             (T, 2, 3)        EE moment Mx,My,Mz (world axes) [L, R]
   obs/timestamp                (T,)             seconds since first kept frame
   obs/images/<camera_name>     (T, H, W, 3)     uint8 RGB

Force = residual-torque method (robot-only inverse dynamics from the MJCF model):
   tau_res = tau_measured - rnea(q, qdot, qddot);  W = pinv(J_hand^T) @ tau_res .
The MuJoCo model is used ONLY via its xml path (no nova/pinocchio dependency).
"""
import argparse
import os
from pathlib import Path

import av
import h5py
import mujoco
import numpy as np
from scipy.signal import savgol_filter

DEFAULT_MODEL = "/home/peng33/supernova/nova/robot/openarm1/models/openarm.xml"
ARM_JOINTS = [f"openarm_{{side}}_joint{k}" for k in range(1, 8)]  # 7 arm joints / side


def load_streams(low_dim_dir: Path):
    def L(name):
        return np.load(low_dim_dir / f"{name}.npy")

    streams = {}
    for side in ("left", "right"):
        streams[f"{side}_ang"] = L(f"follower_{side}_state_actual_angle_rad")      # (N,8)
        streams[f"{side}_vel"] = L(f"follower_{side}_state_actual_velocity_radps")
        streams[f"{side}_tau"] = L(f"follower_{side}_state_actual_torque_nm")
        streams[f"{side}_t"] = L(f"follower_{side}_state_timestamp")[:, :7].mean(1)  # (N,) epoch
        streams[f"{side}_cmd_q"] = L(f"follower_{side}_mit_command_q")               # (M,8)
        streams[f"{side}_cmd_t"] = L(f"follower_{side}_mit_command_timestamp")       # (M,) epoch
    return streams


def read_video(mp4_path: Path):
    """Return (frames RGB uint8 [F,H,W,3], epoch_seconds [F])."""
    frames, ts = [], []
    with av.open(str(mp4_path)) as container:
        stream = container.streams.video[0]
        tb = stream.time_base
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            ts.append(float(frame.pts * tb))               # pts were written as epoch microseconds
            frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames), np.asarray(ts)


def interp_cols(tq, ts, arr):
    return np.stack([np.interp(tq, ts, arr[:, j]) for j in range(arr.shape[1])], axis=1)


def smooth_deriv(x, t):
    """Smoothed time-derivative along axis 0 (per column)."""
    n = len(t)
    dt = np.median(np.diff(t))
    if n >= 7:
        win = min(11, n if n % 2 == 1 else n - 1)
        xf = savgol_filter(x, win, 2, axis=0)
        return savgol_filter(x, win, 2, deriv=1, delta=dt, axis=0), xf
    return np.gradient(x, t, axis=0), x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir", help="folder with low_dim_npys/ and camera_head.mp4")
    ap.add_argument("--output", default=None, help="output hdf5 path")
    ap.add_argument("--episode_id", type=int, default=0)
    ap.add_argument("--model_path", default=DEFAULT_MODEL)
    ap.add_argument("--camera_name", default="camera_head_color")
    ap.add_argument("--video_name", default="camera_head.mp4")
    args = ap.parse_args()

    ep = Path(args.episode_dir)
    out = args.output or str(ep / f"episode_{args.episode_id}.hdf5")
    s = load_streams(ep / "low_dim_npys")
    frames, vid_t = read_video(ep / args.video_name)
    print(f"video: {len(frames)} frames {frames.shape[1:]}, "
          f"states L/R: {len(s['left_t'])}/{len(s['right_t'])}")

    # master timeline = video frames clipped to the common time window
    lo = max(vid_t[0], s["left_t"][0], s["right_t"][0], s["left_cmd_t"][0], s["right_cmd_t"][0])
    hi = min(vid_t[-1], s["left_t"][-1], s["right_t"][-1], s["left_cmd_t"][-1], s["right_cmd_t"][-1])
    keep = (vid_t >= lo) & (vid_t <= hi)
    t = vid_t[keep]
    frames = frames[keep]
    T = len(t)
    print(f"kept {T} frames in [{lo:.3f},{hi:.3f}] epoch, {1/np.median(np.diff(t)):.1f} Hz")

    # resample every stream onto the frame timeline
    ang = {sd: interp_cols(t, s[f"{sd}_t"], s[f"{sd}_ang"]) for sd in ("left", "right")}
    vel = {sd: interp_cols(t, s[f"{sd}_t"], s[f"{sd}_vel"]) for sd in ("left", "right")}
    tau = {sd: interp_cols(t, s[f"{sd}_t"], s[f"{sd}_tau"]) for sd in ("left", "right")}
    cmd = {sd: interp_cols(t, s[f"{sd}_cmd_t"], s[f"{sd}_cmd_q"]) for sd in ("left", "right")}

    # MuJoCo model: robot-only rigid-body dynamics + frame kinematics
    m = mujoco.MjModel.from_xml_path(args.model_path)
    m.dof_damping[:] = 0
    m.dof_frictionloss[:] = 0
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    d = mujoco.MjData(m)

    def qadr(side):
        return np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j.format(side=side))]
                         for j in ARM_JOINTS])

    dofs = {"left": qadr("left"), "right": qadr("right")}  # nq==nv, addr coincide
    hand = {sd: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{sd}_hand") for sd in ("left", "right")}
    base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "openarm_body_link0")

    # smoothed accelerations for the inertial term
    acc = {sd: smooth_deriv(vel[sd][:, :7], t)[0] for sd in ("left", "right")}

    wrench = {sd: np.zeros((T, 6)) for sd in ("left", "right")}
    ee = {sd: np.zeros((T, 4, 4)) for sd in ("left", "right")}
    base = np.zeros((T, 4, 4))
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))

    def pose(bid):
        Tm = np.eye(4)
        Tm[:3, :3] = d.xmat[bid].reshape(3, 3)
        Tm[:3, 3] = d.xpos[bid]
        return Tm

    for i in range(T):
        d.qpos[:] = 0
        d.qvel[:] = 0
        d.qacc[:] = 0
        for sd in ("left", "right"):
            d.qpos[dofs[sd]] = ang[sd][i, :7]
            d.qvel[dofs[sd]] = vel[sd][i, :7]
            d.qacc[dofs[sd]] = acc[sd][i]
        mujoco.mj_inverse(m, d)                         # qfrc_inverse = M q'' + C q' + g
        for sd in ("left", "right"):
            tau_res = tau[sd][i, :7] - d.qfrc_inverse[dofs[sd]]
            mujoco.mj_jacBody(m, d, jacp, jacr, hand[sd])
            J = np.vstack([jacp, jacr])[:, dofs[sd]]    # 6x7
            wrench[sd][i] = np.linalg.pinv(J.T) @ tau_res
            ee[sd][i] = pose(hand[sd])
        base[i] = pose(base_id)

    # assemble IWS-format episode dict ([left, right] stacking along axis 1)
    stack = lambda a, b: np.stack([a, b], axis=1)
    episode = {
        "action": np.concatenate([cmd["left"], cmd["right"]], axis=1).astype(np.float32),
        "robot_bases": stack(base, base).astype(np.float32),
        "obs": {
            "joint_pos": np.concatenate([ang["left"], ang["right"]], axis=1).astype(np.float32),
            "joint_vel": np.concatenate([vel["left"], vel["right"]], axis=1).astype(np.float32),
            "ee_pos": stack(ee["left"], ee["right"]).astype(np.float32),
            "wrist_force": stack(wrench["left"][:, :3], wrench["right"][:, :3]).astype(np.float32),
            "wrist_torque": stack(wrench["left"][:, 3:], wrench["right"][:, 3:]).astype(np.float32),
            "timestamp": (t - t[0]).astype(np.float64),
            "images": {args.camera_name: frames.astype(np.uint8)},
        },
    }

    def write(g, val):
        for k, v in val.items():
            if isinstance(v, dict):
                write(g.create_group(k), v)
            elif k == args.camera_name:
                g.create_dataset(k, data=v, chunks=(1,) + v.shape[1:], dtype="uint8")
            else:
                g.create_dataset(k, data=v)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with h5py.File(out, "w") as f:
        write(f, episode)
    print(f"wrote {out}  (T={T})  |F_R| peak={np.linalg.norm(wrench['right'][:, :3], axis=1).max():.1f} N")


if __name__ == "__main__":
    main()
