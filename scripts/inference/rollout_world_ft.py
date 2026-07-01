#!/usr/bin/env python
"""Per-episode open-loop world-model rollout for world_ft (Stage-2 + torque head).

For each ``episode_*.hdf5`` under ``<data_dir>/<split>/`` this:
  1. encodes the first frame,
  2. autoregressively rolls the *recorded* action sequence through the Stage-2
     latent dynamics for the WHOLE episode,
  3. decodes the predicted latents back to RGB, and
  4. predicts the right-arm joint torque (teacher-forced on the GT latents — the
     exact same estimator the Stage-2 ``validation/torque_rmse_nm`` metric uses).

Outputs, one set per episode, under ``--out``:
  episode_<i>_rollout.mp4   pred | gt, per frame, browser-playable H.264
  episode_<i>_torque.png    8-joint pred-vs-gt torque curves (+ per-joint RMSE)

Why this exists: the built-in Stage-2 ``validation`` renders one video *per sliding
window* (``validation_vis/video_0, _1, ...``), and with long episodes that is
hundreds of near-duplicate overlapping windows. This script instead does exactly
one full rollout per episode — what you actually want to eyeball new recordings.

Usage:
  python scripts/inference/rollout_world_ft.py \
    --ckpt outputs/world_ft_stage_2/<ts>/checkpoints/last.ckpt \
    --data_dir data/world_ft_eval --split val \
    --out data/wm_rollout

  # quick single-episode smoke test, only first 120 frames:
  python scripts/inference/rollout_world_ft.py --ckpt ... --data_dir ... \
    --episodes 0 --max_frames 120
"""
import argparse
import glob
import subprocess
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from torch.nn.attention import SDPBackend
from yixuan_utilities.draw_utils import center_crop

from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm
from interactive_world_sim.algorithms.latent_dynamics.latent_world_model_torque import (
    LatentWorldModelTorque,
)


# --------------------------------------------------------------------------- #
# model loading (mirrors scripts/inference/teleoperate_keyboard.py:load_model,
# but with the torque subclass so the torque head weights are restored)
# --------------------------------------------------------------------------- #
def load_torque_model(ckpt_path: str, device: str) -> tuple[LatentWorldModelTorque, object]:
    cfg_path = Path(ckpt_path).parent.parent / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Missing {cfg_path}. This script needs the Hydra config saved next to "
            f"the run (outputs/<run>/.hydra/config.yaml)."
        )
    cfg = OmegaConf.load(cfg_path)
    dtype = torch.float32 if "dtype" not in cfg.algorithm else cfg.algorithm.dtype
    cfg.algorithm.load_ae = None  # weights come from the ckpt, not the AE loader
    model = LatentWorldModelTorque.load_from_checkpoint(
        ckpt_path,
        cfg=cfg.algorithm,
        map_location=device,
        dtype=dtype,
        strict=False,
        weights_only=False,
    )
    model.dynamics = model.dynamics.to(dtype)
    model.eval()

    # This repo's attention restricts A100/L20 to the FLASH SDPA kernel, which only
    # accepts bf16/fp16 and raises "No available kernel" on fp32 inference. Re-enable
    # the MATH kernel (dtype-agnostic) so fp32 rollouts work on any GPU.
    n_patched = 0
    for m in model.modules():
        if hasattr(m, "cuda_backends"):
            m.cuda_backends = [
                SDPBackend.MATH,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
            ]
            n_patched += 1
    print(f"[rollout] enabled MATH SDPA kernel on {n_patched} attention modules")
    return model, cfg


# --------------------------------------------------------------------------- #
# episode io / preprocessing (matches RealAlohaDataset image pipeline)
# --------------------------------------------------------------------------- #
def preprocess_frames(frames_uint8: np.ndarray, res: int) -> torch.Tensor:
    """(T,H,W,3) uint8 -> (T,3,res,res) float32 in [0,1], center-crop + area resize."""
    out = []
    for f in frames_uint8:
        c = center_crop(f, (res, res))
        c = cv2.resize(c, (res, res), interpolation=cv2.INTER_AREA)
        out.append(c.astype(np.float32) / 255.0)
    return torch.from_numpy(np.stack(out).transpose(0, 3, 1, 2))  # T,3,H,W


def load_episode(path: Path, obs_keys: list[str]) -> tuple[dict, np.ndarray, np.ndarray | None]:
    with h5py.File(path, "r") as f:
        imgs = {k: f["obs"]["images"][k][()] for k in obs_keys}
        action = f["action"][()].astype(np.float32)
        torque = (
            f["obs"]["joint_torque"][()].astype(np.float32)
            if "joint_torque" in f["obs"]
            else None
        )
    return imgs, action, torque


# --------------------------------------------------------------------------- #
# rollout (mirrors LatentWorldModel.validation_step stage-2 branch, B=1)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def rollout_episode(
    model: LatentWorldModelTorque,
    obs_by_key: dict[str, torch.Tensor],
    action_np: np.ndarray,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (z_gt, z_seq, action_bt), all with a leading batch dim of 1."""
    obs = torch.cat(
        [model.normalizer[k].normalize(obs_by_key[k][None].to(device)) for k in model.obs_keys],
        dim=2,
    ).float()  # (1, T, 3*V, H, W)
    action_bt = model.normalizer["action"].normalize(
        torch.from_numpy(action_np)[None].to(device)
    ).float()  # (1, T, A)

    xs = rearrange(obs, "b t c h w -> (b t) c h w")
    z_gt = model.encoder_forward(xs)
    z_gt = rearrange(z_gt, "(b t) c h w -> b t c h w", b=1)

    # --- open-loop latent rollout, identical structure to validation_step ---
    z_0 = z_gt[:, 0]
    horizon = z_gt.shape[1]  # roll the whole episode in one sliding-window pass
    z_seq_ls = []
    z_last = z_0.clone()
    for i in range(1, action_bt.shape[1], horizon):
        action_chunk = action_bt[:, i : i + horizon]
        init = action_chunk.shape[1]
        if init < horizon:
            action_chunk = torch.nn.functional.pad(
                action_chunk, (0, 0, 0, horizon - init), mode="replicate"
            )
        z_seq = model.dynamics_forward(z_last[:, None], action_chunk)
        z_seq = z_seq[:, :init]
        z_seq_ls.append(z_seq)
        z_last = z_seq[:, -1].clone()
    z_seq = torch.cat(z_seq_ls, 1)
    z_seq = torch.cat([z_0.unsqueeze(1), z_seq], 1)  # (1, T, C, H, W)
    return z_gt, z_seq, action_bt


@torch.no_grad()
def decode_view0(model: LatentWorldModelTorque, z_seq: torch.Tensor, res: int) -> np.ndarray:
    """z_seq (1,T,C,H,W) -> (T,res,res,3) uint8 RGB for the first camera view."""
    z_flat = rearrange(z_seq, "b t c h w -> (b t) c h w")
    imgs = render_img_cm(model, z_flat, res, model.normalizer, num_views=model.num_views)
    imgs = imgs[:, :3]  # first view
    imgs = imgs.permute(0, 2, 3, 1).clamp(0, 1).float().cpu().numpy()  # T,H,W,3
    return (imgs * 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# writers
# --------------------------------------------------------------------------- #
def annotate(bgr: np.ndarray, text: str) -> None:
    cv2.putText(bgr, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)


def write_mp4_h264(frames_rgb: np.ndarray, out_path: Path, fps: int) -> None:
    T, H, W, _ = frames_rgb.shape
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
        "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-movflags", "+faststart", str(out_path),
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert p.stdin is not None
    try:
        for frame in frames_rgb:
            p.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed — is it installed and on PATH?")


def make_rollout_video(pred: np.ndarray, gt: np.ndarray, out_path: Path, fps: int, upscale: int) -> None:
    """pred/gt: (T,res,res,3) uint8. Writes a pred|gt side-by-side H.264 mp4."""
    T = min(len(pred), len(gt))
    res = pred.shape[1]
    new = res * upscale
    frames = np.empty((T, new, new * 2, 3), dtype=np.uint8)
    for t in range(T):
        p = cv2.resize(pred[t], (new, new), interpolation=cv2.INTER_NEAREST)
        g = cv2.resize(gt[t], (new, new), interpolation=cv2.INTER_NEAREST)
        pb = cv2.cvtColor(p, cv2.COLOR_RGB2BGR)
        gb = cv2.cvtColor(g, cv2.COLOR_RGB2BGR)
        annotate(pb, f"pred t={t}")
        annotate(gb, "gt")
        frames[t] = cv2.cvtColor(np.concatenate([pb, gb], axis=1), cv2.COLOR_BGR2RGB)
    write_mp4_h264(frames, out_path, fps)


def make_torque_plot(pred: np.ndarray, gt: np.ndarray, out_path: Path, title: str) -> float:
    """pred/gt: (T,8) N·m. Saves an 8-panel curve figure. Returns overall RMSE."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T, D = pred.shape
    ncol, nrow = 4, (D + 3) // 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.2 * nrow), squeeze=False)
    for j in range(D):
        ax = axes[j // ncol][j % ncol]
        ax.plot(gt[:, j], lw=1.3, label="gt")
        ax.plot(pred[:, j], lw=1.0, ls="--", label="pred")
        rmse = float(((pred[:, j] - gt[:, j]) ** 2).mean() ** 0.5)
        ax.set_title(f"joint {j}  rmse={rmse:.2f} N·m", fontsize=8)
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.legend(fontsize=7)
    for j in range(D, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    overall = float(((pred - gt) ** 2).mean() ** 0.5)
    fig.suptitle(f"{title}   overall rmse={overall:.3f} N·m", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return overall


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="Stage-2 (torque) checkpoint .ckpt")
    ap.add_argument("--data_dir", required=True, help="dataset dir containing <split>/episode_*.hdf5")
    ap.add_argument("--split", default="val", help="subdir under data_dir (default: val)")
    ap.add_argument("--out", default="wm_rollout", help="output dir")
    ap.add_argument("--episodes", type=int, nargs="*", default=None,
                    help="only these episode indices (default: all)")
    ap.add_argument("--max_frames", type=int, default=None, help="cap rollout length (debug)")
    ap.add_argument("--fps", type=int, default=10, help="video fps (default: 10 = collection rate)")
    ap.add_argument("--upscale", type=int, default=4, help="integer upscale for the video")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x), replace=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    model, cfg = load_torque_model(args.ckpt, device)
    obs_keys = list(model.obs_keys)
    res = int(cfg.dataset.resolution)
    has_torque_head = (
        getattr(model, "torque_predictor", None) is not None
        or getattr(model, "torque_head", None) is not None
    )
    print(f"[rollout] device={device} obs_keys={obs_keys} res={res} "
          f"torque_head={'yes' if has_torque_head else 'no'}")

    split_dir = Path(args.data_dir) / args.split
    paths = sorted(glob.glob(str(split_dir / "episode_*.hdf5")),
                   key=lambda p: int(Path(p).stem.split("_")[-1]))
    if not paths:
        raise FileNotFoundError(f"No episode_*.hdf5 in {split_dir}")
    if args.episodes is not None:
        keep = set(args.episodes)
        paths = [p for p in paths if int(Path(p).stem.split("_")[-1]) in keep]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for path in paths:
        path = Path(path)
        eid = int(path.stem.split("_")[-1])
        imgs_np, action_np, torque_np = load_episode(path, obs_keys)
        T = action_np.shape[0]
        if args.max_frames is not None:
            T = min(T, args.max_frames)
        action_np = action_np[:T]
        obs_by_key = {k: preprocess_frames(imgs_np[k][:T], res) for k in obs_keys}

        t0 = time.time()
        want_torque = has_torque_head and torque_np is not None
        with torch.no_grad():
            z_gt, z_seq, action_bt = rollout_episode(model, obs_by_key, action_np, device)
            pred = decode_view0(model, z_seq, res)
            torque_pred_t = model._val_torque_pred(z_gt, action_bt) if want_torque else None
        gt = (obs_by_key[obs_keys[0]].permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

        vid_path = out_dir / f"episode_{eid}_rollout.mp4"
        make_rollout_video(pred, gt, vid_path, args.fps, args.upscale)

        rmse = None
        if want_torque:
            torque_pred = torque_pred_t[0].float().cpu().numpy()
            torque_gt = torque_np[:T]
            n = min(len(torque_pred), len(torque_gt))
            png_path = out_dir / f"episode_{eid}_torque.png"
            rmse = make_torque_plot(
                torque_pred[:n], torque_gt[:n], png_path, title=f"episode_{eid}"
            )
        elif has_torque_head and torque_np is None:
            print(f"[rollout] episode_{eid}: no obs/joint_torque -> skipping torque plot")

        dt = time.time() - t0
        msg = f"[rollout] episode_{eid}: T={T} -> {vid_path.name} ({dt:.1f}s)"
        if rmse is not None:
            msg += f"  torque_rmse={rmse:.3f} N·m"
        print(msg)
        summary.append((eid, T, rmse))

    print("\n=== summary ===")
    for eid, T, rmse in summary:
        r = f"{rmse:.3f}" if rmse is not None else "n/a"
        print(f"  episode_{eid}: frames={T}  torque_rmse_nm={r}")
    if any(r is not None for _, _, r in summary):
        vals = [r for _, _, r in summary if r is not None]
        print(f"  mean torque_rmse_nm over {len(vals)} episodes = {np.mean(vals):.3f}")
    print(f"\nOutputs in: {out_dir}")


if __name__ == "__main__":
    main()
