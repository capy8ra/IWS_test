#!/usr/bin/env python
"""World-model "virtual follower" worker — milestone 1: visual only, no force yet.

Reads the real RIGHT-leader joint angles from redis (published by the nova damiao
controller in supernova), drives the trained world model as a virtual follower, and
shows the decoded predicted camera view in a window. The human moves the real leader
and watches the simulated follower move on screen.

Seeding (fixed start pose):
  * the sim latent is seeded from frame 0 of a chosen validation episode,
  * the action context is seeded from that episode's action[0],
  * the real leader must be homed to the SAME pose first
    (supernova: projects/world_ft/leader_home_and_float.py --seed-episode <same hdf5>).

Gripper: joint 8 is NOT used. The live action's 8th dim is held at the seed
episode's action[0][7] (kept in-distribution, matches the static-gripper seed).

Env: runs in the WORLD_FT (iws) env (torch + interactive_world_sim). It needs
nova.redis_client, which only imports redis+numpy — make it importable with:
    PYTHONPATH=/path/to/supernova python scripts/inference/wm_follower_worker.py ...

Usage:
  PYTHONPATH=/path/to/supernova python scripts/inference/wm_follower_worker.py \
    --ckpt outputs/world_ft_stage_2/<ts>/checkpoints/last.ckpt \
    --seed-episode data/world_ft_v3/val/episode_0.hdf5
"""
import argparse
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

try:
    from nova.redis_client import RedisClient
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Could not import nova.redis_client. Run with the supernova repo on PYTHONPATH, "
        "e.g. `PYTHONPATH=/path/to/supernova python scripts/inference/wm_follower_worker.py ...`\n"
        f"(original error: {e})"
    )


def load_model(ckpt_path: str, device: str) -> tuple[LatentWorldModelTorque, object]:
    cfg_path = Path(ckpt_path).parent.parent / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path} (need the run's .hydra/config.yaml)")
    cfg = OmegaConf.load(cfg_path)
    dtype = torch.float32 if "dtype" not in cfg.algorithm else cfg.algorithm.dtype
    cfg.algorithm.load_ae = None
    model = LatentWorldModelTorque.load_from_checkpoint(
        ckpt_path, cfg=cfg.algorithm, map_location=device,
        dtype=dtype, strict=False, weights_only=False,
    )
    model.dynamics = model.dynamics.to(dtype)
    model.eval()
    # Re-enable the MATH SDPA kernel so fp32 inference works on any GPU (harmless on
    # non-A100 where MATH is already allowed; required if ever run on A100).
    for m in model.modules():
        if hasattr(m, "cuda_backends"):
            m.cuda_backends = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]
    return model, cfg


@torch.no_grad()
def seed_from_episode(
    model: LatentWorldModelTorque, seed_episode: str, res: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return (z_ctx (1,1,C,H,W), a_ctx (1,1,A) normalized, seed_gripper)."""
    obs_key = model.obs_keys[0]
    with h5py.File(seed_episode, "r") as f:
        frame0 = f["obs"]["images"][obs_key][0]      # (H,W,3) uint8
        action0 = f["action"][0].astype(np.float32)  # (8,) right q_des
    img = center_crop(frame0, (res, res))
    img = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)
    img_t = model.normalizer[obs_key].normalize(img_t)
    z0 = model.encoder_forward(img_t)[:, None]                            # (1,1,C,H,W)
    a0 = model.normalizer["action"].normalize(
        torch.from_numpy(action0)[None, None].to(device)
    ).clamp(-1.0, 1.0)                                                    # (1,1,A)
    return z0, a0, float(action0[7])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="Stage-2 (torque) checkpoint .ckpt")
    ap.add_argument("--seed-episode", required=True, help="hdf5 to seed the fixed start pose/scene")
    ap.add_argument("--redis-host", default="localhost")
    ap.add_argument("--redis-port", type=int, default=6379)
    ap.add_argument("--leader-right", default="leader_right")
    ap.add_argument("--hz", type=float, default=10.0, help="sim step rate (training cadence = 10)")
    ap.add_argument("--n-context", type=int, default=None, help="latent context frames (default: model n_tokens)")
    ap.add_argument("--display-size", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args.ckpt, device)
    res = int(cfg.dataset.resolution)
    n_ctx = args.n_context or int(getattr(model, "n_tokens", 10))
    angle_key = f"{args.leader_right}_state:actual_angle_rad"
    print(f"[wm] device={device} obs_keys={list(model.obs_keys)} res={res} n_ctx={n_ctx}")

    z_ctx, a_ctx, seed_gripper = seed_from_episode(model, args.seed_episode, res, device)
    print(f"[wm] seeded from {args.seed_episode} (gripper held at {seed_gripper:+.3f})")

    client = RedisClient(host=args.redis_host, port=args.redis_port)
    win = "world_model_follower (pred)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.display_size, args.display_size)

    # show the seeded frame once before any motion
    def _show(z_last: torch.Tensor, label: str) -> None:
        with torch.no_grad():
            img = render_img_cm(model, z_last, res, model.normalizer, num_views=model.num_views)
        img = img[0, :3].permute(1, 2, 0).clamp(0, 1).float().cpu().numpy()
        img = (img * 255).astype(np.uint8)
        img = cv2.resize(img, (args.display_size, args.display_size), interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(win, img)
        cv2.waitKey(1)

    _show(z_ctx[:, -1], "seed")
    print("[wm] waiting for leader state on redis ...")

    dt_target = 1.0 / args.hz
    step = 0
    while True:
        t0 = time.time()
        ang = client.stream_get_batch({angle_key: np.ndarray})[angle_key]
        if ang is None:
            time.sleep(0.02)
            continue

        # live action: 7 leader arm joints + held seed gripper (8th dim, not driven)
        action_live = np.concatenate([ang[:7].astype(np.float32), [seed_gripper]])
        a_new = model.normalizer["action"].normalize(
            torch.from_numpy(action_live)[None, None].to(device)
        ).clamp(-1.0, 1.0).float()

        with torch.no_grad():
            action_seq = torch.cat([a_ctx, a_new], dim=1)           # (1, L+1, A)
            z_seq = model.dynamics_forward(z_ctx, action_seq)        # (1, L+1, C,H,W)
            z_new = z_seq[:, -1:]
        z_ctx = torch.cat([z_ctx, z_new], dim=1)[:, -n_ctx:]
        a_ctx = torch.cat([a_ctx, a_new], dim=1)[:, -n_ctx:]

        _show(z_ctx[:, -1], f"step {step}")

        step += 1
        elapsed = time.time() - t0
        if elapsed < dt_target:
            time.sleep(dt_target - elapsed)
        elif step % 20 == 0:
            print(f"[wm] step {step}: {1.0/elapsed:.1f} Hz (slower than {args.hz:.0f} Hz target)")

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
