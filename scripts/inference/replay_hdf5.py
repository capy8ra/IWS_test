"""Replay a saved episode HDF5 as an H.264 MP4 that plays in VS Code / browsers.

Usage:
    python scripts/inference/replay_hdf5.py PATH_TO_EPISODE.hdf5
        # → writes PATH_TO_EPISODE.mp4 next to the input

    python scripts/inference/replay_hdf5.py PATH_TO_EPISODE.hdf5 -o out.mp4 --fps 10 --upscale 4
"""

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np


def load_episode(path: Path) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Return (frames RGB uint8 [T,H,W,3], action [T, A] or None, image-key name)."""
    with h5py.File(path, "r") as f:
        image_grp = f["obs"]["images"]
        keys = list(image_grp.keys())
        if not keys:
            raise RuntimeError(f"No image streams in {path}")
        key = keys[0]
        frames = image_grp[key][()]
        action = f["action"][()] if "action" in f else None
    return frames, action, key


def annotate(frame_rgb: np.ndarray, text_lines: list[str]) -> np.ndarray:
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    y = 18
    for line in text_lines:
        cv2.putText(img, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
        y += 16
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_mp4_h264(frames_rgb: np.ndarray, out_path: Path, fps: int) -> None:
    """Pipe raw RGB frames into ffmpeg to produce a browser-playable H.264 MP4."""
    T, H, W, _ = frames_rgb.shape
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-movflags", "+faststart",
        str(out_path),
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hdf5", type=Path, help="Episode HDF5 to replay")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="Output MP4 (default: same path as input, .mp4 extension)")
    ap.add_argument("--fps", type=int, default=10,
                    help="Playback frame rate (default: 10, matches collection rate)")
    ap.add_argument("--upscale", type=int, default=4,
                    help="Integer upscale factor (default: 4 → 128→512)")
    ap.add_argument("--no-overlay", action="store_true",
                    help="Disable text overlay (step, action vector)")
    args = ap.parse_args()

    frames, action, key = load_episode(args.hdf5)
    T, H, W, _ = frames.shape
    print(f"Loaded {T} frames {H}x{W} from '{key}'"
          f"{f', action shape {action.shape}' if action is not None else ', no action'}")

    if args.upscale > 1:
        new_size = (W * args.upscale, H * args.upscale)
        frames = np.stack([
            cv2.resize(f, new_size, interpolation=cv2.INTER_NEAREST) for f in frames
        ])

    if not args.no_overlay:
        annotated = np.empty_like(frames)
        for t in range(T):
            lines = [f"t={t}/{T-1}"]
            if action is not None:
                a = action[t].reshape(-1)
                lines.append("a=[" + ", ".join(f"{x:+.2f}" for x in a) + "]")
            annotated[t] = annotate(frames[t], lines)
        frames = annotated

    out_path = args.out or args.hdf5.with_suffix(".mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_mp4_h264(frames, out_path, args.fps)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
