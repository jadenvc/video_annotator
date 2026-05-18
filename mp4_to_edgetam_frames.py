#!/usr/bin/env python3
import os
import json
import math
import shutil
import argparse
import subprocess
from pathlib import Path

import cv2


def extract_with_ffmpeg(
    mp4_path: Path,
    out_dir: Path,
    jpeg_q: int = 2,          # ffmpeg: 2 (best) .. 31 (worst)
    fps: float | None = None,
    scale_max: int | None = None,
):
    """
    Uses ffmpeg to write out_dir/0.jpg, 1.jpg, ...
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    vf_parts = []
    if fps is not None:
        vf_parts.append(f"fps={fps}")
    if scale_max is not None:
        # scale longest side to scale_max, preserve aspect
        vf_parts.append(f"scale='if(gte(iw,ih),{scale_max},-2)':'if(gte(iw,ih),-2,{scale_max})'")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(mp4_path),
        "-start_number",
        "0",
    ]
    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]

    # -q:v controls JPEG quality (lower is better)
    cmd += ["-q:v", str(jpeg_q), str(out_dir / "%d.jpg")]

    subprocess.run(cmd, check=True)


def extract_with_opencv(
    mp4_path: Path,
    out_dir: Path,
    every: int = 1,
    target_fps: float | None = None,
    jpeg_quality: int = 95,   # OpenCV: 0..100 (higher is better)
    scale_max: int | None = None,
):
    """
    Pure Python fallback: reads frames with OpenCV and saves as 0.jpg, 1.jpg, ...
    If target_fps is set, approximates by skipping frames.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {mp4_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None

    if target_fps is not None and src_fps > 0:
        # approximate stride to reach target fps
        stride = max(1, int(round(src_fps / target_fps)))
    else:
        stride = 1

    stride = max(stride, every)

    out_idx = 0
    in_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if (in_idx % stride) == 0:
            if scale_max is not None:
                h, w = frame.shape[:2]
                long_side = max(w, h)
                if long_side > scale_max:
                    s = scale_max / long_side
                    new_w = int(round(w * s))
                    new_h = int(round(h * s))
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            out_path = out_dir / f"{out_idx}.jpg"
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            out_idx += 1

        in_idx += 1

    cap.release()

    return {
        "source_fps": src_fps,
        "source_total_frames": total,
        "saved_frames": out_idx,
        "stride": stride,
    }


def main():
    ap = argparse.ArgumentParser(description="Convert MP4 -> EdgeTAM/SAM2 JPEG folder (0.jpg,1.jpg,...)")
    ap.add_argument("mp4", type=str, help="Input .mp4 path")
    ap.add_argument("out_dir", type=str, help="Output frames directory")
    ap.add_argument("--no-ffmpeg", action="store_true", help="Force OpenCV extraction even if ffmpeg exists")
    ap.add_argument("--fps", type=float, default=None, help="Optional output fps (ffmpeg exact, OpenCV approximate)")
    ap.add_argument("--scale-max", type=int, default=None, help="Optional max size of longest side (e.g., 1080)")
    ap.add_argument("--ffmpeg-q", type=int, default=2, help="ffmpeg JPEG quality: 2(best)..31(worst)")
    ap.add_argument("--opencv-q", type=int, default=95, help="OpenCV JPEG quality: 0..100 (higher is better)")
    ap.add_argument("--every", type=int, default=1, help="OpenCV: keep every Nth frame (ignored for ffmpeg)")
    args = ap.parse_args()

    mp4_path = Path(args.mp4).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not mp4_path.exists():
        raise SystemExit(f"Input not found: {mp4_path}")

    ffmpeg_ok = (shutil.which("ffmpeg") is not None) and (not args.no_ffmpeg)

    meta = {
        "input_mp4": str(mp4_path),
        "output_dir": str(out_dir),
        "used": None,
        "fps_arg": args.fps,
        "scale_max": args.scale_max,
    }

    if ffmpeg_ok:
        extract_with_ffmpeg(
            mp4_path=mp4_path,
            out_dir=out_dir,
            jpeg_q=args.ffmpeg_q,
            fps=args.fps,
            scale_max=args.scale_max,
        )
        meta["used"] = "ffmpeg"
    else:
        info = extract_with_opencv(
            mp4_path=mp4_path,
            out_dir=out_dir,
            every=args.every,
            target_fps=args.fps,
            jpeg_quality=args.opencv_q,
            scale_max=args.scale_max,
        )
        meta["used"] = "opencv"
        meta.update(info)

    # write a small metadata file (optional, harmless)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done. Frames in: {out_dir}")
    print(f"Example files: {out_dir / '0.jpg'}, {out_dir / '1.jpg'}")


if __name__ == "__main__":
    main()
