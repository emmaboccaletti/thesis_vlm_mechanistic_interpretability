"""
Extract N uniformly-sampled frames from every IM_*.mp4 and NIM_*.mp4 in the MOMENTS dataset.

Output structure:
  {output_dir}/{clip_id}/{im|nim}/{group_idx}/{clip_name}/frame_{00..N-1}.png

Usage:
  python extract_moments_frames.py \
      --moments_dir ~/thesis_project/data/MOMENTS \
      --output_dir  ~/thesis_project/data/MOMENTS/frames \
      --n_frames 10
"""

import argparse
import glob
import os

import cv2
import numpy as np
from tqdm import tqdm


def extract_frames(mp4_path: str, output_dir: str, n_frames: int) -> int:
    """Extract n_frames uniformly from mp4_path, write as frame_00.png ... frame_{n-1}.png.
    Returns the number of frames actually written (may be < n_frames for very short clips)."""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(mp4_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        return 0

    actual_n = min(n_frames, total)
    indices = np.linspace(0, total - 1, actual_n, dtype=int)

    written = 0
    for i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(output_dir, f"frame_{i:02d}.png")
        cv2.imwrite(out_path, frame)
        written += 1

    cap.release()
    return written


def iter_clips(moments_dir: str):
    """Yield (mp4_path, clip_type, clip_id, group_idx, clip_name) for every clip."""
    for clip_type, prefix in [("im", "IM"), ("nim", "NIM")]:
        pattern = os.path.join(
            moments_dir, "*",
            "important-moments" if clip_type == "im" else "non-important-moments",
            "*", f"{prefix}_*.mp4"
        )
        for mp4_path in sorted(glob.glob(pattern)):
            parts = mp4_path.split(os.sep)
            # .../{clip_id}/{im|nim-folder}/{group_idx}/IM_N.mp4
            clip_id   = parts[-4]
            group_idx = parts[-2]
            clip_name = os.path.splitext(parts[-1])[0]
            yield mp4_path, clip_type, clip_id, group_idx, clip_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moments_dir", required=True,
                        help="Path to the MOMENTS root directory")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write extracted frame PNGs")
    parser.add_argument("--n_frames", type=int, default=10,
                        help="Number of frames to extract per clip (default: 10)")
    args = parser.parse_args()

    clips = list(iter_clips(args.moments_dir))
    print(f"Found {len(clips)} clips (IM + NIM). Extracting {args.n_frames} frames each.")

    skipped = 0
    for mp4_path, clip_type, clip_id, group_idx, clip_name in tqdm(clips):
        out_dir = os.path.join(
            args.output_dir, clip_id, clip_type, group_idx, clip_name
        )
        written = extract_frames(mp4_path, out_dir, args.n_frames)
        if written == 0:
            print(f"  WARNING: 0 frames written for {mp4_path}")
            skipped += 1

    total = len(clips) - skipped
    print(f"Done. Extracted frames for {total} clips ({skipped} skipped).")
    print(f"Frames saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
