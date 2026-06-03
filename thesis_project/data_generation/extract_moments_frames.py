"""
Extract N uniformly-sampled frames from clips listed in an annotation CSV.

Output structure:
  {output_dir}/{clip_id}/{im|nim}/{group_idx}/{clip_name}/frame_{00..N-1}.png

Usage:
  python extract_moments_frames.py \
      --annotation_csv ~/thesis_project/data/MOMENTS_categories/goals_annotation_only_IM.csv \
      --moments_dir    ~/thesis_project/data/MOMENTS \
      --output_dir     ~/thesis_project/data/MOMENTS/frames \
      --n_frames 10

  # To extract for multiple CSVs at once, pass them comma-separated:
  --annotation_csv goals_annotation_only_IM.csv,category_annotation.csv
"""

import argparse
import csv
import os

import cv2
import numpy as np
from tqdm import tqdm


def extract_frames(mp4_path: str, output_dir: str, n_frames: int) -> int:
    """Extract n_frames uniformly from mp4_path. Returns number of frames written."""
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
        cv2.imwrite(os.path.join(output_dir, f"frame_{i:02d}.png"), frame)
        written += 1

    cap.release()
    return written


def load_clips_from_csv(csv_path: str, moments_dir: str):
    """Read annotation CSV and yield (mp4_path, out_key) for each row."""
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            yield row["mp4_path"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_csv", required=True,
                        help="Annotation CSV (or comma-separated list of CSVs)")
    parser.add_argument("--moments_dir", required=True,
                        help="Path to MOMENTS root directory")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write extracted frame PNGs")
    parser.add_argument("--n_frames", type=int, default=10)
    args = parser.parse_args()

    # Collect unique mp4 paths across all CSVs
    mp4_paths = set()
    for csv_path in args.annotation_csv.split(","):
        for mp4_path in load_clips_from_csv(csv_path.strip(), args.moments_dir):
            mp4_paths.add(mp4_path)

    print(f"Found {len(mp4_paths)} unique clips to extract. "
          f"Extracting {args.n_frames} frames each.")

    skipped = 0
    for mp4_path in tqdm(sorted(mp4_paths)):
        # Derive output dir from mp4 path structure:
        # .../MOMENTS/{clip_id}/{im|nim-folder}/{group_idx}/{clip_name}.mp4
        parts = mp4_path.split(os.sep)
        clip_id   = parts[-4]
        folder    = parts[-3]
        clip_type = "im" if folder == "important-moments" else "nim"
        group_idx = parts[-2]
        clip_name = os.path.splitext(parts[-1])[0]

        out_dir = os.path.join(args.output_dir, clip_id, clip_type, group_idx, clip_name)
        written = extract_frames(mp4_path, out_dir, args.n_frames)
        if written == 0:
            print(f"  WARNING: 0 frames — {mp4_path}")
            skipped += 1

    total = len(mp4_paths) - skipped
    print(f"Done. Extracted frames for {total} clips ({skipped} failed).")
    print(f"Frames saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
