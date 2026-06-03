"""
Parse category_annotations.json into a clean annotation CSV.

Creates:
  goals_annotation_only_IM.csv  — GOAL IMs (positive) vs. non-GOAL IMs (negative)
  category_annotation.csv — all IMs (positive) vs. all NIMs (negative)

Usage:
  python parse_category_annotations.py \
      --annotations /home/eboccaletti/thesis_project/data/MOMENTS/category_annotations.json \
      --moments_dir /home/eboccaletti/thesis_project/data/MOMENTS \
      --output_dir  /home/eboccaletti/thesis_project/data/MOMENTS
"""

import argparse
import csv
import json
import os


def parse_id(entry: str):
    """Parse 'clip_id-group_idx-clip_name' → (clip_id, group_idx, clip_name).
    Handles clip_ids that contain hyphens by splitting from the right."""
    parts = entry.rsplit("-", 2)
    return parts[0], parts[1], parts[2]


def mp4_path(moments_dir: str, clip_id: str, group_idx: str, clip_name: str,
             is_im: bool) -> str:
    subfolder = "important-moments" if is_im else "non-important-moments"
    return os.path.join(moments_dir, clip_id, subfolder, group_idx, f"{clip_name}.mp4")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True,
                        help="Path to category_annotations.json")
    parser.add_argument("--moments_dir", required=True,
                        help="Path to MOMENTS root directory")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write output CSVs")
    args = parser.parse_args()

    with open(args.annotations) as f:
        data = json.load(f)

    # Print summary of all categories
    print("=== Category Summary ===")
    for cat, entries in data.items():
        n_im  = len(entries.get("IMs",  []))
        n_nim = len(entries.get("NIMs", []))
        print(f"  {cat:<25} IMs: {n_im:>3}   NIMs: {n_nim:>3}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- goals_annotation_only_IM.csv ---
    goal_ims = {entry for entry in data.get("GOAL", {}).get("IMs", [])}
    other_ims = {
        entry
        for cat, entries in data.items()
        if cat != "GOAL"
        for entry in entries.get("IMs", [])
        if entry not in goal_ims
    }

    rows = []
    for entry in sorted(goal_ims):
        clip_id, group_idx, clip_name = parse_id(entry)
        rows.append({
            "clip_id": clip_id, "group_idx": group_idx, "clip_name": clip_name,
            "event_type": "goal", "is_im": True,
            "mp4_path": mp4_path(args.moments_dir, clip_id, group_idx, clip_name, True),
            "label": "goal",
        })
    for entry in sorted(other_ims):
        clip_id, group_idx, clip_name = parse_id(entry)
        rows.append({
            "clip_id": clip_id, "group_idx": group_idx, "clip_name": clip_name,
            "event_type": "other_im", "is_im": True,
            "mp4_path": mp4_path(args.moments_dir, clip_id, group_idx, clip_name, True),
            "label": "not_goal",
        })

    goal_csv = os.path.join(args.output_dir, "goals_annotation_only_IM.csv")
    with open(goal_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "clip_id", "group_idx", "clip_name", "event_type", "mp4_path", "label"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    n_goals    = sum(1 for r in rows if r["label"] == "goal")
    n_not_goal = sum(1 for r in rows if r["label"] == "not_goal")
    print(f"goals_annotation_only_IM.csv → {n_goals} goals, {n_not_goal} non-goals")
    print(f"  Saved to: {goal_csv}\n")

    # --- category_annotation.csv (all IMs vs all NIMs) ---
    im_rows, nim_rows = [], []
    for cat, entries in data.items():
        for entry in entries.get("IMs", []):
            clip_id, group_idx, clip_name = parse_id(entry)
            im_rows.append({
                "clip_id": clip_id, "group_idx": group_idx, "clip_name": clip_name,
                "event_type": cat,
                "mp4_path": mp4_path(args.moments_dir, clip_id, group_idx, clip_name, True),
                "label": "important",
            })
        for entry in entries.get("NIMs", []):
            clip_id, group_idx, clip_name = parse_id(entry)
            nim_rows.append({
                "clip_id": clip_id, "group_idx": group_idx, "clip_name": clip_name,
                "event_type": cat,
                "mp4_path": mp4_path(args.moments_dir, clip_id, group_idx, clip_name, False),
                "label": "not_important",
            })

    imp_csv = os.path.join(args.output_dir, "category_annotation.csv")
    with open(imp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "clip_id", "group_idx", "clip_name", "event_type", "mp4_path", "label"
        ])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in im_rows + nim_rows)

    print(f"category_annotation.csv → {len(im_rows)} IMs, {len(nim_rows)} NIMs")
    print(f"  Saved to: {imp_csv}")


if __name__ == "__main__":
    main()
