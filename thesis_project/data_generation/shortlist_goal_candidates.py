"""
Scan MOMENTS important-moment JSON files for goal-related keywords to produce
a shortlist of candidate clips to manually review.

Outputs a CSV with columns:
  clip_id, group_idx, clip_name, local_desc, matched_keywords, similarity, json_path

Usage:
  python shortlist_goal_candidates.py \
      --moments_dir ~/thesis_project/data/MOMENTS \
      --output candidates_goals.csv \
      --min_similarity 0.5
"""

import argparse
import csv
import glob
import json
import os
import re

GOAL_KEYWORDS = [
    r"\bgoal\b", r"\bscores?\b", r"\bscored\b", r"\bgoes in\b",
    r"\binto the net\b", r"\bback of the net\b", r"\bit's in\b",
    r"\bfound the net\b", r"\bslots? (it )?home\b", r"\bputs? it (away|in)\b",
    r"\bgoooal\b", r"\bGOAL\b",
]

PENALTY_KEYWORDS = [r"\bpenalty\b", r"\bspot kick\b", r"\bfrom the spot\b"]
RED_CARD_KEYWORDS = [r"\bred card\b", r"\bsent off\b", r"\boff he goes\b"]

ALL_KEYWORD_GROUPS = {
    "goal": GOAL_KEYWORDS,
    "penalty": PENALTY_KEYWORDS,
    "red_card": RED_CARD_KEYWORDS,
}


def find_matches(text: str) -> dict[str, list[str]]:
    text_lower = text.lower()
    matches = {}
    for event_type, patterns in ALL_KEYWORD_GROUPS.items():
        found = [p for p in patterns if re.search(p, text_lower)]
        if found:
            matches[event_type] = found
    return matches


def iter_im_jsons(moments_dir: str, min_similarity: float):
    pattern = os.path.join(moments_dir, "*", "important-moments", "*", "*_v1.json")
    for json_path in sorted(glob.glob(pattern)):
        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception:
            continue

        sim = data.get("similarity", 0.0)
        if sim < min_similarity:
            continue

        parts = json_path.split(os.sep)
        clip_id   = parts[-4]
        group_idx = parts[-2]
        clip_name = os.path.splitext(parts[-1])[0].replace("_v1", "")

        yield clip_id, group_idx, clip_name, data["local"], sim, json_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moments_dir", required=True)
    parser.add_argument("--output", default="candidates_goals.csv")
    parser.add_argument("--min_similarity", type=float, default=0.5)
    args = parser.parse_args()

    rows = []
    for clip_id, group_idx, clip_name, local_desc, sim, json_path in iter_im_jsons(
        args.moments_dir, args.min_similarity
    ):
        matches = find_matches(local_desc)
        if matches:
            matched_str = "; ".join(
                f"{etype}:{','.join(pats)}" for etype, pats in matches.items()
            )
            rows.append({
                "clip_id": clip_id,
                "group_idx": group_idx,
                "clip_name": clip_name,
                "matched_events": "|".join(matches.keys()),
                "matched_keywords": matched_str,
                "similarity": f"{sim:.3f}",
                "local_desc": local_desc,
                "json_path": json_path,
                "mp4_path": json_path.replace("_v1.json", ".mp4"),
                "label": "",  # fill in manually: goal / not_goal / penalty / red_card / unsure
            })

    rows.sort(key=lambda r: r["matched_events"])

    with open(args.output, "w", newline="") as f:
        fieldnames = [
            "clip_id", "group_idx", "clip_name", "matched_events",
            "matched_keywords", "similarity", "local_desc", "mp4_path",
            "json_path", "label",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_scanned = sum(1 for _ in iter_im_jsons(args.moments_dir, args.min_similarity))
    print(f"Scanned {total_scanned} IM clips (similarity >= {args.min_similarity})")
    print(f"Found {len(rows)} keyword matches")
    print(f"  goal candidates:     {sum(1 for r in rows if 'goal' in r['matched_events'])}")
    print(f"  penalty candidates:  {sum(1 for r in rows if 'penalty' in r['matched_events'])}")
    print(f"  red card candidates: {sum(1 for r in rows if 'red_card' in r['matched_events'])}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
