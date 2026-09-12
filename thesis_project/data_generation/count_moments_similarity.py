"""
Count transcript-similarity statistics across the MOMENTS JSON tree.

This scans every JSON file under the MOMENTS root, reads the top-level
`similarity` value when present, and reports threshold counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--moments_root",
        default=".",
        help="Root directory containing the MOMENTS transcript JSON tree.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=[0.9, 0.8, 0.5],
        help="Similarity thresholds to count. The default reports >=0.9, >=0.8, and <0.5.",
    )
    return parser.parse_args()


def load_similarity_values(root: Path) -> List[float]:
    values: List[float] = []
    for path in root.rglob("*.json"):
        try:
            with path.open("r") as f:
                data = json.load(f)
        except Exception:
            continue

        if not isinstance(data, dict) or "similarity" not in data:
            continue

        try:
            values.append(float(data["similarity"]))
        except Exception:
            continue
    return values


def main() -> None:
    args = parse_args()
    root = Path(args.moments_root)
    if not root.exists():
        raise FileNotFoundError(f"Missing MOMENTS root: {root}")

    values = load_similarity_values(root)
    print(f"files_with_similarity {len(values)}")

    if values:
        for threshold in args.thresholds:
            if threshold == 0.5:
                print(f"< 0.5: {sum(v < 0.5 for v in values)}")
            else:
                print(f">= {threshold}: {sum(v >= threshold for v in values)}")
        print(f"min {min(values)}")
        print(f"max {max(values)}")
        print(f"mean {sum(values) / len(values)}")
    else:
        for threshold in args.thresholds:
            if threshold == 0.5:
                print("< 0.5: 0")
            else:
                print(f">= {threshold}: 0")
        print("min None")
        print("max None")
        print("mean None")


if __name__ == "__main__":
    main()
