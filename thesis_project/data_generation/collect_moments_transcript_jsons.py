"""
Collect the transcript JSON files needed for locally available MOMENTS frame
sequences.

This helper scans an extracted-frame tree like:

  thesis_project/data/MOMENTS_frames/frames/<clip_id>/<im|nim>/<group_idx>/<clip_name>/frame_00.png

and infers the matching transcript JSON paths under a raw MOMENTS tree like:

  thesis_project/data/MOMENTS/<clip_id>/<important-moments|non-important-moments>/<group_idx>/<clip_name>_v1.json

or `_v2.json` / `.json`.

The script does not copy files. It prints the exact transcript paths that are
still missing locally, so you can fetch only those JSONs from Snellius.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class SampleSpec:
    clip_id: str
    clip_type: str
    group_idx: str
    clip_name: str


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_frames_root = repo_root / "thesis_project" / "data" / "MOMENTS_frames" / "frames"
    default_moments_root = repo_root / "thesis_project" / "data" / "MOMENTS"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames_root",
        default=str(default_frames_root),
        help="Root of the already extracted MOMENTS frames.",
    )
    parser.add_argument(
        "--moments_root",
        default=str(default_moments_root),
        help="Root of the raw MOMENTS tree that contains transcript JSONs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, only report this many missing transcript specs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional file to write the missing JSON paths to, one per line.",
    )
    return parser.parse_args()


def discover_sample_specs(frames_root: Path) -> List[SampleSpec]:
    specs = []
    seen = set()
    for frame_path in sorted(frames_root.rglob("frame_*.png")):
        sample_dir = frame_path.parent
        try:
            clip_name = sample_dir.name
            group_idx = sample_dir.parent.name
            clip_type = sample_dir.parent.parent.name
            clip_id = sample_dir.parent.parent.parent.name
        except IndexError:
            continue

        if clip_type not in {"im", "nim"}:
            continue

        key = (clip_id, clip_type, group_idx, clip_name)
        if key in seen:
            continue
        seen.add(key)
        specs.append(
            SampleSpec(
                clip_id=clip_id,
                clip_type=clip_type,
                group_idx=group_idx,
                clip_name=clip_name,
            )
        )
    return specs


def transcript_candidates(moments_root: Path, spec: SampleSpec) -> List[Path]:
    raw_folder = "important-moments" if spec.clip_type == "im" else "non-important-moments"
    base = moments_root / spec.clip_id / raw_folder / spec.group_idx / spec.clip_name
    return [
        base.with_name(base.name + "_v1.json"),
        base.with_name(base.name + "_v2.json"),
        base.with_suffix(".json"),
    ]


def format_status(spec: SampleSpec, candidate: Path, found: bool) -> str:
    status = "FOUND" if found else "MISSING"
    return (
        f"{status}  {spec.clip_id}/{spec.clip_type}/{spec.group_idx}/{spec.clip_name}  ->  "
        f"{candidate}"
    )


def main() -> None:
    args = parse_args()
    frames_root = Path(args.frames_root).expanduser().resolve()
    moments_root = Path(args.moments_root).expanduser().resolve()

    if not frames_root.exists():
        raise FileNotFoundError(f"Frames root not found: {frames_root}")
    if not moments_root.exists():
        raise FileNotFoundError(f"MOMENTS root not found: {moments_root}")

    specs = discover_sample_specs(frames_root)
    missing: List[Path] = []
    present = 0

    for spec in specs:
        candidates = transcript_candidates(moments_root, spec)
        found = next((path for path in candidates if path.exists()), None)
        if found is not None:
            present += 1
            print(format_status(spec, found, True))
            continue

        print(format_status(spec, candidates[0], False))
        for path in candidates[1:]:
            print(f"         alternate candidate: {path}")
        missing.append(candidates[0])

        if args.limit > 0 and len(missing) >= args.limit:
            break

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            for path in missing:
                f.write(str(path) + "\n")
        print(f"\nWrote {len(missing)} missing JSON paths to {output_path}")

    print(
        f"\nSummary: {present} clip specs already have transcript JSONs locally, "
        f"{len(missing)} are missing."
    )


if __name__ == "__main__":
    main()
