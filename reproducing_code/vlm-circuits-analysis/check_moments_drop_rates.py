#!/usr/bin/env python3
"""Report token-length drop rates for MOMENTS CSVs.

This is a lightweight checker for the MOMENTS Qwen pipeline. It tokenizes the
clean and counterfactual prompts the same way the analysis code does, then
reports how many rows would be kept or dropped if we required exact sequence
length matches.

By default it checks the three language-sensitive MOMENTS modes:
  - random_pair
  - language_only
  - both

You can also pass `--modes vision_only` if you want to confirm that the visual
mode remains length-stable.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Sequence

from PIL import Image
from transformers import AutoProcessor

from general_utils import get_image_size_for_model, load_image_for_model
from moments_utils import _build_qwen_chat_prompt, _resolve_paths, _split_paths


DEFAULT_MODEL_NAME = "qwen2-7b-vl-instruct"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    default_data_dir = repo_root / "data" / "moments_goal"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=str(default_data_dir),
        help="Directory containing the MOMENTS CSV files.",
    )
    parser.add_argument(
        "--model_path",
        default="/home/eboccaletti/models/qwen2vl7b",
        help="HF model path for the Qwen processor/tokenizer.",
    )
    parser.add_argument(
        "--model_name",
        default=DEFAULT_MODEL_NAME,
        help="Model name used to determine the image resize target.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["random_pair", "language_only", "both"],
        choices=["random_pair", "language_only", "vision_only", "both"],
        help="Which MOMENTS CSV modes to inspect.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="If set, only inspect the first N frames from each sequence.",
    )
    parser.add_argument(
        "--show_examples",
        type=int,
        default=5,
        help="How many mismatched rows to print per mode.",
    )
    return parser.parse_args()


def _trim(paths: Sequence[str], max_images: int | None) -> List[str]:
    if max_images is None:
        return list(paths)
    if max_images < 1:
        raise ValueError("--max_images must be at least 1")
    return list(paths[:max_images])


def _load_images(paths: Iterable[str], model_name: str, target_size) -> List[Image.Image]:
    return [
        load_image_for_model(path, model_name, target_size=target_size)
        for path in paths
    ]


def _tokenized_length(processor, prompt: str, images: List[Image.Image]) -> int:
    if images:
        inputs = processor(
            images,
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
    else:
        inputs = processor.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
    return int(inputs["input_ids"].shape[1])


def _build_prompt(processor, prompt_text: str, n_images: int, language_only: bool) -> str:
    if language_only or not hasattr(processor, "apply_chat_template"):
        return prompt_text
    return _build_qwen_chat_prompt(processor, prompt_text, n_images)


def inspect_mode(
    *,
    processor,
    csv_path: Path,
    model_name: str,
    max_images: int | None,
    show_examples: int,
    language_only: bool,
) -> None:
    total_rows = 0
    kept_rows = 0
    dropped_rows = 0
    mismatched_rows = 0
    missing_cf_rows = 0
    examples = []

    image_size = get_image_size_for_model(model_name)

    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            total_rows += 1
            prompt_text = row["prompt"]
            cf_prompt_text = row.get("cf_prompt") or ""
            if not cf_prompt_text:
                missing_cf_rows += 1
                dropped_rows += 1
                continue

            image_paths = _trim(_resolve_paths(_split_paths(row.get("image_paths", "")), str(csv_path)), max_images)
            cf_image_paths = _trim(_resolve_paths(_split_paths(row.get("cf_image_paths", "")), str(csv_path)), max_images)

            if language_only:
                images: List[Image.Image] = []
                cf_images: List[Image.Image] = []
            else:
                images = _load_images(image_paths, model_name, image_size)
                cf_images = _load_images(cf_image_paths, model_name, image_size)

            prompt = _build_prompt(processor, prompt_text, len(image_paths), language_only)
            cf_prompt = _build_prompt(processor, cf_prompt_text, len(cf_image_paths), language_only)

            prompt_len = _tokenized_length(processor, prompt, images)
            cf_prompt_len = _tokenized_length(processor, cf_prompt, cf_images)

            if prompt_len == cf_prompt_len:
                kept_rows += 1
                continue

            mismatched_rows += 1
            dropped_rows += 1
            if len(examples) < show_examples:
                examples.append(
                    (
                        row.get("clip_id", ""),
                        row.get("group_idx", ""),
                        row.get("clip_name", ""),
                        prompt_len,
                        cf_prompt_len,
                    )
                )

    drop_rate = (dropped_rows / total_rows) if total_rows else 0.0
    print(f"\n{csv_path.name}")
    print(f"  total rows: {total_rows}")
    print(f"  kept rows: {kept_rows}")
    print(f"  dropped rows: {dropped_rows}")
    print(f"  mismatched rows: {mismatched_rows}")
    print(f"  missing cf rows: {missing_cf_rows}")
    print(f"  drop rate: {drop_rate:.4f}")
    if examples:
        print("  examples:")
        for clip_id, group_idx, clip_name, prompt_len, cf_prompt_len in examples:
            print(
                f"    - {clip_id}/{group_idx}/{clip_name}: "
                f"{prompt_len} vs {cf_prompt_len}"
            )


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    for mode in args.modes:
        csv_path = data_dir / f"{mode}_data.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing MOMENTS CSV: {csv_path}")
        inspect_mode(
            processor=processor,
            csv_path=csv_path,
            model_name=args.model_name,
            max_images=args.max_images,
            show_examples=args.show_examples,
            language_only=(mode == "language_only"),
        )


if __name__ == "__main__":
    main()
