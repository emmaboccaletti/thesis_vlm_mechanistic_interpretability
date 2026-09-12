import csv
import os
import logging
import sys
from typing import Dict, List, Optional

from PIL import Image

from vision_language_prompts import VLPrompt

sys.path.append("./third_party/TransformerLens")

import transformer_lens as lens

from general_utils import (
    get_image_size_for_model,
    get_single_token_tokens,
    load_image_for_model,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


MOMENTS_CF_MODES = {"random_pair", "language_only", "vision_only", "both"}
MOMENTS_LABELS = ["yes", "no"]
MOMENTS_EVENT_TYPE_LABELS = ["goal", "corner", "shot"]


def _split_paths(field: str) -> List[str]:
    if field is None:
        return []
    field = field.strip()
    if not field:
        return []
    return [p for p in field.split("|") if p]


def _resolve_paths(image_paths: List[str], data_csv_path: str) -> List[str]:
    data_csv_dir = os.path.dirname(os.path.abspath(data_csv_path))
    resolved = []
    for path in image_paths:
        candidates = []
        normalized = path.lstrip("/")
        anchor_candidates = [
            "thesis_project/",
            "reproducing_code/",
            "tmp_moments",
            "tmp/",
        ]
        for anchor in anchor_candidates:
            if anchor in normalized:
                candidates.append(os.path.join(REPO_ROOT, normalized[normalized.index(anchor) :]))
        if os.path.isabs(path):
            candidates.extend(
                [
                    path,
                    os.path.join(REPO_ROOT, normalized),
                    os.path.join(data_csv_dir, normalized),
                ]
            )
        else:
            candidates.extend(
                [
                    os.path.join(REPO_ROOT, path),
                    os.path.join(data_csv_dir, path),
                    os.path.join(REPO_ROOT, normalized),
                ]
            )

        chosen = None
        for candidate in candidates:
            if os.path.exists(candidate):
                chosen = candidate
                break
        resolved.append(chosen or candidates[0])
    return resolved


def _load_image_sequence(
    image_paths: List[str],
    model_name: str,
    target_size: Optional[tuple] = None,
) -> List[Image.Image]:
    return [load_image_for_model(path, model_name, target_size=target_size) for path in image_paths]


def _prompt_token_length(model, prompt: str, images: List[Image.Image]) -> int:
    """
    Return the tokenized length of a MOMENTS prompt with its associated images.

    We use the same model-side tokenization path that the analysis code uses, so
    this length check matches the tensor shapes seen during attribution.
    """
    tokenized = model.to_tokens(
        prompt,
        images if images else None,
        prepend_bos=False,
        truncate=False,
    )
    return int(tokenized.shape[1])


def _build_qwen_chat_prompt(processor, prompt_text: str, n_images: int) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"} for _ in range(n_images)
            ]
            + [
                {
                    "type": "text",
                    "text": prompt_text,
                }
            ],
        }
    ]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


def load_moments_vl_prompts_list(
    data_csv_path: str,
    model: Optional[lens.HookedVLTransformer] = None,
    processor=None,
    language_only: bool = False,
    correct_preds_only: bool = True,
    image_size=None,
    max_images: Optional[int] = None,
    check_counterfactual_alignment: bool = True,
) -> List[VLPrompt]:
    """
    Load MOMENTS prompts from a CSV produced by create_moments_dataset.py.

    For MOMENTS we do not rely on model-generated predictions, so
    `correct_preds_only` is intentionally ignored. The argument is kept for API
    compatibility with the other task loaders.

    Set `check_counterfactual_alignment=False` for clean behavioral evaluation,
    where a clean row should not be excluded because its unused counterfactual
    has a token-length mismatch.
    """
    if not os.path.exists(data_csv_path):
        raise FileNotFoundError(f"Missing MOMENTS CSV: {data_csv_path}")

    if model is None:
        raise ValueError("model must be provided to load MOMENTS data")

    if image_size is None:
        image_size = get_image_size_for_model(model.model_name.lower())

    vl_prompts: List[VLPrompt] = []
    total_rows = 0
    kept_rows = 0
    alignment_checked_rows = 0
    dropped_length_mismatches = 0
    with open(data_csv_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            total_rows += 1
            prompt_text = row["prompt"]
            answer = row["answer"]
            cf_prompt_text = row.get("cf_prompt") or None
            cf_answer = row.get("cf_answer") or None
            image_paths = _resolve_paths(
                _split_paths(row.get("image_paths", "")), data_csv_path
            )
            cf_image_paths = _resolve_paths(
                _split_paths(row.get("cf_image_paths", "")), data_csv_path
            )

            if max_images is not None:
                if max_images < 1:
                    raise ValueError("max_images must be at least 1")
                image_paths = image_paths[:max_images]
                cf_image_paths = cf_image_paths[:max_images]

            images = []
            cf_images = None
            if not language_only:
                images = _load_image_sequence(
                    image_paths, model.model_name, target_size=image_size
                )
                if cf_image_paths:
                    cf_images = _load_image_sequence(
                        cf_image_paths, model.model_name, target_size=image_size
                    )
            else:
                images = []
                if cf_image_paths:
                    cf_images = []

            prompt = prompt_text
            cf_prompt = cf_prompt_text
            if processor is not None and not language_only:
                prompt = _build_qwen_chat_prompt(processor, prompt_text, len(image_paths))
                if cf_prompt_text:
                    cf_prompt = _build_qwen_chat_prompt(
                        processor, cf_prompt_text, len(cf_image_paths)
                    )

            metadata = {
                "clip_id": row.get("clip_id"),
                "group_idx": row.get("group_idx"),
                "clip_name": row.get("clip_name"),
                "event_type": row.get("event_type"),
                "label": row.get("label"),
                "similarity": row.get("similarity"),
                "local_text": row.get("local_text"),
                "global_text": row.get("global_text"),
                "cf_mode": row.get("cf_mode"),
                "cf_prompt_state": row.get("cf_prompt_state"),
                "cf_prompt_changes": row.get("cf_prompt_changes"),
                "image_paths": image_paths,
                "cf_image_paths": cf_image_paths,
            }

            if check_counterfactual_alignment and row.get("cf_mode") in {
                "random_pair",
                "language_only",
                "vision_only",
                "both",
            } and cf_prompt and (
                row.get("cf_mode") != "vision_only" or cf_image_paths
            ):
                alignment_checked_rows += 1
                prompt_len = _prompt_token_length(model, prompt, images)
                cf_prompt_len = _prompt_token_length(model, cf_prompt, cf_images or [])
                if prompt_len != cf_prompt_len:
                    dropped_length_mismatches += 1
                    logging.info(
                        "Skipping MOMENTS %s row %s/%s/%s due to token-length mismatch (%d vs %d)",
                        row.get("cf_mode"),
                        row.get("clip_id"),
                        row.get("group_idx"),
                        row.get("clip_name"),
                        prompt_len,
                        cf_prompt_len,
                    )
                    continue
                metadata["prompt_token_length"] = prompt_len
                metadata["cf_prompt_token_length"] = cf_prompt_len

            vl_prompts.append(
                VLPrompt(
                    prompt,
                    images,
                    answer,
                    cf_prompt=cf_prompt,
                    cf_images=cf_images,
                    cf_answer=cf_answer,
                    metadata=metadata,
                )
            )
            kept_rows += 1

    logging.info(
        "Loaded MOMENTS prompts from %s: %d rows seen, %d rows kept, %d rows dropped",
        data_csv_path,
        total_rows,
        kept_rows,
        total_rows - kept_rows,
    )
    if alignment_checked_rows > 0:
        logging.info(
            "MOMENTS aligned counterfactual rows seen: %d, matched and kept: %d, dropped for token-length mismatch: %d",
            alignment_checked_rows,
            alignment_checked_rows - dropped_length_mismatches,
            dropped_length_mismatches,
        )
    return vl_prompts


def load_moments_parallel_l_prompts(
    vl_prompts: List[VLPrompt], processor
) -> List[VLPrompt]:
    """
    Create textual-only counterparts for MOMENTS prompts by removing images.
    """
    parallel_l_prompts: List[VLPrompt] = []
    for vl_prompt in vl_prompts:
        parallel_l_prompts.append(
            VLPrompt(
                vl_prompt.prompt,
                [],
                str(vl_prompt.answer),
                cf_prompt=vl_prompt.cf_prompt,
                cf_images=[] if vl_prompt.cf_prompt is not None else None,
                cf_answer=vl_prompt.cf_answer,
                metadata=vl_prompt.metadata,
            )
        )
    return parallel_l_prompts


def get_moments_limited_labels(processor):
    """
    Return token strings for the yes/no labels used by MOMENTS.
    """
    labels = get_single_token_tokens(
        processor,
        ["yes", "no", " yes", " no", "Yes", "No"],
    )
    if len(labels) < 2:
        raise ValueError(
            "Could not find single-token yes/no labels for the MOMENTS task."
        )
    # Preserve the canonical yes/no order if possible.
    label_set = set(labels)
    ordered = [label for label in ["yes", "no", " yes", " no", "Yes", "No"] if label in label_set]
    return ordered[:2]


def get_moments_event_type_limited_labels(processor):
    """
    Return token strings for the MOMENTS event-type labels.
    """
    labels = get_single_token_tokens(
        processor,
        ["goal", "corner", "shot", " goal", " corner", " shot", "Goal", "Corner", "Shot"],
    )
    if len(labels) < 3:
        raise ValueError(
            "Could not find single-token goal/corner/shot labels for the MOMENTS event-type task."
        )
    label_set = set(labels)
    ordered = [
        label
        for label in ["goal", "corner", "shot", " goal", " corner", " shot", "Goal", "Corner", "Shot"]
        if label in label_set
    ]
    # Preserve canonical order if possible.
    canonical = []
    seen = set()
    for candidate in ordered:
        lowered = candidate.strip().lower()
        if lowered not in seen:
            seen.add(lowered)
            canonical.append(candidate)
    return canonical[:3]
