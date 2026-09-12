"""Evaluate pairwise MOMENTS event-type behavior.

The evaluator keeps the clean transcript and image, replaces the original
yes/no question with a two-way event-type question, and compares only the
final-token logits for the two labels in each pair.
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import Counter
from pathlib import Path

import torch

from analysis_utils import load_model
from moments_utils import load_moments_vl_prompts_list


CLASS_TO_TOKEN = {
    "GOAL": "goal",
    "CORNER/THROW-IN": "corner",
    "SHOT-ON-TARGET": "shot",
}
TOKEN_TO_CLASS = {token: label for label, token in CLASS_TO_TOKEN.items()}

PAIR_SPECS = [
    ("goal_vs_corner", "GOAL", "CORNER/THROW-IN"),
    ("goal_vs_shot", "GOAL", "SHOT-ON-TARGET"),
    ("corner_vs_shot", "CORNER/THROW-IN", "SHOT-ON-TARGET"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="qwen2-7b-vl-instruct")
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--torch_dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Torch dtype used when loading the model.",
    )
    parser.add_argument(
        "--csv",
        default="data/moments_goal/vision_only_data.csv",
        help="CSV providing clean prompts, images, and event_type labels.",
    )
    parser.add_argument("--language_only", action="store_true")
    parser.add_argument(
        "--check_counterfactual_alignment",
        action="store_true",
        help="Drop clean rows whose unused counterfactual has a token mismatch.",
    )
    parser.add_argument(
        "--output_csv",
        default="data/moments_goal/results/qwen2-7b-vl-instruct/behavior_event_type_pairwise_clean.csv",
    )
    parser.add_argument(
        "--summary_csv",
        default=None,
        help="Optional summary CSV path. Defaults to output_csv with _summary suffix.",
    )
    return parser.parse_args()


def event_type_pair_question(prompt: str, left_label: str, right_label: str) -> str:
    suffix = " Is this a goal? Answer yes or no."
    if suffix not in prompt:
        raise ValueError(f"Expected MOMENTS goal question suffix in prompt: {prompt!r}")
    left_token = CLASS_TO_TOKEN[left_label]
    right_token = CLASS_TO_TOKEN[right_label]
    return (
        prompt.rsplit(suffix, 1)[0]
        + f" What type of football event is shown? Answer {left_token} or {right_token}."
    )


def single_token_ids(model) -> dict[str, int]:
    ids = {}
    for token in CLASS_TO_TOKEN.values():
        token_ids = model.to_tokens(token, prepend_bos=False).view(-1).tolist()
        if len(token_ids) != 1:
            raise RuntimeError(f"{token!r} is not one model token: {token_ids}")
        ids[token] = int(token_ids[0])
    return ids


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary_rows = []
    for pair_name, pair_rows in group_by_pair(rows).items():
        correct = sum(bool(row["correct"]) for row in pair_rows)
        summary_rows.append(
            {
                "pair": pair_name,
                "group": "overall",
                "target": "all",
                "n": len(pair_rows),
                "correct": correct,
                "accuracy": correct / len(pair_rows),
            }
        )
        for target, count in Counter(row["target"] for row in pair_rows).items():
            target_rows = [row for row in pair_rows if row["target"] == target]
            target_correct = sum(bool(row["correct"]) for row in target_rows)
            summary_rows.append(
                {
                    "pair": pair_name,
                    "group": "target",
                    "target": target,
                    "n": count,
                    "correct": target_correct,
                    "accuracy": target_correct / count,
                }
            )
    return summary_rows


def group_by_pair(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["pair"]), []).append(row)
    return grouped


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = getattr(torch, args.torch_dtype)

    model, processor = load_model(
        args.model_name,
        args.model_path,
        device=device,
        use_tlens_wrapper=True,
        extra_hooks=False,
        torch_dtype=torch_dtype,
    )
    model.eval()
    token_ids = single_token_ids(model)
    logging.info("Constrained event-type token IDs: %s", token_ids)

    prompts = load_moments_vl_prompts_list(
        args.csv,
        model=model,
        processor=processor,
        language_only=args.language_only,
        correct_preds_only=False,
        check_counterfactual_alignment=args.check_counterfactual_alignment,
    )
    logging.info(
        "Counterfactual alignment filtering: %s",
        "enabled" if args.check_counterfactual_alignment else "disabled",
    )
    logging.info("Loaded %d retained clean prompts", len(prompts))

    rows = []
    for pair_name, left_label, right_label in PAIR_SPECS:
        pair_labels = {left_label, right_label}
        pair_tokens = [CLASS_TO_TOKEN[left_label], CLASS_TO_TOKEN[right_label]]
        pair_prompts = [
            (index, prompt)
            for index, prompt in enumerate(prompts)
            if str((prompt.metadata or {}).get("event_type", "")).strip().upper() in pair_labels
        ]
        logging.info("Evaluating %s on %d prompts", pair_name, len(pair_prompts))

        for original_index, prompt in pair_prompts:
            metadata = prompt.metadata or {}
            target = str(metadata.get("event_type", "")).strip().upper()
            if target not in CLASS_TO_TOKEN:
                raise ValueError(f"Unknown event_type at row {original_index}: {target!r}")
            classification_prompt = event_type_pair_question(prompt.prompt, left_label, right_label)
            with torch.no_grad():
                logits = model([classification_prompt], [prompt.images])[:, -1, :]
                constrained_logits = torch.stack(
                    [logits[:, token_ids[token]] for token in pair_tokens],
                    dim=-1,
                )
                prediction_index = int(constrained_logits.argmax(dim=-1).item())

            predicted_token = pair_tokens[prediction_index]
            prediction = TOKEN_TO_CLASS[predicted_token]
            rows.append(
                {
                    "pair": pair_name,
                    "row_index": original_index,
                    "clip_id": metadata.get("clip_id", ""),
                    "group_idx": metadata.get("group_idx", ""),
                    "clip_name": metadata.get("clip_name", ""),
                    "target": target,
                    "prediction": prediction,
                    "correct": prediction == target,
                    "left_label": left_label,
                    "right_label": right_label,
                    "left_token": pair_tokens[0],
                    "right_token": pair_tokens[1],
                    "left_logit": float(constrained_logits[0, 0].item()),
                    "right_logit": float(constrained_logits[0, 1].item()),
                }
            )

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = summarize_rows(rows)
    summary_output = (
        Path(args.summary_csv)
        if args.summary_csv
        else output.with_name(output.stem + "_summary.csv")
    )
    with summary_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"pairwise rows evaluated: {len(rows)}")
    for row in summary_rows:
        if row["group"] == "overall":
            print(
                f"{row['pair']} accuracy: {row['accuracy']:.4f} "
                f"({row['correct']}/{row['n']})"
            )
    print(
        "confusion counts (pair, target, prediction):",
        dict(Counter((row["pair"], row["target"], row["prediction"]) for row in rows)),
    )
    print(f"per-example output: {output}")
    print(f"summary output: {summary_output}")


if __name__ == "__main__":
    main()
