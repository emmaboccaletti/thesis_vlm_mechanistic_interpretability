"""Evaluate clean Qwen2 behavior on the three MOMENTS event types.

The evaluator keeps the clean transcript and image, replaces the original
yes/no question with an explicit three-way event-type question, and compares
only the final-token logits for the canonical labels goal, corner, and shot.
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
        default="data/moments_goal/results/qwen2-7b-vl-instruct/behavior_event_type_clean.csv",
    )
    return parser.parse_args()


def event_type_question(prompt: str) -> str:
    suffix = " Is this a goal? Answer yes or no."
    if suffix not in prompt:
        raise ValueError(f"Expected MOMENTS goal question suffix in prompt: {prompt!r}")
    return (
        prompt.rsplit(suffix, 1)[0]
        + " What type of football event is shown? Answer goal, corner, or shot."
    )


def single_token_ids(model) -> dict[str, int]:
    ids = {}
    for token in CLASS_TO_TOKEN.values():
        token_ids = model.to_tokens(token, prepend_bos=False).view(-1).tolist()
        if len(token_ids) != 1:
            raise RuntimeError(f"{token!r} is not one model token: {token_ids}")
        ids[token] = int(token_ids[0])
    return ids


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, processor = load_model(
        args.model_name,
        args.model_path,
        device=device,
        use_tlens_wrapper=True,
        extra_hooks=False,
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
    logging.info("Evaluating %d retained clean prompts", len(prompts))

    rows = []
    for index, prompt in enumerate(prompts):
        metadata = prompt.metadata or {}
        target = str(metadata.get("event_type", "")).strip().upper()
        if target not in CLASS_TO_TOKEN:
            raise ValueError(f"Unknown event_type at row {index}: {target!r}")

        classification_prompt = event_type_question(prompt.prompt)
        with torch.no_grad():
            logits = model([classification_prompt], [prompt.images])[:, -1, :]
            constrained_logits = torch.stack(
                [logits[:, token_ids[token]] for token in CLASS_TO_TOKEN.values()],
                dim=-1,
            )
            prediction_index = int(constrained_logits.argmax(dim=-1).item())

        predicted_token = list(CLASS_TO_TOKEN.values())[prediction_index]
        prediction = TOKEN_TO_CLASS[predicted_token]
        rows.append(
            {
                "row_index": index,
                "clip_id": metadata.get("clip_id", ""),
                "group_idx": metadata.get("group_idx", ""),
                "clip_name": metadata.get("clip_name", ""),
                "target": target,
                "prediction": prediction,
                "correct": prediction == target,
                "goal_logit": float(constrained_logits[0, 0].item()),
                "corner_logit": float(constrained_logits[0, 1].item()),
                "shot_logit": float(constrained_logits[0, 2].item()),
            }
        )

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    targets = Counter(row["target"] for row in rows)
    predictions = Counter(row["prediction"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    class_accuracies = {}
    for label in CLASS_TO_TOKEN:
        class_rows = [row for row in rows if row["target"] == label]
        class_accuracies[label] = sum(row["correct"] for row in class_rows) / len(class_rows)

    print(f"rows evaluated: {len(rows)}")
    print(f"target counts: {dict(targets)}")
    print(f"prediction counts: {dict(predictions)}")
    print(f"constrained event-type accuracy: {correct / len(rows):.4f} ({correct}/{len(rows)})")
    for label, accuracy in class_accuracies.items():
        print(f"{label} accuracy: {accuracy:.4f}")
    print(f"balanced accuracy: {sum(class_accuracies.values()) / len(class_accuracies):.4f}")
    print(
        "confusion counts (target, prediction):",
        dict(Counter((row["target"], row["prediction"]) for row in rows)),
    )
    print(f"per-example output: {output}")


if __name__ == "__main__":
    main()
