"""Evaluate clean Qwen2 behavior on the MOMENTS goal/no-goal task.

The primary metric is constrained yes/no accuracy: at the final answer
position, choose only between the token logits for yes and no. This avoids
counting an unrelated high-probability token as a valid task answer.
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
        help="CSV providing clean prompts and labels.",
    )
    parser.add_argument("--language_only", action="store_true")
    parser.add_argument(
        "--check_counterfactual_alignment",
        action="store_true",
        help="Drop clean rows whose unused counterfactual has a token-length mismatch.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--output_csv",
        default="data/moments_goal/results/qwen2-7b-vl-instruct/behavior_clean.csv",
    )
    return parser.parse_args()


def canonical_token_ids(model, processor) -> dict[str, int]:
    ids = {}
    for label in ("yes", "no"):
        token_ids = model.to_tokens(label, prepend_bos=False).view(-1).tolist()
        if len(token_ids) != 1:
            raise RuntimeError(f"{label!r} is not one model token: {token_ids}")
        ids[label] = int(token_ids[0])
    return ids


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
    label_ids = canonical_token_ids(model, processor)
    logging.info("Constrained answer token IDs: %s", label_ids)

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
        with torch.no_grad():
            logits = model([prompt.prompt], [prompt.images])[:, -1, :]
            constrained_logits = torch.stack(
                [logits[:, label_ids["yes"]], logits[:, label_ids["no"]]], dim=-1
            )
            constrained_index = int(constrained_logits.argmax(dim=-1).item())
            constrained_prediction = ("yes", "no")[constrained_index]
            unrestricted_id = int(logits.argmax(dim=-1).item())

        target = str(prompt.answer).strip().lower()
        metadata = prompt.metadata or {}
        rows.append(
            {
                "row_index": index,
                "clip_id": metadata.get("clip_id", ""),
                "group_idx": metadata.get("group_idx", ""),
                "clip_name": metadata.get("clip_name", ""),
                "target": target,
                "prediction_constrained_yes_no": constrained_prediction,
                "constrained_correct": constrained_prediction == target,
                "unrestricted_token_id": unrestricted_id,
                "yes_logit": float(constrained_logits[0, 0].item()),
                "no_logit": float(constrained_logits[0, 1].item()),
                "prompt_token_length": metadata.get("prompt_token_length", ""),
            }
        )

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    correct = sum(row["constrained_correct"] for row in rows)
    targets = Counter(row["target"] for row in rows)
    predictions = Counter(row["prediction_constrained_yes_no"] for row in rows)
    print(f"rows evaluated: {total}")
    print(f"target counts: {dict(targets)}")
    print(f"prediction counts: {dict(predictions)}")
    print(f"constrained yes/no accuracy: {correct / total:.4f} ({correct}/{total})")
    for label in ("yes", "no"):
        class_rows = [row for row in rows if row["target"] == label]
        class_correct = sum(row["constrained_correct"] for row in class_rows)
        print(
            f"{label} accuracy: {class_correct / len(class_rows):.4f} "
            f"({class_correct}/{len(class_rows)})"
        )
    confusion = Counter((row["target"], row["prediction_constrained_yes_no"]) for row in rows)
    print("confusion counts (target, prediction):", dict(confusion))
    print(f"per-example output: {output}")


if __name__ == "__main__":
    main()
