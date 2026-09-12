#!/usr/bin/env python3
"""Inspect raw LD baselines behind selected MOMENTS faithfulness scores."""

import argparse
import csv
import logging
from pathlib import Path

import torch

from analysis_utils import load_dataset, load_model
from evaluation_utils import circuit_faithfulness
from general_utils import get_top_scoring_components, set_deterministic


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name", default="qwen2-7b-vl-instruct")
    parser.add_argument("--ap_ig_steps", type=int, default=5)
    parser.add_argument("--percentages", nargs="+", type=float, default=[0.3, 0.5, 0.9])
    parser.add_argument(
        "--eval_prompt_index",
        type=int,
        default=None,
        help="Evaluate only this zero-based evaluation-prompt index.",
    )
    parser.add_argument(
        "--output_csv",
        default="data/moments_goal/results/qwen2-7b-vl-instruct/language_only_ld_baselines.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_deterministic(42)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    device = "cuda"
    model, processor = load_model(
        args.model_name,
        args.model_path,
        device,
        use_tlens_wrapper=True,
        extra_hooks=True,
        torch_dtype=torch.bfloat16,
    )
    _, _, eval_prompts = load_dataset(
        model,
        processor,
        "moments_goal",
        args.model_name,
        language_only=True,
        seed=42,
        train_test_split_ratio=0.75,
        moments_cf_mode="language_only",
    )
    if args.eval_prompt_index is not None:
        if not 0 <= args.eval_prompt_index < len(eval_prompts):
            raise ValueError(
                f"eval_prompt_index must be in [0, {len(eval_prompts) - 1}]"
            )
        eval_prompts = [eval_prompts[args.eval_prompt_index]]

    scores_path = Path(
        f"data/moments_goal/results/{args.model_name}/node_scores/"
        f"nap_ig_l_ig={args.ap_ig_steps}_metric=LD.pt"
    )
    scores = torch.load(scores_path, weights_only=True)
    scores = {key: value.abs() for key, value in scores.items()}
    seq_len = scores["blocks.0.attn.hook_z"].shape[0]

    model.cfg.ungroup_grouped_query_attention = True
    model.set_use_split_qkv_input(True)
    model.set_use_attn_result(True)
    model.set_use_hook_mlp_in(True)

    percentages = sorted(args.percentages)
    output_rows = []
    for requested_percent in percentages:
        n_mlp_neurons = int(requested_percent * model.cfg.d_mlp * model.cfg.n_layers * seq_len)
        n_heads = int(requested_percent * model.cfg.n_heads * model.cfg.n_layers * seq_len)
        top_heads, top_mlps, _, _ = get_top_scoring_components(
            model, scores, n_heads, n_mlp_neurons
        )
        _, details = circuit_faithfulness(
            model,
            top_heads + top_mlps,
            eval_prompts,
            metric="LD",
            verbose=True,
            return_details=True,
        )
        for row in details:
            row["requested_circuit_percent"] = requested_percent * 100
            output_rows.append(row)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "requested_circuit_percent",
        "prompt_index",
        "clip_id",
        "group_idx",
        "clip_name",
        "good_ld",
        "bad_ld",
        "ablated_ld",
        "denominator",
        "normalized_score",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} per-sample baseline rows to {output_path}")


if __name__ == "__main__":
    main()
