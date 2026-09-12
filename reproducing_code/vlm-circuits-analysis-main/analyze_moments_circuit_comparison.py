"""Compare saved MOMENTS circuits without rerunning model inference.

The comparison is deliberately hook-level: node-score files contain tensors
for hooks, so this script ranks hooks by maximum absolute attribution and
reports top-k overlap, a random-overlap baseline, layer composition, and
faithfulness summaries. It is an exploratory structural comparison, not a
causal interchange test.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", default="data/moments_goal/results")
    parser.add_argument(
        "--output_dir",
        default="figures/moments_circuit_comparison",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random_draws", type=int, default=10000)
    parser.add_argument("--top_k", type=int, nargs="+", default=[5, 10, 20])
    return parser.parse_args()


def result_specs(root: Path) -> dict[str, dict[str, Path]]:
    return {
        "language_only": {
            "label": "Language only",
            "scores": root / "qwen2-7b-vl-instruct_language_only_nap_ig_l=5/node_scores/nap_ig_l_ig=5_metric=LD.pt",
            "faithfulness": root / "qwen2-7b-vl-instruct_language_only_nap_ig_l=5/faithfulness_LD_l_node_circuit.pt",
        },
        "vision_only": {
            "label": "Vision only",
            "scores": root / "qwen2-7b-vl-instruct_vision_only_nap_ig_vl=5/node_scores/nap_ig_vl_ig=5_metric=LD.pt",
            "faithfulness": root / "qwen2-7b-vl-instruct_vision_only_nap_ig_vl=5/faithfulness_LD_vl_node_circuit.pt",
        },
        "both_multimodal": {
            "label": "Both (multimodal)",
            "scores": root / "qwen2-7b-vl-instruct/node_scores/nap_ig_vl_ig=5_metric=LD.pt",
            "faithfulness": root / "qwen2-7b-vl-instruct/faithfulness_LD_vl_node_circuit.pt",
        },
        "both_text_only": {
            "label": "Both (text only control)",
            "scores": root / "qwen2-7b-vl-instruct/node_scores/nap_ig_l_ig=5_metric=LD.pt",
            "faithfulness": root / "qwen2-7b-vl-instruct/faithfulness_LD_l_node_circuit.pt",
        },
    }


def load_scores(path: Path) -> dict[str, dict[str, float | int]]:
    raw = torch.load(path, weights_only=True)
    rows = {}
    for hook, tensor in raw.items():
        values = tensor.detach().float().cpu().abs()
        match = re.search(r"blocks\.(\d+)", str(hook))
        if match is None:
            continue
        rows[str(hook)] = {
            "layer": int(match.group(1)),
            "max_abs_score": float(values.max().item()),
            "mean_abs_score": float(values.mean().item()),
            "num_values": int(values.numel()),
        }
    return rows


def ranked_hooks(scores: dict[str, dict], key: str) -> list[str]:
    return sorted(scores, key=lambda hook: (-scores[hook][key], hook))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def faithfulness_rows(path: Path, condition: str) -> list[dict]:
    percentages, values, completed = torch.load(path, weights_only=True)
    values = values.detach().float().cpu().diag().tolist()
    completed = completed.detach().cpu().bool().diag().tolist()
    return [
        {
            "condition": condition,
            "circuit_percent": 100 * float(percent),
            "faithfulness_LD": float(value),
            "completed": bool(done),
        }
        for percent, value, done in zip(percentages, values, completed)
    ]


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    root = Path(args.results_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    specs = result_specs(root)
    scores = {}
    faithfulness = []
    for name, spec in specs.items():
        if not spec["scores"].exists() or not spec["faithfulness"].exists():
            print(f"Skipping {name}: missing score or faithfulness file")
            continue
        scores[name] = load_scores(spec["scores"])
        faithfulness.extend(faithfulness_rows(spec["faithfulness"], name))
        print(f"Loaded {name}: {len(scores[name])} hooks")

    if len(scores) < 2:
        raise RuntimeError("At least two complete result conditions are required.")

    all_hooks = sorted(set().union(*(set(table) for table in scores.values())))
    overlap_rows = []
    names = list(scores)
    for i, left in enumerate(names):
        left_ranked = ranked_hooks(scores[left], "max_abs_score")
        for right in names[i + 1 :]:
            right_ranked = ranked_hooks(scores[right], "max_abs_score")
            for k in args.top_k:
                left_set = set(left_ranked[:k])
                right_set = set(right_ranked[:k])
                intersection = len(left_set & right_set)
                union = len(left_set | right_set)
                random_jaccards = []
                for _ in range(args.random_draws):
                    a = set(random.sample(all_hooks, min(k, len(all_hooks))))
                    b = set(random.sample(all_hooks, min(k, len(all_hooks))))
                    random_jaccards.append(len(a & b) / max(1, len(a | b)))
                random_jaccards.sort()
                overlap_rows.append(
                    {
                        "left": left,
                        "right": right,
                        "top_k": k,
                        "left_available_hooks": len(left_ranked),
                        "right_available_hooks": len(right_ranked),
                        "intersection": intersection,
                        "jaccard": intersection / max(1, union),
                        "random_mean_jaccard": sum(random_jaccards) / len(random_jaccards),
                        "random_p025_jaccard": random_jaccards[int(0.025 * len(random_jaccards))],
                        "random_p975_jaccard": random_jaccards[int(0.975 * len(random_jaccards))],
                    }
                )

    write_csv(
        output / "top_k_overlap_random_baseline.csv",
        overlap_rows,
        list(overlap_rows[0]),
    )

    top_rows = []
    for name, table in scores.items():
        for rank, hook in enumerate(ranked_hooks(table, "max_abs_score"), start=1):
            top_rows.append({"condition": name, "rank": rank, "hook": hook, **table[hook]})
    write_csv(
        output / "ranked_hook_scores.csv",
        top_rows,
        ["condition", "rank", "hook", "layer", "max_abs_score", "mean_abs_score", "num_values"],
    )

    layer_rows = []
    for name, table in scores.items():
        for layer in sorted({row["layer"] for row in table.values()}):
            layer_hooks = [row for row in table.values() if row["layer"] == layer]
            layer_rows.append(
                {
                    "condition": name,
                    "layer": layer,
                    "max_abs_score": max(row["max_abs_score"] for row in layer_hooks),
                    "mean_abs_score": sum(row["mean_abs_score"] for row in layer_hooks) / len(layer_hooks),
                    "hook_count": len(layer_hooks),
                }
            )
    write_csv(
        output / "layer_composition.csv",
        layer_rows,
        ["condition", "layer", "max_abs_score", "mean_abs_score", "hook_count"],
    )
    write_csv(
        output / "faithfulness_summary.csv",
        faithfulness,
        ["condition", "circuit_percent", "faithfulness_LD", "completed"],
    )
    print(f"Saved circuit comparison outputs to {output.resolve()}")


if __name__ == "__main__":
    main()
