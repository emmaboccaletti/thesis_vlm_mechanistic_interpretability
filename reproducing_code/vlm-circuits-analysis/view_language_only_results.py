#!/usr/bin/env python3
"""Summarize and plot Qwen2 MOMENTS language-only circuit results."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch


BLOCK_RE = re.compile(r"blocks\.(\d+)\.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot faithfulness and summarize language-only node attributions."
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("data/moments_goal/results/qwen2-7b-vl-instruct"),
        help="Directory containing faithfulness_LD_l_node_circuit.pt and node_scores.",
    )
    parser.add_argument(
        "--ig_steps",
        type=int,
        default=5,
        help="Integrated-gradients step count used in the node-score filename.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("figures/language_only_results"),
        help="Directory for plots, CSV output, and the text summary.",
    )
    return parser.parse_args()


def load_results(results_dir: Path, ig_steps: int):
    faithfulness_path = results_dir / "faithfulness_LD_l_node_circuit.pt"
    scores_path = results_dir / "node_scores" / f"nap_ig_l_ig={ig_steps}_metric=LD.pt"
    if not faithfulness_path.exists():
        raise FileNotFoundError(f"Missing faithfulness file: {faithfulness_path}")
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing node-score file: {scores_path}")

    percentages, faithfulness, completed = torch.load(
        faithfulness_path, weights_only=True
    )
    scores = torch.load(scores_path, weights_only=True)
    return percentages, faithfulness, completed, scores, faithfulness_path, scores_path


def layer_number(hook_name: str) -> int | None:
    match = BLOCK_RE.search(str(hook_name))
    return int(match.group(1)) if match else None


def score_rows(scores: dict) -> list[dict]:
    rows = []
    for hook_name, tensor in scores.items():
        layer = layer_number(str(hook_name))
        if layer is None:
            continue
        values = tensor.detach().float().abs().cpu()
        if values.ndim < 2:
            continue

        # For MLPs this is [position, neuron]. For attention z it is
        # [position, head, d_head]. Reduce only the feature dimensions.
        position_scores = values.reshape(values.shape[0], -1).mean(dim=1)
        flat_index = int(values.reshape(-1).argmax())
        rows.append(
            {
                "hook": str(hook_name),
                "layer": layer,
                "shape": "x".join(str(dim) for dim in values.shape),
                "mean_abs_score": float(values.mean()),
                "max_abs_score": float(values.max()),
                "mean_position_score": float(position_scores.mean()),
                "top_flat_index": flat_index,
            }
        )
    return sorted(rows, key=lambda row: row["max_abs_score"], reverse=True)


def write_summary(output_dir: Path, percentages, faithfulness, completed, rows, paths):
    values = faithfulness.diag().detach().float().cpu().tolist()
    done = completed.diag().detach().cpu().tolist()
    summary_path = output_dir / "summary.txt"
    with summary_path.open("w") as handle:
        handle.write("Qwen2 MOMENTS goal language-only results\n")
        handle.write(f"Faithfulness file: {paths[0]}\n")
        handle.write(f"Node-score file: {paths[1]}\n")
        handle.write(f"Completed checkpoints: {sum(done)}/{len(done)}\n\n")
        handle.write("Faithfulness (normalized LD score)\n")
        for percent, value, is_done in zip(percentages, values, done):
            handle.write(f"{float(percent):.3f}\t{value:.6f}\tcompleted={bool(is_done)}\n")
        handle.write("\nTop attribution entries by maximum absolute score\n")
        for row in rows[:20]:
            handle.write(
                f"{row['hook']}\tlayer={row['layer']}\t"
                f"shape={row['shape']}\tmax={row['max_abs_score']:.6g}\n"
            )
    return summary_path


def write_csv(output_dir: Path, rows: list[dict]) -> Path:
    path = output_dir / "top_node_scores.csv"
    fields = list(rows[0]) if rows else ["hook", "layer"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def make_plots(output_dir: Path, percentages, faithfulness, rows) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    x = [float(value) * 100 for value in percentages]
    y = faithfulness.diag().detach().float().cpu().numpy()

    faithfulness_path = output_dir / "faithfulness_language_only.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker="o", linewidth=2, color="#0173b2")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(1, color="#d55e00", linestyle="--", linewidth=1)
    ax.set_xlabel("Nodes included in circuit (%)")
    ax.set_ylabel("Normalized faithfulness (LD)")
    ax.set_title("Qwen2 MOMENTS goal: language-only circuit")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(faithfulness_path, dpi=180)
    plt.close(fig)

    layer_values: dict[int, float] = {}
    for row in rows:
        layer_values[row["layer"]] = max(
            layer_values.get(row["layer"], 0.0), row["max_abs_score"]
        )
    layers = sorted(layer_values)
    layer_path = output_dir / "attribution_by_layer.png"
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(layers, [layer_values[layer] for layer in layers], color="#029e73")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Maximum absolute attribution")
    ax.set_title("Language-only attribution strength by layer")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(layer_path, dpi=180)
    plt.close(fig)
    return faithfulness_path, layer_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    percentages, faithfulness, completed, scores, faith_path, scores_path = load_results(
        args.results_dir, args.ig_steps
    )
    rows = score_rows(scores)
    summary_path = write_summary(
        args.output_dir,
        percentages,
        faithfulness,
        completed,
        rows,
        (faith_path, scores_path),
    )
    csv_path = write_csv(args.output_dir, rows)
    faith_plot, layer_plot = make_plots(args.output_dir, percentages, faithfulness, rows)
    print(f"Completed checkpoints: {int(completed.diag().sum())}/{len(percentages)}")
    print(f"Faithfulness plot: {faith_plot}")
    print(f"Layer plot: {layer_plot}")
    print(f"Summary: {summary_path}")
    print(f"Node CSV: {csv_path}")


if __name__ == "__main__":
    main()
