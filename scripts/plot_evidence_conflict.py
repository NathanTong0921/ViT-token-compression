#!/usr/bin/env python3
"""Plot the frozen-CLIP evidence-conflict screen from saved small artifacts."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"pets": "#0072B2", "flowers": "#D55E00"}
LABELS = {"pets": "Pets", "flowers": "Flowers"}
BUDGETS = [0, 4, 16, 49, 98, 196]


def original_curve(rows, layer):
    values = []
    for keep in BUDGETS:
        methods = (lambda name: name == "cls_only") if keep == 0 else (
            (lambda name: name == "all") if keep == 196 else lambda name: name.startswith("random_"))
        selected = [float(r["binary_accuracy"]) for r in rows
                    if int(r["layer"]) == layer and int(r["keep"]) == keep and methods(r["method"])]
        assert len(selected) == (1 if keep in (0, 196) else 4)
        values.append(100 * np.mean(selected))
    return values


def plot(input_dir, output_prefix):
    review = json.loads((input_dir / "independent-review.json").read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42,
                         "axes.titlesize": 11, "legend.fontsize": 8.5})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8))
    fig.subplots_adjust(left=.075, right=.985, bottom=.30, top=.87, wspace=.28)
    for dataset in ("pets", "flowers"):
        with (input_dir / dataset / "original_summary.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        for layer in (3, 6):
            axes[0].plot(np.arange(len(BUDGETS)), original_curve(rows, layer),
                         color=COLORS[dataset], linestyle="--" if layer == 3 else "-",
                         marker="o" if layer == 3 else "s", markersize=4,
                         linewidth=1.7, label=f"{LABELS[dataset]}, block {layer}")
    axes[0].axhline(50, color=".65", linestyle=":", linewidth=1, zorder=0)
    axes[0].set(title="a  Original images: fixed random selection",
                ylabel="Binary accuracy (%)", xlabel="Retained patch tokens (CLS always retained)",
                ylim=(45, 102), yticks=[50, 60, 70, 80, 90, 100])
    axes[0].set_xticks(np.arange(len(BUDGETS)), ["0\n(CLS only)", "4", "16", "49", "98", "196\n(all)"])
    axes[0].legend(loc="lower right", frameon=True, framealpha=.94, edgecolor="none")
    axes[0].grid(axis="y", alpha=.18)

    width = .18
    for group, (dataset, layer) in enumerate((d, l) for d in ("pets", "flowers") for l in (3, 6)):
        selected = {r["keep"]: r["native_all_four_correct"]
                    for r in review["datasets"][dataset]["four_way"] if r["layer"] == layer}
        x = np.arange(2) + (group - 1.5) * width
        heights = [100 * selected[k]["conflict_fraction"] for k in (16, 49)]
        axes[1].bar(x, heights, width=width * .9, color=COLORS[dataset],
                    alpha=.7 if layer == 3 else 1., hatch="//" if layer == 3 else None,
                    edgecolor="white", linewidth=.8, label=f"{LABELS[dataset]}, block {layer}")
        for xi, height, keep in zip(x, heights, (16, 49)):
            count = selected[keep]
            axes[1].text(xi, height + .45, f'{count["conflict_count"]}/{count["images"]}',
                         ha="center", va="bottom", fontsize=8)
    axes[1].set(title="b  Four-task conflict in the fixed mask library",
                ylabel="Four-task conflicts (%)",
                xlabel="Retained patch tokens", ylim=(0, 18.7), yticks=[0, 5, 10, 15])
    axes[1].set_xticks([0, 1], ["16", "49"])
    axes[1].legend(loc="upper right", ncol=2, frameon=False, columnspacing=.9, handlelength=1.7)
    axes[1].grid(axis="y", alpha=.18)
    axes[1].set_axisbelow(True)
    fig.text(.075, .08,
             "a: Mean of four fixed random masks; 0 and 196 use CLS-only and all-patch inference.\n"
             "b: All four tasks are individually feasible, but no one of the same 35 masks answers all four.\n"
             "Only images where full CLIP answers all four correctly are included; labels show image counts.",
             fontsize=8.8, va="center", linespacing=1.45)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output_prefix.with_suffix(f".{extension}"), dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/evidence-conflict-v1"))
    parser.add_argument("--output", type=Path, help="Output prefix, without extension")
    args = parser.parse_args()
    plot(args.input, args.output or args.input / "evidence_conflict")
