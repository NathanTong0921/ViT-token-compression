"""Compare fixed direct-training effects on development and expanded evaluation."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COMPARISONS = (
    ("direct_C_minus_direct_U", "Direct C − direct U"),
    ("direct_gap_minus_kl_gap", "Change in C−U gap\n(direct versus KL)"),
    ("direct_C_minus_kl_C", "Direct C − KL C"),
    ("direct_U_minus_kl_U", "Direct U − KL U"),
)


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    panels = [("Development: 1,020 images", args.development, "#8A94A6", -.12),
              ("Existing test panel: 6,149 images", args.expanded, "#2166A5", .12)]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), sharey=True)
    fig.subplots_adjust(left=.24, right=.98, top=.75, bottom=.20, wspace=.23)
    fig.suptitle("Direct classification training across evaluation panels", fontsize=16,
                 fontweight="bold", y=.98)
    fig.text(.5, .885, "Flowers102 · CLIP B/16 · 49 tokens · fixed 1,200-step weights · three seeds",
             ha="center", fontsize=10, color="#596573")
    rows = []
    for ai, (metric, multiplier, xlabel) in enumerate((
            ("accuracy", 100., "Accuracy difference (percentage points)"),
            ("raw_gap", 1000., "Cosine-gap difference (× 10⁻³)"))):
        ax = axes[ai]
        extrema = []
        for panel_name, source, color, offset in panels:
            summary = json.loads(source.read_text())
            means, lower, upper = [], [], []
            for key, label in COMPARISONS:
                value = summary["contrasts"][key][metric]
                ci = value.get("two_primary_adjusted_ci95", value["ci95"])
                means.append(value["mean"] * multiplier)
                lower.append(ci[0] * multiplier)
                upper.append(ci[1] * multiplier)
                rows.append(dict(panel=panel_name, contrast=key, metric=metric,
                                 mean=means[-1], low=lower[-1], high=upper[-1],
                                 scale=multiplier,
                                 interval="two-primary adjusted" if "two_primary_adjusted_ci95" in value else "ordinary 95%"))
            means, lower, upper = map(np.asarray, (means, lower, upper))
            ax.errorbar(means, np.arange(len(COMPARISONS))+offset,
                        xerr=np.stack((means-lower, upper-means)), fmt="o", color=color,
                        capsize=3, markersize=5, linewidth=1.5, label=panel_name)
            extrema.extend(lower); extrema.extend(upper)
        ax.axvline(0, ls="--", lw=1, color="#737D8B")
        ax.grid(axis="x", color="#E7EBF0", linewidth=.7)
        ax.set_axisbelow(True)
        span = max(abs(float(x)) for x in extrema) * 1.10
        ax.set_xlim(-span, span)
        ax.set_xlabel(xlabel)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.set_ylim(len(COMPARISONS)-.5, -.5)
    axes[0].set_yticks(np.arange(len(COMPARISONS)), [label for _, label in COMPARISONS])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5,.85), ncol=2,
               frameon=False, fontsize=10)
    fig.text(.24, .075,
             "Image-cluster bootstrap intervals, averaging the same three trained seeds.\n"
             "Top two contrasts: adjustment for the two primary comparisons; bottom two: ordinary 95% intervals.",
             color="#596573", fontsize=9, va="center")
    for extension in ("png", "pdf"):
        fig.savefig(args.output / f"development_and_expanded.{extension}", dpi=240, facecolor="white")
    plt.close(fig)
    with (args.output / "source_values.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, default=Path("results/direct-classification-v1/summary.json"))
    parser.add_argument("--expanded", type=Path, default=Path("results/direct-classification-panel-v1/summary.json"))
    parser.add_argument("--output", type=Path, default=Path("results/direct-classification-panel-v1/figures"))
    main(parser.parse_args())
