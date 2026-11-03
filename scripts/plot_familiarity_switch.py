"""Plot the predeclared same-image/task familiarity-switch contrasts."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def run(root):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), sharey=True)
    rows = []
    definitions = [("pets", "real_minus_permuted_mean", "Pets49: real − identity", "#176A9D"),
                   ("pets", "real_minus_wide", "Pets49: real − shared", "#AB6125"),
                   ("flowers", "real_minus_permuted_mean", "Flowers98: real − identity", "#176A9D"),
                   ("flowers", "real_minus_wide", "Flowers98: real − shared", "#AB6125")]
    summaries = {name: json.loads((root / name / "evaluation" / "summary.json").read_text())
                 for name in ("pets", "flowers")}
    for col, metric in enumerate(("correct", "margin")):
        ax = axes[col]
        scale = 1. if metric == "correct" else 1000.
        for index, (dataset, contrast, label, color) in enumerate(definitions):
            saved = summaries[dataset]["metrics"][metric]["contrasts"][contrast]
            value = saved["seen_minus_held"]
            point, low, high = value["estimate"] * scale, *(np.array(value["ci95"]) * scale)
            y = len(definitions) - 1 - index
            ax.errorbar(point, y, xerr=[[point - low], [high - point]], fmt="o", color=color,
                        markersize=5, capsize=3, elinewidth=1.4, zorder=3)
            seed_values = np.array(saved["seen_minus_held_per_seed"]) * scale
            ax.scatter(seed_values, np.full(len(seed_values), y + .17), marker="|", color=color,
                       s=30, alpha=.65, zorder=2)
            rows.append(dict(dataset=dataset, contrast=contrast, metric=metric,
                             estimate=value["estimate"], lower95=value["ci95"][0], upper95=value["ci95"][1],
                             seed_2041=saved["seen_minus_held_per_seed"][0], seed_2042=saved["seen_minus_held_per_seed"][1],
                             seed_2043=saved["seen_minus_held_per_seed"][2]))
        ax.axvline(0, color="#3E4750", lw=.8, linestyle="--")
        ax.axhline(1.5, color="#CED2D6", lw=.7)
        ax.grid(axis="x", alpha=.15)
        ax.set_yticks(np.arange(len(definitions))[::-1], [item[2] for item in definitions])
        ax.set_ylim(-.45, 3.55)
        ax.set_xlabel("Seen − held change in benefit (accuracy, pp)" if metric == "correct"
                      else "Seen − held change in benefit (margin × 10³)")
        ax.set_title("Accuracy" if metric == "correct" else "Geometric margin", weight="bold")
    fig.suptitle("Does conditional benefit follow training familiarity?", weight="bold", fontsize=12, y=.98)
    fig.legend(handles=[Line2D([0], [0], marker="o", color="#25344B", label="Mean and image-bootstrap 95% interval"),
                        Line2D([0], [0], marker="|", linestyle="none", color="#647080", label="Individual paired seeds")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.54, .025), fontsize=8)
    fig.text(.54, .005, "Identical image/task rows and class codes across fits; both candidate familiarity roles exchange.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .14, 1, .93), w_pad=2.0)
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"familiarity_switch_effects.{suffix}", dpi=230, bbox_inches="tight")
    plt.close(fig)
    with (root / "plot_data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/familiarity-switch-v1"))
    run(parser.parse_args().results_root)
