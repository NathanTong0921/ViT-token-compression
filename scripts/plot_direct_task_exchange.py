"""Plot own/exchanged task outputs and the direct-versus-KL mapping effect."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(args):
    summary = json.loads(args.source.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    fig.subplots_adjust(left=.065, right=.98, top=.72, bottom=.24, wspace=.40)
    fig.suptitle("Does direct training increase the value of the correct task?",
                 fontsize=15, fontweight="bold", y=.97)
    fig.text(.5, .875, "Flowers102 · same images and target classes · average over three alternative tasks per image",
             ha="center", color="#596573", fontsize=10)
    styles = (("C", "own", "Correct-task C", "#2166A5", "o"),
              ("C", "swapped", "Exchanged-task C", "#C77C27", "x"),
              ("U", "own", "Image-only U", "#768292", "s"))
    ranges = []
    for ax, budget in zip(axes[:2], summary["budgets"]):
        for arm, context, label, color, marker in styles:
            values = [summary["methods"][f"budget{budget}_{stage}_{arm}_{context}"]["accuracy"]["mean"] * 100
                      for stage in ("initial", "kl", "direct")]
            ranges.extend(values)
            ax.plot(range(3), values, marker=marker, label=label, color=color,
                    markersize=5, linewidth=1.7)
        ax.set_xticks(range(3), ["Initial", "KL", "Direct"])
        ax.set_title(f"{budget} retained tokens", fontsize=11, pad=10)
        ax.grid(axis="y", color="#E7EBF0")
        ax.set_axisbelow(True)
    spread = max(ranges) - min(ranges)
    for ax in axes[:2]:
        ax.set_ylim(min(ranges) - .15 * spread, max(ranges) + .15 * spread)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(frameon=False, fontsize=9, ncol=3, bbox_to_anchor=(0, -.25), loc="upper left")
    points, intervals = [], []
    for budget in summary["budgets"]:
        result = summary["contrasts"][f"budget{budget}_mapping_change_direct_vs_kl"]["accuracy"]
        points.append(100 * result["mean"])
        intervals.append(np.asarray(result["two_primary_adjusted_ci95"]) * 100)
    points, intervals = np.asarray(points), np.asarray(intervals)
    error = np.stack((points - intervals[:, 0], intervals[:, 1] - points))
    axes[2].errorbar(points, [0, 1], xerr=error, fmt="o", color="#2166A5", capsize=3, markersize=6)
    axes[2].set_yticks([0, 1], [f"{budget} tokens" for budget in summary["budgets"]])
    axes[2].set_ylim(1.5, -.5)
    axes[2].axvline(0, color="#768292", linestyle="--", linewidth=1)
    axes[2].set_title("Change in correct-task value", fontsize=11, pad=10)
    axes[2].set_xlabel("(Own − exchanged): direct − KL (pp)", fontsize=9)
    axes[2].grid(axis="x", color="#E7EBF0")
    axes[2].set_axisbelow(True)
    lo, hi = min(0, intervals.min()), max(0, intervals.max())
    width = max(hi - lo, .05)
    axes[2].set_xlim(lo - .15 * width, hi + .15 * width)
    fig.text(.065, .06, "Curves show means of three fixed seeds. Right: paired image-bootstrap intervals adjusted over both budgets.\n"
             "Task exchange changes the selector's candidate description; classification still uses the original target classes.",
             color="#596573", fontsize=9)
    for extension in ("png", "pdf"):
        fig.savefig(args.output / f"task_exchange.{extension}", dpi=240, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/direct-task-exchange-v1/summary.json"))
    parser.add_argument("--output", type=Path, default=Path("results/direct-task-exchange-v1/figures"))
    main(parser.parse_args())
