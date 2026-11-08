"""Plot validation KL and development classification for PR checkpoint choice."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(results_dir):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10,
                         "svg.fonttype": "none"})
    datasets = (("pets", "Oxford-IIIT Pets"), ("flowers", "Flowers102"))
    seed_colors = ("#3169A5", "#B27924", "#8055A3")
    methods = (("pr_final", "PR\nEpoch 50", "#7B8790"),
               ("pr_best_val_kl", "PR\nMinimum val KL", "#3169A5"),
               ("image_only_wide", "Strong shared U", "#267A68"),
               ("conditioned", "Conditioned C", "#8055A3"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.1), sharey="row")
    for column, (dataset, title) in enumerate(datasets):
        root = results_dir / dataset
        analysis = json.loads((root / "analysis/analysis.json").read_text())
        with (root / "training/training_history.csv").open() as handle:
            history = list(csv.DictReader(handle))
        ax = axes[0, column]
        for selection, color in zip(analysis["checkpoint_selection"], seed_colors):
            seed = selection["seed"]
            rows = sorted([r for r in history if int(r["seed"]) == seed],
                          key=lambda r: int(r["epoch"]))
            epochs = [int(r["epoch"]) for r in rows]
            losses = [float(r["val_kl"]) for r in rows]
            selected = selection["selected_epoch"]
            ax.plot(epochs, losses, color=color, lw=1.7,
                    label=f"Seed {seed}: minimum at epoch {selected}")
            ax.scatter([selected], [selection["selected_val_kl"]], s=44,
                       color=color, edgecolor="white", linewidth=.8, zorder=4)
        ax.set(title=f"({chr(97 + column)}) {title}: validation loss",
               xlabel="Epoch", xlim=(0, 51), ylim=(.23, .34),
               xticks=[1, 10, 20, 30, 40, 50], yticks=[.24, .26, .28, .30, .32, .34])
        ax.legend(loc="upper left", frameon=False, fontsize=8.7, handlelength=2)

        ax = axes[1, column]
        table = {row["method"]: row for row in analysis["table"]}
        for x, (method, label, color) in enumerate(methods):
            estimate = table[method]["accuracy"]
            mean = 100 * estimate["mean"]
            low, high = 100 * np.asarray(estimate["ci95"])
            ax.errorbar(x, mean, yerr=[[mean-low], [high-mean]], color=color,
                        fmt="o", markersize=6, elinewidth=1.6, capsize=4, zorder=3)
            ax.text(x, high + .18, f"{mean:.2f}", ha="center", va="bottom", fontsize=9.5)
        ax.set(title=f"({chr(99 + column)}) {title}: development accuracy",
               xticks=range(len(methods)), xticklabels=[m[1] for m in methods],
               xlim=(-.5, 3.5), ylim=(90, 100), yticks=[90, 92, 94, 96, 98, 100])
    axes[0, 0].set_ylabel("Validation KL")
    axes[1, 0].set_ylabel("Task/class-balanced accuracy (%)")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#9AA4AA")
        ax.tick_params(length=3, color="#9AA4AA")
        ax.grid(axis="y", color="#C5CDD2", linewidth=.7, alpha=.6)
        ax.set_axisbelow(True)
    fig.suptitle("Patch Ranking: validation loss and checkpoint choice", fontsize=17, y=.98)
    fig.text(.5, .938, "Frozen CLIP ViT-B/16  ·  Block 6  ·  98/196 patches  ·  Development data",
             ha="center", color="#4F5B64", fontsize=10.5)
    fig.text(.075, .038,
             "Accuracy: means over three seeds; bars are image-bootstrap 95% confidence intervals. Both accuracy axes span 90–100%.\n"
             "Checkpoint selection uses validation KL. These development results precede the reserved final test.",
             ha="left", va="bottom", fontsize=9, color="#4F5B64", linespacing=1.5)
    fig.subplots_adjust(left=.075, right=.98, bottom=.15, top=.88, hspace=.38, wspace=.15)
    for suffix in ("png", "svg"):
        fig.savefig(results_dir / f"checkpoint_comparison.{suffix}", dpi=220,
                    facecolor="white", bbox_inches="tight", pad_inches=.16)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results/patch-ranking-checkpoints-v1")
    args = parser.parse_args()
    plot(args.results_dir)
