#!/usr/bin/env python3
"""Plot the six fixed partition diagnostics without pooling overlapping images."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run(root, partitions):
    data = {(dataset, partition): json.loads((root / f"{dataset}-p{partition}" / "summary.json").read_text())
            for dataset in ("pets", "flowers") for partition in partitions}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.1))
    fig.subplots_adjust(left=.085, right=.985, bottom=.17, top=.88, hspace=.43, wspace=.24)
    csv_rows = []
    colors = ["#444444", "#0072B2", "#D55E00"]
    for column, dataset in enumerate(("pets", "flowers")):
        for j, kind in enumerate(("raw", "global", "task")):
            values = [data[(dataset, p)]["correction"][kind]["contrasts"]["accuracy_pp"]["real_minus_identity"]["seen_minus_held"] for p in partitions]
            for i, (partition, value) in enumerate(zip(partitions, values)):
                x = i + (j - 1) * .15
                axes[0, column].plot([x, x], value["ci95"], color=colors[j], linewidth=1.3)
                axes[0, column].plot(x, value["estimate"], marker=["o", "s", "^"][j], color=colors[j], markersize=5,
                                     label={"raw": "Raw", "global": "Global correction", "task": "Task correction"}[kind] if i == 0 else None)
                csv_rows.append({"dataset": dataset, "partition": partition, "metric": "role_accuracy_pp", "variant": kind,
                                 "estimate": value["estimate"], "ci_low": value["ci95"][0], "ci_high": value["ci95"][1]})
        for j, (key, label) in enumerate((("midpoint_shift", "Candidate-direction shift"), ("half_separation_change", "Half-separation change"))):
            for i, partition in enumerate(partitions):
                value = data[(dataset, partition)]["decomposition"]["real_minus_identity"][key]
                x = i + (j - .5) * .16
                axes[1, column].plot([x, x], np.asarray(value["ci95"]) * 1000, color=colors[j + 1], linewidth=1.3)
                axes[1, column].plot(x, value["estimate"] * 1000, marker=["s", "D"][j], color=colors[j + 1], markersize=5,
                                     label=label if i == 0 else None)
                csv_rows.append({"dataset": dataset, "partition": partition, "metric": "decomposition_geometric_margin", "variant": key,
                                 "estimate": value["estimate"], "ci_low": value["ci95"][0], "ci_high": value["ci95"][1]})
        axes[0, column].set_title("Pets / 49 patches" if dataset == "pets" else "Flowers / 98 patches")
        for row in range(2):
            axis = axes[row, column]
            axis.axhline(0, color=".5", linestyle="--", linewidth=.8)
            axis.set_xticks(range(len(partitions)), [f"{p}\n" + ("exploratory" if p == 2099 else "new partition") for p in partitions])
            axis.set_xlim(-.45, len(partitions) - .55)
            axis.grid(axis="y", alpha=.18)
    fig.suptitle("Historical partition 2099 only" if partitions == [2099] else "Fixed class partitions", y=.97, fontsize=12)
    axes[0, 0].set_ylabel("Familiarity-linked H0 choice shift (pp)")
    axes[1, 0].set_ylabel("Score change (geometric margin ×10⁻³)")
    axes[0, 0].legend(loc="best", fontsize=8, frameon=False)
    axes[1, 0].legend(loc="best", fontsize=8, frameon=False)
    fig.text(.085, .06,
             "Real text minus fixed identity controls; H0 choice shift equals the previous role-oriented accuracy contrast.\n"
             "Offsets use five disjoint image folds and full-CLIP reference scores, without labels in fitting.\n"
             "Intervals condition on the fixed task panels and fitted offsets; correction effects do not establish an internal prior mechanism.",
             fontsize=8.5, va="center", linespacing=1.4)
    for extension in ("png", "pdf"):
        fig.savefig(root / f"familiarity_prior.{extension}", dpi=220, facecolor="white")
    plt.close(fig)
    with (root / "plot_data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/familiarity-prior-v1"))
    parser.add_argument("--partitions", type=int, nargs="+", default=[2099, 2100, 2101])
    args = parser.parse_args()
    run(args.root, args.partitions)
