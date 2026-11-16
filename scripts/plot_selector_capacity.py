"""Plot absolute performance and paired conditioning gains across capacity."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_selector_capacity import CAPACITIES, PRIMARY


def plot_metric(summaries, metric, output):
    factor = 100 if metric == "accuracy" else 1000
    absolute_label = "Accuracy (%)" if metric == "accuracy" else "Geometric margin (×1,000)"
    gap_label = "C − U accuracy (pp)" if metric == "accuracy" else "C − U margin (×1,000)"
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex="col")
    fig.subplots_adjust(left=.10, right=.97, top=.85, bottom=.17, hspace=.20, wspace=.27)
    fig.suptitle("How does selector capacity change the value of task conditioning?",
                 fontsize=14, fontweight="bold", y=.975)
    fig.text(.5, .925, "Frozen CLIP ViT-B/16 · block 6 · 49 retained patches · three fixed seeds",
             fontsize=10, color="#596573", ha="center")
    for column, dataset in enumerate(("pets", "flowers")):
        result = summaries[dataset]
        pair_x = []
        for arm, name, color, marker in (("C", "Task-conditioned C", "#2166A5", "o"),
                                          ("U", "Image-only U", "#768292", "s")):
            x, values, bounds = [], [], []
            for capacity in CAPACITIES:
                method = f"{capacity}_{arm}"
                x.append(result["method_metadata"][method]["parameter_count"] / 1000)
                value = result["methods"][method][metric]
                values.append(value["mean"] * factor)
                bounds.append(np.asarray(value["ci95"]) * factor)
            x, values, bounds = np.asarray(x), np.asarray(values), np.asarray(bounds)
            axes[0, column].errorbar(x, values, yerr=np.stack((values-bounds[:,0], bounds[:,1]-values)),
                color=color, marker=marker, capsize=3, linewidth=1.6, markersize=5, label=name)
            pair_x.append(x)
        pair_x = np.mean(pair_x, axis=0)
        points, limits = [], []
        for capacity in CAPACITIES:
            value = result["contrasts"][f"{capacity}_C_minus_U"][metric]
            points.append(value["mean"] * factor)
            limits.append(np.asarray(value["ci95"]) * factor)
        points, limits = np.asarray(points), np.asarray(limits)
        axes[1, column].errorbar(pair_x, points,
            yerr=np.stack((points-limits[:,0], limits[:,1]-points)),
            color="#2166A5", marker="o", capsize=3, linewidth=1.6, markersize=5)
        axes[1, column].axhline(0, color="#768292", linestyle="--", linewidth=1)
        interaction = result["contrasts"][PRIMARY][metric]
        interval = np.asarray(interaction["two_primary_adjusted_ci95"]) * factor
        annotation = (f"Large − small change in C−U: {factor*interaction['mean']:+.3f}\n"
                      f"Adjusted interval [{interval[0]:+.3f}, {interval[1]:+.3f}]")
        axes[1, column].text(.02, .98, annotation, transform=axes[1,column].transAxes,
                             va="top", ha="left", fontsize=9,
                             bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
        # Reserve vertical headroom for the paired-interaction annotation.
        low, high = axes[1,column].get_ylim()
        axes[1,column].set_ylim(low, high + .35 * max(high-low, .01))
        axes[0,column].set_title("Oxford-IIIT Pets" if dataset == "pets" else "Flowers102", fontsize=12)
        axes[1,column].set_xlabel("Selector parameters (thousands)")
        axes[1,column].set_xticks(pair_x, [f"{x:.0f}" for x in pair_x])
        for ax in axes[:,column]:
            ax.grid(axis="y", color="#E7EBF0")
            ax.set_axisbelow(True)
    axes[0,0].set_ylabel(absolute_label)
    axes[1,0].set_ylabel(gap_label)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5,.07), ncol=2, frameon=False, fontsize=10)
    fig.text(.10,.025,"Top: image-bootstrap intervals. Bottom: paired differences; large−small intervals adjust over both datasets.\n"
             "Capacity pairs use closely matched parameter counts; exact counts and all seeds accompany the figure.",
             fontsize=8.5,color="#596573")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"capacity_{metric}.{extension}", dpi=240, facecolor="white")
    plt.close(fig)


def plot_training(rows, output):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), sharex="col")
    fig.subplots_adjust(left=.09, right=.97, top=.85, bottom=.15, hspace=.22, wspace=.27)
    fig.suptitle("Teacher fit across selector capacities", fontsize=15, fontweight="bold", y=.97)
    fig.text(.5,.91,"Same KL objective and fixed training updates · three-seed means",ha="center",fontsize=10,color="#596573")
    for column,dataset in enumerate(("pets","flowers")):
        for arm,name,color,marker in (("conditioned","Task-conditioned C","#2166A5","o"),
                                       ("image_only_wide","Image-only U","#768292","s")):
            group = sorted([row for row in rows if row["dataset"] == dataset and row["arm"] == arm],
                           key=lambda row:int(row["parameter_count"]))
            x=np.asarray([float(row["parameter_count"])/1000 for row in group])
            for row_index,metric in enumerate(("final_train_kl","final_val_kl")):
                axes[row_index,column].plot(x,[float(row[metric]) for row in group],
                    marker=marker,color=color,label=name,linewidth=1.6)
        axes[0,column].set_title("Oxford-IIIT Pets" if dataset == "pets" else "Flowers102",fontsize=12)
        axes[1,column].set_xlabel("Selector parameters (thousands)")
        axes[1,column].set_xticks(x,[f"{v:.0f}" for v in x])
        for ax in axes[:,column]:
            ax.grid(axis="y",color="#E7EBF0"); ax.set_axisbelow(True)
    axes[0,0].set_ylabel("Final training KL")
    axes[1,0].set_ylabel("Final development KL")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.025),ncol=2,frameon=False)
    for suffix in ("png","pdf"):
        fig.savefig(output/f"capacity_teacher_fit.{suffix}",dpi=240,facecolor="white")
    plt.close(fig)


def main(args):
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "pdf.fonttype":42,
                         "axes.spines.top":False, "axes.spines.right":False})
    args.output.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.source.read_text())[args.panel]
    for metric in ("accuracy", "geometric_margin"):
        plot_metric(summary, metric, args.output)
    with (args.source.parent / "training_fit_summary.csv").open() as handle:
        plot_training(list(csv.DictReader(handle)), args.output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("results/selector-capacity-v1/summary.json"))
    p.add_argument("--panel", choices=("dev","test"), default="test")
    p.add_argument("--output", type=Path, default=Path("results/selector-capacity-v1/figures"))
    main(p.parse_args())
