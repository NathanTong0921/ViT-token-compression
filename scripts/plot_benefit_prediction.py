"""Plot absolute and within-image predictive value on held images and tasks."""

import argparse
import csv
from pathlib import Path

import numpy as np


SETTINGS = (("pets_b16_49", "Pets · 49 tokens"),
            ("pets_b16_98", "Pets · 98 tokens"),
            ("flowers_b16_49", "Flowers · 49 tokens"),
            ("flowers_b16_98", "Flowers · 98 tokens"))
PANELS = (("primary", "Conditioned vs shared", r"$C_s-U_s$"),
          ("mix", "Task vs shared mixtures", r"$CU_{s,t}-UU_{s,t}$"),
          ("increment", "Extra gain from mixing",
           r"$(CU_{s,t}-C_s)-(UU_{s,t}-U_s)$"))
OUTCOMES = (("geom", "Geometric margin", "#2166ac", "o", .10),
            ("raw", "Raw score gap", "#d97925", "s", -.10))
METRICS = (("selection_r2", "Prediction $R^2$"),
           ("selection_increment_within_r2", "Within-image incremental $R^2$"))


def plot(source, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source, output = Path(source), Path(output)
    with (source / "predictive_summary.csv").open(newline="") as handle:
        rows = {(row["setting"], row["panel"], row["outcome"]): row
                for row in csv.DictReader(handle) if row["mode"] == "double_holdout"}

    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.4), sharex="row", sharey=True)
    fig.subplots_adjust(left=.145, right=.985, bottom=.235, top=.755,
                        wspace=.18, hspace=.38)
    y = np.arange(len(SETTINGS))[::-1]

    for ri, (metric, xlabel) in enumerate(METRICS):
        for ci, (panel, title, formula) in enumerate(PANELS):
            ax = axes[ri, ci]
            ax.axvline(0, color="#555555", lw=1, zorder=1)
            ax.grid(axis="x", color="#e9e9e9")
            ax.set_axisbelow(True)
            for outcome, label, color, marker, offset in OUTCOMES:
                selected = [rows[(setting, panel, outcome)] for setting, _ in SETTINGS]
                means = [float(row[metric]) for row in selected]
                low = [float(row[metric + "_ci_low"]) for row in selected]
                high = [float(row[metric + "_ci_high"]) for row in selected]
                ax.hlines(y + offset, low, high, color=color, lw=1.5, zorder=2)
                ax.scatter(means, y + offset, color=color, marker=marker,
                           s=34, label=label, zorder=3)
            ax.set_yticks(y, [label for _, label in SETTINGS])
            ax.set_ylim(-.45, len(SETTINGS) - .55)
            ax.set_xlabel(xlabel, labelpad=8)
            ax.tick_params(axis="y", length=0)
            ax.tick_params(axis="x", labelsize=9)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            if ri == 0:
                ax.set_title(title + "\n" + formula, fontsize=11, pad=11)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.565, .89),
               ncol=2, frameon=False, columnspacing=2.0)
    fig.suptitle("Predicting gains beyond shared token selection", y=.98,
                 fontsize=15, fontweight="bold")
    fig.text(.565, .922, "Held-out images and class pairs · frozen CLIP ViT-B/16",
             ha="center", fontsize=11, color="#444444")
    fig.text(.145, .13,
             "Points pool three seeds (left) or six ordered seed pairs (middle/right); bars are ordinary 95% image-bootstrap CIs.\n"
             "Top: prediction error relative to the training-mean predictor. Negative values indicate worse prediction.\n"
             "Bottom: added value of selection features beyond text distance and shared boundary gap, after centering within each image.\n"
             "A positive increment does not imply positive absolute prediction R².",
             fontsize=9, linespacing=1.65, va="center")
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"benefit_prediction.{extension}", dpi=220,
                    facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("results/conditioning-benefit-v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/conditioning-benefit-v1"))
    args = parser.parse_args()
    plot(args.source, args.output)
