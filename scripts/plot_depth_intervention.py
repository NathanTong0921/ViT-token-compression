"""Plot depth-dependent conditioning gains alongside the shared/full baseline."""

import argparse
import json
from pathlib import Path

import numpy as np


DATASETS = (("pets", "Oxford Pets"), ("flowers", "Flowers102"))
DEPTHS = (3, 6, 9)
CONTRASTS = (("conditioned_minus_shared", "Conditioning gain: C − U", "#2166ac"),
             ("cu_minus_uu", "Mixture gain: CU − UU", "#d97925"))
METRICS = {
    "geom_margin": {"scale": 1., "gain_label": "Geometric margin gain",
                    "absolute_label": "Geometric margin", "basename": "depth_geometric_margin"},
    "accuracy": {"scale": 100., "gain_label": "Accuracy gain (percentage points)",
                 "absolute_label": "Accuracy (%)", "basename": "depth_accuracy"},
}


def values_by_depth(dataset, metric, contrast=None, method=None):
    """Keep all three depths and use the ordinary paired-image intervals."""
    values = []
    for depth in DEPTHS:
        point = dataset["by_depth"][str(depth)]
        if contrast is not None:
            stats = point["contrasts"][contrast][metric]
        else:
            stats = next(row[metric] for row in point["table"] if row["method"] == method)
        values.append([stats["mean"], *stats["ci95"]])
    return np.asarray(values) * METRICS[metric]["scale"]


def draw_intervals(ax, values, color, label=None):
    mean, low, high = values.T
    ax.plot(DEPTHS, mean, marker="o", markersize=5.5, lw=1.7,
            color=color, label=label, zorder=3)
    ax.vlines(DEPTHS, low, high, color=color, lw=1.4, zorder=2)
    for depth, lower, upper in zip(DEPTHS, low, high):
        ax.hlines([lower, upper], depth - .12, depth + .12, color=color, lw=1.1)


def plot_metric(summary, output, metric):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    details = METRICS[metric]
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 8.), sharex=True)
    fig.subplots_adjust(left=.095, right=.98, top=.81, bottom=.25,
                        wspace=.31, hspace=.32)
    gain_bounds, absolute_bounds = [0.], []
    for ri, (dataset_name, label) in enumerate(DATASETS):
        dataset = summary["datasets"][dataset_name]
        for ci, (contrast, title, color) in enumerate(CONTRASTS):
            ax = axes[ri, ci]
            values = values_by_depth(dataset, metric, contrast=contrast)
            draw_intervals(ax, values, color)
            gain_bounds.extend(values[:, 1:].ravel())
            ax.axhline(0, color="#666666", lw=1, zorder=1)
            if ri == 0:
                ax.set_title(title, fontsize=11, pad=12)
        axes[ri, 0].set_ylabel(label + "\n" + details["gain_label"], labelpad=11)

        ax = axes[ri, 2]
        shared = values_by_depth(dataset, metric, method="shared")
        full_stats = dataset["full_reference"][metric]
        full_mean, full_low, full_high = np.asarray([full_stats["mean"], *full_stats["ci95"]]) * details["scale"]
        ax.axhspan(full_low, full_high, color="#777777", alpha=.13, zorder=0)
        ax.axhline(full_mean, color="#777777", lw=1.6, ls="--", label="Full model (no pruning)")
        draw_intervals(ax, shared, "#252525", label="Shared selector U")
        absolute_bounds.extend([*shared[:, 1:].ravel(), full_low, full_high])
        ax.set_ylabel(details["absolute_label"], labelpad=8)
        if ri == 0:
            ax.set_title("Shared baseline and full model", fontsize=11, pad=12)
        for ax in axes[ri]:
            ax.set_xticks(DEPTHS)
            ax.set_xlim(2.65, 9.35)
            ax.grid(axis="y", color="#e9e9e9")
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", labelsize=9)

    # Keep the same gain scale across both contrasts and both datasets; the
    # separate absolute-performance column uses its own common scale.
    for bounds, columns in ((gain_bounds, (0, 1)), (absolute_bounds, (2,))):
        low, high = min(bounds), max(bounds)
        padding = .09 * (high - low or .01)
        for ri in range(2):
            for ci in columns:
                axes[ri, ci].set_ylim(low - padding, high + padding)

    handles, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.54, .90),
               ncol=2, frameon=False)
    fig.suptitle("How pruning depth changes the value of conditioning", y=.98,
                 fontsize=15, fontweight="bold")
    fig.text(.54, .934, "Frozen CLIP ViT-B/16 · 49 retained patch tokens · fixed training protocol",
             ha="center", fontsize=11, color="#444444")
    fig.supxlabel("Prune after Transformer block", y=.185, fontsize=11)
    fig.text(.095, .105,
             "C − U pools three paired seeds; CU − UU pools six ordered seed pairs. Bars show ordinary 95% image-bootstrap intervals.\n"
             "Each depth uses its own fixed-step selectors and teacher; images, tasks, teacher probe masks and training settings are held fixed.\n"
             "The right column shows the shared selector's absolute performance against the unchanged full model.\n"
             "Depth effects are quantified separately by paired changes relative to block 6, with family-adjusted intervals.",
             fontsize=9, linespacing=1.65, va="center")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{details['basename']}.{extension}", dpi=220, facecolor="white")
    plt.close(fig)


def plot(source, output):
    source, output = Path(source), Path(output)
    summary = json.loads((source / "summary.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        plot_metric(summary, output, metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/depth-intervention-v1"))
    parser.add_argument("--output", type=Path, default=Path("results/depth-intervention-v1"))
    args = parser.parse_args()
    plot(args.source, args.output)
