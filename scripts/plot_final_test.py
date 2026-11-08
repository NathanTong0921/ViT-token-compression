"""Plot paired contrasts and exchange/geometry results from the reserved test."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


BLUE, ORANGE, GREEN = "#246B99", "#C17022", "#237C66"


def setting_order(summary):
    datasets = {"pets": 0, "flowers": 1}
    models = {"ViT-B/16": 0, "ViT-L/14": 1}
    return sorted(summary["settings"].values(), key=lambda result: (
        datasets[result["setting"]["dataset"]], models[result["setting"]["model"]],
        result["setting"]["keep_patch_tokens"]))


def setting_label(result):
    setting = result["setting"]
    dataset = {"pets": "Pets", "flowers": "Flowers102"}[setting["dataset"]]
    model = setting["model"].replace("ViT-", "")
    return f"{dataset}  ·  {model}  ·  k={setting['keep_patch_tokens']}"


def point_interval(ax, row, estimate, scale, color, marker="o"):
    low, high = scale * np.asarray(estimate["ci95"])
    mean = scale * estimate["mean"]
    ax.hlines(row, low, high, color=color, linewidth=1.8, zorder=3)
    ax.vlines([low, high], row - .055, row + .055, color=color, linewidth=1.3, zorder=3)
    ax.scatter([mean], [row], color=color, marker=marker, s=38,
               edgecolor="white", linewidth=.6, zorder=4)


def style_rows(axes, results):
    n = len(results)
    for ax in axes:
        ax.axvline(0, color="#737F87", linewidth=1, linestyle=(0, (3, 3)), zorder=2)
        for row in range(n):
            if row % 2 == 0:
                ax.axhspan(row - .45, row + .45, color="#F2F5F7", zorder=0)
            if row and results[row]["setting"]["dataset"] != results[row - 1]["setting"]["dataset"]:
                ax.axhline(row - .5, color="#BBC5CB", linewidth=.8)
        ax.set(yticks=np.arange(n), ylim=(n - .55, -.55))
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#AAB5BD")
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=3, color="#AAB5BD")
        ax.grid(axis="x", color="#D8DFE3", linewidth=.55, alpha=.6)
        ax.set_axisbelow(True)
        ax.margins(x=.14)
    axes[0].set_yticklabels([setting_label(result) for result in results], fontsize=10)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)


def legend_handle(label, color, marker):
    return Line2D([0], [0], color=color, marker=marker, linewidth=1.8,
                  markersize=6, label=label)


def save(fig, output, name):
    for suffix in ("png", "svg"):
        fig.savefig(output / f"{name}.{suffix}", dpi=220, facecolor="white",
                    bbox_inches="tight", pad_inches=.16)
    plt.close(fig)


def plot_conditioning(results, output):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, max(6.1, .56 * len(results) + 2.1)),
                             sharey=True)
    comparisons = (("conditioned_minus_image_only", BLUE, "o", -.13),
                   ("conditioned_minus_image_only_wide", ORANGE, "s", .13))
    for row, result in enumerate(results):
        for name, color, marker, shift in comparisons:
            if name in result["contrasts"]:
                for ax, metric, scale in zip(axes, ("accuracy", "margin"), (100, 1000)):
                    point_interval(ax, row + shift, result["contrasts"][name][metric],
                                   scale, color, marker)
    style_rows(axes, results)
    axes[0].set(title="(a) Accuracy", xlabel="C − comparator (percentage points)")
    axes[1].set(title="(b) Geometric classification margin", xlabel="C − comparator (×10⁻³)")
    fig.suptitle("Does task conditioning improve selection on the reserved test?", fontsize=15.5, y=.985)
    fig.text(.5, .935, "Fixed model weights  ·  Held-out images and class combinations",
             ha="center", fontsize=10.5, color="#4F5C65")
    fig.legend(handles=[legend_handle("C − I: original image-only comparator", BLUE, "o"),
                        legend_handle("C − U: wider image-only comparator", ORANGE, "s")],
               loc="upper center", bbox_to_anchor=(.61, .89), ncol=1,
               frameon=False, fontsize=10)
    missing = [setting_label(r) for r in results
               if "conditioned_minus_image_only_wide" not in r["contrasts"]]
    note = "Means over three paired seeds; bars show image-bootstrap 95% confidence intervals. Positive values favor C."
    if missing:
        note += "\nU was not evaluated for " + "; ".join(missing) + "."
    fig.text(.02, .025, note, fontsize=9, color="#4F5C65", va="bottom", linespacing=1.5)
    fig.subplots_adjust(left=.24, right=.985, bottom=.17, top=.76, wspace=.16)
    save(fig, output, "conditioning_contrasts")


def plot_exchange_geometry(results, output):
    fig, axes = plt.subplots(1, 3, figsize=(15, max(6.1, .56 * len(results) + 2.1)), sharey=True)
    for row, result in enumerate(results):
        exchange = result["exchange_own_minus_swapped"]
        point_interval(axes[0], row, exchange["accuracy"], 100, GREEN)
        point_interval(axes[1], row, exchange["margin"], 1000, GREEN)
        for name, color, marker, shift in (("difficulty_adjusted", BLUE, "o", -.13),
                                           ("within_image", ORANGE, "s", .13)):
            estimate = result["h2"][name]
            if estimate["angle_estimable_beyond_controls"]:
                point_interval(axes[2], row + shift,
                               {"mean": estimate["angle_slope"], "ci95": estimate["ci95"]},
                               1000, color, marker)
            else:
                axes[2].text(.98, row + shift, "not estimable", fontsize=8, color=color,
                             ha="right", va="center", transform=axes[2].get_yaxis_transform())
    style_rows(axes, results)
    axes[0].set(title="(a) Task exchange: accuracy", xlabel="Own − exchanged task (percentage points)")
    axes[1].set(title="(b) Task exchange: margin", xlabel="Own − exchanged task (×10⁻³)")
    axes[2].set(title="(c) Angle–exchange-loss association", xlabel="Margin slope per 10° (×10⁻³)")
    fig.suptitle("Task exchange and geometric association on the reserved test", fontsize=15.5, y=.985)
    fig.text(.5, .935, "Fixed model weights  ·  Held-out images and class combinations",
             ha="center", fontsize=10.5, color="#4F5C65")
    axes[2].legend(handles=[legend_handle("Adjusted for full-model margins", BLUE, "o"),
                           legend_handle("Also adjusted within image", ORANGE, "s")],
                   loc="lower center", bbox_to_anchor=(.5, 1.12), frameon=False, fontsize=9)
    fig.text(.02, .025,
             "Three paired seeds averaged before analysis; bars show image-bootstrap 95% confidence intervals.\n"
             "Exchange effects compare task-specific selections on the same image. H2 slopes describe associations with task-direction angles.",
             fontsize=9, color="#4F5C65", va="bottom", linespacing=1.5)
    fig.subplots_adjust(left=.195, right=.985, bottom=.17, top=.76, wspace=.17)
    save(fig, output, "exchange_and_geometry")


def plot(analysis_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((analysis_dir / "analysis.json").read_text())
    results = setting_order(summary)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11.5, "axes.labelsize": 10,
                         "svg.fonttype": "none"})
    plot_conditioning(results, output)
    plot_exchange_geometry(results, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.analysis_dir, args.output)
