"""Plot candidate-count effects from the paired development analysis."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


COLORS = {"full_model": "#56616B", "conditioned": "#246B99", "image_only_wide": "#C17022"}
DATASET_LABELS = {"pets": "Pets", "flowers": "Flowers102", "cub": "CUB-200-2011"}


def run(summary, output):
    output.mkdir(parents=True, exist_ok=True)
    rows = sorted(summary["settings"].values(), key=lambda x: (x["dataset"] != "pets", x["dataset"], x["way"]))
    labels = [f"{DATASET_LABELS.get(r['dataset'], r['dataset'])} · {r['way']} classes" for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, max(4.7, len(rows) * .6 + 2.1)), sharey=True)
    for i, row in enumerate(rows):
        table = {m["method"]: m for m in row["table"]}
        for method, marker, shift in (("full_model", "D", -.17), ("image_only_wide", "s", 0), ("conditioned", "o", .17)):
            axes[0].scatter(100 * table[method]["accuracy"]["mean"], i + shift, s=35,
                            c=COLORS[method], marker=marker, zorder=3)
        for axis, metric, scale in zip(axes[1:], ("accuracy", "geometric_margin"), (100, 1000)):
            stats = row["conditioned_minus_image_only_wide"][metric]
            family = np.array(stats.get("bonferroni_family_ci95", stats["ci95"])) * scale
            ordinary = np.array(stats["ci95"]) * scale
            axis.hlines(i, *family, color=COLORS["conditioned"], linewidth=1, alpha=.65)
            axis.hlines(i, *ordinary, color=COLORS["conditioned"], linewidth=2.5)
            axis.scatter(stats["mean"] * scale, i, color=COLORS["conditioned"], s=36, zorder=3)
    for axis in axes:
        for i in range(len(rows)):
            if i % 2 == 0:
                axis.axhspan(i - .45, i + .45, color="#F0F4F6", zorder=0)
        axis.set(yticks=np.arange(len(rows)), ylim=(len(rows) - .5, -.5))
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#AAB5BD")
        axis.tick_params(axis="y", length=0)
        axis.grid(axis="x", color="#D8DFE3", linewidth=.5, alpha=.8)
        axis.set_axisbelow(True)
        axis.margins(x=.18)
    axes[0].set_yticklabels(labels, fontsize=10)
    axes[0].set(title="(a) Classification", xlabel="Accuracy (%)")
    axes[1].set(title="(b) Conditioning gain", xlabel="C − U accuracy (percentage points)")
    axes[2].set(title="(c) Decision margin", xlabel="C − U geometric margin (×10⁻³)")
    for axis in axes[1:]:
        axis.axvline(0, color="#747F87", linewidth=1, linestyle=(0, (3, 3)))
        axis.tick_params(axis="y", labelleft=False)
    fig.suptitle("Does candidate-set size change conditioning gains?", fontsize=15, y=.98)
    fig.text(.57, .90, "Development evaluation · Frozen CLIP B/16 · Block 6 · 98 retained tokens",
             ha="center", fontsize=10, color="#4F5C65")
    handles = [Line2D([], [], linestyle="", marker=m, color=COLORS[key], label=label, markersize=6)
               for key, m, label in (("full_model", "D", "Full CLIP"),
                                     ("image_only_wide", "s", "U: strong shared"),
                                     ("conditioned", "o", "C: conditioned"))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.56, .85), ncol=3, frameon=False, fontsize=10)
    seed_counts = sorted({len(row["seeds"]) for row in rows})
    count = str(seed_counts[0]) if len(seed_counts) == 1 else "/".join(map(str, seed_counts))
    fig.text(.02, .025, f"Means over {count} paired training seeds. Thick bars: image-bootstrap 95% intervals; thin bars: family-adjusted intervals.\n"
             "Each classification task weights its candidate classes equally. Positive C − U values favor conditioning.",
             fontsize=9, color="#4F5C65", linespacing=1.5)
    fig.subplots_adjust(left=.17, right=.98, top=.70, bottom=.22, wspace=.25)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"candidate_count_effect.{suffix}", dpi=220, bbox_inches="tight", facecolor="white", pad_inches=.15)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.analysis.read_text()), args.output)
