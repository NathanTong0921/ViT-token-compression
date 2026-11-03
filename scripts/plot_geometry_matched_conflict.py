#!/usr/bin/env python3
"""Plot primary and secondary effects from the frozen development-pool matches."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/geometry-matched-conflict-devpool-v1"))
    args = parser.parse_args()
    summary = json.loads((args.input / "summary.json").read_text())["datasets"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    fig.subplots_adjust(left=.08, right=.98, top=.85, bottom=.27, wspace=.30)
    conditions = [(3, 16), (6, 16), (3, 49), (6, 49)]
    for dataset, color, marker, offset in (("pets", "#0072B2", "o", -.10),
                                            ("flowers", "#D55E00", "s", .10)):
        rows = [summary[dataset][f"native_both_correct_layer{layer}_k{keep}"] for layer, keep in conditions]
        for axis, metric, scale in ((axes[0], "conflict", 1), (axes[1], "mean_margin_gap", 1000)):
            points = np.asarray([r["outcomes"][metric]["different_minus_same"] for r in rows]) * scale
            intervals = np.asarray([r["outcomes"][metric]["paired_image_bootstrap_ci95"] for r in rows]) * scale
            axis.errorbar(np.arange(4) + offset, points,
                          yerr=np.stack((points - intervals[:, 0], intervals[:, 1] - points)),
                          fmt=marker, color=color, markersize=5, capsize=3, linewidth=1.3,
                          label=f'{dataset.capitalize()} (n={rows[0]["composites"]})')
    for axis in axes:
        axis.axhline(0, color=".55", linestyle="--", linewidth=1)
        axis.set_xticks(range(4), ["16 / block 3", "16 / block 6", "49 / block 3", "49 / block 6"], fontsize=9)
        axis.set_xlabel("Retained patch budget / pruning location")
        axis.grid(axis="y", alpha=.2)
        axis.set_xlim(-.4, 3.4)
    axes[0].set(title="a  Primary: extra joint-selection conflict",
                ylabel="Extra conflict (percentage points)", ylim=(-.35, 6.4))
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    axes[1].set(title="b  Secondary: extra margin tradeoff",
                ylabel="Extra geometric-margin tradeoff (×10⁻³)", ylim=(-1.2, 16))
    fig.text(.08, .10,
             "Effects compare different-object minus same-object pairs, matched within each image using native scores only.\n"
             "Shown: native-correct pairs from the expanded train+val pool; fixed 35-mask library; paired-composite 95% bootstrap.\n"
             "Zero-conflict groups have zero empirical bootstrap width; they do not establish equivalence.",
             fontsize=8.5, va="center", linespacing=1.4)
    for suffix in ("png", "pdf"):
        fig.savefig(args.input / f"geometry_matched_conflict.{suffix}", dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
