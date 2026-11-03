"""Plot the frozen core-protection intervention from saved analysis summaries."""

import argparse
import csv
import json
from pathlib import Path


PANELS = (("pets_b16_49", "Oxford Pets · 49 tokens"),
          ("pets_b16_98", "Oxford Pets · 98 tokens"),
          ("flowers_b16_49", "Flowers102 · 49 tokens"),
          ("flowers_b16_98", "Flowers102 · 98 tokens"))
FRACTIONS = (0., .5, .75, 1.)
Y_SPAN = .0045


def plot(source, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.4))
    fig.subplots_adjust(left=.105, right=.98, bottom=.24, top=.81, hspace=.58, wspace=.28)
    colors = {"top": "#2166ac", "random": "#d97925"}
    rows = []
    for ax, (setting, title) in zip(axes.flat, PANELS):
        summary_path = source / setting / "summary.json"
        summary = json.loads(summary_path.read_text())
        table = {r["method"]: r for r in summary["classification"]["table"]}
        plotted_values = []
        for kind, style, marker in (("top", "-", "o"), ("random", "--", "D")):
            methods = ("conditioned", f"{kind}_50", f"{kind}_75", "wide")
            values = [table[m]["margin"]["mean"] for m in methods]
            plotted_values.extend(values)
            ax.plot(FRACTIONS, values, color=colors[kind], linestyle=style,
                    marker=marker, linewidth=1.8, markersize=5, zorder=3)
            for fraction, method, value in zip(FRACTIONS, methods, values):
                estimate = table[method]["margin"]
                rows.append({"setting": setting, "curve": kind, "nominal_protected_fraction": fraction,
                             "protected_tokens": summary["protected_counts"][str(round(fraction * 100))],
                             "method": method, "margin_mean": value,
                             "ordinary_ci95_low": estimate["ci95"][0],
                             "ordinary_ci95_high": estimate["ci95"][1],
                             "source": str(summary_path)})
        c, u = table["conditioned"]["margin"]["mean"], table["wide"]["margin"]["mean"]
        ax.scatter([0], [c], marker="o", s=34, color="#242424", zorder=4)
        ax.scatter([1], [u], marker="s", s=34, color="#242424", zorder=4)
        middle = (min(plotted_values) + max(plotted_values)) / 2
        ax.set_ylim(middle - Y_SPAN / 2, middle + Y_SPAN / 2)
        ax.set_xlim(-.04, 1.04)
        ax.set_xticks(FRACTIONS, ["0\nC", "0.50", "0.75", "1\nU"])
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        ax.grid(axis="y", color="#e7e7e7", linewidth=.65)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=10)
        if ax in axes[1]:
            ax.set_xlabel("Protected shared-core fraction", labelpad=1)
        ax.set_ylabel("Mean geometric margin")
    fig.suptitle("Shared core + task-specific fill", fontsize=15, fontweight="bold", y=.97)
    fig.text(.5, .921, "Frozen CLIP ViT-B/16 · block 6 · fixed token count within each panel",
             ha="center", fontsize=10, color="#444444")
    legend = [Line2D([0], [0], color=colors["top"], marker="o", linewidth=1.8, label="Highest-ranked shared core"),
              Line2D([0], [0], color=colors["random"], marker="D", linestyle="--", linewidth=1.8,
                     label="Random subset of shared top-k")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .897), ncol=2, frameon=False, fontsize=10)
    fig.text(.105, .075,
             "C: original conditioned selector; U: wide image-only selector. The remaining slots follow C's ranking.\n"
             "Means over 3 paired training seeds; random cores also average 3 mask seeds. Fractions round down to tokens.\n"
             "All y-axis windows span 0.0045; absolute levels differ. Curves show means, not confidence intervals.\n"
             "Pets98 and Flowers49 complete the exploratory 2×2 extension of the initial Pets49 / Flowers98 pilot.",
             ha="left", va="center", fontsize=8.7, linespacing=1.5, color="#444444")
    stem = output / "core_protection"
    fig.savefig(stem.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return stem


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/core-protection-v1"))
    parser.add_argument("--output", type=Path, default=Path("results/core-protection-v1"))
    args = parser.parse_args()
    print(plot(args.source, args.output))
