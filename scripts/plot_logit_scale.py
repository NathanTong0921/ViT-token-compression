"""Plot unchanged-ranking scale diagnostics against the same hybrid-mask target."""

import argparse
import csv
from pathlib import Path


SETTINGS = (("pets_b16_49", "Pets · 49 tokens"),
            ("flowers_b16_49", "Flowers · 49 tokens"),
            ("flowers_b16_98", "Flowers · 98 tokens"))
ARMS = (("task", "Task student"), ("image", "Image student"))


def plot(source, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source, output = Path(source), Path(output)
    with (source / "summary.csv").open(newline="") as handle:
        rows = {(row["setting"], row["arm"]): row for row in csv.DictReader(handle)}
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.1), sharex=True, sharey="col")
    fig.subplots_adjust(left=.09, right=.98, top=.78, bottom=.20,
                        wspace=.27, hspace=.24)
    for column, (setting, title) in enumerate(SETTINGS):
        for row_index, (arm, label) in enumerate(ARMS):
            ax, row = axes[row_index, column], rows[(setting, arm)]
            for prefix, name, color, marker in (
                ("original", "Original-target training", "#2166ac", "o"),
                ("hybrid", "Hybrid-target training", "#d97925", "s"),
            ):
                values = [float(row[f"{prefix}_{stage}"])
                          for stage in ("raw", "global", "row")]
                ax.plot(range(3), values, color=color, marker=marker, label=name,
                        lw=2, markersize=5.5)
            ax.set_xlim(-.13, 2.13)
            ax.set_ylim(bottom=0)
            ax.set_xticks(range(3), ["Raw", "Global scale", "Per-row\n+ global"])
            ax.grid(axis="y", color="#e9e9e9")
            ax.set_axisbelow(True)
            if row_index == 0:
                ax.set_title(title, loc="left", fontsize=11, pad=10)
            if column == 0:
                ax.set_ylabel(f"{label}\nKL to hybrid target", labelpad=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.54, .90),
               ncol=2, frameon=False, fontsize=10)
    fig.suptitle("How much does logit scale account for the target-fit gap?",
                 fontsize=15, fontweight="bold", y=.98)
    fig.text(.09, .085,
             "Each model keeps its own top-k selections unchanged across the three points.\n"
             "Positive scales; five-fold calibration by image; averages of three fixed seeds.",
             fontsize=9, linespacing=1.6, va="center")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"logit_scale.{extension}", dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("results/hybrid-logit-scale-v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/hybrid-logit-scale-v1"))
    args = parser.parse_args()
    plot(args.source, args.output)
