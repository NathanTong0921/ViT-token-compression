"""Plot saved alignment slopes against their assignment-permutation references.

The gray bars are central reference ranges, not confidence intervals. This script
only presents statistics saved by analyze_geometry_permutation.py.
"""

import argparse
import csv
import json
from pathlib import Path


PRIMARY = "difficulty_adjusted_slope"
MODEL_ORDER = {"ViT-B/16": 0, "ViT-L/14": 1}
DATASET_ORDER = {"pets": 0, "flowers": 1}
DATASET_NAMES = {"pets": "Pets", "flowers": "Flowers102"}


def plotted_rows(summary):
    assert summary["status"] == "complete", "Plot the completed seven-setting analysis"
    rows = []
    for key, value in summary["settings"].items():
        setting = value["setting"]
        statistic = value["statistics"][PRIMARY]
        lower, median, upper = statistic["null_quantiles_025_50_975"]
        rows.append({
            "setting_id": key, "model": setting["model"], "dataset": setting["dataset"],
            "retained_patch_tokens": setting["keep_patch_tokens"],
            "statistic": PRIMARY, "unit": "alignment slope per 10 degrees",
            "observed": statistic["observed"], "reference_mean": statistic["null_mean"],
            "reference_q025": lower, "reference_median": median, "reference_q975": upper,
            "upper_tail_p": statistic["upper_tail_p"],
            "bonferroni_upper_tail_p": statistic["bonferroni_upper_tail_p"],
            "permutation_draws": summary["config"]["draws"],
            "primary_family_size": summary["config"]["family_size"],
        })
    return sorted(rows, key=lambda row: (MODEL_ORDER[row["model"]],
                                         DATASET_ORDER[row["dataset"]], row["retained_patch_tokens"]))


def run(source, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    summary = json.loads(Path(source).read_text())
    rows = plotted_rows(summary)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "primary_alignment_reference"
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8.7, 5.7))
    fig.subplots_adjust(left=.28, right=.79, bottom=.28, top=.89)
    positions, headers = [], []
    previous, current = None, 0.
    for row in rows:
        if row["model"] != previous:
            if previous is not None:
                current += 1.1
            headers.append((row["model"], current - .7))
            previous = row["model"]
        positions.append(current)
        current += 1.
    for row, y in zip(rows, positions):
        ax.plot([row["reference_q025"], row["reference_q975"]], [y, y],
                color=".66", linewidth=5, solid_capstyle="butt", zorder=2)
        ax.plot([row["reference_q025"], row["reference_q975"]], [y, y],
                linestyle="none", marker="|", markersize=8, color=".40", zorder=3)
        ax.plot(row["observed"], y, "o", color="#16699c", markersize=6.5, zorder=4)
        p = row["bonferroni_upper_tail_p"]
        ax.text(1.08, y, f"{p:.4f}", transform=ax.get_yaxis_transform(),
                va="center", ha="center", fontsize=10)
    ax.text(1.08, -.7, "Adjusted p", transform=ax.get_yaxis_transform(),
            va="center", ha="center", fontsize=10, fontweight="bold")
    for model, y in headers:
        ax.text(-.48, y, model, transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=11, fontweight="bold")
    if len(headers) > 1:
        for _, y in headers[1:]:
            ax.axhline(y - .55, color=".88", linewidth=.8, zorder=0)
    ax.axvline(0, color=".35", linestyle="--", linewidth=.9, zorder=1)
    ax.set_yticks(positions, [f"{DATASET_NAMES[row['dataset']]} · {row['retained_patch_tokens']} tokens"
                             for row in rows])
    ax.set_ylim(current - .35, -1.15)
    ax.margins(x=.10)
    ax.tick_params(axis="y", length=0, pad=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlabel("Difficulty-adjusted alignment slope per 10°", labelpad=9)
    ax.grid(axis="x", color=".92", linewidth=.7, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(".7")
    fig.suptitle("Learned task assignment and randomized references", y=.97, fontsize=13)
    handles = [Line2D([0], [0], marker="o", color="#16699c", linestyle="none",
                      markersize=6.5, label="Observed task assignment"),
               Line2D([0], [0], color=".66", linewidth=5,
                      label="Central 95% permutation reference range")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.51, .125),
               ncol=2, frameon=False, fontsize=9, columnspacing=1.6)
    draws = summary["config"]["draws"]
    family = summary["config"]["family_size"]
    fig.text(.5, .075, f"{draws:,} within-image assignment permutations; gray ranges are not confidence intervals.",
             ha="center", fontsize=9, color=".30")
    fig.text(.5, .04, f"One-sided upper-tail p values, Bonferroni-adjusted across {family} settings. Development data.",
             ha="center", fontsize=9, color=".30")
    for extension in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=240, facecolor="white")
    plt.close(fig)
    return [str(stem.with_suffix(f".{extension}")) for extension in ("png", "pdf", "csv")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Saved analysis.json")
    parser.add_argument("--output", type=Path, help="Figure directory; defaults to the JSON directory")
    args = parser.parse_args()
    print(json.dumps({"outputs": run(args.source, args.output or args.source.parent)}))
