"""Plot learned task/image correction contrasts from saved development summaries."""

import argparse
import csv
import json
from pathlib import Path


SETTINGS = (("pets_b16_49", "Pets · 49 tokens"),
            ("flowers_b16_49", "Flowers · 49 tokens"),
            ("flowers_b16_98", "Flowers · 98 tokens"))
COMPARISONS = (("task_residual_minus_image_residual", "Task − image correction", "#2166ac"),
               ("task_residual_minus_shared", "Task correction − U", "#d97925"),
               ("image_residual_minus_shared", "Image correction − U", "#75945c"))


def plot(source, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = {setting: json.loads((source / setting / "summary.json").read_text()) for setting, _ in SETTINGS}
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 5.8), sharey=True)
    fig.subplots_adjust(left=.20, right=.98, bottom=.25, top=.77, wspace=.28)
    positions = [0, 1.6, 3.2]
    rows = []
    for setting_index, (setting, label) in enumerate(SETTINGS):
        for ci, (contrast, display, color) in enumerate(COMPARISONS):
            values = data[setting]["classification"]["contrasts"][contrast]
            y = positions[setting_index] + (ci - 1) * .3
            row = {"setting": setting, "contrast": contrast}
            for ax, metric, scale in zip(axes, ("accuracy", "margin"), (100., 1000.)):
                estimate = values[metric]
                mean, low, high = [estimate["mean"] * scale, *[v * scale for v in estimate["ci95"]]]
                ax.plot([low, high], [y, y], color=color, linewidth=3, zorder=3)
                if "bonferroni_family_ci95" in estimate:
                    a, b = [v * scale for v in estimate["bonferroni_family_ci95"]]
                    ax.plot([a, b], [y, y], color=color, alpha=.45, linewidth=1.2,
                            marker="|", markersize=5, zorder=2)
                    row[f"{metric}_family_ci95_low"], row[f"{metric}_family_ci95_high"] = a, b
                else:
                    row[f"{metric}_family_ci95_low"], row[f"{metric}_family_ci95_high"] = "", ""
                ax.scatter(mean, y, color=color, s=28, zorder=4)
                row.update({f"{metric}_difference": mean, f"{metric}_ci95_low": low, f"{metric}_ci95_high": high})
            for seed in values["per_seed"]:
                row[f"accuracy_seed_{seed['seed']}_pp"] = seed["accuracy"] * 100
                row[f"margin_seed_{seed['seed']}_x1000"] = seed["margin"] * 1000
            rows.append(row)
    for ax in axes:
        ax.axvline(0, color="#888888", linestyle="--", linewidth=.9)
        ax.set_yticks(positions, [label for _, label in SETTINGS])
        ax.set_ylim(3.85, -.65)
        ax.grid(axis="x", color="#ececec", linewidth=.6)
        ax.set_axisbelow(True)
        for y in (.8, 2.4):
            ax.axhline(y, color="#ececec", linewidth=.7)
    axes[0].set_xlabel("Accuracy difference (percentage points)")
    axes[1].set_xlabel("Geometric margin difference (×10⁻³)")
    axes[0].set_title("Classification accuracy", loc="left", fontsize=11)
    axes[1].set_title("Classification margin", loc="left", fontsize=11)
    fig.suptitle("Learning a correction to a strong shared selector", fontsize=15, fontweight="bold", y=.97)
    fig.legend(handles=[Line2D([0], [0], marker="o", color=color, label=display)
                        for _, display, color in COMPARISONS],
               loc="upper center", bbox_to_anchor=(.59, .90), ncol=3, frameon=False, fontsize=9.2)
    fig.text(.20, .11,
             "Same frozen U; zero-output initialization; paired data, optimizer, and fixed training steps.\n"
             "Task / image corrections use 9,801 / 9,850 learned parameters; totals are 357,200 / 357,249.\n"
             "Points average all three training seeds. Thick intervals: paired image-bootstrap 95%.\n"
             "Thin blue intervals: adjustment for the three primary task-minus-image comparisons, per metric.",
             fontsize=8.8, color="#444444", va="center", ha="left", linespacing=1.5)
    stem = output / "shared_residual"
    fig.savefig(stem.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    with (output / "method-contrasts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return stem


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/shared-residual-v1"))
    parser.add_argument("--output", type=Path, default=Path("results/shared-residual-v1"))
    args = parser.parse_args()
    print(plot(args.source, args.output))
