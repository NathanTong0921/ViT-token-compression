"""Plot paired shared-ranking contrasts with matched actual token replacements."""

import argparse
import csv
import json
from pathlib import Path


SETTINGS = (("pets_b16_49", "Pets · 49 tokens"), ("pets_b16_98", "Pets · 98 tokens"),
            ("flowers_b16_49", "Flowers · 49 tokens"), ("flowers_b16_98", "Flowers · 98 tokens"))


def read_results(source):
    rows = []
    for setting, label in SETTINGS:
        data = json.loads((source / setting / "summary.json").read_text())
        contrasts = data["classification"]["contrasts"]
        counts = {r["method"]: r for r in data["token_diagnostics"]["classification_replacements"]}
        for fraction in (50, 75):
            main = contrasts[f"top_{fraction}_minus_random_{fraction}"]
            margin, accuracy = main["margin"], main["accuracy"]
            row = {"setting": setting, "label": label, "replacement_fraction_percent": fraction,
                   "mean_actual_replacements": counts[f"top_{fraction}"]["replaced_tokens_vs_conditioned"]["mean"],
                   "margin_difference": margin["mean"],
                   "margin_ci95_low": margin["ci95"][0], "margin_ci95_high": margin["ci95"][1],
                   "margin_family_ci95_low": margin["bonferroni_family_ci95"][0],
                   "margin_family_ci95_high": margin["bonferroni_family_ci95"][1],
                   "accuracy_difference_pp": accuracy["mean"] * 100,
                   "accuracy_ci95_low_pp": accuracy["ci95"][0] * 100,
                   "accuracy_ci95_high_pp": accuracy["ci95"][1] * 100}
            for seed in main["per_seed"]:
                row[f"margin_seed_{seed['seed']}"] = seed["margin"]
            for baseline in ("conditioned", "wide"):
                value = contrasts[f"top_{fraction}_minus_{baseline}"]["margin"]
                row[f"margin_vs_{baseline}"] = value["mean"]
                row[f"margin_vs_{baseline}_ci95_low"] = value["ci95"][0]
                row[f"margin_vs_{baseline}_ci95_high"] = value["ci95"][1]
            rows.append(row)
    return rows


def plot(source, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = read_results(source)
    with (output / "primary-contrasts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.1, 6.4), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1.22]})
    fig.subplots_adjust(left=.22, right=.98, bottom=.23, top=.77, wspace=.24)
    labels, positions = [], []
    for index, row in enumerate(rows):
        y = index + (index // 2) * .42
        positions.append(y)
        labels.append(f"{row['label']} · {row['replacement_fraction_percent']}%")
        ax = axes[0]
        ax.plot([row["margin_family_ci95_low"] * 1000, row["margin_family_ci95_high"] * 1000],
                [y, y], color="#98b4cd", linewidth=1.5, marker="|", markersize=5, zorder=2)
        ax.plot([row["margin_ci95_low"] * 1000, row["margin_ci95_high"] * 1000],
                [y, y], color="#2166ac", linewidth=4, zorder=3)
        ax.scatter(row["margin_difference"] * 1000, y, s=27, color="#112b43", zorder=4)
        for baseline, offset, color, marker in (("conditioned", -.13, "#2166ac", "o"),
                                                ("wide", .13, "#d97925", "s")):
            mean = row[f"margin_vs_{baseline}"] * 1000
            low = row[f"margin_vs_{baseline}_ci95_low"] * 1000
            high = row[f"margin_vs_{baseline}_ci95_high"] * 1000
            axes[1].errorbar(mean, y + offset, xerr=[[mean - low], [high - mean]],
                             fmt=marker, color=color, markersize=4.5, elinewidth=1.3, capsize=2)
    for ax in axes:
        ax.axvline(0, color="#888888", linewidth=.9, linestyle="--", zorder=1)
        ax.set_xlabel("Geometric margin difference (×10⁻³)")
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#eeeeee", linewidth=.6)
        ax.set_yticks(positions, labels)
        for index in (2, 4, 6):
            y = (positions[index - 1] + positions[index]) / 2
            ax.axhline(y, color="#eeeeee", linewidth=.7)
    axes[0].set_ylim(positions[-1] + .65, -.65)
    axes[0].set_title("Highest-ranked minus random additions", loc="left", fontsize=11, pad=14)
    axes[1].set_title("Highest-ranked hybrid minus each endpoint", loc="left", fontsize=11, pad=14)
    axes[0].legend(handles=[Line2D([0], [0], color="#2166ac", lw=4, label="Ordinary 95% interval"),
                           Line2D([0], [0], color="#98b4cd", lw=1.5, label="8-comparison family interval")],
                   loc="lower left", bbox_to_anchor=(-.02, 1.09), frameon=False, fontsize=9)
    axes[1].legend(handles=[Line2D([0], [0], color="#2166ac", marker="o", label="vs conditioned (C)"),
                           Line2D([0], [0], color="#d97925", marker="s", label="vs wide shared (U)")],
                   loc="lower left", bbox_to_anchor=(-.02, 1.09), frameon=False, fontsize=9)
    fig.suptitle("Equal actual displacement: does shared ranking help?", fontsize=15, fontweight="bold", y=.975)
    fig.text(.22, .113,
             "Same removed tokens and replacement count in top and random arms; common C ∩ U tokens stay.\n"
             "Both addition policies use U \\ C; fractions specify how much of this shared-only pool to insert.\n"
             "Left: primary contrasts, with paired image-bootstrap intervals. Right: secondary endpoint contrasts,\n"
             "ordinary 95% intervals. Random outcomes average three fixed addition seeds before training-seed averaging.",
             fontsize=9, color="#444444", va="center", ha="left", linespacing=1.5)
    stem = output / "matched_replacement"
    fig.savefig(stem.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return stem


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/matched-replacement-v1"))
    parser.add_argument("--output", type=Path, default=Path("results/matched-replacement-v1"))
    args = parser.parse_args()
    print(plot(args.source, args.output))
