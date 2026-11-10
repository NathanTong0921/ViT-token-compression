"""Plot actual development accuracy and fixed-training-monitor CE."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARMS = (("C", "conditioned", "#2166A5", "o"),
        ("U", "image_only_wide", "#C97927", "s"))
STAGES = ("initial", "kl", "direct")
STAGE_LABELS = ("Initial", "KL continuation", "Direct classification")


def load_monitor(folder):
    initial = json.loads((folder / "initial_train_monitor.json").read_text())
    values = {0: float(initial["ce"])}
    for path in sorted(folder.glob("monitor_*.json")):
        step = int(path.stem.split("_")[-1])
        values[step] = float(json.loads(path.read_text())["ce"])
    return values


def main(source, output):
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((source / "summary.json").read_text())
    with (source / "per_seed.csv").open() as file:
        per_seed = list(csv.DictReader(file))
    seeds = summary["training_seeds"]
    seed_lookup = {(int(row["seed"]), row["name"]): float(row["accuracy"]) * 100
                   for row in per_seed if row["kind"] == "method"}
    table = []
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": "#24323E",
                         "text.color": "#24323E", "axes.edgecolor": "#ADB5BF",
                         "xtick.color": "#46515D", "ytick.color": "#46515D"})
    fig, (accuracy_ax, ce_ax) = plt.subplots(1, 2, figsize=(12.2, 5.5))
    fig.subplots_adjust(left=.075, right=.975, bottom=.22, top=.70, wspace=.30)
    fig.suptitle("Direct classification training vs KL continuation", fontsize=17,
                 fontweight="bold", y=.97)
    fig.text(.5, .905,
             f"Flowers102 · CLIP B/16 · block 6 · {summary.get('keep_patch_tokens', 49)} tokens · {len(seeds)} fixed seeds · {summary['step']:,} extra updates",
             ha="center", color="#66717D", fontsize=11)

    x = np.arange(len(STAGES))
    offsets = np.linspace(-.055, .055, len(seeds))
    accuracy_ax.set_title("A  Development accuracy", loc="left", fontweight="bold", pad=14)
    accuracy_values = []
    for symbol, arm, color, marker in ARMS:
        arm_offset = -.075 if symbol == "C" else .075
        means = []
        for stage_index, stage in enumerate(STAGES):
            name = stage + "_" + symbol
            values = np.asarray([seed_lookup[(seed, name)] for seed in seeds])
            mean = float(summary["methods"][name]["accuracy"]["mean"]) * 100
            accuracy_values.extend(values)
            means.append(mean)
            accuracy_ax.scatter(stage_index + arm_offset + offsets, values,
                                marker=marker, s=22, color=color, alpha=.35, linewidths=0, zorder=3)
            for seed, value in zip(seeds, values):
                table.append(dict(panel="A", measure="accuracy", stage=stage, arm=symbol,
                                  objective=stage, seed=seed, step=0 if stage=="initial" else summary["step"],
                                  value=float(value), unit="percent"))
            table.append(dict(panel="A", measure="accuracy", stage=stage, arm=symbol,
                              objective=stage, seed="mean", step=0 if stage=="initial" else summary["step"],
                              value=mean, unit="percent"))
        accuracy_ax.plot(x + arm_offset, means, color=color, marker=marker, ms=6,
                         lw=1.8, label=f"{symbol}: {'task-conditioned' if symbol == 'C' else 'strong shared'}", zorder=4)
    # Use a broad, explicitly labeled percentage scale instead of magnifying tiny changes.
    lower = min(95., np.floor(min(accuracy_values) / 5.) * 5.)
    accuracy_ax.set_ylim(lower, 100.)
    accuracy_ax.set_xlim(-.45, 2.45)
    accuracy_ax.set_xticks(x, STAGE_LABELS)
    accuracy_ax.set_ylabel("Accuracy (%)")
    accuracy_ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    accuracy_ax.grid(axis="y", color="#E6EAF0", lw=.7)
    accuracy_ax.set_axisbelow(True)
    accuracy_ax.legend(loc="lower left", frameon=False, fontsize=9)
    difference_y = 100. - .11 * (100. - lower)
    for index, stage in enumerate(STAGES):
        gap = (summary["methods"][stage+"_C"]["accuracy"]["mean"] -
               summary["methods"][stage+"_U"]["accuracy"]["mean"]) * 100
        accuracy_ax.text(index, difference_y, f"C−U: {gap:+.3f} pp", ha="center", fontsize=9,
                         color="#46515D")
        table.append(dict(panel="A", measure="C_minus_U_accuracy", stage=stage, arm="C−U",
                          objective=stage, seed="mean", step=0 if stage=="initial" else summary["step"],
                          value=float(gap), unit="percentage_points"))

    ce_ax.set_title("B  Classification loss on a fixed training monitor", loc="left",
                    fontweight="bold", pad=14)
    all_ce = []
    for symbol, arm, color, marker in ARMS:
        for objective, linestyle in (("direct", "-"), ("kl", "--")):
            runs = [load_monitor(source / "training_records" / f"seed_{seed}" / arm / objective)
                    for seed in seeds]
            steps = sorted(runs[0])
            assert all(sorted(run)==steps for run in runs), "Plot matching saved monitor steps across seeds"
            values = np.asarray([[run[step] for step in steps] for run in runs])
            all_ce.extend(values.ravel())
            for seed, row in zip(seeds, values):
                ce_ax.plot(steps, row, color=color, ls=linestyle, lw=.6, alpha=.18)
                for step, value in zip(steps, row):
                    table.append(dict(panel="B", measure="training_monitor_ce", stage="monitor", arm=symbol,
                                      objective=objective, seed=seed, step=step, value=float(value), unit="nats"))
            mean = values.mean(axis=0)
            ce_ax.plot(steps, mean, color=color, ls=linestyle, lw=1.8, marker=marker,
                       ms=4, label=f"{symbol} · {'direct' if objective == 'direct' else 'KL continuation'}")
            for step, value in zip(steps, mean):
                table.append(dict(panel="B", measure="training_monitor_ce", stage="monitor", arm=symbol,
                                  objective=objective, seed="mean", step=step, value=float(value), unit="nats"))
    ce_ax.set_xlabel("Additional optimizer updates")
    ce_ax.set_ylabel("Cross-entropy (nats; lower is better)")
    ce_ax.set_xlim(0, summary["step"])
    ce_ax.set_ylim(0, max(all_ce) * 1.12)
    ce_ax.xaxis.set_major_locator(plt.MaxNLocator(5, integer=True))
    ce_ax.grid(color="#E6EAF0", lw=.7)
    ce_ax.set_axisbelow(True)
    ce_ax.legend(loc="best", frameon=False, fontsize=9, ncol=2)

    fig.text(.075, .105,
             "A  Faint points show individual seeds; connected markers show their means. C−U differences are in percentage points.\n"
             "B  CE uses the same fixed training-monitor examples within each seed, evaluated with native hard token deletion.\n"
             "     Thin curves show seeds; bold curves show means. Lines connect saved monitor evaluations, not batch-loss averages.",
             fontsize=9, color="#46515D", linespacing=1.5, va="center")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"direct_classification.{extension}", dpi=240, facecolor="white")
    plt.close(fig)
    with (output / "source_values.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    (output / "sources.json").write_text(json.dumps(dict(
        summary="../summary.json", per_seed="../per_seed.csv", monitors="../training_records/",
        script="scripts/plot_direct_classification.py", accuracy_unit="percent", difference_unit="percentage_points",
        loss_scope="Fixed training monitor, native gather suffix, per-seed sample stream; not development loss."),
        indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/direct-classification-v1"))
    parser.add_argument("--output", type=Path, default=Path("results/direct-classification-v1/figures"))
    args = parser.parse_args()
    main(args.source, args.output)
