"""Show fixed direct-set training outcomes alongside matched KL training."""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_set_distillation import write_csv


SETTINGS = (("pets_b16_49", "Pets · 49 tokens"),
            ("flowers_b16_49", "Flowers · 49 tokens"),
            ("flowers_b16_98", "Flowers · 98 tokens"))


def read_summary(root, setting):
    direct = root / setting / "summary.json"
    return json.loads((direct if direct.exists() else root / setting / "analysis/summary.json").read_text())


def interval(ax, result, y, color, scale=1):
    low, high = np.asarray(result["ci95"]) * scale
    ax.plot([low, high], [y, y], color=color, lw=3)
    if "bonferroni_family_ci95" in result:
        a, b = np.asarray(result["bonferroni_family_ci95"]) * scale
        ax.plot([a, b], [y, y], color=color, lw=1, marker="|")
    ax.scatter(result["mean"] * scale, y, color=color, s=26, zorder=3)


def plot(source, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = {s: read_summary(source, s) for s, _ in SETTINGS}
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    definitions = (("set_task_minus_set_image", "Set task − set image", "#2166ac"),
                   ("set_task_minus_shared", "Set task − shared U", "#d97925"),
                   ("set_image_minus_shared", "Set image − shared U", "#75945c"),
                   ("objective_by_conditioning_interaction", "Objective × conditioning", "#8e5ca2"))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 6.3), sharey=True)
    fig.subplots_adjust(left=.20, right=.98, bottom=.23, top=.77, wspace=.28)
    contrast_rows, absolute_rows, error_rows = [], [], []
    for si, (setting, _) in enumerate(SETTINGS):
        result = results[setting]
        for ci, (key, _, color) in enumerate(definitions):
            estimate = result["classification"]["contrasts"][key]
            y = si * 1.7 + (ci - 1.5) * .27
            for ax, metric, scale in zip(axes, ("accuracy", "margin"), (100., 1000.)):
                interval(ax, estimate[metric], y, color, scale)
        for key, estimate in result["classification"]["contrasts"].items():
            for metric in ("accuracy", "margin", "raw_gap"):
                contrast_rows.append({"setting": setting, "contrast": key, "metric": metric,
                                      **estimate[metric]})
        for key, estimate in result["classification"]["absolute"].items():
            for metric in ("accuracy", "margin", "raw_gap"):
                absolute_rows.append({"setting": setting, "method": key, "metric": metric,
                                      **estimate[metric]})
        for key, estimate in result["error_transitions"].items():
            for metric in ("corrected", "broken", "net_accuracy", "prediction_disagreement"):
                error_rows.append({"setting": setting, "comparison": key, "metric": metric,
                                   **estimate[metric]})
    for ax, label in zip(axes, ("Accuracy difference (percentage points)",
                                "Geometric margin difference (×10⁻³)")):
        ax.axvline(0, color="#888888", ls="--", lw=.8)
        ax.set_yticks(np.arange(3) * 1.7, [name for _, name in SETTINGS])
        ax.set_ylim(4.15, -.7)
        ax.set_xlabel(label)
        ax.grid(axis="x", color="#eeeeee")
        ax.set_axisbelow(True)
    fig.suptitle("Does directly learning the retained set improve classification?",
                 fontsize=14, fontweight="bold", y=.97)
    fig.legend(handles=[Line2D([0], [0], color=c, marker="o", label=l) for _, l, c in definitions],
               loc="upper center", bbox_to_anchor=(.60, .90), ncol=2, frameon=False, fontsize=9)
    fig.text(.20, .09, "Same fixed hybrid masks; matched task/image branches; all three seeds retained.\n"
             "Thick bars: paired image-bootstrap 95% intervals. Thin blue: three-setting adjustment.\n"
             "Interaction = (task − image) with direct-set loss minus (task − image) with KL.",
             fontsize=8.5, linespacing=1.5, va="center")
    for ext in ("png", "pdf"):
        fig.savefig(output / f"set_classification.{ext}", dpi=220, facecolor="white")
    plt.close(fig)
    write_csv(output / "classification_contrasts.csv", flatten_intervals(contrast_rows))
    write_csv(output / "absolute_classification.csv", flatten_intervals(absolute_rows))
    write_csv(output / "error_transitions.csv", flatten_intervals(error_rows))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 6), sharey=True)
    fig.subplots_adjust(left=.19, right=.98, bottom=.22, top=.76, wspace=.24)
    styles = (("shared", "Shared U", "#999999"), ("kl_task", "KL task", "#9ecae1"),
              ("set_task", "Set task", "#2166ac"), ("kl_image", "KL image", "#f6c58a"),
              ("set_image", "Set image", "#d97925"))
    recovery_rows, fit_rows = [], []
    for si, (setting, _) in enumerate(SETTINGS):
        result = results[setting]
        for mi, (name, _, color) in enumerate(styles):
            estimate = result["recovery"]["absolute"][name]["overlap"]
            interval(axes[0], estimate, si * 1.7 + (mi - 2) * .21, color, 100)
            recovery_rows.append({"setting": setting, "kind": "absolute", "method": name, **estimate})
        for ai, (arm, color) in enumerate((("task", "#2166ac"), ("image", "#d97925"))):
            name = f"set_{arm}_minus_kl_{arm}"
            estimate = result["recovery"]["contrasts"][name]["overlap"]
            interval(axes[1], estimate, si * 1.7 + (ai - .5) * .33, color, 100)
            recovery_rows.append({"setting": setting, "kind": "difference", "method": name, **estimate})
        for split in ("train", "val"):
            for method, metrics in result["fit_diagnostics"][split].items():
                fit_rows.extend({"setting": setting, "split": split, "method": method,
                                 "metric": metric, "mean": estimate["mean"]}
                                for metric, estimate in metrics.items())
    for ax, label in zip(axes, ("Online top-k overlap with teacher (%)", "Set − KL overlap (percentage points)")):
        ax.set_yticks(np.arange(3) * 1.7, [name for _, name in SETTINGS])
        ax.set_ylim(4.1, -.65)
        ax.set_xlabel(label)
        ax.grid(axis="x", color="#eeeeee")
        ax.set_axisbelow(True)
    axes[1].axvline(0, color="#888888", ls="--", lw=.8)
    fig.suptitle("Does direct-set supervision recover more of the same teacher mask?",
                 fontsize=13, fontweight="bold", y=.97)
    fig.legend(handles=[Line2D([0], [0], color=c, marker="o", label=l) for _, l, c in styles],
               loc="upper center", bbox_to_anchor=(.58, .89), ncol=3, frameon=False, fontsize=9)
    fig.text(.19, .08, "Validation images and unseen class combinations; online retained masks.\n"
             "Thick bars: paired image-bootstrap 95% intervals. Thin bars on right: six-comparison adjustment.\n"
             "Cached training/validation fitting diagnostics are retained separately in fit_diagnostics.csv.",
             fontsize=8.5, linespacing=1.5)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"set_recovery.{ext}", dpi=220, facecolor="white")
    plt.close(fig)
    write_csv(output / "set_recovery.csv", flatten_intervals(recovery_rows))
    write_csv(output / "fit_diagnostics.csv", fit_rows)


def flatten_intervals(rows):
    """Keep exported tables numeric with explicit interval bounds."""
    output = []
    for row in rows:
        current = {k: v for k, v in row.items() if k not in ("ci95", "bonferroni_family_ci95")}
        current.update(ci95_low=row["ci95"][0], ci95_high=row["ci95"][1])
        if "bonferroni_family_ci95" in row:
            current.update(adjusted_low=row["bonferroni_family_ci95"][0],
                           adjusted_high=row["bonferroni_family_ci95"][1])
        output.append(current)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/set-distillation-v1"))
    parser.add_argument("--output", type=Path, default=Path("results/set-distillation-v1"))
    args = parser.parse_args()
    plot(args.source, args.output)
