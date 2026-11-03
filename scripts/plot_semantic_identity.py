"""Render paired semantic/identity effects from retained development summaries."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


GROUPS = ["unseen_combinations", "known_known", "known_new", "new_new",
          "known_new_true_known", "known_new_true_new"]
LABELS = ["All-class fit: new pairs", "Heldout fit: known / known",
          "Heldout fit: known / new", "Heldout fit: new / new",
          "Mixed task: true known", "Mixed task: true new"]


def read(path):
    return json.loads(path.read_text())


def audit_panels(root):
    checks = {}
    for name in ("pets", "flowers"):
        protocol = read(root / name / "protocol.json")
        training = read(root / name / "class_holdout" / "mapping_audit.json")
        expanded_protocol = read(root / f"{name}-expanded" / "protocol.json")
        with np.load(root / name / "class_holdout" / "evaluation_panel.npz") as saved:
            original = {key: saved[key].copy() for key in saved.files}
        with np.load(root / f"{name}-expanded" / "evaluation_panel.npz") as saved:
            expanded = {key: saved[key].copy() for key in saved.files}
        original_tasks = set(original["task_ids"])
        from_val = expanded["task_sources"] == "original_val"
        held = set(protocol["held_classes"])
        checks[name] = {
            "held_classes_absent_from_training_images": not bool(held & set(training["training_class_ids"])),
            "training_evaluation_images_disjoint": not bool(set(training["training_image_ids"]) & set(expanded["image_ids"])),
            "training_evaluation_tasks_disjoint": not bool(set(training["training_task_ids"]) & set(expanded["task_ids"])),
            "original_validation_tasks_preserved": set(expanded["task_ids"][from_val]) == original_tasks,
            "added_pairs_all_touch_held_class": bool(np.isin(expanded["class_pairs"][~from_val], list(held)).any(1).all()),
            "known_known_pairs_unchanged": set(original["task_ids"][original["strata"] == "known_known"])
                == set(expanded["task_ids"][expanded["strata"] == "known_known"]),
            "protocols_record_no_final_test_use": not protocol["final_test_used"] and not expanded_protocol["final_test_used"],
            "protocols_record_no_official_test_use": not protocol["official_test_used"] and not expanded_protocol["official_test_used"],
        }
        assert all(checks[name].values()), (name, checks[name])
    return checks


def plot(root):
    root.mkdir(parents=True, exist_ok=True)
    checks = audit_panels(root)
    (root / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.4), sharex="row", sharey=True)
    rows = []
    colors = {"all_class": "#25344B", "expanded": "#176A9D", "original": "#858D96"}
    for col, (dataset, title) in enumerate((("pets", "Oxford Pets · 49 patches"), ("flowers", "Flowers102 · 98 patches"))):
        known = read(root / dataset / "known_classes" / "summary.json")["strata"]
        original = read(root / dataset / "class_holdout" / "class_novelty_details.json")["groups"]
        expanded = read(root / f"{dataset}-expanded" / "class_novelty_details.json")["groups"]
        for row_index, metric in enumerate(("correct", "margin")):
            ax = axes[row_index, col]
            scale = 1. if metric == "correct" else 1000.
            for index, group in enumerate(GROUPS):
                sources = [("all_class", known[group], 0.)] if index == 0 else [
                    ("original", original[group], -.12), ("expanded", expanded[group], .12)]
                for source, summary, offset in sources:
                    effect = summary["metrics"][metric]["contrasts"]["real_minus_permuted_mean"]
                    point, low, high = effect["estimate"] * scale, *(np.asarray(effect["ci95"]) * scale)
                    y = len(GROUPS) - 1 - index + offset
                    ax.errorbar(point, y, xerr=[[point - low], [high - point]], fmt="o",
                                color=colors[source], markerfacecolor="white" if source == "original" else colors[source],
                                markersize=4.5, capsize=2, elinewidth=.9 if source == "original" else 1.5,
                                zorder=2 if source == "original" else 3)
                    rows.append(dict(dataset=dataset, metric=metric, source=source, group=group,
                                     tasks=summary["tasks"], estimate=effect["estimate"],
                                     lower95=effect["ci95"][0], upper95=effect["ci95"][1]))
            ax.axvline(0, color="#3E4750", lw=.8, linestyle="--", zorder=0)
            ax.axhline(4.5, color="#C3C7CC", lw=.6)
            ax.axhline(1.5, color="#C3C7CC", lw=.6)
            ax.grid(axis="x", alpha=.15)
            ax.set_yticks(np.arange(len(GROUPS))[::-1], LABELS)
            ax.set_ylim(-.5, 5.5)
            ax.set_xlabel("Real text − mean permutation (accuracy, pp)" if metric == "correct"
                          else "Real text − mean permutation (margin × 10³)")
            if row_index == 0:
                ax.set_title(title, weight="bold", pad=12)
            if col == 0:
                ax.tick_params(axis="y", labelsize=8.5)
    legend = [Line2D([0], [0], marker="o", color=colors[key], markerfacecolor="white" if key == "original" else colors[key],
                     linestyle="none", label=label, markersize=5)
              for key, label in (("all_class", "All-class training"),
                                 ("expanded", "Class-heldout: expanded panel"),
                                 ("original", "Class-heldout: original panel"))]
    fig.legend(handles=legend, loc="lower center", ncol=3, bbox_to_anchor=(.55, .025), frameon=False)
    fig.suptitle("Semantic text versus stable class identity", fontsize=13, weight="bold", y=.985)
    fig.text(.55, .005, "Means over 3 paired seeds and 2 fixed mappings; bars: paired image-bootstrap 95% intervals.",
             ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, .96), h_pad=2.5, w_pad=2.5)
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"semantic_identity_effects.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    with (root / "plot_data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/semantic-identity-v1"))
    plot(parser.parse_args().results_root)
