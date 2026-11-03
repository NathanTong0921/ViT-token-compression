"""CPU-only class-novelty analysis of original or expanded holdout predictions."""

import argparse
import json
from pathlib import Path

import numpy as np

from semantic_identity_protocol import weighted_image_interval, paired_stratum_interaction


def summarize_holdout_panel(panel, predictions, arms, seeds, held_classes, replicates=2000):
    ids, weights, strata = (panel[key] for key in ("image_ids", "weights", "strata"))
    task_sources = panel.get("task_sources", np.full(len(ids), "original_val"))
    true_new = np.isin(panel["class_ids"], held_classes)
    masks = {str(group): strata == group for group in np.unique(strata)}
    masks["known_new_true_known"] = (strata == "known_new") & ~true_new
    masks["known_new_true_new"] = (strata == "known_new") & true_new
    report = {"final_test_used": False, "groups": {}, "mixed_true_class_interaction": {},
              "uncertainty": "Paired image bootstrap; fixed class partition, task panel, mappings and fitted models. All predetermined seeds/mappings included."}
    metric_arrays = {metric: {arm: np.stack([predictions[(seed, arm)][metric] for seed in seeds])
                             for arm in arms} for metric in ("correct", "margin")}
    for name, mask in masks.items():
        if not mask.any():
            continue
        group = {"examples": int(mask.sum()), "images": int(len(np.unique(ids[mask]))),
                 "tasks": int(len(np.unique(panel["task_ids"][mask]))),
                 "task_sources": {str(source): int(len(np.unique(panel["task_ids"][mask & (task_sources == source)])))
                                  for source in np.unique(task_sources[mask])},
                 "full_model_accuracy_percent": float(np.average(panel["full_correct"][mask], weights=weights[mask]) * 100),
                 "full_model_margin": float(np.average(panel["full_margin"][mask], weights=weights[mask])),
                 "metrics": {}}
        for metric, arrays in metric_arrays.items():
            scale = 100. if metric == "correct" else 1.
            means = {arm: value.mean(0) for arm, value in arrays.items()}
            permuted = np.mean([means[arm] for arm in arms if arm.startswith("permuted_")], axis=0)
            effects = {"real_minus_permuted_mean": means["real_text"] - permuted,
                       "real_minus_wide": means["real_text"] - means["image_only_wide"],
                       "permuted_mean_minus_wide": permuted - means["image_only_wide"]}
            group["metrics"][metric] = {
                "unit": "percentage points" if metric == "correct" else "geometric margin",
                "arms": {arm: {"mean": float(np.average(value.mean(0)[mask], weights=weights[mask]) * scale),
                               "per_seed": [float(np.average(row[mask], weights=weights[mask]) * scale) for row in value]}
                         for arm, value in arrays.items()},
                "contrasts": {contrast: weighted_image_interval(ids[mask], weights[mask], effect[mask] * scale, replicates)
                              for contrast, effect in effects.items()}}
        report["groups"][name] = group
    mixed = strata == "known_new"
    if mixed.any() and true_new[mixed].any() and (~true_new[mixed]).any():
        side_strata = np.where(true_new, "true_new", "true_known")
        for metric, arrays in metric_arrays.items():
            effect = arrays["real_text"].mean(0) - np.mean(
                [value.mean(0) for arm, value in arrays.items() if arm.startswith("permuted_")], axis=0)
            scale = 100. if metric == "correct" else 1.
            report["mixed_true_class_interaction"][metric] = paired_stratum_interaction(
                ids[mixed], weights[mixed], effect[mixed] * scale, side_strata[mixed],
                "true_new", "true_known", replicates)
    return report


def analyze(source, protocol_path, output, replicates=2000):
    protocol = json.loads(protocol_path.read_text())
    with np.load(source / "evaluation_panel.npz") as saved:
        panel = {key: saved[key].copy() for key in saved.files}
    arms, seeds = protocol["arms"], protocol["training_seeds"]
    predictions = {}
    for seed in seeds:
        for arm in arms:
            with np.load(source / f"seed_{seed}_{arm}.npz") as saved:
                predictions[(seed, arm)] = {key: saved[key].copy() for key in ("correct", "margin")}
    report = summarize_holdout_panel(panel, predictions, arms, seeds, protocol["held_classes"], replicates)
    report.update(source=str(source), protocol=str(protocol_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    analyze(args.source, args.protocol, args.output, args.bootstrap_replicates)
