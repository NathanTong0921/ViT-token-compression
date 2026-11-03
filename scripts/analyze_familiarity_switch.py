"""Paired same-image/task effects after exchanging training familiarity."""

import argparse
import json
from pathlib import Path

import numpy as np

from semantic_identity_protocol import weighted_image_interval, paired_stratum_interaction


def summarize_switch(panel, predictions, arms, seeds, replicates=2000):
    weights, ids = panel["weights"], panel["image_ids"]
    h0 = panel["true_original_held"].astype(bool)
    groups = np.where(h0, "H0", "H1")
    sign = np.where(h0, 1., -1.)
    mapping_arms = [a for a in arms if a.startswith("permuted_")]
    result = dict(final_test_used=False, rows=len(weights), images=len(np.unique(ids)),
                  tasks=len(np.unique(panel["task_ids"])),
                  role_definition="H0 is held in original fit and seen in switched fit; H1 reverses. Each evaluated task contains one class from each group.",
                  estimand="Seen-minus-held contrast is the equally weighted mean of oriented within-image/task changes. DID is the H0 partition-change minus H1 partition-change; its point estimate equals twice the seen-minus-held contrast under balanced task/class weights.",
                  uncertainty="Paired image bootstrap conditional on these fixed two partitions, task panel, three seeds and two mappings.",
                  true_group_weight={name: float(weights[groups == name].sum()) for name in ("H0", "H1")},
                  metrics={})
    for metric in ("correct", "margin"):
        scale = 100. if metric == "correct" else 1.
        arrays = {phase: {arm: np.stack([predictions[(phase, seed, arm)][metric].astype(float) for seed in seeds]) * scale
                          for arm in arms} for phase in ("original", "switched")}
        effects = {}
        for phase, values in arrays.items():
            permuted = np.mean([values[arm] for arm in mapping_arms], axis=0)
            effects[phase] = {"real_minus_permuted_mean": values["real_text"] - permuted,
                              "real_minus_wide": values["real_text"] - values["image_only_wide"],
                              "permuted_mean_minus_wide": permuted - values["image_only_wide"]}
            effects[phase].update({f"real_minus_{arm}": values["real_text"] - values[arm] for arm in mapping_arms})
        contrasts = {}
        for contrast in effects["original"]:
            change_by_seed = effects["switched"][contrast] - effects["original"][contrast]
            change = change_by_seed.mean(0)
            oriented = change * sign
            contrasts[contrast] = dict(
                seen_minus_held=weighted_image_interval(ids, weights, oriented, replicates),
                seen_minus_held_per_seed=[float(np.average(row * sign, weights=weights)) for row in change_by_seed],
                difference_in_differences=paired_stratum_interaction(ids, weights, change, groups, "H0", "H1", replicates),
                partition_change_by_true_group={name: weighted_image_interval(ids[groups == name], weights[groups == name],
                    change[groups == name], replicates) for name in ("H0", "H1")},
                phase_effects={phase: {name: float(np.average(effects[phase][contrast].mean(0)[groups == name],
                    weights=weights[groups == name])) for name in ("H0", "H1")} for phase in effects})
        result["metrics"][metric] = dict(unit="percentage points" if metric == "correct" else "geometric margin",
            full_model=float(np.average(panel["full_correct" if metric == "correct" else "full_margin"], weights=weights) * scale),
            absolute_arms={phase: {arm: {name: float(np.average(value.mean(0)[groups == name], weights=weights[groups == name]))
                               for name in ("H0", "H1")} for arm, value in values.items()} for phase, values in arrays.items()},
            contrasts=contrasts)
    return result


def analyze(source, replicates=2000):
    protocol = json.loads((source / "protocol.json").read_text())
    with np.load(source / "evaluation_panel.npz") as saved:
        panel = {key: saved[key].copy() for key in saved.files}
    predictions = {}
    for phase in ("original", "switched"):
        for seed in protocol["training_seeds"]:
            for arm in protocol["arms"]:
                with np.load(source / phase / f"seed_{seed}_{arm}.npz") as saved:
                    predictions[(phase, seed, arm)] = {key: saved[key].copy() for key in ("correct", "margin")}
    result = summarize_switch(panel, predictions, protocol["arms"], protocol["training_seeds"], replicates)
    (source / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    analyze(args.source, args.bootstrap_replicates)
