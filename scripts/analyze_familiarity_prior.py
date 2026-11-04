#!/usr/bin/env python3
"""Diagnose candidate-direction shifts using saved familiarity-switch features.

Calibration uses image IDs, task IDs, model scores and full-CLIP scores only.
Ground-truth labels are used afterward for evaluation and descriptive contrasts.
"""
import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def assign_image_folds(image_ids, count=5, seed=2301):
    unique = np.unique(image_ids)
    order = np.random.default_rng(seed).permutation(len(unique))
    assigned = np.empty(len(unique), dtype=int)
    assigned[order] = np.arange(len(unique)) % count
    lookup = {image_id: int(fold) for image_id, fold in zip(unique, assigned)}
    return np.asarray([lookup[image_id] for image_id in image_ids]), lookup


def crossfit_offsets(q, full_q, task_ids, folds, kind):
    """Predict residual means on disjoint images; no labels/label weights enter."""
    residual = np.asarray(q, dtype=float) - full_q
    offsets = np.empty(len(q), dtype=float)
    records = []
    for fold in np.unique(folds):
        calibration, evaluation = folds != fold, folds == fold
        groups = ["__global__"] if kind == "global" else np.unique(task_ids[evaluation])
        for task in groups:
            train = calibration if kind == "global" else calibration & (task_ids == task)
            held = evaluation if kind == "global" else evaluation & (task_ids == task)
            assert train.any(), f"No calibration rows for task {task} in fold {fold}"
            offset = float(residual[train].mean())
            offsets[held] = offset
            records.append({"kind": kind, "fold": int(fold), "task_id": str(task),
                            "calibration_rows": int(train.sum()), "evaluation_rows": int(held.sum()),
                            "offset": offset})
    return offsets, records


def correct_from_q(q, true_h0, h0_class, h1_class):
    predict_h0 = (q > 0) | ((q == 0) & (h0_class < h1_class))
    return (predict_h0 == true_h0).astype(float)


def task_auc(q, true_h0, task_ids):
    """AUC toward H0; invariant to one additive constant within each task."""
    values = []
    for task in np.unique(task_ids):
        rows = task_ids == task
        positive, negative = q[rows & true_h0], q[rows & ~true_h0]
        assert len(positive) and len(negative)
        differences = positive[:, None] - negative[None, :]
        values.append(float(((differences > 0) + .5 * (differences == 0)).mean()))
    return np.asarray(values)


def image_interval(values, ids, weights, replicates=2000, seed=2302):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    _, inverse = np.unique(ids, return_inverse=True)
    n = inverse.max() + 1
    numerator = np.bincount(inverse, weights=weights * values, minlength=n)
    denominator = np.bincount(inverse, weights=weights, minlength=n)
    rng, boot = np.random.default_rng(seed), []
    for start in range(0, replicates, 100):
        ix = rng.integers(n, size=(min(100, replicates - start), n))
        boot.extend(numerator[ix].sum(1) / denominator[ix].sum(1))
    return {"estimate": float(numerator.sum() / denominator.sum()),
            "ci95": np.quantile(boot, [.025, .975]).tolist()}


def summarize(q, offsets, panel, arms, seeds, full_q, h0_class, h1_class, replicates):
    """q: phase, seed, arm, row. Average model outcomes, never vote ensembles."""
    ids, weights = panel["image_ids"], panel["weights"]
    truth = panel["true_original_held"].astype(bool)
    sign = np.where(truth, 1., -1.)
    rix, wix = arms.index("real_text"), arms.index("image_only_wide")
    pix = [i for i, arm in enumerate(arms) if arm.startswith("permuted_")]
    contrasts = {"real_minus_identity": (rix, pix), "real_minus_wide": (rix, [wix])}
    interval = lambda x: image_interval(x, ids, weights, replicates)
    result = {"decomposition": {}, "correction": {}, "offset_prediction": {}, "raw_task_auc": {}}
    raw_correct = correct_from_q(q, truth, h0_class, h1_class)
    task_rows = []
    for name, (real, controls) in contrasts.items():
        effects = q[:, :, real] - q[:, :, controls].mean(axis=2)
        change = effects[1] - effects[0]
        averaged = change.mean(0)
        mu = [float(np.average(averaged[truth == group], weights=weights[truth == group])) for group in (True, False)]
        result["decomposition"][name] = {
            "H0_image_directional_change": mu[0], "H1_image_directional_change": mu[1],
            "midpoint_shift": interval(averaged),
            "half_separation_change": interval(sign * averaged),
            "midpoint_shift_per_seed": [float(np.average(row, weights=weights)) for row in change],
            "half_separation_change_per_seed": [float(np.average(sign * row, weights=weights)) for row in change],
            "weighted_identity_error": float(abs(np.average(averaged, weights=weights) - .5 * sum(mu)))}
        for j, seed in enumerate(seeds):
            for task in np.unique(panel["task_ids"]):
                selected = panel["task_ids"] == task
                means = [float(change[j, selected & (truth == group)].mean()) for group in (True, False)]
                task_rows.append({"contrast": name, "seed": seed, "task_id": str(task),
                                  "H0_image_change": means[0], "H1_image_change": means[1],
                                  "midpoint_shift": .5 * sum(means), "half_separation_change": .5 * (means[0] - means[1])})
        result["offset_prediction"][name] = {}
        for kind, fitted in offsets.items():
            fit_effect = fitted[:, :, real] - fitted[:, :, controls].mean(axis=2)
            predicted = fit_effect[1] - fit_effect[0]
            null_mse = (change ** 2).mean(0)
            corrected_mse = ((change - predicted) ** 2).mean(0)
            base, residual = float(np.average(null_mse, weights=weights)), float(np.average(corrected_mse, weights=weights))
            result["offset_prediction"][name][kind] = {
                "held_image_MSE_null_zero": base, "held_image_MSE_predicted_offset": residual,
                "fraction_MSE_reduction_vs_zero": 1 - residual / base if base else None,
                "paired_MSE_reduction": interval(null_mse - corrected_mse),
                "per_seed_MSE_reduction_fraction": [float(1 - np.average((a - b) ** 2, weights=weights) / np.average(a ** 2, weights=weights)) if np.any(a) else None
                                                     for a, b in zip(change, predicted)]}
    # AUC is calculated from each raw model separately, then averaged over seeds/mappings.
    auc = np.empty(q.shape[:3] + (len(np.unique(panel["task_ids"])),))
    for p in range(2):
        for s in range(len(seeds)):
            for a in range(len(arms)):
                auc[p, s, a] = task_auc(q[p, s, a], truth, panel["task_ids"])
    result["raw_task_auc"]["absolute_arms"] = {phase: {arm: float(auc[p, :, a].mean())
                                                      for a, arm in enumerate(arms)}
                                               for p, phase in enumerate(("original", "switched"))}
    for name, (real, controls) in contrasts.items():
        advantage = auc[:, :, real] - auc[:, :, controls].mean(axis=2)
        result["raw_task_auc"][name] = {
            "phase_original_advantage": float(advantage[0].mean()),
            "phase_switched_advantage": float(advantage[1].mean()),
            "phase_change": float((advantage[1] - advantage[0]).mean()),
            "phase_change_per_seed": (advantage[1] - advantage[0]).mean(1).tolist(),
            "units": "AUC, equally averaged across fixed tasks; descriptive, no independent-task uncertainty claim"}
    for kind in ("raw", "global", "task"):
        corrected = q if kind == "raw" else q - offsets[kind]
        correct = raw_correct if kind == "raw" else correct_from_q(corrected, truth, h0_class, h1_class)
        outcomes = {"accuracy_pp": correct * 100, "true_margin": corrected * sign}
        result["correction"][kind] = {"absolute_arms": {}, "contrasts": {}}
        for metric, values in outcomes.items():
            result["correction"][kind]["absolute_arms"][metric] = {phase: {arm: float(np.average(values[p, :, a].mean(0), weights=weights))
                                                                           for a, arm in enumerate(arms)}
                                                                   for p, phase in enumerate(("original", "switched"))}
            result["correction"][kind]["contrasts"][metric] = {}
            for name, (real, controls) in contrasts.items():
                effects = values[:, :, real] - values[:, :, controls].mean(axis=2)
                change = effects[1] - effects[0]
                oriented = change * sign
                result["correction"][kind]["contrasts"][metric][name] = {
                    "seen_minus_held": interval(oriented.mean(0)),
                    "seen_minus_held_per_seed": [float(np.average(row, weights=weights)) for row in oriented],
                    "phase_advantages": {phase: float(np.average(effects[p].mean(0), weights=weights))
                                         for p, phase in enumerate(("original", "switched"))}}
        if kind != "raw":
            improvement = (correct - raw_correct) * 100
            result["correction"][kind]["accuracy_change_from_raw"] = {
                phase: {arm: interval(improvement[p, :, a].mean(0)) for a, arm in enumerate(arms)}
                for p, phase in enumerate(("original", "switched"))}
    return result, task_rows, auc


def run(source, output, folds_count=5, fold_seed=2301, replicates=2000, cache_override=None, plan_only=False):
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((source / "protocol.json").read_text())
    with np.load(source / "evaluation_panel.npz") as saved:
        panel = {key: saved[key].copy() for key in saved.files}
    folds, assignments = assign_image_folds(panel["image_ids"], folds_count, fold_seed)
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    cache = cache_override or Path(protocol["config"]["cache_dir"])
    plan = {"source": str(source), "cache": str(cache), "code_commit": code_commit,
            "partition_seed": protocol["partition_seed"], "folds": folds_count, "fold_seed": fold_seed,
            "bootstrap_seed": 2302, "bootstrap_replicates": replicates,
            "calibration": "global and per-task unweighted q_pruned minus q_full means on other image folds; no truth labels or task/class weights used in fitting",
            "orientation": "positive q favors H0, the original held class; orientation follows candidate IDs and frozen training groups",
            "interpretation": "full-CLIP-referenced diagnostic correction; not a deployable speedup or exclusive causal identification of an answer prior",
            "margin_cancellation": "Average corrected margin effects can cancel partly algebraically; held-image accuracy and held-image shift-prediction MSE are the correction evidence",
            "uncertainty": "paired image bootstrap conditional on fixed folds, fitted offsets, training seeds, mappings and task panel; offset fitting is not repeated inside bootstrap",
            "final_test_used": False, "arms": protocol["arms"], "training_seeds": protocol["training_seeds"]}
    write_json(output / "protocol.json", plan)
    with (output / "image_folds.csv").open("w", newline="") as handle:
        writer = csv.writer(handle);writer.writerow(["image_id", "fold"]);writer.writerows(sorted(assignments.items()))
    if plan_only:
        return
    with np.load(cache / "val" / "evidence.npz") as saved:
        text = saved["text_features"].copy()
        full_scores = saved["full_scores"].copy()
        source_ids = saved["image_ids"].copy()
    lookup = {image_id: i for i, image_id in enumerate(source_ids)}
    full_scores = full_scores[np.asarray([lookup[image_id] for image_id in panel["image_ids"]])]
    pairs = panel["class_pairs"]
    first_h0 = np.isin(pairs[:, 0], protocol["original_held_classes"])
    h0_class, h1_class = np.where(first_h0, pairs[:, 0], pairs[:, 1]), np.where(first_h0, pairs[:, 1], pairs[:, 0])
    assert np.isin(h0_class, protocol["original_held_classes"]).all()
    assert np.isin(h1_class, protocol["switched_held_classes"]).all()
    direction_norm = np.linalg.norm(text[h0_class] - text[h1_class], axis=1)
    rows = np.arange(len(pairs))
    score_direction = lambda scores: ((scores[rows, h0_class] - scores[rows, h1_class]) / direction_norm).astype(np.float64)
    full_q = score_direction(full_scores)
    arms, seeds = protocol["arms"], protocol["training_seeds"]
    q = np.empty((2, len(seeds), len(arms), len(rows)), dtype=np.float64)
    offsets = {kind: np.empty_like(q) for kind in ("global", "task")}
    offset_rows, checks = [], []
    truth = panel["true_original_held"].astype(bool)
    sign = np.where(truth, 1., -1.)
    np.testing.assert_array_equal(truth, panel["class_ids"] == h0_class)
    np.testing.assert_allclose(full_q * sign, panel["full_margin"], rtol=1e-5, atol=1e-6)
    for p, phase in enumerate(("original", "switched")):
        for s, seed in enumerate(seeds):
            for a, arm in enumerate(arms):
                path = source / phase / f"seed_{seed}_{arm}.npz"
                with np.load(path) as saved:
                    features = saved["image_features"].copy()
                    stored_margin, stored_correct = saved["margin"].copy(), saved["correct"].copy()
                q[p, s, a] = score_direction(features @ text.T)
                error = float(np.max(np.abs(q[p, s, a] * sign - stored_margin)))
                np.testing.assert_allclose(q[p, s, a] * sign, stored_margin, rtol=1e-5, atol=1e-6)
                np.testing.assert_array_equal(correct_from_q(q[p, s, a], truth, h0_class, h1_class), stored_correct)
                checks.append({"phase": phase, "seed": seed, "arm": arm, "max_margin_reconstruction_error": error,
                               "predictions_match_saved": True})
                for kind in offsets:
                    fitted, records = crossfit_offsets(q[p, s, a], full_q, panel["task_ids"], folds, kind)
                    offsets[kind][p, s, a] = fitted
                    offset_rows.extend({"phase": phase, "seed": seed, "arm": arm, **record} for record in records)
    result, task_rows, auc = summarize(q, offsets, panel, arms, seeds, full_q, h0_class, h1_class, replicates)
    result.update({"protocol": plan, "reconstruction_checks": checks,
                   "rows": len(rows), "images": len(assignments), "tasks": len(np.unique(panel["task_ids"]))})
    previous = json.loads((source / "summary.json").read_text())
    original = previous["metrics"]["margin"]["contrasts"]["real_minus_permuted_mean"]["seen_minus_held"]["estimate"]
    reconstructed = result["decomposition"]["real_minus_identity"]["midpoint_shift"]["estimate"]
    result["original_margin_estimand_check"] = {"existing_seen_minus_held": original, "reconstructed_directional_midpoint_shift": reconstructed,
                                                "absolute_difference": abs(original - reconstructed)}
    np.testing.assert_allclose(original, reconstructed, rtol=1e-5, atol=1e-7)
    write_json(output / "summary.json", result)
    for filename, records in (("offsets.csv", offset_rows), ("task_decomposition.csv", task_rows)):
        with (output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    np.savez_compressed(output / "analysis_arrays.npz", q=q, offsets_global=offsets["global"], offsets_task=offsets["task"],
                        full_q=full_q, task_auc=auc, arms=np.asarray(arms), seeds=np.asarray(seeds),
                        image_ids=panel["image_ids"], task_ids=panel["task_ids"], weights=panel["weights"],
                        true_h0=truth, folds=folds, h0_class=h0_class, h1_class=h1_class)
    print(json.dumps({"source": str(source), "decomposition": result["decomposition"], "offset_prediction": result["offset_prediction"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=2301)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    run(args.source, args.output, args.folds, args.fold_seed, args.bootstrap_replicates, args.cache_dir, args.plan_only)
