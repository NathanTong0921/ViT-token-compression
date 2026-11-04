#!/usr/bin/env python3
"""Independently reconstruct calibration and reported diagnostic point estimates."""
import argparse
import json
from pathlib import Path

import numpy as np


def audit(folder):
    summary = json.loads((folder / "summary.json").read_text())
    with np.load(folder / "analysis_arrays.npz") as saved:
        data = {key: saved[key].copy() for key in saved.files}
    q, ids, tasks, folds, weights = [data[k] for k in ("q", "image_ids", "task_ids", "folds", "weights")]
    y = np.where(data["true_h0"], 1., -1.)
    assert all(len(np.unique(folds[ids == image])) == 1 for image in np.unique(ids))
    unique = np.unique(ids)
    permutation = np.random.default_rng(summary["protocol"]["fold_seed"]).permutation(len(unique))
    expected_fold = np.empty(len(unique), int);expected_fold[permutation] = np.arange(len(unique)) % summary["protocol"]["folds"]
    np.testing.assert_array_equal(folds, expected_fold[np.searchsorted(unique, ids)])
    offset_max_error = {}
    for kind in ("global", "task"):
        observed = data[f"offsets_{kind}"]
        expected = np.empty_like(observed)
        residuals = q - data["full_q"]
        for fold in np.unique(folds):
            groups = [None] if kind == "global" else np.unique(tasks[folds == fold])
            for task in groups:
                train = (folds != fold) & (True if task is None else tasks == task)
                held = (folds == fold) & (True if task is None else tasks == task)
                expected[..., held] = residuals[..., train].mean(axis=-1)[..., None]
        offset_max_error[kind] = float(np.max(np.abs(expected - observed)))
        np.testing.assert_allclose(expected, observed, rtol=0, atol=1e-12)
    arms = data["arms"].tolist()
    r = arms.index("real_text");permuted = [i for i, arm in enumerate(arms) if arm.startswith("permuted_")]
    delta = (q[1, :, r] - q[1][:, permuted].mean(axis=1)) - (q[0, :, r] - q[0][:, permuted].mean(axis=1))
    midpoint = float(np.average(delta.mean(0), weights=weights))
    separation = float(np.average(delta.mean(0) * y, weights=weights))
    np.testing.assert_allclose(midpoint, summary["decomposition"]["real_minus_identity"]["midpoint_shift"]["estimate"], atol=1e-12)
    np.testing.assert_allclose(separation, summary["decomposition"]["real_minus_identity"]["half_separation_change"]["estimate"], atol=1e-12)
    accuracy_roles = {}
    choice_shifts = {}
    for kind in ("raw", "global", "task"):
        value = q if kind == "raw" else q - data[f"offsets_{kind}"]
        predicted_h0 = (value > 0) | ((value == 0) & (data["h0_class"] < data["h1_class"]))
        correctness = (predicted_h0 == data["true_h0"]).astype(float)
        differences = correctness[:, :, r] - correctness[:, :, permuted].mean(axis=2)
        role = float(np.average((differences[1] - differences[0]).mean(0) * y, weights=weights) * 100)
        choices = predicted_h0.astype(float)
        choice_effect = choices[:, :, r] - choices[:, :, permuted].mean(axis=2)
        choice_shift = float(np.average((choice_effect[1] - choice_effect[0]).mean(0), weights=weights) * 100)
        np.testing.assert_allclose(choice_shift, role, atol=1e-10)
        choice_shifts[kind] = choice_shift
        accuracy_roles[kind] = role
        np.testing.assert_allclose(role, summary["correction"][kind]["contrasts"]["accuracy_pp"]["real_minus_identity"]["seen_minus_held"]["estimate"], atol=1e-10)
    return {"partition_seed": summary["protocol"]["partition_seed"], "rows": len(ids), "images": len(unique),
            "image_folds_reconstructed": True, "all_same_image_rows_share_fold": True,
            "offset_max_reconstruction_error": offset_max_error,
            "directional_midpoint_shift": midpoint, "half_separation_change": separation,
            "role_accuracy_effect_pp": accuracy_roles, "H0_choice_frequency_change_pp": choice_shifts,
            "source_feature_reconstruction_checks": len(summary["reconstruction_checks"]),
            "all_source_predictions_match": all(r["predictions_match_saved"] for r in summary["reconstruction_checks"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/familiarity-prior-v1"))
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    folders = sorted(p for p in args.root.iterdir() if p.is_dir() and (p / "summary.json").exists())
    if args.require_all:
        assert {p.name for p in folders} == {f"{d}-p{s}" for d in ("pets", "flowers") for s in (2099, 2100, 2101)}
    output = {p.name: audit(p) for p in folders}
    (args.root / "verification.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
