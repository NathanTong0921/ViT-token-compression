"""Independent NumPy audit of classification and H2 in targeted ViT-L validation.

No project analysis functions are imported. Raw evaluation files supply the
classification outcomes and full-dimensional features; saved analysis files
are only comparison targets. Runs on CPU after both analyses finish.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def close(actual, expected, atol=2e-10):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)


def read_npz(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def prediction_group(path, embedding_dim, reference=False):
    keys = ("method", "seed", "image_ids", "task_ids", "labels", "class_pairs",
            "cosine_scores", "image_features", "margin", "correct")
    with np.load(path) as saved:
        chosen = saved["task_split"] == "unseen"
        chosen &= (saved["method"] == "full_model" if reference else
                   np.isin(saved["method"], ["conditioned", "image_only"]))
        group = {key: saved[key][chosen] for key in keys}
        group["text_features"] = saved["text_features"]
    scores, text, features = [group[k] for k in ("cosine_scores", "text_features", "image_features")]
    assert features.shape[1] == text.shape[1] == embedding_dim
    close(np.linalg.norm(features, axis=1), 1., atol=2e-6)
    close(features @ text.T, scores, atol=1e-6)
    labels, pairs = group["labels"], group["class_pairs"]
    rows = np.arange(len(labels))
    other = np.where(labels == pairs[:, 0], pairs[:, 1], pairs[:, 0])
    margin = (scores[rows, labels] - scores[rows, other]) / np.linalg.norm(text[labels] - text[other], axis=1)
    prediction = pairs[rows, np.argmax(scores[rows[:, None], pairs], axis=1)]
    correct = prediction == labels
    close(margin, group["margin"], atol=1e-7)
    np.testing.assert_array_equal(correct, group["correct"])
    group["outcomes"] = np.column_stack([correct, margin]).astype(float)
    return group


def aligned_indices(group, method, seed):
    rows = np.flatnonzero((group["method"] == method) & (group["seed"] == seed))
    rows = rows[np.lexsort((group["task_ids"][rows], group["image_ids"][rows]))]
    keys = list(zip(group["image_ids"][rows], group["task_ids"][rows]))
    assert len(keys) == len(set(keys))
    return rows, keys


def task_class_weights(task_ids, labels):
    """Equal tasks, half per class side, then equal images within each side."""
    counts = Counter(zip(task_ids, labels))
    tasks = set(task_ids)
    assert all(sum(t == task for t, _ in counts) == 2 for task in tasks)
    return np.asarray([1. / (2 * len(tasks) * counts[t, y]) for t, y in zip(task_ids, labels)])


def paired_bootstrap(ids, weights, values, replicates, seed, family_size=1):
    """Resample whole images, retaining their weighted numerator/denominator."""
    values = np.asarray(values, float)
    unique, inverse = np.unique(ids, return_inverse=True)
    mass = np.bincount(inverse, weights=weights)
    totals = np.zeros((len(unique), values.shape[1]))
    np.add.at(totals, inverse, values * weights[:, None])
    point = totals.sum(axis=0) / mass.sum()
    rng = np.random.default_rng(seed)
    boot = np.empty((replicates, values.shape[1]))
    for i in range(replicates):
        draw = rng.integers(len(unique), size=len(unique))
        boot[i] = totals[draw].sum(axis=0) / mass[draw].sum()
    return point, np.quantile(boot, [.025, .975], axis=0), np.quantile(
        boot, [.025 / family_size, 1 - .025 / family_size], axis=0)


def classification_audit(learned, reference, main, budget):
    seeds = main["training_seeds"]
    base_rows, keys = aligned_indices(reference, "full_model", -1)
    ids, labels, tasks = [reference[k][base_rows] for k in ("image_ids", "labels", "task_ids")]
    weights = task_class_weights(tasks, labels)
    arrays = {}
    for method in ("conditioned", "image_only"):
        assert sorted(np.unique(learned["seed"][learned["method"] == method]).tolist()) == sorted(seeds)
        parts = []
        for seed in seeds:
            rows, current_keys = aligned_indices(learned, method, seed)
            assert current_keys == keys
            np.testing.assert_array_equal(learned["labels"][rows], labels)
            parts.append(learned["outcomes"][rows])
        arrays[method] = np.asarray(parts)
    delta = arrays["conditioned"] - arrays["image_only"]
    point, interval, family_interval = paired_bootstrap(
        ids, weights, delta.mean(axis=0), main["bootstrap_replicates"],
        main["bootstrap_seed"], len(main["budgets"]))
    saved = main["task_splits"]["unseen"]["budgets"][str(budget)]
    contrasts = {}
    for j, metric in enumerate(("accuracy", "margin")):
        target = saved["conditioned_minus_image_only"][metric]
        close(point[j], target["mean"])
        close(interval[:, j], target["ci95"])
        if len(main["budgets"]) > 1:
            close(family_interval[:, j], target["bonferroni_family_ci95"])
        else:
            assert "bonferroni_family_ci95" not in target
        contrasts[metric] = {"mean": float(point[j]), "ci95": interval[:, j].tolist()}
    for method, outcomes in [("full_model", reference["outcomes"][base_rows]),
                             *[(m, a.mean(axis=0)) for m, a in arrays.items()]]:
        target = next(row for row in saved["table"] if row["method"] == method)
        close(weights @ outcomes, [target["accuracy"]["mean"], target["margin"]["mean"]])
    for seed, differences in zip(seeds, delta):
        target = next(row for row in saved["paired_seed_results"] if row["seed"] == seed)
        close(weights @ differences, [target["accuracy_difference"], target["margin_difference"]])
    return {"image_count": len(set(ids)), "image_task_rows_per_arm_and_seed": len(ids),
            "task_count": len(set(tasks)), "conditioned_minus_image_only": contrasts,
            "aggregation": "Task/class-balanced; fixed seeds averaged before paired image resampling."}


def image_weights(ids):
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    return 1. / (len(counts) * counts[inverse])


def regression_features(cases):
    lo = np.minimum(cases["base_margin_a"], cases["base_margin_b"])
    hi = np.maximum(cases["base_margin_a"], cases["base_margin_b"])
    return np.column_stack([lo, hi, lo**2, lo*hi, hi**2, cases["angle_degrees"] / 10.])


def weighted_fit(x, y, weights, intercept=True):
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x - mean)**2, axis=0, weights=weights))
    scale[scale < 1e-12] = 1.
    design = (x - mean) / scale
    if intercept:
        design = np.column_stack([np.ones(len(x)), design])
    root = np.sqrt(weights)
    coef, _, rank, _ = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)
    return coef, int(rank), mean, scale, design


def angle_fit(cases, fixed=False):
    ids = cases["image_ids"]
    x, y = regression_features(cases), cases["exchange_loss"].astype(float).copy()
    if fixed:
        for image in np.unique(ids):
            rows = ids == image
            x[rows] -= x[rows].mean(axis=0)
            y[rows] -= y[rows].mean()
    weights = image_weights(ids)
    coef, rank, _, scale, design = weighted_fit(x, y, weights, intercept=not fixed)
    baseline_rank = np.linalg.matrix_rank(design[:, :-1] * np.sqrt(weights)[:, None])
    return {"angle_slope": float(coef[-1] / scale[-1]), "design_rank": rank,
            "design_columns": design.shape[1], "angle_estimable_beyond_controls": bool(rank > baseline_rank)}


def geometry_from_predictions(learned, reference, cases, embedding_dim):
    """Rebuild each saved pair from its raw image/task/seed identities."""
    lookup = {(str(image), str(task), int(seed), str(method)): i for i, (image, task, seed, method) in
              enumerate(zip(learned["image_ids"], learned["task_ids"], learned["seed"], learned["method"]))}
    full = {(str(image), str(task)): i for i, (image, task) in
            enumerate(zip(reference["image_ids"], reference["task_ids"]))}
    assert len(lookup) == len(learned["method"]) and len(full) == len(reference["method"])
    counts = Counter((str(i), int(s)) for i, s, m in zip(learned["image_ids"], learned["seed"], learned["method"])
                     if m == "conditioned")
    assert len(cases["image_ids"]) == sum(n * (n - 1) // 2 for n in counts.values())
    result = {key: np.empty(len(cases["image_ids"])) for key in
              ("exchange_loss", "base_margin_a", "base_margin_b", "angle_degrees", "displacement_norm")}
    max_identity_error = 0.
    text = learned["text_features"].astype(float)
    for start in range(0, len(cases["image_ids"]), 512):
        end = min(start + 512, len(cases["image_ids"]))
        keys = list(zip(cases["image_ids"][start:end], cases["task_a"][start:end],
                        cases["task_b"][start:end], cases["seed"][start:end]))
        a = [lookup[str(i), str(t), int(s), "conditioned"] for i, t, _, s in keys]
        b = [lookup[str(i), str(t), int(s), "conditioned"] for i, _, t, s in keys]
        ia = [lookup[str(i), str(t), int(s), "image_only"] for i, t, _, s in keys]
        ib = [lookup[str(i), str(t), int(s), "image_only"] for i, _, t, s in keys]
        f = [full[str(i), str(t)] for i, t, _, _ in keys]
        np.testing.assert_array_equal(learned["image_features"][ia], learned["image_features"][ib])
        y = learned["labels"][a]
        np.testing.assert_array_equal(y, learned["labels"][b])
        np.testing.assert_array_equal(y, cases["labels"][start:end])
        directions = []
        for rows, field in ((a, "competitor_a"), (b, "competitor_b")):
            pairs = learned["class_pairs"][rows]
            competitor = np.where(y == pairs[:, 0], pairs[:, 1], pairs[:, 0])
            np.testing.assert_array_equal(competitor, cases[field][start:end])
            direction = text[y] - text[competitor]
            directions.append(direction / np.linalg.norm(direction, axis=1)[:, None])
        ua, ub = directions
        delta = learned["image_features"][a].astype(float) - learned["image_features"][b]
        z_full = reference["image_features"][f].astype(float)
        assert delta.shape[1] == embedding_dim
        direction_delta = ua - ub
        loss = .5 * np.sum(delta * direction_delta, axis=1)
        displacement = np.linalg.norm(delta, axis=1)
        separation = np.linalg.norm(direction_delta, axis=1)
        valid = (displacement > 1e-12) & (separation > 1e-12)
        alignment = np.zeros(len(loss))
        alignment[valid] = np.sum(delta[valid] * direction_delta[valid], axis=1) / (displacement[valid] * separation[valid])
        reconstructed = displacement * (separation / 2) * alignment
        max_identity_error = max(max_identity_error, float(np.max(np.abs(loss - reconstructed))))
        result["exchange_loss"][start:end] = loss
        result["base_margin_a"][start:end] = np.sum(z_full * ua, axis=1)
        result["base_margin_b"][start:end] = np.sum(z_full * ub, axis=1)
        result["angle_degrees"][start:end] = np.degrees(np.arccos(np.clip(np.sum(ua * ub, axis=1), -1, 1)))
        result["displacement_norm"][start:end] = displacement
    for key, value in result.items():
        close(value, cases[key])
    close(result["exchange_loss"], cases["exchange_loss_from_scores"], atol=1e-6)
    return {"embedding_dim": embedding_dim, "per_seed_pair_count": len(cases["image_ids"]),
            "identity_max_abs_error": max_identity_error, "image_only_exchange_max_abs_difference": 0.}


def held_image_audit(cases, saved_predictions, target, fold_seed):
    ids, labels = cases["image_ids"], cases["labels"]
    x, y = regression_features(cases), cases["exchange_loss"]
    unique, first = np.unique(ids, return_index=True)
    rng, assignments = np.random.default_rng(fold_seed), {}
    for label in sorted(np.unique(labels[first])):
        images = unique[labels[first] == label].copy()
        rng.shuffle(images)
        assignments.update({str(image): i % target["fold_count"] for i, image in enumerate(images)})
    fold = np.asarray([assignments[str(image)] for image in ids])
    assert assignments == target["fold_assignment"]
    np.testing.assert_array_equal(fold, saved_predictions["fold"])
    predictions = {k: np.empty(len(ids)) for k in ("difficulty_only", "difficulty_plus_angle", "training_mean")}
    for f in range(target["fold_count"]):
        train, val = fold != f, fold == f
        weights = image_weights(ids[train])
        coef, _, mean, scale, design = weighted_fit(x[train], y[train], weights)
        validation = np.column_stack([np.ones(val.sum()), (x[val] - mean) / scale])
        root = np.sqrt(weights)
        base = np.linalg.lstsq(design[:, :-1] * root[:, None], y[train] * root, rcond=None)[0]
        predictions["difficulty_plus_angle"][val] = validation @ coef
        predictions["difficulty_only"][val] = validation[:, :-1] @ base
        predictions["training_mean"][val] = np.average(y[train], weights=weights)
    for key, value in predictions.items():
        close(value, saved_predictions[key])
    weights = image_weights(ids)
    base, angle, reference = [float(weights @ (y - predictions[k])**2) for k in
                              ("difficulty_only", "difficulty_plus_angle", "training_mean")]
    r2 = (base - angle) / reference
    close([base, angle, r2], [target["difficulty_only_mse"], target["angle_augmented_mse"],
                             target["incremental_out_of_fold_r2"]])
    return {"independent_refit_matches_saved_predictions": True, "difficulty_only_mse": base,
            "angle_augmented_mse": angle, "incremental_out_of_fold_r2": r2}


def geometry_audit(learned, reference, folder, summary, budget, embedding_dim):
    per_seed = read_npz(folder / "per_seed_cases.npz")
    averaged = read_npz(folder / "seed_mean_cases.npz")
    findings = geometry_from_predictions(learned, reference, per_seed, embedding_dim)
    keys = list(zip(averaged["image_ids"], averaged["task_a"], averaged["task_b"]))
    assert len(keys) == len(set(keys))
    assert sorted(np.unique(per_seed["seed"]).tolist()) == sorted(summary["training_seeds"])
    losses = []
    target = summary["budgets"][str(budget)]
    for seed in summary["training_seeds"]:
        rows = np.flatnonzero(per_seed["seed"] == seed)
        rows = rows[np.lexsort((per_seed["task_b"][rows], per_seed["task_a"][rows], per_seed["image_ids"][rows]))]
        assert list(zip(*[per_seed[k][rows] for k in ("image_ids", "task_a", "task_b")])) == keys
        for key in ("labels", "competitor_a", "competitor_b", "keep_patch_tokens"):
            np.testing.assert_array_equal(per_seed[key][rows], averaged[key])
        for key in ("base_margin_a", "base_margin_b", "angle_degrees"):
            close(per_seed[key][rows], averaged[key])
        losses.append(per_seed["exchange_loss"][rows])
        subset = {k: value[rows] for k, value in per_seed.items()}
        saved_seed = next(item for item in target["per_seed"] if item["seed"] == seed)
        close(angle_fit(subset)["angle_slope"], saved_seed["difficulty_adjusted"]["angle_slope"])
    close(np.mean(losses, axis=0), averaged["exchange_loss"])
    np.testing.assert_array_equal(averaged["seed_count"], len(losses))
    for name, fixed in (("primary_difficulty_adjusted", False), ("image_fixed_effects_sensitivity", True)):
        fit, saved = angle_fit(averaged, fixed), target[name]
        for key, value in fit.items():
            close(value, saved[key])
        assert saved["family_size"] == summary["family_size"]
        close(saved["familywise_interval_level"], 1 - .05 / summary["family_size"])
        if summary["family_size"] == 1:
            close(saved["ci95"], saved["ci_familywise_budgets"])
        assert saved["positive_after_budget_adjustment"] == (
            fit["angle_estimable_beyond_controls"] and saved["ci_familywise_budgets"][0] > 0)
        findings[name] = {**fit, "reported_ci95": saved["ci95"]}
    cv = read_npz(folder / "cross_fitted_predictions.npz")
    for key in ("image_ids", "task_a", "task_b", "exchange_loss"):
        np.testing.assert_array_equal(cv[key], averaged[key])
    findings["held_image_prediction"] = held_image_audit(averaged, cv, target["held_image_prediction"], summary["fold_seed"])
    return findings


def run(root, results, output, budget=128, embedding_dim=768, source=None):
    main = json.loads((results / "analysis.json").read_text())
    geometry = json.loads((results / "geometry" / "analysis.json").read_text())
    source = source or root / main["sources"][str(budget)]["evaluation"]
    assert main["training_seeds"] == geometry["training_seeds"]
    assert geometry["family_size"] == len(main["budgets"])
    learned = prediction_group(source / "full" / "predictions.npz", embedding_dim)
    reference = prediction_group(source / "references" / "predictions.npz", embedding_dim, reference=True)
    np.testing.assert_array_equal(learned["text_features"], reference["text_features"])
    report = {"status": "passed", "budget": budget, "source": str(source),
              "audit_scope": "Independent raw-outcome reconstruction, weighted image-bootstrap contrasts, full-feature geometry, H2 point slopes and held-image refits.",
              "not_recomputed": "H2 slope-bootstrap and held-image R2 intervals; slope interval levels and decision consistency are checked.",
              "classification": classification_audit(learned, reference, main, budget),
              "h2": geometry_audit(learned, reference, results / "geometry" / f"budget_{budget}",
                                   geometry, budget, embedding_dim)}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--source", type=Path, help="Override saved raw-evaluation path, for archive verification.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=768)
    args = parser.parse_args()
    results = args.root / args.results
    report = run(args.root, results, args.output or results / "scientific_audit.json",
                 args.budget, args.embedding_dim, args.source)
    print(json.dumps({"status": report["status"], "classification": report["classification"],
                      "h2": report["h2"]["primary_difficulty_adjusted"]}, indent=2))
