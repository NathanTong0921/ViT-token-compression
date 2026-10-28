"""Finite-library specialization potential versus learned conditioning gains.

The decomposition is an accounting identity, not a mechanism discovery. The
augmented diagnostic library contains the learned selections. Prediction uses
only the separate fixed random-probe library, which does not include them.
"""

import numpy as np

from .selector_geometry import (
    CONTROL_NAMES, _fit, _scaled_design, difficulty_features,
    image_fold_assignments, image_weights,
)


def library_potential(margin_a, margin_b):
    """Search one identical finite library for separate versus shared masks."""
    a, b = np.asarray(margin_a, float), np.asarray(margin_b, float)
    assert a.shape == b.shape and a.ndim == 2
    separate = (a.max(axis=1) + b.max(axis=1)) / 2
    shared = ((a + b) / 2).max(axis=1)
    gap = separate - shared
    assert np.all(gap >= -1e-12)
    return {"separate": separate, "shared": shared, "gap": gap}


def regret_decomposition(separate, shared, conditioned, image_only):
    """Use common population/weights and a library containing both learned arms.

    Both oracle references know the true label and both tasks. They are not
    deployable models or global optima over all token subsets.
    """
    separate, shared, conditioned, image_only = [np.asarray(x, float) for x in
                                               (separate, shared, conditioned, image_only)]
    assert separate.shape == shared.shape == conditioned.shape == image_only.shape
    potential = separate - shared
    conditional_regret, shared_regret = separate - conditioned, shared - image_only
    assert np.min(potential) >= -1e-6
    assert np.min(conditional_regret) >= -1e-6
    assert np.min(shared_regret) >= -1e-6
    gain = conditioned - image_only
    difference = conditional_regret - shared_regret
    residual = gain - (potential - difference)
    np.testing.assert_allclose(residual, 0, atol=1e-12, rtol=0)
    return {"augmented_specialization_potential": potential,
            "conditioned_regret": conditional_regret,
            "image_only_regret": shared_regret,
            "regret_difference": difference,
            "learned_gain": gain, "decomposition_residual": residual}


def average_seed_cases(exchange, seeds):
    """Align task pairs exactly, then average trained arms before resampling."""
    indices = []
    for seed in seeds:
        ix = np.flatnonzero(exchange["seed"] == seed)
        assert len(ix)
        order = np.lexsort((exchange["task_b"][ix], exchange["task_a"][ix], exchange["image_ids"][ix]))
        indices.append(ix[order])
    metadata = ["image_ids", "task_a", "task_b", "labels", "competitor_a", "competitor_b",
                "keep_patch_tokens", "angle_degrees", "base_margin_a", "base_margin_b"]
    first = indices[0]
    result = {k: exchange[k][first].copy() for k in metadata}
    for ix in indices[1:]:
        for key in metadata:
            np.testing.assert_array_equal(exchange[key][ix], result[key])
    for prefix, field in [("", "conditioned"), ("image_only_", "image_only")]:
        for metric, suffix in [("margin", "margin"), ("correct", "accuracy")]:
            per_seed = np.stack([.5 * (exchange[f"{prefix}{metric}_aa"][ix].astype(float) +
                                      exchange[f"{prefix}{metric}_bb"][ix].astype(float)) for ix in indices])
            result[f"{field}_{suffix}"] = per_seed.mean(axis=0)
            for seed, values in zip(seeds, per_seed):
                result[f"{field}_{suffix}_seed_{seed}"] = values
    result["learned_accuracy_gain"] = result["conditioned_accuracy"] - result["image_only_accuracy"]
    result["seed_count"] = np.full(len(first), len(seeds), dtype=np.int8)
    return result


def image_means(ids, values):
    values = np.asarray(values, float)
    unique, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    output = np.zeros((len(unique), values.shape[1]))
    np.add.at(output, inverse, values)
    return unique, output / counts[:, None]


def bootstrap_intervals(values, family_size=2):
    alpha = .05 / family_size
    return {"ci95": np.percentile(values, [2.5, 97.5], axis=0),
            "ci_familywise_budgets": np.percentile(values, [100 * alpha / 2, 100 * (1-alpha/2)], axis=0)}


def summarize_columns(cases, columns, replicates=2000, seed=2084, family_size=2):
    unique, per_image = image_means(cases["image_ids"], np.column_stack([cases[k] for k in columns]))
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, len(columns)))
    for start in range(0, replicates, 100):
        n = min(100, replicates-start)
        ix = rng.integers(0, len(unique), (n, len(unique)))
        draws[start:start+n] = per_image[ix].mean(axis=1)
    intervals = bootstrap_intervals(draws, family_size)
    return {k: {"mean": float(per_image[:, j].mean()),
                **{name: values[:, j].tolist() for name, values in intervals.items()}}
            for j, k in enumerate(columns)}


def probe_gap_association(cases, replicates=2000, seed=2084, family_size=2):
    """Image-weighted partial association; the probe gap remains label-informed."""
    ids = np.asarray(cases["image_ids"])
    target = np.asarray(cases["learned_gain"], float)
    features = predictor_features(cases)
    weights = image_weights(ids)
    design, (_, scale) = _scaled_design(features, weights)
    coefficient, rank, singular = _fit(design, target, weights)
    _, base_rank, _ = _fit(design[:, :-1], target, weights)
    unique, inverse = np.unique(ids, return_inverse=True)
    p = design.shape[1]
    xx, xy = np.zeros((len(unique), p, p)), np.zeros((len(unique), p))
    np.add.at(xx, inverse, weights[:, None, None] * design[:, :, None] * design[:, None, :])
    np.add.at(xy, inverse, weights[:, None] * design * target[:, None])
    rng = np.random.default_rng(seed)
    boot = np.empty((replicates, p - 1))
    deficient = 0
    for start in range(0, replicates, 100):
        n = min(100, replicates-start)
        draws = rng.integers(0, len(unique), (n, len(unique)))
        multiplicity = np.zeros((n, len(unique)))
        np.add.at(multiplicity, (np.repeat(np.arange(n), len(unique)), draws.ravel()), 1.)
        bxx, bxy = np.einsum("bi,ijk->bjk", multiplicity, xx), multiplicity @ xy
        for j in range(n):
            fit, _, boot_rank, _ = np.linalg.lstsq(bxx[j], bxy[j], rcond=1e-12)
            deficient += boot_rank < p
            boot[start+j] = fit[1:] / scale
    intervals = bootstrap_intervals(boot, family_size)
    names = CONTROL_NAMES + ["angle_per_10_degrees", "probe_gap_per_0_01_margin"]
    coefficients = {name: {"estimate": float(coefficient[j+1] / scale[j]),
                           **{k: value[:, j].tolist() for k, value in intervals.items()}}
                    for j, name in enumerate(names)}
    residual_gap = features[:, -1] - design[:, :-1] @ _fit(design[:, :-1], features[:, -1], weights)[0]
    return {"outcome": "pair_mean_conditioned_minus_image_only_margin",
            "predictor_scope": "Label-informed pure-probe library diagnostic; not available for deployment.",
            "controls": CONTROL_NAMES + ["angle_per_10_degrees"],
            "coefficients": coefficients,
            "primary_gap_coefficient": coefficients[names[-1]],
            "gap_unit": "0.01 geometric margin",
            "design_rank": rank, "design_columns": p,
            "gap_estimable_beyond_controls": bool(rank > base_rank),
            "residual_gap_standard_deviation_units0_01": float(np.sqrt(np.sum(weights * residual_gap**2))),
            "scaled_design_condition_number": float(singular[0]/singular[-1]) if singular[-1] > 0 else None,
            "bootstrap_rank_deficient_replicates": int(deficient),
            "family_size": family_size, "familywise_interval_level": 1-.05/family_size,
            "positive_after_budget_adjustment": bool(rank > base_rank and intervals["ci_familywise_budgets"][0, -1] > 0)}


def predictor_features(cases):
    return np.column_stack([difficulty_features(cases["base_margin_a"], cases["base_margin_b"]),
                            np.asarray(cases["angle_degrees"], float)/10,
                            np.asarray(cases["probe_gap_512"], float)/.01])


def cross_validated_gap_prediction(cases, folds=5, seed=2060, replicates=2000,
                                   bootstrap_seed=2084, family_size=2):
    """Hold out whole images; fit all scaling and regression only on training folds."""
    ids, target = np.asarray(cases["image_ids"]), np.asarray(cases["learned_gain"], float)
    features = predictor_features(cases)
    fold = image_fold_assignments(ids, cases["labels"], folds, seed)
    base, augmented, training_mean = [np.empty(len(ids)) for _ in range(3)]
    fit_records = []
    for f in range(folds):
        train, hold = fold != f, fold == f
        assert train.any() and hold.any()
        train_ids, hold_ids = np.unique(ids[train]), np.unique(ids[hold])
        assert not set(train_ids) & set(hold_ids)
        weights = image_weights(ids[train])
        train_x, scaling = _scaled_design(features[train], weights)
        hold_x, _ = _scaled_design(features[hold], np.ones(hold.sum()), scaling=scaling)
        fit_full, rank, _ = _fit(train_x, target[train], weights)
        fit_base, base_rank, _ = _fit(train_x[:, :-1], target[train], weights)
        base[hold], augmented[hold] = hold_x[:, :-1] @ fit_base, hold_x @ fit_full
        training_mean[hold] = np.sum(weights * target[train])
        fit_records.append({"fold": f, "training_image_count": len(train_ids), "held_image_count": len(hold_ids),
                            "training_feature_mean": scaling[0].tolist(), "training_feature_scale": scaling[1].tolist(),
                            "baseline_coefficients_scaled": fit_base.tolist(), "augmented_coefficients_scaled": fit_full.tolist(),
                            "baseline_rank": base_rank, "augmented_rank": rank})
    base_error, full_error = (target-base)**2, (target-augmented)**2
    columns = np.column_stack([base_error, full_error, base_error-full_error, (target-training_mean)**2])
    unique, per_image = image_means(ids, columns)
    point = per_image.mean(axis=0)
    rng = np.random.default_rng(bootstrap_seed)
    boot = np.empty((replicates, 2))
    for start in range(0, replicates, 100):
        n = min(100, replicates-start)
        ix = rng.integers(0, len(unique), (n, len(unique)))
        values = per_image[ix].mean(axis=1)
        boot[start:start+n, 0] = values[:, 2]
        boot[start:start+n, 1] = np.divide(values[:, 2], values[:, 3], out=np.zeros(n), where=values[:, 3] > 1e-24)
    intervals = bootstrap_intervals(boot, family_size)
    return {"fold_count": folds, "fold_seed": seed, "image_count": len(unique), "pair_count": len(ids),
            "baseline": "Both full-model margins with symmetric quadratic terms plus angle.",
            "added_predictor": "Pure-probe 512-mask specialization gap; label-informed diagnostic.",
            "baseline_mse": float(point[0]), "gap_augmented_mse": float(point[1]),
            "mse_improvement": float(point[2]),
            "mse_improvement_ci95": intervals["ci95"][:, 0].tolist(),
            "mse_improvement_ci_familywise_budgets": intervals["ci_familywise_budgets"][:, 0].tolist(),
            "incremental_out_of_fold_r2": float(point[2]/point[3]) if point[3] > 1e-24 else None,
            "incremental_out_of_fold_r2_ci95": intervals["ci95"][:, 1].tolist() if point[3] > 1e-24 else None,
            "incremental_out_of_fold_r2_ci_familywise_budgets": intervals["ci_familywise_budgets"][:, 1].tolist() if point[3] > 1e-24 else None,
            "family_size": family_size, "familywise_interval_level": 1-.05/family_size,
            "prediction_uncertainty_scope": "Image bootstrap of fixed out-of-fold predictions, without model refitting.",
            "fold_assignment": {str(i): int(fold[np.flatnonzero(ids==i)[0]]) for i in unique},
            "fold_models": fit_records,
            "out_of_fold_predictions": {"baseline": base, "gap_augmented": augmented,
                                        "training_mean": training_mean, "fold": fold}}
