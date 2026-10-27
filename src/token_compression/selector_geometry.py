"""Difficulty-adjusted task geometry for paired token-selection exchanges.

The symmetric exchange loss obeys an exact identity, not a new mechanism:
L = 0.5 * (z_A-z_B) dot (u_A-u_B)
  = ||z_A-z_B|| * sin(theta/2) * alignment.
The primary analysis adds angle to a fixed quadratic model of both uncompressed
signed margins. All uncertainty is clustered by image, conditional on the
observed task collection and training seeds.
"""

import numpy as np


CONTROL_NAMES = ["margin_min", "margin_max", "margin_min_squared",
                 "margin_min_times_max", "margin_max_squared"]


def exchange_geometry(z_a, z_b, z_full, text, labels, competitor_a, competitor_b):
    """Return full-dimensional exchange geometry without class-score projection."""
    z_a, z_b, z_full, text = [np.asarray(x, dtype=np.float64)
                             for x in (z_a, z_b, z_full, text)]
    labels, competitor_a, competitor_b = [np.asarray(x, dtype=np.int64)
                                          for x in (labels, competitor_a, competitor_b)]
    d_a, d_b = text[labels] - text[competitor_a], text[labels] - text[competitor_b]
    norm_a, norm_b = np.linalg.norm(d_a, axis=1), np.linalg.norm(d_b, axis=1)
    assert np.all(norm_a > 0) and np.all(norm_b > 0)
    u_a, u_b = d_a / norm_a[:, None], d_b / norm_b[:, None]
    direction_delta, displacement = u_a - u_b, z_a - z_b
    direction_norm = np.linalg.norm(direction_delta, axis=1)
    displacement_norm = np.linalg.norm(displacement, axis=1)
    dot = np.einsum("ij,ij->i", displacement, direction_delta)
    alignment = np.full(len(dot), np.nan)
    valid = (displacement_norm > 1e-12) & (direction_norm > 1e-12)
    alignment[valid] = dot[valid] / (displacement_norm[valid] * direction_norm[valid])
    cosines = np.clip(np.einsum("ij,ij->i", u_a, u_b), -1., 1.)
    angle = np.degrees(np.arccos(cosines))
    loss_a = np.einsum("ij,ij->i", displacement, u_a)
    loss_b = -np.einsum("ij,ij->i", displacement, u_b)
    loss = (loss_a + loss_b) / 2
    reconstructed = displacement_norm * (direction_norm / 2) * np.nan_to_num(alignment)
    return {
        "angle_degrees": angle,
        "base_margin_a": np.einsum("ij,ij->i", z_full, u_a),
        "base_margin_b": np.einsum("ij,ij->i", z_full, u_b),
        "direction_norm_a": norm_a, "direction_norm_b": norm_b,
        "sin_half_angle": direction_norm / 2,
        "displacement_norm": displacement_norm,
        "alignment": alignment, "alignment_valid": valid,
        "exchange_loss": loss, "loss_a": loss_a, "loss_b": loss_b,
        "identity_error": np.abs(loss - reconstructed),
        "feature_displacement": displacement,
        "direction_displacement": direction_delta,
    }


def image_weights(image_ids):
    """Equal total mass per image, equal mass per pair within an image."""
    _, inverse, counts = np.unique(image_ids, return_inverse=True, return_counts=True)
    return 1.0 / (len(counts) * counts[inverse])


def difficulty_features(margin_a, margin_b):
    """Symmetric quadratic controls retain both signed, continuous margins."""
    lo, hi = np.minimum(margin_a, margin_b), np.maximum(margin_a, margin_b)
    return np.column_stack([lo, hi, lo * lo, lo * hi, hi * hi])


def demean_by_image(values, image_ids):
    values = np.asarray(values, dtype=float)
    vector = values.ndim == 1
    if vector:
        values = values[:, None]
    _, inverse, counts = np.unique(image_ids, return_inverse=True, return_counts=True)
    means = np.zeros((len(counts), values.shape[1]))
    np.add.at(means, inverse, values)
    result = values - (means / counts[:, None])[inverse]
    return result[:, 0] if vector else result


def _scaled_design(features, weights, intercept=True, scaling=None):
    if scaling is None:
        mean = np.sum(weights[:, None] * features, axis=0) / weights.sum()
        scale = np.sqrt(np.sum(weights[:, None] * (features - mean) ** 2, axis=0) / weights.sum())
        scale[scale < 1e-12] = 1.
    else:
        mean, scale = scaling
    design = (features - mean) / scale
    if intercept:
        design = np.column_stack([np.ones(len(features)), design])
    return design, (mean, scale)


def _fit(design, target, weights):
    root = np.sqrt(weights)
    coefficients, _, rank, singular = np.linalg.lstsq(design * root[:, None], target * root, rcond=None)
    return coefficients, int(rank), singular


def adjusted_angle_association(cases, outcome="exchange_loss", replicates=2000,
                               seed=2061, image_fixed_effects=False):
    """Estimate angle slope and image-bootstrap intervals under fixed controls.

    95% intervals are pointwise. 98.333...% intervals use a Bonferroni correction
    for the three predeclared token budgets; only primary loss slopes use them
    for the study's directional H2 assessment.
    """
    ids = np.asarray(cases["image_ids"])
    target = np.asarray(cases[outcome], dtype=float)
    features = np.column_stack([difficulty_features(cases["base_margin_a"], cases["base_margin_b"]),
                                np.asarray(cases["angle_degrees"]) / 10.])
    assert np.all(np.isfinite(target)) and np.all(np.isfinite(features))
    weights = image_weights(ids)
    if image_fixed_effects:
        features, target = demean_by_image(features, ids), demean_by_image(target, ids)
    design, (_, scale) = _scaled_design(features, weights, intercept=not image_fixed_effects)
    coefficients, rank, singular = _fit(design, target, weights)
    baseline_coefficients, baseline_rank, _ = _fit(design[:, :-1], target, weights)
    fitted, baseline_fitted = design @ coefficients, design[:, :-1] @ baseline_coefficients
    unique, inverse = np.unique(ids, return_inverse=True)
    p = design.shape[1]
    cluster_xx = np.zeros((len(unique), p, p))
    cluster_xy = np.zeros((len(unique), p))
    np.add.at(cluster_xx, inverse, weights[:, None, None] * design[:, :, None] * design[:, None, :])
    np.add.at(cluster_xy, inverse, weights[:, None] * design * target[:, None])
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replicates)
    deficient = 0
    for start in range(0, replicates, 100):
        n = min(100, replicates - start)
        draws = rng.integers(0, len(unique), (n, len(unique)))
        counts = np.zeros((n, len(unique)))
        np.add.at(counts, (np.repeat(np.arange(n), len(unique)), draws.ravel()), 1.)
        xx = np.einsum("bi,ijk->bjk", counts, cluster_xx)
        xy = counts @ cluster_xy
        for j in range(n):
            value, _, boot_rank, _ = np.linalg.lstsq(xx[j], xy[j], rcond=1e-12)
            deficient += boot_rank < p
            bootstrap[start + j] = value[-1] / scale[-1]
    point = float(coefficients[-1] / scale[-1])
    lo95, hi95 = np.percentile(bootstrap, [2.5, 97.5])
    lo_family, hi_family = np.percentile(bootstrap, [100 * .05 / (2 * 3), 100 * (1 - .05 / (2 * 3))])
    residual_angle = features[:, -1] - design[:, :-1] @ _fit(design[:, :-1], features[:, -1], weights)[0]
    residual_sd = float(np.sqrt(np.sum(weights * residual_angle ** 2)))
    weighted_mean = np.sum(weights * target)
    total = float(np.sum(weights * (target - weighted_mean) ** 2))
    mse = float(np.sum(weights * (target - fitted) ** 2))
    baseline_mse = float(np.sum(weights * (target - baseline_fitted) ** 2))
    return {
        "outcome": outcome, "image_fixed_effects": bool(image_fixed_effects),
        "image_count": len(unique), "pair_count": len(ids),
        "controls": CONTROL_NAMES, "angle_unit": "10 degrees",
        "angle_slope": point, "ci95": [float(lo95), float(hi95)],
        "ci_familywise_three_budgets": [float(lo_family), float(hi_family)],
        "familywise_interval_level": 1 - .05 / 3,
        "angle_estimable_beyond_controls": bool(rank > baseline_rank),
        "positive_after_three_budget_adjustment": bool(rank > baseline_rank and lo_family > 0),
        "design_rank": rank, "design_columns": p,
        "scaled_design_condition_number": float(singular[0] / singular[-1]) if singular[-1] > 0 else None,
        "bootstrap_rank_deficient_replicates": int(deficient),
        "residual_angle_standard_deviation_units10degrees": residual_sd,
        "within_image_angle_varying_images": int(sum(np.ptp(features[ids == i, -1]) > 1e-10 for i in unique)),
        "weighted_mse": mse, "difficulty_only_weighted_mse": baseline_mse,
        "in_sample_incremental_r2": (baseline_mse - mse) / total if total > 0 else None,
    }


def image_fold_assignments(image_ids, labels, folds=5, seed=2060):
    """Class-stratified fixed image folds, shared across seeds and token budgets."""
    unique, first = np.unique(image_ids, return_index=True)
    image_labels = np.asarray(labels)[first]
    rng = np.random.default_rng(seed)
    assignments = {}
    for label in sorted(np.unique(image_labels)):
        images = unique[image_labels == label].copy()
        rng.shuffle(images)
        assignments.update({str(image_id): i % folds for i, image_id in enumerate(images)})
    return np.asarray([assignments[str(i)] for i in image_ids], dtype=np.int8)


def cross_validated_angle_prediction(cases, outcome="exchange_loss", folds=5, seed=2060,
                                     replicates=2000, bootstrap_seed=2061):
    """Held-image prediction gain within a fixed development-task collection.

    Models and preprocessing are fitted on four folds only. Bootstrap intervals
    resample images in fixed out-of-fold predictions; they do not refit the CV
    models or represent final-test confirmation.
    """
    ids, target = np.asarray(cases["image_ids"]), np.asarray(cases[outcome], float)
    features = np.column_stack([difficulty_features(cases["base_margin_a"], cases["base_margin_b"]),
                                np.asarray(cases["angle_degrees"]) / 10.])
    fold = image_fold_assignments(ids, cases["labels"], folds, seed)
    base_prediction, angle_prediction, mean_prediction = [np.empty(len(ids)) for _ in range(3)]
    for f in range(folds):
        train, validation = fold != f, fold == f
        weights = image_weights(ids[train])
        train_x, scaling = _scaled_design(features[train], weights)
        validation_x, _ = _scaled_design(features[validation], np.ones(validation.sum()), scaling=scaling)
        angle_prediction[validation] = validation_x @ _fit(train_x, target[train], weights)[0]
        base_prediction[validation] = validation_x[:, :-1] @ _fit(train_x[:, :-1], target[train], weights)[0]
        mean_prediction[validation] = np.sum(weights * target[train])
    weights = image_weights(ids)
    squared_base, squared_angle = (target - base_prediction) ** 2, (target - angle_prediction) ** 2
    reference_squared = (target - mean_prediction) ** 2
    improvement = squared_base - squared_angle
    unique, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    per_image = np.zeros((len(unique), 4))
    np.add.at(per_image, inverse, np.column_stack([squared_base, squared_angle, improvement, reference_squared]))
    per_image /= counts[:, None]
    point = per_image.mean(axis=0)
    rng = np.random.default_rng(bootstrap_seed)
    boot = np.empty((replicates, 2))
    for start in range(0, replicates, 100):
        n = min(100, replicates - start)
        draw = rng.integers(0, len(unique), (n, len(unique)))
        value = per_image[draw].mean(axis=1)
        boot[start:start + n, 0] = value[:, 2]
        boot[start:start + n, 1] = np.divide(value[:, 2], value[:, 3], out=np.zeros(n), where=value[:, 3] > 1e-24)
    interval = np.percentile(boot, [2.5, 97.5], axis=0)
    return {
        "fold_count": folds, "fold_seed": seed, "image_count": len(unique), "pair_count": len(ids),
        "difficulty_only_mse": float(point[0]), "angle_augmented_mse": float(point[1]),
        "mse_improvement": float(point[2]), "mse_improvement_ci95": interval[:, 0].tolist(),
        "incremental_out_of_fold_r2": float(point[2] / point[3]) if point[3] > 1e-24 else None,
        "incremental_out_of_fold_r2_ci95": interval[:, 1].tolist() if point[3] > 1e-24 else None,
        "prediction_uncertainty_scope": "Image resampling of fixed cross-fitted predictions; model-fitting variability not resampled.",
        "fold_assignment": {str(i): int(fold[np.flatnonzero(ids == i)[0]]) for i in unique},
        "out_of_fold_predictions": {"difficulty_only": base_prediction,
                                    "difficulty_plus_angle": angle_prediction,
                                    "training_mean": mean_prediction, "fold": fold},
    }
