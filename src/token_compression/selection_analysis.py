"""Image-clustered, exploratory analysis of the offline selection pilot.

Exchange rows contain one image, two tasks sharing its true class, and one
token budget. Their accuracies describe this anchor-image panel, not a balanced
binary benchmark. Label-informed candidate searches diagnose available choices;
they do not estimate deployable selector performance.
"""

from collections import defaultdict

import numpy as np


SHARED_METHODS = ("shared_proxy", "attention", "representation", "random_0",
                  "shared_search_mean", "shared_search_maxmin")


def row_metrics(rows):
    """Return per-row paired metrics; accuracy values are fractions."""
    def column(key):
        return np.asarray([float(row[key]) for row in rows], dtype=np.float64)

    metrics = {}
    for source, metric in (("margin", "margin"), ("correct", "accuracy")):
        own = (column(f"{source}_task_a_a") + column(f"{source}_task_b_b")) / 2
        swapped = (column(f"{source}_task_b_a") + column(f"{source}_task_a_b")) / 2
        metrics[f"own_{metric}"] = own
        metrics[f"swapped_{metric}"] = swapped
        metrics[f"exchange_{metric}_loss"] = own - swapped
        searched = (column(f"{source}_task_search_a_a")
                    + column(f"{source}_task_search_b_b")) / 2
        metrics[f"task_search_own_{metric}"] = searched
        for method in SHARED_METHODS:
            shared = (column(f"{source}_{method}_a") + column(f"{source}_{method}_b")) / 2
            metrics[f"{method}_{metric}"] = shared
            metrics[f"own_minus_{method}_{metric}"] = own - shared
            if method.startswith("shared_"):
                metrics[f"task_search_own_minus_{method}_{metric}"] = searched - shared
    metrics["overlap_fraction_task_ab"] = column("overlap_fraction_task_ab")
    metrics["selection_feature_distance"] = column("selection_feature_distance")
    metrics["exchange_alignment"] = column("exchange_alignment")
    return metrics


def image_means(image_ids, values):
    """Average rows within image, returning sorted IDs and per-image values."""
    image_ids = np.asarray(image_ids)
    unique, inverse, counts = np.unique(image_ids, return_inverse=True, return_counts=True)
    values = np.asarray(values, dtype=np.float64)
    sums = np.zeros((len(unique),) + values.shape[1:], dtype=np.float64)
    np.add.at(sums, inverse, values)
    divisor = counts.reshape((-1,) + (1,) * (values.ndim - 1))
    return unique, sums / divisor


def cluster_summary(image_ids, values, bootstrap_replicates=1000, seed=2030):
    """Percentile CI from resampling images, retaining all their paired rows."""
    unique, per_image = image_means(image_ids, values)
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(unique), size=(bootstrap_replicates, len(unique)))
    boot_means = per_image[draws].mean(axis=1)
    low, high = np.quantile(boot_means, [.025, .975], axis=0)
    return {"row_count": len(image_ids), "image_count": len(unique),
            "mean": np.mean(per_image, axis=0), "ci95_low": low, "ci95_high": high}


def grouped_folds(image_ids, n_splits=5, seed=2030):
    """Assign entire images to deterministic folds, never individual rows."""
    image_ids = np.asarray(image_ids)
    unique = np.unique(image_ids)
    n_splits = min(n_splits, len(unique))
    if n_splits < 2:
        raise ValueError("Grouped cross-validation requires at least two images.")
    shuffled = np.random.default_rng(seed).permutation(unique)
    mapping = {image_id: fold for fold, group in enumerate(np.array_split(shuffled, n_splits))
               for image_id in group}
    return np.asarray([mapping[image_id] for image_id in image_ids])


def image_weights(image_ids):
    _, inverse, counts = np.unique(image_ids, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def least_squares(design, target, weights):
    root = np.sqrt(weights)
    coefficients, _, rank, _ = np.linalg.lstsq(design * root[:, None], target * root, rcond=None)
    return coefficients, int(rank)


def h2_design(rows, include_feature_distance=False):
    a = np.asarray([float(row["base_margin_a"]) for row in rows])
    b = np.asarray([float(row["base_margin_b"]) for row in rows])
    mean, difference = (a + b) / 2, np.abs(a - b)
    columns = [np.ones(len(rows)), mean, difference, mean ** 2,
               difference ** 2, mean * difference]
    names = ["intercept", "baseline_margin_mean", "baseline_margin_abs_difference",
             "baseline_margin_mean_squared", "baseline_margin_abs_difference_squared",
             "baseline_margin_mean_times_abs_difference"]
    if include_feature_distance:
        columns.append(np.asarray([float(row["selection_feature_distance"]) for row in rows]))
        names.append("selection_feature_distance")
    angle = np.asarray([float(row["angle_degrees"]) / 90 for row in rows])
    return np.column_stack(columns), angle, names


def fit_h2(rows, target, include_feature_distance=False, bootstrap_replicates=300, seed=2030):
    """Image-equal weighted OLS and image-grouped out-of-fold prediction."""
    image_ids = np.asarray([row["image_id"] for row in rows])
    controls, angle, names = h2_design(rows, include_feature_distance)
    design = np.column_stack([controls, angle])
    weights = image_weights(image_ids)
    coefficients, rank = least_squares(design, target, weights)
    control_fit, _ = least_squares(controls, target, weights)
    angle_fit, _ = least_squares(controls, angle, weights)
    residual_angle = angle - controls @ angle_fit
    residual_target = target - controls @ control_fit
    identifiable = float(np.dot(weights, residual_angle ** 2)) > 1e-12

    folds = grouped_folds(image_ids, seed=seed)
    predicted_controls, predicted_angle = np.empty(len(rows)), np.empty(len(rows))
    fold_details = []
    for fold in np.unique(folds):
        test, train = folds == fold, folds != fold
        train_weights = image_weights(image_ids[train])
        base_coef, base_rank = least_squares(controls[train], target[train], train_weights)
        full_coef, full_rank = least_squares(design[train], target[train], train_weights)
        predicted_controls[test] = controls[test] @ base_coef
        predicted_angle[test] = design[test] @ full_coef
        fold_details.append({"fold": int(fold), "test_image_count": len(np.unique(image_ids[test])),
                             "train_image_count": len(np.unique(image_ids[train])),
                             "controls_rank": base_rank, "with_angle_rank": full_rank})
    _, squared_controls = image_means(image_ids, (target - predicted_controls) ** 2)
    _, squared_angle = image_means(image_ids, (target - predicted_angle) ** 2)

    unique, inverse = np.unique(image_ids, return_inverse=True)
    rng = np.random.default_rng(seed)
    boot_coefficients, deficient = [], 0
    for _ in range(bootstrap_replicates):
        multiplicity = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))
        boot_weights = weights * multiplicity[inverse]
        estimate, boot_rank = least_squares(design, target, boot_weights)
        deficient += boot_rank < design.shape[1]
        angle_estimate, _ = least_squares(controls, angle, boot_weights)
        if np.dot(boot_weights, (angle - controls @ angle_estimate) ** 2) > 1e-12:
            boot_coefficients.append(estimate[-1])
    interval = (np.quantile(boot_coefficients, [.025, .975]).tolist()
                if boot_coefficients and identifiable else None)
    result = {
        "status": "exploratory", "row_count": len(rows), "image_count": len(unique),
        "estimator": "least squares with inverse within-image row-count weights",
        "control_columns": names, "angle_scale": "angle_degrees / 90",
        "coefficients": {name: float(value) for name, value in zip(names + ["angle_per_90_degrees"], coefficients)},
        "design_rank": rank, "design_column_count": design.shape[1],
        "rank_deficient": rank < design.shape[1],
        "angle_identifiable_after_controls": bool(identifiable),
        "angle_coefficient": float(coefficients[-1]) if identifiable else None,
        "angle_coefficient_ci95": interval,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_identifiable_replicates": len(boot_coefficients),
        "bootstrap_rank_deficient_replicates": int(deficient),
        "cv_seed": seed, "cv_folds": len(fold_details), "fold_details": fold_details,
        "cv_controls_mse": float(np.mean(squared_controls)),
        "cv_controls_plus_angle_mse": float(np.mean(squared_angle)),
        "cv_mse_reduction_from_angle": float(np.mean(squared_controls - squared_angle)),
        "interpretation": (
            "Descriptive association, not causality. Rank-deficient fits use the least-squares "
            "minimum-norm solution. Repeated task rows are clustered by image. "
            "Cross-validation holds out images, not unseen class combinations."
        ),
    }
    return result, residual_angle, residual_target


def summarize_single_tasks(rows):
    """Separate anchor-panel summaries from tasks observed on both class sides."""
    groups = defaultdict(list)
    for row in rows:
        groups[(int(row["keep_patch_tokens"]), row["method"])].append(row)
    results = []
    for (budget, method), selected in sorted(groups.items()):
        by_task = defaultdict(list)
        for row in selected:
            by_task[row["task_id"]].append(row)
        balanced = []
        for task_rows in by_task.values():
            by_class = defaultdict(list)
            for row in task_rows:
                by_class[int(row["class_id"])].append(float(row["correct"]))
            if len(by_class) == 2:
                balanced.append(np.mean([np.mean(values) for values in by_class.values()]))
        results.append({
            "keep_patch_tokens": budget, "method": method, "row_count": len(selected),
            "image_count": len({row["image_id"] for row in selected}), "task_count": len(by_task),
            "panel_accuracy_micro": float(np.mean([float(row["correct"]) for row in selected])),
            "panel_accuracy_task_macro": float(np.mean([
                np.mean([float(row["correct"]) for row in task_rows]) for task_rows in by_task.values()])),
            "two_sided_task_count": len(balanced),
            "two_sided_task_macro_balanced_accuracy": float(np.mean(balanced)) if balanced else None,
        })
    return results


def analyze_rows(rows, bootstrap_replicates=1000, h2_bootstrap_replicates=300, seed=2030):
    by_budget = defaultdict(list)
    for row in rows:
        by_budget[int(row["keep_patch_tokens"])].append(row)
    results = []
    for budget, selected in sorted(by_budget.items()):
        metrics = row_metrics(selected)
        names = list(metrics)
        ids = [row["image_id"] for row in selected]
        aggregate = cluster_summary(ids, np.column_stack(list(metrics.values())), bootstrap_replicates, seed)
        summaries = {name: {"mean": float(aggregate["mean"][i]),
                            "ci95": [float(aggregate["ci95_low"][i]), float(aggregate["ci95_high"][i])]}
                     for i, name in enumerate(names)}
        raw, _, _ = fit_h2(selected, metrics["exchange_margin_loss"], False, h2_bootstrap_replicates, seed)
        distance_adjusted, _, _ = fit_h2(selected, metrics["exchange_margin_loss"], True, h2_bootstrap_replicates, seed)
        # Zero displacement gives an undefined alignment, recorded as 0 by the
        # runner. Exclude those rows from the normalized association analysis.
        aligned_rows = [row for row in selected if float(row["selection_feature_distance"]) > 1e-12
                        and abs(np.sin(np.deg2rad(float(row["angle_degrees"])) / 2)) > 1e-12]
        if len({row["image_id"] for row in aligned_rows}) >= 2:
            aligned_target = np.asarray([float(row["exchange_alignment"]) for row in aligned_rows])
            alignment, _, _ = fit_h2(aligned_rows, aligned_target, False, h2_bootstrap_replicates, seed)
        else:
            alignment = {"status": "insufficient nonzero-displacement images", "row_count": len(aligned_rows)}
        results.append({
            "keep_patch_tokens": budget, "row_count": len(selected), "image_count": aggregate["image_count"],
            "metrics": summaries,
            "max_abs_projection_identity_residual": float(max(abs(float(row["exchange_projection_residual"])) for row in selected)),
            "h2": {"raw_exchange_loss": raw, "feature_distance_adjusted_exchange_loss": distance_adjusted,
                   "normalized_exchange_alignment": alignment,
                   "alignment_excluded_zero_scale_rows": len(selected) - len(aligned_rows)},
        })
    return {
        "aggregation": "mean within each image, then equal-weight mean across images",
        "interval": "paired image-cluster bootstrap percentile 95% interval",
        "bootstrap_replicates": bootstrap_replicates, "bootstrap_seed": seed,
        "accuracy_scope": "Shared-anchor image panel; not the full balanced binary benchmark.",
        "selection_scope": "Label-informed offline diagnostics, not deployable predictions or proof of learnability.",
        "h2_geometry": (
            "Exchange loss = 0.5 * (z_A-z_B) dot (d_A-d_B), bounded in absolute value by "
            "||z_A-z_B|| * sin(theta/2). Normalized alignment removes this mechanical scale "
            "where nonzero; neither normalization nor regression establishes causality."
        ),
        "budgets": results,
    }
