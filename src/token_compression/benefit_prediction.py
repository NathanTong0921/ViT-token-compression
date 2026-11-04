"""Small weighted regressions for held-image, held-task benefit prediction."""

import numpy as np


def fit_weighted_ridge(X, y, weights, ridge=1e-3):
    """Fit weighted MSE + ridge * ||beta||² after training-only standardization.

    Weights are normalized to sum to one, so repeating observations does not
    change the penalty strength. The intercept is unpenalized. A one-dimensional
    target produces a vector beta and scalar intercept; multiple targets produce
    a coefficient matrix and intercept vector.
    """
    X, y, weights = (np.asarray(value, dtype=np.float64)
                     for value in (X, y, weights))
    weights = weights / weights.sum()
    mean = weights @ X
    constant = np.ptp(X, axis=0) == 0
    mean[constant] = X[0, constant]
    centered = X - mean
    scale = np.sqrt(weights @ (centered * centered))
    scale[constant] = 1.0
    standardized = centered / scale
    intercept = weights @ y
    target = y - intercept
    gram = standardized.T @ (weights[:, None] * standardized)
    gram += ridge * np.eye(X.shape[1])
    weighted_target = weights * target if y.ndim == 1 else weights[:, None] * target
    beta = np.linalg.solve(gram, standardized.T @ weighted_target)
    return {"mean": mean, "scale": scale, "beta": beta,
            "intercept": intercept, "ridge": float(ridge)}


def predict_ridge(fit, X):
    """Predict using the exact feature transform fitted on training rows."""
    standardized = (np.asarray(X, dtype=np.float64) - fit["mean"]) / fit["scale"]
    return standardized @ fit["beta"] + fit["intercept"]


def _group_folds(values, nfolds, seed):
    groups, inverse = np.unique(values, return_inverse=True)
    order = np.random.default_rng(seed).permutation(len(groups))
    assignments = np.empty(len(groups), dtype=np.int64)
    assignments[order] = np.arange(len(groups)) % nfolds
    return assignments[inverse]


def two_way_folds(image_ids, task_ids, nfolds=5, image_seed=2351, task_seed=2352):
    """Assign image and task groups independently; repeated seed rows stay together."""
    return (_group_folds(image_ids, nfolds, image_seed),
            _group_folds(task_ids, nfolds, task_seed))


def iter_two_way_splits(image_fold, task_fold, nfolds=5):
    """Each test cell holds out both its images and its tasks from fitting."""
    image_fold, task_fold = np.asarray(image_fold), np.asarray(task_fold)
    for i in range(nfolds):
        for j in range(nfolds):
            train = (image_fold != i) & (task_fold != j)
            test = (image_fold == i) & (task_fold == j)
            yield i, j, train, test


def weighted_center(values, group_ids, weights):
    """Subtract each group's weighted mean while preserving the input shape."""
    values, weights = np.asarray(values, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    _, inverse = np.unique(group_ids, return_inverse=True)
    totals = np.bincount(inverse, weights=weights)
    if values.ndim == 1:
        sums = np.bincount(inverse, weights=weights * values)
        return values - (sums / totals)[inverse]
    sums = np.zeros((len(totals), values.shape[1]))
    np.add.at(sums, inverse, weights[:, None] * values)
    return values - (sums / totals[:, None])[inverse]
