"""Additive patch utilities fitted to masks at a specified token budget."""

from math import comb

import numpy as np


def sample_masks(num_patches, keep_count, count, seed, exclude=None):
    """Sample distinct, uniform fixed-size masks, optionally excluding a bank.

    Rejection of duplicate masks gives sampling without replacement. The exact
    combination count also covers small toy designs where a requested bank may
    exhaust the available masks.
    """
    if not 0 <= keep_count <= num_patches or count < 0:
        raise ValueError("Require 0 <= keep_count <= num_patches and count >= 0")
    seen = set()
    if exclude is not None:
        excluded = np.asarray(exclude, dtype=bool)
        if excluded.ndim != 2 or excluded.shape[1] != num_patches:
            raise ValueError("Excluded masks must have shape (N, num_patches)")
        # Masks at other budgets cannot overlap the requested fixed-size bank.
        seen.update(np.packbits(row).tobytes() for row in excluded
                    if np.count_nonzero(row) == keep_count)
    if count > comb(num_patches, keep_count) - len(seen):
        raise ValueError("Requested more distinct masks than remain available")
    rng = np.random.default_rng(seed)
    masks = np.zeros((count, num_patches), dtype=bool)
    index = 0
    while index < count:
        mask = np.zeros(num_patches, dtype=bool)
        mask[rng.choice(num_patches, keep_count, replace=False)] = True
        key = np.packbits(mask).tobytes()
        if key not in seen:
            masks[index] = mask
            seen.add(key)
            index += 1
    return masks


def ridge_projector(masks, ridge=0.01):
    """Precompute a shared linear fit for many images and class responses.

    Columns are centered and standardized before solving the ridge objective
    ``mean squared residual + ridge * squared coefficient norm``. Returned
    coefficients are rescaled to the original 0/1 mask units. An unpenalized
    intercept absorbs the response mean. Positive ridge resolves the fixed-k
    design's unavoidable constant-direction ambiguity.
    """
    if ridge <= 0:
        raise ValueError("Fixed-size mask designs require positive ridge")
    values = np.asarray(masks, dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    if np.any(std == 0):
        raise ValueError("Every patch must vary across the fitted mask bank")
    standardized = (values - mean) / std
    sample_count, num_patches = values.shape
    covariance = standardized.T @ standardized / sample_count
    covariance.flat[::num_patches + 1] += ridge
    projection = np.linalg.solve(covariance, standardized.T / sample_count)
    projection /= std[:, None]
    return {"projection": projection, "mask_mean": mean, "mask_std": std,
            "ridge": ridge, "sample_count": sample_count}


def fit_responses(projector, responses):
    """Fit class cosine scores of shape (masks, classes)."""
    responses = np.asarray(responses, dtype=np.float64)
    coefficients = projector["projection"] @ responses
    intercept = responses.mean(axis=0) - projector["mask_mean"] @ coefficients
    return coefficients, intercept


def predict_masks(masks, coefficients, intercept):
    """Predict each class score for masks with shape (N, patches)."""
    return np.asarray(masks, dtype=np.float64) @ coefficients + intercept


def task_coefficients(coefficients, text_features, true_class, competitors):
    """Convert class-score coefficients to true-class-oriented margin utilities.

    Columns correspond to the supplied competitors, with geometric margins
    normalized by the distance between the two class text embeddings.
    """
    coefficients = np.asarray(coefficients, dtype=np.float64)
    text_features = np.asarray(text_features, dtype=np.float64)
    competitors = np.asarray(competitors, dtype=np.int64)
    direction_norms = np.linalg.norm(
        text_features[true_class] - text_features[competitors], axis=1
    )
    return (coefficients[:, true_class, None] - coefficients[:, competitors]) / direction_norms
