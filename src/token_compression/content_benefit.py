"""Task-contrast measurements from a train-fitted prefix readout."""

import numpy as np

from token_compression.benefit_prediction import fit_weighted_ridge


FEATURE_NAMES = (
    "u_omitted_contrast_energy", "c_minus_u_retained_energy",
    "u_surrogate_gap_loss", "c_minus_u_surrogate_gap",
    "full_surrogate_abs_gap", "cu_surrogate_prediction_disagreement",
)


def fit_content_readout(inputs, targets, weights, ridge=.01):
    """Fit one affine map, with standardization and weights from training only."""
    fit = fit_weighted_ridge(inputs, targets, weights, ridge=ridge)
    matrix = fit["beta"] / fit["scale"][:, None]
    bias = fit["intercept"] - fit["mean"] @ matrix
    return {**fit, "matrix": matrix, "bias": bias}


def normalize_rows(values):
    values = np.asarray(values, dtype=np.float64)
    return values / np.linalg.norm(values, axis=-1, keepdims=True)


def content_features(patches, c_indices, u_indices, directions, matrix, bias):
    """Return six label-free features for aligned image/task rows.

    patches: [rows, spatial_tokens, width], directions: [rows, text_width]
    are unit class-a minus class-b text directions. The affine readout is fitted
    on image means, so its patch decomposition is a surrogate, not a claim of
    independently localized patch semantics.
    """
    patches = np.asarray(patches, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    means = patches.mean(axis=1)
    c_means = np.take_along_axis(patches, c_indices[:, :, None], axis=1).mean(axis=1)
    u_means = np.take_along_axis(patches, u_indices[:, :, None], axis=1).mean(axis=1)
    z_full = normalize_rows(means @ matrix + bias)
    z_c = normalize_rows(c_means @ matrix + bias)
    z_u = normalize_rows(u_means @ matrix + bias)
    full_gap = np.sum(z_full * directions, axis=1)
    c_gap = np.sum(z_c * directions, axis=1)
    u_gap = np.sum(z_u * directions, axis=1)
    orientation = np.where(full_gap >= 0, 1., -1.)

    # Centering removes the common image component and the affine intercept.
    axes = directions @ matrix.T
    patch_contrast = np.einsum("npd,nd->np", patches - means[:, None], axes)
    energy = patch_contrast ** 2
    total = energy.sum(axis=1)
    c_energy = np.take_along_axis(energy, c_indices, axis=1).sum(axis=1)
    u_energy = np.take_along_axis(energy, u_indices, axis=1).sum(axis=1)
    omitted = np.divide(total - u_energy, total, out=np.zeros_like(total), where=total > 0)
    retained_difference = np.divide(c_energy - u_energy, total,
                                   out=np.zeros_like(total), where=total > 0)
    return np.column_stack((
        omitted, retained_difference, orientation * (full_gap - u_gap),
        orientation * (c_gap - u_gap), np.abs(full_gap),
        (np.sign(c_gap) != np.sign(u_gap)).astype(float),
    ))
