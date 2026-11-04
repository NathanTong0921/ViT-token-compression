"""Positive logit-scale diagnostics for uniform retained-token targets.

Scaling by a positive scalar preserves the entire token ranking. The only fitted
quantity here is that scalar; selectors and target masks are held fixed.
"""

import numpy as np
from scipy.optimize import brentq


def _prepare(logits, mask_indices):
    centered = np.asarray(logits, dtype=np.float64)
    centered = centered - centered.mean(axis=1, keepdims=True)
    masks = np.asarray(mask_indices)
    target_mean = np.take_along_axis(centered, masks, axis=1).mean(axis=1)
    return centered, target_mean, masks.shape[1]


def _row_kl(centered, target_mean, k, alpha):
    scaled = alpha * centered
    maxima = scaled.max(axis=1)
    log_normalizer = maxima + np.log(np.exp(scaled - maxima[:, None]).sum(axis=1))
    return log_normalizer - alpha * target_mean - np.log(k)


def _mean_derivative(centered, target_mean, weights, alpha):
    scaled = alpha * centered
    scaled -= scaled.max(axis=1, keepdims=True)
    probabilities = np.exp(scaled)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return float(weights @ ((probabilities * centered).sum(axis=1) - target_mean))


def row_kl(logits, mask_indices, alpha=1.0):
    """Return KL(uniform mask || softmax(alpha * logits)) for each row.

    Inputs are logits [rows, patches] and distinct mask indices [rows, k]. A
    positive alpha changes probability concentration without changing ranking.
    Row means are removed for numerical stability; softmax is shift invariant.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive to preserve token ranking")
    return _row_kl(*_prepare(logits, mask_indices), alpha)


def mean_kl_derivative(logits, mask_indices, weights, alpha):
    """Derivative of the weighted mean KL with respect to positive alpha."""
    if alpha <= 0:
        raise ValueError("alpha must be positive to preserve token ranking")
    centered, target_mean, _ = _prepare(logits, mask_indices)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    return _mean_derivative(centered, target_mean, weights, alpha)


def fit_global_scale(logits, mask_indices, weights, bounds=(1e-4, 1e4)):
    """Minimize weighted mean KL over one positive scale shared by all rows.

    KL is convex in alpha: its second derivative is the weighted softmax
    variance of the logits. Endpoint derivative signs therefore identify a
    constrained boundary optimum; otherwise Brent's method finds the derivative
    root in log-alpha coordinates. The returned boundary flag is None, 'lower',
    or 'upper'; nfev counts derivative evaluations. No labels are used.
    """
    lower, upper = bounds
    if not 0 < lower < upper:
        raise ValueError("bounds must satisfy 0 < lower < upper")
    centered, target_mean, k = _prepare(logits, mask_indices)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    nfev = 0

    def derivative(log_alpha):
        nonlocal nfev
        nfev += 1
        return _mean_derivative(centered, target_mean, weights, np.exp(log_alpha))

    lo, hi = np.log(lower), np.log(upper)
    if derivative(lo) >= 0:
        alpha, boundary = lower, "lower"
    elif derivative(hi) <= 0:
        alpha, boundary = upper, "upper"
    else:
        alpha = float(np.exp(brentq(derivative, lo, hi, xtol=1e-11, rtol=1e-12)))
        boundary = None
    return {
        "alpha": float(alpha),
        "loss": float(weights @ _row_kl(centered, target_mean, k, alpha)),
        "boundary": boundary,
        "nfev": nfev,
    }
