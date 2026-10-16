"""Classification geometry for unit-normalized CLIP embeddings."""

import numpy as np


def binary_margin(cosine_scores, text_features, true_class, competitor):
    """Return the signed cosine gap and distance to the binary decision plane.

    The separating direction points toward the true class, independently of
    the order in which the two candidates are stored. Negative means wrong.
    """
    direction = text_features[true_class] - text_features[competitor]
    gap = cosine_scores[true_class] - cosine_scores[competitor]
    return float(gap), float(gap / np.linalg.norm(direction))


def direction_angle(text_features, anchor, competitor_a, competitor_b):
    """Angle in degrees between two tasks sharing the same true-class anchor."""
    a = text_features[anchor] - text_features[competitor_a]
    b = text_features[anchor] - text_features[competitor_b]
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
