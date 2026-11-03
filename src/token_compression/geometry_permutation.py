"""Within-image task-assignment references for normalized exchange alignment.

Each draw permutes the existing task-conditioned feature vectors, preserving the
complete graph of feature displacements while changing their correspondence to
fixed text directions. Seed features are averaged before computing alignment;
their mean is never renormalized. This module does not load or select data.
"""

from dataclasses import dataclass
import json

import numpy as np

from .selector_geometry import (
    _scaled_design, demean_by_image, difficulty_features, image_weights,
)


STATISTIC_NAMES = ("difficulty_adjusted_slope", "image_fixed_effects_slope",
                   "mean_alignment")


@dataclass
class ImageGeometry:
    image_id: str
    namespace: str
    task_ids: np.ndarray
    mean_features: np.ndarray
    task_directions: np.ndarray
    full_margins: np.ndarray
    edges: np.ndarray
    angle_degrees: np.ndarray
    direction_norm: np.ndarray
    feature_norm: np.ndarray
    alignment_lookup: np.ndarray
    edge_index: np.ndarray
    edge_sign: np.ndarray

    def _assigned_edges(self, permutation):
        permutation = np.asarray(permutation, dtype=np.int64)
        a = permutation[..., self.edges[:, 0]]
        b = permutation[..., self.edges[:, 1]]
        return self.edge_index[a, b], self.edge_sign[a, b]

    def alignment(self, permutation=None):
        """Alignment for one assignment, or a matrix of assignments [draw, task]."""
        if permutation is None:
            permutation = np.arange(len(self.task_ids))
        indices, signs = self._assigned_edges(permutation)
        return self.alignment_lookup[np.arange(len(self.edges)), indices] * signs

    def displacement_norms(self, permutation=None):
        if permutation is None:
            permutation = np.arange(len(self.task_ids))
        indices, _ = self._assigned_edges(permutation)
        return self.feature_norm[indices]


def prepare_image(image_id, task_ids, seed_features, task_directions, full_margins,
                  namespace=""):
    """Build a complete task graph in canonical task-ID order.

    Inputs are [seed, task, feature], [task, feature] unit text directions, and
    [task] signed full-model margins. Every edge must have defined alignment;
    the fixed protocol does not drop zero-length edges or change support.
    """
    task_ids = np.asarray(task_ids, dtype=str)
    order = np.argsort(task_ids)
    features = np.asarray(seed_features, dtype=np.float64)
    directions = np.asarray(task_directions, dtype=np.float64)[order]
    margins = np.asarray(full_margins, dtype=np.float64)[order]
    assert features.ndim == 3 and features.shape[1:] == directions.shape
    assert len(task_ids) >= 2 and len(np.unique(task_ids)) == len(task_ids)
    assert margins.shape == (len(task_ids),)
    assert np.all(np.isfinite(features)) and np.all(np.isfinite(directions))
    assert np.all(np.isfinite(margins))
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1., rtol=0, atol=1e-10)
    mean = features[:, order].mean(axis=0, dtype=np.float64)
    edges = np.column_stack(np.triu_indices(len(task_ids), 1))
    a, b = edges.T
    delta_z, delta_u = mean[a] - mean[b], directions[a] - directions[b]
    norm_z, norm_u = np.linalg.norm(delta_z, axis=1), np.linalg.norm(delta_u, axis=1)
    assert np.all(norm_z > 1e-12), f"Undefined feature alignment in image {image_id}"
    assert np.all(norm_u > 1e-12), f"Undefined text alignment in image {image_id}"
    lookup = (delta_u / norm_u[:, None]) @ (delta_z / norm_z[:, None]).T
    angle = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", directions[a], directions[b]), -1., 1.)))
    index = np.full((len(task_ids), len(task_ids)), -1, dtype=np.int64)
    sign = np.zeros(index.shape, dtype=np.int8)
    index[a, b] = index[b, a] = np.arange(len(edges))
    sign[a, b], sign[b, a] = 1, -1
    return ImageGeometry(str(image_id), str(namespace), task_ids[order], mean,
                         directions, margins, edges, angle, norm_u, norm_z,
                         lookup, index, sign)


def image_permutations(image, draws=1999, seed=2094):
    """Uniform permutations keyed independently of image iteration/model order.

    The namespace, image ID and sorted task IDs identify the panel. The key is
    encoded directly as UTF-8 seed entropy, without Python's process-random hash.
    Fixed points, including the identity permutation, are allowed.
    """
    key = json.dumps([image.namespace, image.image_id, image.task_ids.tolist()],
                     ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), *key]))
    return np.asarray([rng.permutation(len(image.task_ids)) for _ in range(draws)],
                      dtype=np.int64).reshape(draws, len(image.task_ids))


def _slope_coefficients(features, weights, ids, fixed_effects):
    if fixed_effects:
        features = demean_by_image(features, ids)
    design, (_, scale) = _scaled_design(features, weights, intercept=not fixed_effects)
    root = np.sqrt(weights)
    weighted = design * root[:, None]
    # Match the default lstsq rank threshold used by adjusted_angle_association.
    u, singular, vh = np.linalg.svd(weighted, full_matrices=False)
    valid = singular > np.finfo(weighted.dtype).eps * max(weighted.shape) * singular[0]
    rank = int(valid.sum())
    baseline_rank = int(np.linalg.matrix_rank(weighted[:, :-1]))
    assert rank > baseline_rank, "Angle is not estimable beyond the fixed controls"
    coefficients = (vh[valid, -1] / singular[valid]) @ u[:, valid].T
    coefficients *= root / scale[-1]
    if fixed_effects:
        # Apply the same within-image centering to the target via its linear map.
        coefficients = demean_by_image(coefficients, ids)
    return coefficients, {"design_rank": rank, "design_columns": design.shape[1],
                          "baseline_rank": baseline_rank,
                          "angle_unit": "10 degrees"}


@dataclass
class PreparedStatistics:
    coefficients: np.ndarray
    image_slices: list
    cases: dict
    metadata: dict

    def evaluate(self, alignments):
        """Evaluate three statistics from a list of per-image alignment arrays."""
        return np.concatenate(alignments, axis=-1) @ self.coefficients.T


def prepare_statistics(images):
    """Precompute exact linear maps for weighted slopes and image-equal mean."""
    cases = {key: [] for key in ("image_ids", "base_margin_a", "base_margin_b", "angle_degrees")}
    slices, start = [], 0
    for image in images:
        count = len(image.edges)
        cases["image_ids"].extend([json.dumps([image.namespace, image.image_id])] * count)
        cases["base_margin_a"].extend(image.full_margins[image.edges[:, 0]])
        cases["base_margin_b"].extend(image.full_margins[image.edges[:, 1]])
        cases["angle_degrees"].extend(image.angle_degrees)
        slices.append(slice(start, start + count))
        start += count
    cases = {key: np.asarray(value) for key, value in cases.items()}
    ids = cases["image_ids"]
    assert len(np.unique(ids)) == len(images), "Each image must occur once"
    weights = image_weights(ids)
    features = np.column_stack([difficulty_features(cases["base_margin_a"], cases["base_margin_b"]),
                                cases["angle_degrees"] / 10.])
    ordinary, ordinary_meta = _slope_coefficients(features, weights, ids, False)
    fixed, fixed_meta = _slope_coefficients(features, weights, ids, True)
    metadata = {"difficulty_adjusted_slope": ordinary_meta,
                "image_fixed_effects_slope": fixed_meta,
                "image_count": len(images), "pair_count": len(ids)}
    return PreparedStatistics(np.vstack([ordinary, fixed, weights]), slices, cases, metadata)


def upper_tail_p(observed, reference):
    """Finite Monte Carlo upper tail, including ties and the observed assignment."""
    reference = np.asarray(reference)
    return (1 + np.count_nonzero(reference >= observed, axis=0)) / (len(reference) + 1)


def run_permutations(images, draws=1999, seed=2094):
    """Return observed statistics and the fixed within-image permutation reference.

    Multiplicity across model/dataset/budget settings belongs to the caller.
    No model is trained or evaluated here; only existing vectors are reassigned.
    """
    statistics = prepare_statistics(images)
    observed_alignments = [image.alignment() for image in images]
    observed = statistics.evaluate(observed_alignments)
    reference = np.zeros((draws, len(STATISTIC_NAMES)), dtype=np.float64)
    for image, section in zip(images, statistics.image_slices):
        assignments = image_permutations(image, draws, seed)
        reference += image.alignment(assignments) @ statistics.coefficients[:, section].T
    return {"statistic_names": list(STATISTIC_NAMES), "observed": observed,
            "null_statistics": reference, "upper_tail_p": upper_tail_p(observed, reference),
            "per_image_observed_alignment": observed_alignments,
            "metadata": statistics.metadata, "draws": draws, "seed": seed}
