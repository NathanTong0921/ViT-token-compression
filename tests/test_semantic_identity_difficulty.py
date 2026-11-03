"""Check the fixed-effects computation against an explicit dummy regression."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_semantic_identity_difficulty import adjusted_effect


class SemanticIdentityDifficultyTests(unittest.TestCase):
    def test_within_regression_and_cluster_variance_match_explicit_task_dummies(self):
        rng = np.random.default_rng(73)
        tasks = np.repeat(np.arange(12), 20)
        new = np.tile(np.repeat([False, True], 10), 12)
        images = np.tile(np.arange(40), 6)
        margin = rng.normal(size=len(tasks)) + .2 * new
        weights = rng.uniform(.5, 2., len(tasks))
        y = rng.normal(size=len(tasks)) + .7 * new + 2 * margin + rng.normal(size=12)[tasks]
        result = adjusted_effect(y, new, margin, weights, tasks, images)
        center = np.average(margin, weights=weights)
        z = (margin - center) / np.sqrt(np.average((margin - center) ** 2, weights=weights))
        x = np.column_stack((np.eye(12)[tasks], new, z, z * z))
        inverse = np.linalg.inv(x.T @ (weights[:, None] * x))
        beta = inverse @ (x.T @ (weights * y))
        residual = y - x @ beta
        cluster_scores = np.zeros((40, 15))
        np.add.at(cluster_scores, images, weights[:, None] * x * residual[:, None])
        correction = 40 / 39 * (len(y) - 1) / (len(y) - 15)
        covariance = correction * inverse @ (cluster_scores.T @ cluster_scores) @ inverse
        self.assertAlmostEqual(result["adjusted_true_new_minus_known"], beta[12], places=12)
        self.assertAlmostEqual(result["cluster_standard_error"], np.sqrt(covariance[12, 12]), places=12)


if __name__ == "__main__":
    unittest.main()
