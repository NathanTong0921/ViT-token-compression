import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from extract_benefit_features import absolute_outcomes, feature_matrix, selected_positions


class BenefitFeatureTests(unittest.TestCase):
    def test_per_row_positive_scale_and_shift_leave_all_features_unchanged(self):
        c = np.asarray([[4, 4, 1, 0, -1, -2], [2, 0, 4, 1, 3, -1]], float)
        u = np.asarray([[4, 1, 3, -1, 0, -2], [1, 2, 4, 0, 3, -1]], float)
        donor = u[:, ::-1].copy()
        distance = np.asarray([0.5, 1.2])
        baseline = feature_matrix(c, u, distance, 3, donor)
        changed = feature_matrix(c * np.asarray([[0.2], [8]]) + 9,
                                 u * np.asarray([[7], [0.1]]) - 3,
                                 distance, 3, donor * 2.7 + 17)
        np.testing.assert_allclose(changed, baseline, atol=1e-13, rtol=0)

    def test_known_disagreement_boundary_and_donor_features(self):
        c = np.asarray([[6, 5, 4, 3, 2, 1]], float)
        u = np.asarray([[6, 2, 4, 5, 3, 1]], float)
        donor = c.copy()
        result = feature_matrix(c, u, np.asarray([0.5]), 3, donor)[0]
        self.assertAlmostEqual(result[0], 0.5)
        self.assertAlmostEqual(result[1], 2)
        self.assertAlmostEqual(result[2], 1 / u.std())
        self.assertAlmostEqual(result[3], 1 / 3)
        self.assertAlmostEqual(result[4], 1 / c.std())
        self.assertAlmostEqual(result[6], 1 / (3 * u.std()))
        self.assertAlmostEqual(result[7], -1 / 3)
        np.testing.assert_array_equal(selected_positions(c, 3), [[0, 1, 2]])

    def test_candidate_order_is_symmetric_but_outcomes_use_true_class(self):
        scores = np.asarray([[0.7, 0.7, 0.1], [0.1, 0.4, 0.2]])
        labels = np.asarray([0, 2])
        pairs = np.asarray([[0, 1], [1, 2]])
        distance = np.asarray([0.5, 2.0])
        values = absolute_outcomes(scores, labels, pairs, distance)
        np.testing.assert_allclose(values, [[1, 0, 0], [0, -0.1, -0.2]])
        np.testing.assert_array_equal(values, absolute_outcomes(scores, labels, pairs[:, ::-1], distance))


if __name__ == "__main__":
    unittest.main()
