import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_routing_headroom import (aggregate_image_values, correctness_indicators,
    endpoint_summary, stability_indicators)


class RoutingHeadroomTests(unittest.TestCase):
    def test_oracle_improvement_is_smaller_one_sided_rescue_mass(self):
        c = np.asarray([[[1., .3, .2], [1., .1, .05], [0., -.2, -.1], [0., -.1, -.05]]])
        u = np.asarray([[[1., .2, .1], [0., -.2, -.1], [1., .3, .2], [0., -.05, -.02]]])
        weights = np.asarray([.1, .2, .3, .4])
        result = endpoint_summary(c, u, weights)
        indicators = correctness_indicators(c, u)
        np.testing.assert_allclose(result["oracle_gain"][:, 0],
            np.minimum(indicators["conditioned_only"] @ weights, indicators["shared_only"] @ weights))
        self.assertAlmostEqual(result["oracle"][0, 0], .6)
        self.assertAlmostEqual(result["oracle_gain"][0, 0], .2)
        # A margin oracle need not choose the same endpoint as an accuracy oracle.
        self.assertAlmostEqual(result["oracle"][0, 1], .12)

    def test_equal_correctness_has_zero_accuracy_space_despite_margin_space(self):
        c = np.asarray([[[1., .4, .4], [0., -.3, -.3]]])
        u = np.asarray([[[1., .2, .2], [0., -.1, -.1]]])
        result = endpoint_summary(c, u, [.5, .5])
        self.assertEqual(result["oracle_gain"][0, 0], 0.)
        self.assertAlmostEqual(result["oracle_gain"][0, 1], .1)

    def test_seed_stability_does_not_count_ties_as_opposite_direction(self):
        c = np.zeros((3, 5, 3)); u = np.zeros_like(c)
        c[:, :, 0] = [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 0, 0, 0, 1]]
        u[:, :, 0] = [[0, 0, 0, 1, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 0]]
        result = stability_indicators(c, u)
        np.testing.assert_array_equal(result["conditioned_only_majority"], [1, 1, 0, 0, 0])
        np.testing.assert_array_equal(result["conditioned_only_all"], [1, 0, 0, 0, 0])
        np.testing.assert_array_equal(result["shared_only_all"], [0, 0, 0, 1, 0])
        np.testing.assert_array_equal(result["direction_reverses"], [0, 0, 1, 0, 0])

    def test_image_aggregation_preserves_weighted_total_with_repeated_tasks(self):
        x = np.asarray([[1., 2.], [3., 4.], [-2., 5.]])
        w = np.asarray([.1, .6, .3])
        ids = np.asarray(["a", "b", "a"])
        unique, sums, totals = aggregate_image_values(x, w, ids)
        np.testing.assert_array_equal(unique, ["a", "b"])
        np.testing.assert_allclose(sums.sum(axis=0), w @ x)
        np.testing.assert_allclose(totals, [.4, .6])
        np.testing.assert_allclose(sums[0], x[[0, 2]].T @ w[[0, 2]])


if __name__ == "__main__":
    unittest.main()
