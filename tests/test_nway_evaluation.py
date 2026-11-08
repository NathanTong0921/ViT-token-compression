"""Classification and weighting checks for variable candidate-set evaluation."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_nway_selectors import classification_outcomes, packed_predictions, validate_panel
from evaluate_selectors import binary_outcomes


class NWayEvaluationTests(unittest.TestCase):
    def test_binary_agrees_with_existing_outcomes_including_ties(self):
        rng = np.random.default_rng(52)
        text = rng.normal(size=(6, 9))
        text /= np.linalg.norm(text, axis=1, keepdims=True)
        scores = rng.normal(size=(4, 6))
        pairs = np.array([[3, 1], [2, 5], [4, 0], [5, 1]])
        labels = np.array([1, 5, 4, 5])
        scores[0, [1, 3]] = .2
        margin, correct = binary_outcomes(scores, labels, np.sort(pairs, axis=1), text)
        actual = classification_outcomes(scores, labels, pairs, text)
        np.testing.assert_array_equal(actual["correct"], correct)
        np.testing.assert_allclose(actual["geometric_margin"], margin)
        self.assertEqual(actual["predicted"][0], 1)

    def test_multiclass_uses_nearest_boundary_and_is_order_invariant(self):
        text = np.eye(6)
        scores = np.array([[.2, .5, .9, .1, .3, .9],
                           [.1, .4, .3, .0, .6, .5]])
        candidates = np.array([[5, 4, 3, 2, 1], [0, 1, 2, 3, 4]])
        labels = np.array([5, 4])
        result = classification_outcomes(scores, labels, candidates, text)
        np.testing.assert_array_equal(result["predicted"], [2, 4])
        np.testing.assert_allclose(result["margin"], [0., .2])
        np.testing.assert_allclose(result["geometric_margin"], [0., .2 / np.sqrt(2)])
        reverse = classification_outcomes(scores, labels, candidates[:, ::-1], text)
        for key in result:
            np.testing.assert_array_equal(result[key], reverse[key])

    def test_geometric_margin_can_choose_a_different_rival(self):
        text = np.array([[1., 0.], [.8, .6], [-1., 0.]])
        scores = np.array([[.9, .8, .7]])
        actual = classification_outcomes(scores, np.array([0]), np.array([[0, 1, 2]]), text)
        self.assertAlmostEqual(actual["margin"][0], .1)
        self.assertAlmostEqual(actual["geometric_margin"][0], .1)

    def test_task_class_balancing_and_row_packing(self):
        # Different image counts per class must still give equal class weights.
        labels = np.array([0, 1, 1, 1, 2, 2])
        panel = {"image_indices": np.arange(6), "task_indices": np.zeros(6, int),
                 "candidates": np.tile([0, 1, 2], (6, 1)), "labels": labels,
                 "image_ids": np.array([f"i{i}" for i in range(6)]),
                 "task_ids": np.full(6, "t0"),
                 "weights": np.array([1/3, 1/9, 1/9, 1/9, 1/6, 1/6])}
        validate_panel(panel)
        scores = np.eye(3)[labels]
        outcomes = classification_outcomes(scores, labels, panel["candidates"], np.eye(3))
        packed = packed_predictions(panel, [("conditioned", 11, outcomes),
                                             ("image_only_wide", 11, outcomes)], np.eye(3))
        np.testing.assert_array_equal(packed["labels"], np.tile(labels, 2))
        self.assertAlmostEqual(packed["weights"][:6] @ packed["correct"][:6], 1.)
        panel["weights"] = np.ones(6) / 6
        with self.assertRaises(AssertionError):
            validate_panel(panel)


if __name__ == "__main__":
    unittest.main()
