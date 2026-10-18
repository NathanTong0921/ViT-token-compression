import unittest
from itertools import combinations

import numpy as np

from token_compression.budget_ranking import (
    fit_responses, predict_masks, ridge_projector, sample_masks, task_coefficients,
)


class BudgetRankingTests(unittest.TestCase):
    def test_sampled_masks_are_unique_exact_budget_and_reproducible(self):
        train = sample_masks(196, 49, 512, seed=10)
        np.testing.assert_array_equal(train, sample_masks(196, 49, 512, seed=10))
        self.assertEqual(train.dtype, np.bool_)
        self.assertEqual(train.shape, (512, 196))
        np.testing.assert_array_equal(train.sum(axis=1), np.full(512, 49))
        self.assertEqual(len(np.unique(train, axis=0)), 512)
        test = sample_masks(196, 49, 128, seed=11, exclude=train)
        self.assertEqual(len(np.unique(np.concatenate([train, test]), axis=0)), 640)

    def test_exclusions_exhaust_small_design_without_impossible_retry(self):
        train = sample_masks(4, 2, 5, seed=2)
        test = sample_masks(4, 2, 1, seed=3, exclude=train)
        self.assertEqual(len(np.unique(np.concatenate([train, test]), axis=0)), 6)
        with self.assertRaisesRegex(ValueError, "remain available"):
            sample_masks(4, 2, 2, seed=3, exclude=train)
        # Repeated exclusions count once; masks at other budgets cannot collide.
        exclude = np.concatenate([train, train[:1], np.zeros((1, 4), dtype=bool)])
        np.testing.assert_array_equal(test, sample_masks(4, 2, 1, seed=3, exclude=exclude))
        self.assertEqual(sample_masks(4, 2, 0, seed=3).shape, (0, 4))

    def test_ridge_projector_matches_standardized_ridge_equations(self):
        masks = sample_masks(7, 3, 16, seed=10)
        responses = np.random.default_rng(11).normal(size=(16, 3))
        projector = ridge_projector(masks, ridge=.2)
        coefficients, intercept = fit_responses(projector, responses)
        x = masks.astype(float)
        standardized = (x - x.mean(axis=0)) / x.std(axis=0)
        centered_y = responses - responses.mean(axis=0)
        beta = np.linalg.solve(
            standardized.T @ standardized + len(masks) * .2 * np.eye(7),
            standardized.T @ centered_y,
        )
        expected = standardized @ beta + responses.mean(axis=0)
        np.testing.assert_allclose(predict_masks(masks, coefficients, intercept), expected,
                                   rtol=1e-12, atol=1e-12)

    def test_linear_response_predicts_independent_fixed_budget_masks(self):
        train = sample_masks(8, 4, 40, seed=3)
        holdout = sample_masks(8, 4, 30, seed=4, exclude=train)
        coefficients = np.random.default_rng(5).normal(size=(8, 3))
        intercept = np.asarray([.3, -.8, 1.5])
        y = predict_masks(train, coefficients, intercept)
        fitted, fitted_intercept = fit_responses(ridge_projector(train, ridge=1e-9), y)
        np.testing.assert_allclose(predict_masks(holdout, fitted, fitted_intercept),
                                   predict_masks(holdout, coefficients, intercept), atol=3e-8)

    def test_constant_coefficient_ambiguity_does_not_change_ranking(self):
        masks = np.zeros((20, 6), dtype=bool)
        for row, indices in zip(masks, combinations(range(6), 3)):
            row[list(indices)] = True
        coefficients = np.asarray([-3., -1., 0., 2., 4., 7.])[:, None]
        shifted = coefficients + 8.
        y = predict_masks(masks, coefficients, np.asarray([.5]))
        np.testing.assert_array_equal(y, predict_masks(masks, shifted, np.asarray([.5 - 24.])))
        fitted, _ = fit_responses(ridge_projector(masks, ridge=1e-8), y)
        np.testing.assert_array_equal(np.argsort(fitted[:, 0]), np.argsort(coefficients[:, 0]))
        np.testing.assert_allclose(fitted.mean(axis=0), 0., atol=1e-6)
        offset_fitted, _ = fit_responses(ridge_projector(masks, ridge=.01), y + 20.)
        original_fitted, _ = fit_responses(ridge_projector(masks, ridge=.01), y)
        np.testing.assert_allclose(offset_fitted, original_fitted, atol=1e-11)

    def test_task_utilities_use_label_direction_and_text_normalization(self):
        coefficients = np.asarray([[.4, .1, -.3], [.2, -.2, .6]])
        text = np.eye(3)
        actual = task_coefficients(coefficients, text, true_class=1, competitors=[2, 0])
        expected = np.asarray([[.4, -.3], [-.8, -.4]]) / np.sqrt(2.)
        np.testing.assert_allclose(actual, expected)
        reversed_direction = task_coefficients(coefficients, text, true_class=2, competitors=[1])
        np.testing.assert_allclose(actual[:, :1], -reversed_direction)

    def test_invariant_columns_cannot_define_patch_utilities(self):
        with self.assertRaisesRegex(ValueError, "Every patch must vary"):
            ridge_projector(np.asarray([[1, 0, 1], [0, 1, 1]], dtype=bool))
        with self.assertRaisesRegex(ValueError, "positive ridge"):
            ridge_projector(sample_masks(4, 2, 6, seed=1), ridge=0)


if __name__ == "__main__":
    unittest.main()
