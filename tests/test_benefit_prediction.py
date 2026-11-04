import unittest

import numpy as np

from token_compression.benefit_prediction import (
    fit_weighted_ridge, iter_two_way_splits, predict_ridge,
    two_way_folds, weighted_center,
)


class BenefitPredictionTests(unittest.TestCase):
    def test_two_way_holdout_excludes_both_groups_and_covers_every_row(self):
        images = np.repeat([f"image_{i}" for i in range(10)], 10)
        tasks = np.tile([f"task_{i}" for i in range(10)], 10)
        # Three trained seeds are repeated measurements, not independent folds.
        images, tasks = np.tile(images, 3), np.tile(tasks, 3)
        image_fold, task_fold = two_way_folds(images, tasks)
        np.testing.assert_array_equal(image_fold[:100], image_fold[100:200])
        np.testing.assert_array_equal(task_fold[:100], task_fold[200:])
        coverage = np.zeros(len(images), dtype=int)
        cells = 0
        for i, j, train, test in iter_two_way_splits(image_fold, task_fold):
            self.assertTrue(train.any() and test.any())
            self.assertFalse(set(images[train]) & set(images[test]))
            self.assertFalse(set(tasks[train]) & set(tasks[test]))
            self.assertTrue(np.all(image_fold[test] == i))
            self.assertTrue(np.all(task_fold[test] == j))
            coverage += test
            cells += 1
        self.assertEqual(cells, 25)
        np.testing.assert_array_equal(coverage, np.ones(len(images), dtype=int))

    def test_prediction_does_not_fit_heldout_scale_and_constant_column_is_zero(self):
        X = np.column_stack((np.arange(6.), np.full(6, 0.1)))
        y = np.arange(6.) + 2.
        weights = np.asarray([1., 2., 3., 2., 1., 1.])
        fit = fit_weighted_ridge(X, y, weights)
        expected_mean = weights @ X[:, 0] / weights.sum()
        expected_std = np.sqrt(weights @ ((X[:, 0] - expected_mean) ** 2) / weights.sum())
        self.assertAlmostEqual(fit["mean"][0], expected_mean)
        self.assertAlmostEqual(fit["scale"][0], expected_std)
        self.assertEqual(fit["scale"][1], 1.)
        self.assertEqual(fit["beta"][1], 0.)
        before = {key: np.array(value, copy=True) for key, value in fit.items()}
        held = np.asarray([[1e6, 0.1], [-1e6, 200.]])
        prediction = predict_ridge(fit, held)
        expected = ((held - before["mean"]) / before["scale"]) @ before["beta"] + before["intercept"]
        np.testing.assert_array_equal(prediction, expected)
        for key in fit:
            np.testing.assert_array_equal(fit[key], before[key])

    def test_weighted_duplication_does_not_change_fit_or_regularization(self):
        rng = np.random.default_rng(12)
        X, y = rng.normal(size=(24, 4)), rng.normal(size=(24, 2))
        weights = rng.uniform(.1, 2., size=24)
        base = fit_weighted_ridge(X, y, weights)
        repeats = np.arange(24) % 3 + 1
        duplicated = fit_weighted_ridge(np.repeat(X, repeats, axis=0),
                                       np.repeat(y, repeats, axis=0),
                                       np.repeat(weights / repeats, repeats))
        for key in ("mean", "scale", "beta", "intercept"):
            np.testing.assert_allclose(base[key], duplicated[key], atol=1e-14, rtol=0)

    def test_known_multitarget_linear_signal_and_weighted_intercept(self):
        rng = np.random.default_rng(20)
        X, held = rng.normal(size=(100, 3)), rng.normal(size=(12, 3))
        coefficients = np.asarray([[.5, 2.], [-1., .1], [.2, -3.]])
        intercept = np.asarray([4., -2.])
        y, weights = X @ coefficients + intercept, rng.uniform(.2, 2., size=100)
        fit = fit_weighted_ridge(X, y, weights, ridge=1e-12)
        np.testing.assert_allclose(predict_ridge(fit, held), held @ coefficients + intercept,
                                   atol=2e-11, rtol=0)
        np.testing.assert_allclose(weights @ (predict_ridge(fit, X) - y), 0., atol=1e-11, rtol=0)
        one = fit_weighted_ridge(X, y[:, 0], weights, ridge=1e-12)
        self.assertEqual(predict_ridge(one, held).shape, (12,))
        np.testing.assert_allclose(predict_ridge(one, held), predict_ridge(fit, held)[:, 0], atol=1e-13, rtol=0)

    def test_group_centering_has_zero_weighted_mean_and_removes_group_offsets(self):
        ids = np.asarray(["a", "a", "b", "a", "b", "c"])
        values = np.arange(12., dtype=float).reshape(6, 2)
        weights = np.asarray([1., 3., 1., 2., 4., 2.])
        centered = weighted_center(values, ids, weights)
        for group in np.unique(ids):
            chosen = ids == group
            np.testing.assert_allclose(weights[chosen] @ centered[chosen], 0., atol=1e-14, rtol=0)
        shifted = values + np.asarray([10., 10., -7., 10., -7., 3.])[:, None]
        np.testing.assert_allclose(weighted_center(shifted, ids, weights), centered, atol=3e-15, rtol=0)
        np.testing.assert_allclose(weighted_center(values[:, 0], ids, weights), centered[:, 0], atol=1e-14, rtol=0)


if __name__ == "__main__":
    unittest.main()
