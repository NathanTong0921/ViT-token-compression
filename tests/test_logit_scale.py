import unittest

import numpy as np

from token_compression.logit_scale import fit_global_scale, mean_kl_derivative, row_kl


class LogitScaleTests(unittest.TestCase):
    def setUp(self):
        self.logits = np.asarray([[2., 1., -1., -2.], [3., 0., 1., -1.],
                                 [1.5, .1, -1.5, -.1]])
        self.masks = np.asarray([[0, 2], [0, 3], [0, 3]])
        self.weights = np.asarray([1., 2., 3.])

    def test_kl_matches_dense_distribution_and_is_shift_invariant(self):
        alpha = .73
        probabilities = np.exp(alpha * self.logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        direct = -np.log(2) - np.log(np.take_along_axis(
            probabilities, self.masks, axis=1)).mean(axis=1)
        np.testing.assert_allclose(row_kl(self.logits, self.masks, alpha), direct,
                                   rtol=0, atol=1e-14)
        shifted = self.logits + np.asarray([1024., -512., 128.])[:, None]
        np.testing.assert_allclose(row_kl(shifted, self.masks, alpha), direct,
                                   rtol=0, atol=2e-14)
        self.assertTrue(np.isfinite(row_kl(self.logits, self.masks, 1e4)).all())

    def test_derivative_matches_finite_difference_and_stationary_optimum(self):
        alpha, h = .6, 1e-5
        w = self.weights / self.weights.sum()
        numeric = w @ (row_kl(self.logits, self.masks, alpha+h)
                       - row_kl(self.logits, self.masks, alpha-h)) / (2*h)
        derivative = mean_kl_derivative(self.logits, self.masks, self.weights, alpha)
        self.assertAlmostEqual(numeric, derivative, places=9)
        fit = fit_global_scale(self.logits, self.masks, self.weights)
        self.assertIsNone(fit["boundary"])
        self.assertAlmostEqual(mean_kl_derivative(self.logits, self.masks,
                                                self.weights, fit["alpha"]), 0., places=10)
        self.assertLess(fit["loss"], w @ row_kl(self.logits, self.masks))

    def test_positive_scale_preserves_full_ranking_and_ties(self):
        logits = np.asarray([[2., 2., 1., -1.], [-3., 0., 1., 1.]])
        ranking = np.argsort(-logits, axis=1, kind="stable")
        for alpha in (1e-4, .2, 3.7, 1e4):
            np.testing.assert_array_equal(np.argsort(-alpha * logits, axis=1,
                                                    kind="stable"), ranking)
        with self.assertRaises(ValueError):
            row_kl(self.logits, self.masks, 0)

    def test_rescaled_logits_recover_distribution_and_scale_decomposition(self):
        old, new = 3.7 * self.logits, self.logits
        old_fit = fit_global_scale(old, self.masks, self.weights)
        new_fit = fit_global_scale(new, self.masks, self.weights)
        self.assertAlmostEqual(old_fit["alpha"] * 3.7, new_fit["alpha"], places=10)
        np.testing.assert_allclose(old * old_fit["alpha"], new * new_fit["alpha"],
                                   rtol=0, atol=1e-10)
        w = self.weights / self.weights.sum()
        old_raw, new_raw = (float(w @ row_kl(z, self.masks)) for z in (old, new))
        gain = old_raw - new_raw
        residual = old_fit["loss"] - new_fit["loss"]
        scale = (old_raw - old_fit["loss"]) - (new_raw - new_fit["loss"])
        self.assertGreater(gain, 0)
        self.assertAlmostEqual(residual, 0., places=12)
        self.assertAlmostEqual(gain, scale + residual, places=12)

    def test_boundary_optima_uniform_target_and_perfect_order(self):
        logits = self.logits[:1]
        all_tokens = np.arange(logits.shape[1])[None]
        uniform = fit_global_scale(logits, all_tokens, np.ones(1))
        self.assertEqual(uniform["boundary"], "lower")
        self.assertEqual(uniform["alpha"], 1e-4)
        self.assertLess(uniform["loss"], 1e-6)
        perfect = fit_global_scale(logits, np.asarray([[0]]), np.ones(1), (1e-4, 3.))
        self.assertEqual(perfect["boundary"], "upper")
        self.assertEqual(perfect["alpha"], 3.)
        self.assertLess(perfect["loss"], row_kl(logits, np.asarray([[0]]))[0])


if __name__ == "__main__":
    unittest.main()
