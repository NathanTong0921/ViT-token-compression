import unittest

import numpy as np

from token_compression.content_benefit import (
    content_features, fit_content_readout, normalize_rows,
)


class ContentBenefitTest(unittest.TestCase):
    def test_affine_patch_readout_matches_image_mean(self):
        rng = np.random.default_rng(27)
        patches = rng.normal(size=(40, 8, 5))
        target = rng.normal(size=(40, 3))
        fit = fit_content_readout(patches.mean(axis=1), target, np.ones(40))
        image = patches.mean(axis=1) @ fit["matrix"] + fit["bias"]
        token_average = (patches @ fit["matrix"] + fit["bias"]).mean(axis=1)
        np.testing.assert_allclose(image, token_average, atol=1e-14)
        standard = ((patches.mean(axis=1)-fit["mean"])/fit["scale"]) @ fit["beta"] + fit["intercept"]
        np.testing.assert_allclose(image, standard, atol=1e-14)

    def test_candidate_order_invariance_and_identical_masks(self):
        rng = np.random.default_rng(28)
        patches = rng.normal(size=(3, 8, 5))
        matrix, bias = rng.normal(size=(5, 3)), rng.normal(size=3)
        direction = normalize_rows(rng.normal(size=(3, 3)))
        c = np.array([[0, 1, 2, 3]]*3)
        u = np.array([[4, 5, 6, 7]]*3)
        x = content_features(patches, c, u, direction, matrix, bias)
        reverse = content_features(patches, c, u, -direction, matrix, bias)
        np.testing.assert_allclose(x, reverse)
        same = content_features(patches, c, c, direction, matrix, bias)
        np.testing.assert_allclose(same[:, [1, 3, 5]], 0)

    def test_energy_is_fraction_and_centered(self):
        patches = np.array([[[1., 0.], [-1., 0.], [0., 0.], [0., 0.]]])
        matrix, bias = np.eye(2), np.array([.3, 1.])
        direction = np.array([[1., 0.]])
        c, u = np.array([[0, 1]]), np.array([[2, 3]])
        x = content_features(patches, c, u, direction, matrix, bias)
        np.testing.assert_allclose(x[0, :2], [1., 1.])
        shifted = content_features(patches + 7, c, u, direction, matrix, bias)
        np.testing.assert_allclose(x[:, :2], shifted[:, :2])


if __name__ == "__main__":
    unittest.main()
