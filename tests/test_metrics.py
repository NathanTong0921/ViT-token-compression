import unittest

import numpy as np

from token_compression.metrics import binary_margin, direction_angle


class GeometryTests(unittest.TestCase):
    def test_signed_margin_and_reversed_target(self):
        text = np.array([[1., 0.], [0., 1.]])
        scores = np.array([0.8, 0.6])
        gap, margin = binary_margin(scores, text, 0, 1)
        self.assertAlmostEqual(gap, 0.2)
        self.assertAlmostEqual(margin, 0.2 / np.sqrt(2))
        reverse_gap, reverse_margin = binary_margin(scores, text, 1, 0)
        self.assertAlmostEqual(reverse_gap, -gap)
        self.assertAlmostEqual(reverse_margin, -margin)

    def test_directions_share_true_class_anchor(self):
        text = np.array([[1., 0.], [0., 1.], [0., -1.]])
        self.assertAlmostEqual(direction_angle(text, 0, 1, 2), 90.)
        self.assertAlmostEqual(direction_angle(text, 0, 1, 1), 0., places=5)
        self.assertAlmostEqual(direction_angle(text, 0, 1, 2),
                               direction_angle(text, 0, 2, 1))


if __name__ == "__main__":
    unittest.main()
