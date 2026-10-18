import unittest

import numpy as np

from token_compression.selector_scale import (balanced_example_weights,
                                               balanced_training_stream,
                                               map_stream_images,
                                               validate_image_partition)


class SelectorScaleTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([0, 1, 1, 1, 2, 2])
        self.ids = np.array(["a", "d", "b", "c", "f", "e"])
        self.pairs = np.array([[0, 1], [1, 2]])

    def test_exact_task_side_counts_and_reproducibility(self):
        stream = balanced_training_stream(self.pairs, 4000, 2041)
        groups = 2 * stream["pair_indices"] + stream["sides"]
        np.testing.assert_array_equal(np.bincount(groups), [1000] * 4)
        repeated = balanced_training_stream(self.pairs, 4000, 2041)
        for key in stream:
            np.testing.assert_array_equal(stream[key], repeated[key])

    def test_unequal_image_counts_do_not_change_task_or_side_distribution(self):
        stream = balanced_training_stream(self.pairs, 40000, 2)
        selected = map_stream_images(self.labels, self.ids, stream, self.pairs)
        draw_labels = self.pairs[stream["pair_indices"], stream["sides"]]
        np.testing.assert_array_equal(self.labels[selected], draw_labels)
        np.testing.assert_array_equal(np.bincount(draw_labels), [10000, 20000, 10000])
        counts = np.bincount(selected, minlength=6)
        # Class 1 has three available images and occurs in both tasks.
        self.assertLess(np.max(np.abs(counts[1:4] / 20000 - 1 / 3)), 0.02)

    def test_nested_coverage_preserves_shared_task_class_stream(self):
        stream = balanced_training_stream(self.pairs, 400, 7)
        small = ["a", "b", "e"]
        full_images = map_stream_images(self.labels, self.ids, stream, self.pairs)
        small_images = map_stream_images(self.labels, self.ids, stream, self.pairs, small)
        np.testing.assert_array_equal(self.labels[full_images], self.labels[small_images])
        self.assertEqual(set(self.ids[small_images]), set(small))
        # Within-class ordering is by ID, not by the cache's array order.
        fixed = {"pair_indices": np.array([0, 0]), "sides": np.array([1, 1]),
                 "image_quantiles": np.array([0.01, 0.99])}
        np.testing.assert_array_equal(
            self.ids[map_stream_images(self.labels, self.ids, fixed, self.pairs)], ["b", "d"])

    def test_evaluation_weights_balance_each_task_and_class(self):
        images = np.array([0, 1, 2, 3, 1, 2, 3, 4, 5])
        tasks = np.array([0] * 4 + [1] * 5)
        weights = balanced_example_weights(self.labels, self.pairs, images, tasks)
        self.assertAlmostEqual(weights.sum(), 1.)
        for task, pair in enumerate(self.pairs):
            for label in pair:
                mask = (tasks == task) & (self.labels[images] == label)
                self.assertAlmostEqual(weights[mask].sum(), .25)

    def test_image_holdout_and_nested_panel_are_checked(self):
        validate_image_partition(self.ids, ["held_out"], ["a", "b", "e"])
        with self.assertRaises(AssertionError):
            validate_image_partition(self.ids, ["a"], ["a"])
        with self.assertRaises(AssertionError):
            validate_image_partition(self.ids, ["held_out"], ["missing"])


if __name__ == "__main__":
    unittest.main()
