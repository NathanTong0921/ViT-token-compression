import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from token_compression.pets import balanced_task_rows
from token_compression.selection import (
    deletion_windows,
    make_candidate_bank,
    select_panel,
    task_patch_utilities,
    window_to_patch_scores,
)


class SelectionTests(unittest.TestCase):
    def test_overlapping_windows_preserve_row_major_positions_and_average_coverage(self):
        windows = deletion_windows(grid_size=3, window_size=2)
        np.testing.assert_array_equal(windows, [
            [0, 1, 3, 4], [1, 2, 4, 5], [3, 4, 6, 7], [4, 5, 7, 8]
        ])
        expected = np.array([1, 2, 3, 3, 4, 5, 5, 6, 7])
        np.testing.assert_allclose(
            window_to_patch_scores([1, 3, 5, 7], windows, num_patches=9), expected
        )
        matrix = np.array([[1, -1], [3, -3], [5, -5], [7, -7]])
        np.testing.assert_allclose(
            window_to_patch_scores(matrix, windows, num_patches=9),
            np.stack([expected, -expected], axis=1),
        )
        official = deletion_windows()
        self.assertEqual(official.shape, (169, 4))
        self.assertEqual(set(official.reshape(-1)), set(range(196)))

    def test_task_utilities_are_signed_and_normalized_toward_true_class(self):
        texts = np.array([[1., 0.], [0., 1.], [-1., 0.]])
        full = np.array([.2, .8, -.2])
        deleted = np.array([[.1, .5, .0], [.2, .9, -.4]])
        windows = np.array([[0, 1], [1, 2]])
        utilities = task_patch_utilities(full, deleted, texts, 1, [0, 2], windows)
        # Each row's margin is toward class 1, regardless of competitor ID.
        # Window 0 harms both tasks; window 1 improves both margins.
        drops = np.array([[.2, .5], [-.1, -.3]]) / np.sqrt(2)
        np.testing.assert_allclose(utilities, [drops[0], drops.mean(axis=0), drops[1]])
        self.assertTrue(np.all(utilities[0] > 0))
        self.assertTrue(np.all(utilities[2] < 0))

    def test_candidate_bank_has_fixed_budget_deterministic_ties_and_seeded_neighbors(self):
        n, k = 20, 7
        task_a = np.zeros(n)
        task_b = np.arange(n, dtype=float)
        bank, names = make_candidate_bank(task_a, task_b, -task_b, task_b, k, 12)
        self.assertEqual(bank.shape, (64, k))
        self.assertEqual(names[:10], [
            "task_a", "task_b", "shared_proxy", "attention", "representation",
            "mix_025", "mix_075", "random_0", "random_1", "random_2"
        ])
        np.testing.assert_array_equal(bank[0], np.arange(k))
        np.testing.assert_array_equal(bank[1], np.arange(n - k, n))
        self.assertTrue(np.all(np.diff(bank, axis=1) > 0))
        self.assertTrue(np.all((bank >= 0) & (bank < n)))
        for index, neighbor in enumerate(bank[10:]):
            initial = bank[index % 10]
            self.assertEqual(len(set(neighbor) - set(initial)), 1)
            self.assertEqual(len(set(initial) - set(neighbor)), 1)
        repeated, repeated_names = make_candidate_bank(task_a, task_b, -task_b, task_b, k, 12)
        np.testing.assert_array_equal(bank, repeated)
        self.assertEqual(names, repeated_names)
        other, _ = make_candidate_bank(task_a, task_b, -task_b, task_b, k, 13)
        np.testing.assert_array_equal(bank[:7], other[:7])
        self.assertFalse(np.array_equal(bank[7:], other[7:]))
        # Duplicate initial rankings are intentionally retained in the bank.
        np.testing.assert_array_equal(bank[0], bank[3])

    def test_panel_matches_existing_balanced_order_and_excludes_other_splits(self):
        images = [
            {"image_id": f"c{class_id}_{i:02}", "class_id": class_id, "split": "val"}
            for class_id, count in [(0, 7), (1, 5), (2, 6), (3, 4)]
            for i in range(count)
        ]
        images.append({"image_id": "train_image", "class_id": 1, "split": "train"})
        tasks = [
            {"task_id": "01_03", "class_a": 1, "class_b": 3, "split": "val"},
            {"task_id": "01_02", "class_a": 1, "class_b": 2, "split": "val"},
            {"task_id": "00_01", "class_a": 0, "class_b": 1, "split": "train"},
        ]
        selected, selected_tasks = select_panel(images, tasks, 3)
        self.assertEqual(Counter(row["class_id"] for row in selected), {1: 3, 2: 3, 3: 3})
        self.assertEqual([row["image_id"] for row in selected],
                         sorted(row["image_id"] for row in selected))
        self.assertEqual([row["task_id"] for row in selected_tasks], ["01_02", "01_03"])
        full_rows = list(balanced_task_rows(images, tasks, "val", "val"))
        for task in selected_tasks:
            for class_id in (task["class_a"], task["class_b"]):
                expected = [row["image_id"] for row in full_rows
                            if row["task_id"] == task["task_id"] and row["class_id"] == class_id][:3]
                actual = [row["image_id"] for row in selected if row["class_id"] == class_id]
                self.assertEqual(set(actual), set(expected))
        repeated = select_panel(list(reversed(images)), list(reversed(tasks)), 3)
        self.assertEqual((selected, selected_tasks), repeated)


if __name__ == "__main__":
    unittest.main()
