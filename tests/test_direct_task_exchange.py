import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_direct_task_exchange import (
    METHODS, CONTRAST_NAMES, donor_rows, exchange_outcomes, exchange_contrasts, selection_overlap,
)


class DirectTaskExchangeTests(unittest.TestCase):
    def test_donors_stay_with_image_and_exclude_target_task(self):
        ids = np.asarray(["b", "a", "b", "a", "a", "b", "b", "a"])
        tasks = np.asarray(["t1", "t1", "t2", "t2", "t3", "t3", "t4", "t4"])
        labels = np.where(ids == "a", 0, 1)
        donors = donor_rows(ids, tasks, labels)
        for row in range(len(ids)):
            self.assertNotIn(row, donors[row])
            self.assertEqual(set(donors[row]), set(np.flatnonzero(ids == ids[row])) - {row})
        with self.assertRaises(AssertionError):
            donor_rows(ids[:-1], tasks[:-1], labels[:-1])

    def test_outcomes_are_averaged_after_target_classification(self):
        ids = np.asarray(["image"] * 4)
        tasks = np.asarray(["task1", "task2", "task3", "task4"])
        labels = np.zeros(4, dtype=int)
        pairs = np.asarray([[0, 1], [0, 2], [0, 3], [0, 4]])
        scores = np.zeros((4, 5))
        scores[:, 0] = [.7, .2, .2, -1.]
        # Donor 1 and 2 correctly classify target pair (0,1), donor 3 does not.
        # Their mean score is negative: averaging logits would wrongly return zero.
        donors = donor_rows(ids, tasks, labels)
        result = exchange_outcomes(scores, labels, pairs, np.eye(5), 1., donors)
        self.assertEqual(result[0, 0, 0], 1.)
        self.assertAlmostEqual(result[1, 0, 0], 2. / 3.)
        self.assertAlmostEqual(result[1, 0, 1], -.2)

    def test_image_only_scores_have_zero_exchange_effect(self):
        scores = np.tile([.8, .2, .3, .9, .1], (4, 1))
        pairs = np.asarray([[0, 1], [0, 2], [0, 3], [0, 4]])
        labels = np.zeros(4, dtype=int)
        donors = donor_rows(np.asarray(["i"] * 4), np.asarray(["a", "b", "c", "d"]), labels)
        outcome = exchange_outcomes(scores, labels, pairs, np.eye(5), 2., donors)
        np.testing.assert_allclose(outcome[0], outcome[1], atol=1e-15)

    def test_selection_overlap_averages_each_donor_intersection(self):
        selected = np.asarray([[0, 1], [0, 2], [2, 3], [0, 1]])
        donors = donor_rows(np.asarray(["i"] * 4), np.asarray(["a", "b", "c", "d"]),
                            np.zeros(4, dtype=int))
        overlap = selection_overlap(selected, donors, patch_count=4)
        np.testing.assert_allclose(overlap, [.5, .5, 1. / 6., .5])

    def test_mapping_change_separates_own_gain_and_swapped_deterioration(self):
        values = np.zeros((1, 6, 2, 1, 4))
        # C own improves by 1; swapped C falls by 1; U improves by 2.
        # Mapping effect therefore grows by 2 while the C-U increment is -1.
        for method, own, swapped in (("kl_C", 7, 6), ("direct_C", 8, 5),
                                     ("kl_U", 5, 5), ("direct_U", 7, 7)):
            values[:, METHODS.index(method), 0] = own
            values[:, METHODS.index(method), 1] = swapped
        contrasts = exchange_contrasts(values)
        expected = dict(mapping_change_direct_vs_kl=2, own_C_change_direct_vs_kl=1,
                        swapped_C_change_direct_vs_kl=-1, U_change_direct_vs_kl=2,
                        CU_change_direct_vs_kl=-1, swapped_remaining_change=-3)
        for name, target in expected.items():
            np.testing.assert_array_equal(contrasts[:, CONTRAST_NAMES.index(name)], target)


if __name__ == "__main__":
    unittest.main()
