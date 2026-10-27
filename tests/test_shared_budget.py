import unittest
import numpy as np

from token_compression.shared_budget import search_common_library


class SharedBudgetTests(unittest.TestCase):
    def test_shared_tradeoff_and_feasibility_are_distinct(self):
        text = np.eye(3)
        # Candidates 0/1 specialize; candidate 2 solves both. Mean-margin
        # maximization need not select the both-correct candidate.
        scores = np.array([[.1, -.9, .2], [.1, .2, -.9], [.1, 0., 0.]])
        result = search_common_library(scores, 0, [1, 2], text)
        np.testing.assert_array_equal(result['selected_candidate_indices'], [0, 1, 0])
        self.assertTrue(result['any_shared_both_correct'])
        self.assertFalse(result['shared_both_correct'])
        self.assertEqual(result['separate_accuracy'], 1.)
        self.assertEqual(result['shared_accuracy'], .5)
        self.assertGreater(result['separate_minus_shared_margin'], 0.)

    def test_shared_optimum_can_solve_both_without_specialization(self):
        result = search_common_library(np.array([[.8, .1, .2], [.3, .2, .2]]), 0, [1, 2], np.eye(3))
        np.testing.assert_array_equal(result['selected_candidate_indices'], [0, 0, 0])
        self.assertEqual(result['separate_minus_shared_margin'], 0.)

    def test_ties_follow_class_index_and_negative_margins_remain(self):
        result = search_common_library(np.array([[.2, .2, .4]]), 1, [0, 2], np.eye(3))
        self.assertEqual(result['shared_accuracy'], 0.)
        self.assertLess(result['shared_margin'], 0.)


if __name__ == '__main__':
    unittest.main()
