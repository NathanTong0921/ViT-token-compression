import json
import unittest

import numpy as np

from token_compression.pruning_eval import evaluate_variant


class PruningEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.images = [
            {"image_id": "a", "class_id": 0, "split": "val"},
            {"image_id": "b", "class_id": 0, "split": "val"},
            {"image_id": "c", "class_id": 1, "split": "val"},
            {"image_id": "d", "class_id": 1, "split": "val"},
            {"image_id": "e", "class_id": 2, "split": "val"},
        ]
        self.tasks = [{"task_id": "00_01", "class_a": 0, "class_b": 1, "split": "train"}]
        self.labels = np.array([0, 0, 1, 1, 2])
        self.baseline = np.array([
            [.8, .2, .1], [.2, .8, .1], [.2, .8, .1], [.1, .9, .2], [.1, .2, .8]
        ], dtype=np.float32)
        self.text = np.eye(3, dtype=np.float32)

    def evaluate(self, scores, edges=(0., .3, .5)):
        return evaluate_variant(scores, self.baseline, self.text, self.labels,
                                [row["image_id"] for row in self.images], self.images,
                                self.tasks, "val", ["train"], 2028, edges)

    def test_new_errors_and_recoveries_are_not_net_accuracy(self):
        scores = self.baseline.copy()
        scores[0] = [.2, .8, .1]  # A new error.
        scores[1] = [.8, .2, .1]  # A recovered baseline error.
        summary, tasks, strata = self.evaluate(scores)
        self.assertEqual(summary["net_accuracy_change_pp"], 0.)
        self.assertEqual(summary["baseline_correct_to_wrong_count"], 1)
        self.assertEqual(summary["baseline_wrong_to_correct_count"], 1)
        binary = summary["binary"]["train"]
        self.assertEqual(binary["new_errors"], 1)
        self.assertEqual(binary["recoveries"], 1)
        self.assertAlmostEqual(binary["new_error_rate_among_baseline_correct"], 1 / 3)
        self.assertEqual(binary["recovery_rate_among_baseline_wrong"], 1.)
        self.assertEqual(tasks[0]["n"], 4)
        self.assertEqual(tasks[0]["baseline_correct"], 3)
        self.assertEqual(strata[0]["stratum"], "Q1")
        self.assertEqual(strata[0]["n"], 1)
        self.assertEqual(strata[0]["baseline_correct"], 0)
        self.assertEqual(strata[0]["recoveries"], 1)
        self.assertLess(strata[0]["mean_geometric_margin_loss"], 0.)
        json.dumps((summary, tasks, strata), allow_nan=False)

    def test_keep_all_has_zero_change_and_empty_strata_are_explicit(self):
        summary, tasks, strata = self.evaluate(self.baseline.copy())
        self.assertEqual(summary["net_accuracy_change_pp"], 0.)
        self.assertEqual(summary["baseline_correct_to_wrong_count"], 0)
        self.assertEqual(summary["baseline_wrong_to_correct_count"], 0)
        self.assertEqual(tasks[0]["mean_geometric_margin_loss"], 0.)
        binary = summary["binary"]["train"]
        self.assertEqual(binary["new_errors"], 0)
        self.assertEqual(binary["recoveries"], 0)
        self.assertEqual(binary["mean_geometric_margin_loss"], 0.)
        self.assertEqual(strata[1]["n"], 0)
        self.assertIsNone(strata[1]["accuracy"])
        self.assertIsNone(strata[1]["mean_geometric_margin_loss"])
        json.dumps((summary, tasks, strata), allow_nan=False)

    def test_boundary_uses_higher_quartile_and_ties_follow_candidate_order(self):
        self.baseline[0] = [.5, .5, .1]  # Class 0 wins the tie.
        self.baseline[2] = [.5, .5, .1]  # Class 1 loses the tie.
        _, tasks, strata = self.evaluate(self.baseline.copy(), edges=(0., .3, .5))
        self.assertEqual(strata[1]["n"], 2)
        self.assertEqual(strata[1]["baseline_correct"], 1)
        self.assertEqual(tasks[0]["baseline_correct"], 2)

    def test_unequal_class_counts_are_balanced_per_task(self):
        self.tasks = [{"task_id": "00_02", "class_a": 0, "class_b": 2, "split": "train"}]
        summary, tasks, _ = self.evaluate(self.baseline.copy())
        self.assertEqual(tasks[0]["n"], 2)
        self.assertEqual(summary["binary"]["train"]["example_count"], 2)


if __name__ == "__main__":
    unittest.main()
