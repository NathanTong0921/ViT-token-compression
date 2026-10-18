import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_selectors.py"
SPEC = importlib.util.spec_from_file_location("evaluate_selectors", SCRIPT)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class SelectorEvaluationTests(unittest.TestCase):
    def test_same_image_panel_has_only_complete_balanced_pairs(self):
        ids = np.array(["a1", "b1", "c1", "a2", "b2", "c2"])
        labels = np.array([0, 1, 2, 0, 1, 2])
        seen = [{"task_id": "00_01", "class_a": 0, "class_b": 1},
                {"task_id": "00_03", "class_a": 0, "class_b": 3},
                {"task_id": "01_02", "class_a": 1, "class_b": 2}]
        unseen = [{"task_id": "00_02", "class_a": 2, "class_b": 0}]
        rows = evaluation.evaluation_examples(ids, labels, seen, unseen)
        self.assertEqual(len(rows), 12)
        self.assertNotIn("00_03", {row["task_id"] for row in rows})
        for task in {row["task_id"] for row in rows}:
            selected = [row for row in rows if row["task_id"] == task]
            self.assertEqual(len(selected), 4)
            self.assertEqual(sum(row["class_id"] == row["class_a"] for row in selected), 2)
            for row in selected:
                self.assertLess(row["class_a"], row["class_b"])
                self.assertEqual(row["image_id"], ids[row["image_index"]])
        self.assertEqual({row["task_split"] for row in rows}, {"seen", "unseen"})

    def test_seen_and_unseen_task_overlap_is_not_silently_evaluated(self):
        task = [{"task_id": "00_01", "class_a": 0, "class_b": 1}]
        with self.assertRaises(AssertionError):
            evaluation.evaluation_examples(["a", "b"], [0, 1], task, task)

    def test_margin_orientation_and_index_ties_match_binary_classifier(self):
        text = np.eye(3)
        scores = np.array([[.4, .4, -.1], [.4, .4, -.1], [.2, -.3, .8]])
        labels = np.array([0, 1, 2])
        pairs = np.array([[0, 1], [0, 1], [0, 2]])
        margins, correct = evaluation.binary_outcomes(scores, labels, pairs, text)
        np.testing.assert_array_equal(correct, [1, 0, 1])
        np.testing.assert_allclose(margins, [0, 0, .6 / np.sqrt(2)])

    def test_utility_correlations_are_shift_scale_invariant_and_mark_constants(self):
        utilities = np.array([[1., 3., 2.], [1., 3., 2.], [1., 3., 2.]])
        logits = np.array([[5., 9., 7.], [-1., -3., -2.], [7., 7., 7.]])
        values, valid = evaluation.utility_correlations(logits, utilities)
        np.testing.assert_allclose(values, [1, -1, 0], atol=1e-12)
        np.testing.assert_array_equal(valid, [True, True, False])


if __name__ == "__main__":
    unittest.main()
