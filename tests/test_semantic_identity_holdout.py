"""Coverage and class-side controls for the supplementary development panel."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_semantic_identity_holdout import expanded_tasks
from analyze_semantic_identity_holdout import summarize_holdout_panel


class SemanticIdentityHoldoutTests(unittest.TestCase):
    def test_expansion_uses_only_pairs_removed_from_actual_training(self):
        def row(name, a, b):
            return dict(task_id=name, class_a=str(a), class_b=str(b))
        train = [row("train01", 0, 1), row("train02", 0, 2), row("train23", 2, 3)]
        val = [row("val13", 1, 3), row("val04", 0, 4)]
        expanded = expanded_tasks(train, val, [2, 3])
        self.assertEqual([r["task_id"] for r in expanded], ["val13", "val04", "train02", "train23"])
        self.assertNotIn("train01", [r["task_id"] for r in expanded])
        self.assertEqual([r["task_source"] for r in expanded],
                         ["original_val", "original_val", "removed_train", "removed_train"])
        known_pairs = [r for r in expanded if int(r["class_a"]) not in (2, 3)
                       and int(r["class_b"]) not in (2, 3)]
        self.assertTrue(all(r["task_source"] == "original_val" for r in known_pairs))

    def test_mixed_true_class_split_preserves_tasks_and_detects_asymmetry(self):
        n = 40
        panel = dict(image_ids=np.array([f"image{i}" for i in range(n)]),
                     weights=np.full(n, 1 / n), strata=np.full(n, "known_new"),
                     class_ids=np.repeat([0, 1, 2, 3], 10),
                     task_ids=np.tile(np.repeat(["taskA", "taskB"], 10), 2),
                     task_sources=np.tile(np.repeat(["original_val", "removed_train"], 10), 2),
                     full_correct=np.ones(n), full_margin=np.full(n, .5))
        arms = ["real_text", "permuted_2097", "permuted_2098", "image_only_wide"]
        predictions = {(2041, arm): {"correct": np.ones(n), "margin": np.full(n, .2)} for arm in arms}
        predictions[(2041, "real_text")]["margin"][20:] = .3
        report = summarize_holdout_panel(panel, predictions, arms, [2041], [2, 3], replicates=30)
        known = report["groups"]["known_new_true_known"]
        new = report["groups"]["known_new_true_new"]
        self.assertEqual(known["tasks"], 2)
        self.assertEqual(new["tasks"], 2)
        self.assertEqual(new["task_sources"], {"original_val": 1, "removed_train": 1})
        self.assertEqual(new["full_model_accuracy_percent"], 100.)
        self.assertAlmostEqual(known["metrics"]["margin"]["contrasts"]["real_minus_permuted_mean"]["estimate"], 0.)
        self.assertAlmostEqual(new["metrics"]["margin"]["contrasts"]["real_minus_permuted_mean"]["estimate"], .1)
        self.assertAlmostEqual(report["mixed_true_class_interaction"]["margin"]["estimate"], .1)

    def test_original_panel_defaults_to_original_validation_source(self):
        n = 20
        panel = dict(image_ids=np.array([f"image{i}" for i in range(n)]), weights=np.full(n, 1 / n),
                     strata=np.full(n, "new_new"), class_ids=np.repeat([2, 3], 10),
                     task_ids=np.full(n, "val23"), full_correct=np.ones(n), full_margin=np.ones(n))
        arms = ["real_text", "permuted_2097", "permuted_2098", "image_only_wide"]
        predictions = {(2041, arm): {"correct": np.ones(n), "margin": np.ones(n)} for arm in arms}
        report = summarize_holdout_panel(panel, predictions, arms, [2041], [2, 3], replicates=20)
        self.assertEqual(report["groups"]["new_new"]["task_sources"], {"original_val": 1})
        self.assertEqual(report["mixed_true_class_interaction"], {})


if __name__ == "__main__":
    unittest.main()
