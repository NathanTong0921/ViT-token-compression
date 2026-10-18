import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluation = load_script("evaluate_selector_scale")
analysis = load_script("analyze_selector_scale")


class ScaleEvaluationTests(unittest.TestCase):
    def test_unequal_class_counts_are_equal_task_equal_class_weighted(self):
        ids, labels = ["a1", "a2", "b1", "c1"], [0, 0, 1, 2]
        seen = [{"task_id": "00_01", "class_a": 0, "class_b": 1}]
        unseen = [{"task_id": "00_02", "class_a": 0, "class_b": 2},
                  {"task_id": "01_02", "class_a": 1, "class_b": 2}]
        rows = evaluation.evaluation_examples(ids, labels, seen, unseen)
        for split, count in (("seen", 1), ("unseen", 2)):
            part = [r for r in rows if r["task_split"] == split]
            self.assertAlmostEqual(sum(r["task_class_weight"] for r in part), 1)
            for task in {r["task_id"] for r in part}:
                for label in {r["class_id"] for r in part if r["task_id"] == task}:
                    weight = sum(r["task_class_weight"] for r in part if r["task_id"] == task and r["class_id"] == label)
                    self.assertAlmostEqual(weight, 1 / (2 * count))

    def test_task_leakage_between_splits_is_rejected(self):
        task = [{"task_id": "00_01", "class_a": 0, "class_b": 1}]
        with self.assertRaises(AssertionError):
            evaluation.evaluation_examples(["a", "b"], [0, 1], task, task)

    def test_panel_excludes_incomplete_tasks_and_reweights_class_counts(self):
        rows = [
            {"image_id": "a1", "class_id": 0, "task_id": "01", "old_panel": 1},
            {"image_id": "a2", "class_id": 0, "task_id": "01", "old_panel": 1},
            {"image_id": "b", "class_id": 1, "task_id": "01", "old_panel": 1},
            {"image_id": "a1", "class_id": 0, "task_id": "02", "old_panel": 1},
            {"image_id": "c", "class_id": 2, "task_id": "02", "old_panel": 0},
        ]
        weights = analysis.panel_weights(rows, "old")
        np.testing.assert_allclose(weights, [.25, .25, .5, 0, 0])

    def test_weighted_cluster_bootstrap_preserves_its_weighted_estimand(self):
        result = analysis.weighted_cluster_summary(["a", "a", "b"], [.1, .1, .8], [0., 0., 1.], 200, 7)
        self.assertAlmostEqual(float(result["mean"][0]), .8)
        self.assertEqual(result["image_count"], 2)
        self.assertEqual(result["row_count"], 3)
        np.testing.assert_allclose([result["ci95_low"][0], result["ci95_high"][0]], [0., 1.])

    def test_repeating_seed_identical_rows_does_not_change_uncertainty(self):
        ids = np.array(["a", "a", "b", "c"])
        weights = np.array([.1, .2, .3, .4])
        values = np.array([.1, .3, .7, .9])
        original = analysis.weighted_cluster_summary(ids, weights, values, 300, 9)
        repeated = analysis.weighted_cluster_summary(np.tile(ids, 3), np.tile(weights / 3, 3), np.tile(values, 3), 300, 9)
        for key in ("mean", "ci95_low", "ci95_high"):
            np.testing.assert_allclose(original[key], repeated[key], atol=1e-14)

    def test_paired_coverage_interaction_and_old_new_panels(self):
        ids = ["a0", "a1", "a2", "b0", "b1", "c0", "c1"]
        labels = [0, 0, 0, 1, 1, 2, 2]
        seen = [{"task_id": "00_01", "class_a": 0, "class_b": 1}]
        unseen = [{"task_id": "00_02", "class_a": 0, "class_b": 2},
                  {"task_id": "01_02", "class_a": 1, "class_b": 2}]
        examples = evaluation.evaluation_examples(ids, labels, seen, unseen)

        def rows(method, seed, gain):
            return [{**r, "method": method, "seed": seed, "margin": gain + r["class_id"] / 10,
                     "correct": 1, "old_panel": int(r["image_id"].endswith("0"))} for r in examples]

        references = sum([rows(m, -1, 1) for m in ("full_model", "attention", "label_informed_teacher")], [])
        seeds = [1, 2, 3]
        small = sum([rows(m, s, .2 if m == "conditioned" else 0) for s in seeds
                     for m in ("conditioned", "image_only")], [])
        full = sum([rows(m, s, .5 if m == "conditioned" else .1) for s in seeds
                    for m in ("conditioned", "image_only")], [])
        result = analysis.analyze_rows(references, {"small": small, "full": full}, seeds, 40, 9)
        for panel in ("all", "old", "new"):
            effects = result["task_splits"]["unseen"]["panels"][panel]["task_class_balanced_effects"]
            estimate = effects["change_in_conditioning_advantage.margin"]
            self.assertAlmostEqual(estimate["mean"], .2)
            np.testing.assert_allclose(estimate["ci95"], [.2, .2], atol=1e-14)
            self.assertAlmostEqual(effects["conditioned_full_minus_small.margin"]["mean"], .3)
            self.assertAlmostEqual(effects["image_only_full_minus_small.margin"]["mean"], .1)
        self.assertEqual(result["task_splits"]["unseen"]["panels"]["old"]["eligible_image_count"], 3)
        self.assertEqual(result["task_splits"]["unseen"]["panels"]["new"]["eligible_image_count"], 4)


if __name__ == "__main__":
    unittest.main()
