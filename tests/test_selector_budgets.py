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
analysis = load_script("analyze_selector_budgets")


class SelectorBudgetTests(unittest.TestCase):
    def test_random_sets_are_nested_across_budgets_and_reproducible(self):
        a = evaluation.random_rankings(8, 196, 2055)
        b = evaluation.random_rankings(8, 196, 2055)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.dtype, np.float32)
        self.assertFalse(np.array_equal(a, evaluation.random_rankings(8, 196, 2056)))
        order = np.argsort(-a, axis=1, kind="stable")
        for row in order:
            self.assertTrue(set(row[:49]) < set(row[:98]) < set(row[:147]))

    def test_bootstrap_preserves_weighted_estimand_and_widens_family_interval(self):
        ids = np.array(["a", "a", "b", "c", "d", "e"])
        weights = np.array([.05, .05, .2, .2, .25, .25])
        values = np.array([0., 0., .4, .6, .7, 1.])
        result = analysis.weighted_estimates(ids, weights, values, ["value"], 500, 9, 3)["value"]
        self.assertAlmostEqual(result["mean"], float(np.dot(weights, values)))
        self.assertLessEqual(result["bonferroni_family_ci95"][0], result["ci95"][0])
        self.assertGreaterEqual(result["bonferroni_family_ci95"][1], result["ci95"][1])
        self.assertAlmostEqual(result["individual_interval_level"], 1 - .05 / 3)

    def test_repeated_seed_rows_do_not_shrink_image_intervals(self):
        ids = np.array(["a", "a", "b", "c"])
        weights = np.array([.1, .1, .3, .5])
        values = np.array([.1, .3, .4, .8])
        first = analysis.weighted_estimates(ids, weights, values, ["value"], 400, 8, 3)
        repeated = analysis.weighted_estimates(np.tile(ids, 3), np.tile(weights / 3, 3),
                                                np.tile(values, 3), ["value"], 400, 8, 3)
        for key in ("mean", "ci95", "bonferroni_family_ci95"):
            np.testing.assert_allclose(first["value"][key], repeated["value"][key], atol=1e-14)

    def test_prediction_alignment_rejects_missing_or_duplicate_pairs(self):
        rows = [{"image_id": "a", "task_id": "01", "method": "conditioned", "task_split": "unseen",
                 "seed": 1, "correct": 1, "margin": .2}]
        with self.assertRaises(AssertionError):
            analysis.aligned_runs(rows, "conditioned", "unseen", [("a", "01"), ("b", "01")], [1])
        with self.assertRaises(AssertionError):
            analysis.aligned_runs(rows * 2, "conditioned", "unseen", [("a", "01")], [1])

    def test_budget_effects_and_interactions_with_unequal_class_counts(self):
        ids, labels = ["a0", "a1", "a2", "b0", "b1", "c0", "c1"], [0, 0, 0, 1, 1, 2, 2]
        seen = [{"task_id": "01", "class_a": 0, "class_b": 1}]
        unseen = [{"task_id": "02", "class_a": 0, "class_b": 2}, {"task_id": "12", "class_a": 1, "class_b": 2}]
        examples = evaluation.evaluation_examples(ids, labels, seen, unseen)
        seeds, masks = [1, 2, 3], [5, 6, 7]
        rows_by_budget = {}
        for budget, gain in [(49, .1), (98, .3), (147, .6)]:
            records = []
            for method in analysis.METHODS:
                runs = seeds if method in ("conditioned", "image_only") else masks if method == "random" else [-1]
                for seed in runs:
                    for r in examples:
                        margin = 1 + r["class_id"] / 10 + (gain if method == "conditioned" else 0)
                        records.append({**r, "method": method, "seed": seed, "correct": 1,
                                        "margin": margin, "old_panel": 1})
            # Reversing records checks alignment rather than relying on CSV order.
            rows_by_budget[budget] = records[::-1]
        result = analysis.analyze_rows(rows_by_budget, seeds, masks, 100, 6)
        unseen_result = result["task_splits"]["unseen"]
        self.assertEqual(unseen_result["image_count"], 7)
        self.assertEqual(unseen_result["image_task_examples"], 9)
        for budget, gain in [(49, .1), (98, .3), (147, .6)]:
            estimate = unseen_result["budgets"][str(budget)]["conditioned_minus_image_only"]["margin"]
            self.assertAlmostEqual(estimate["mean"], gain)
            np.testing.assert_allclose(estimate["ci95"], [gain, gain], atol=1e-14)
            np.testing.assert_allclose(estimate["bonferroni_family_ci95"], [gain, gain], atol=1e-14)
        changes = unseen_result["secondary_change_in_conditioning_advantage"]
        self.assertAlmostEqual(changes["98_minus_49"]["margin"]["mean"], .2)
        self.assertAlmostEqual(changes["147_minus_49"]["margin"]["mean"], .5)

    def test_common_full_model_must_match_across_budgets(self):
        examples = evaluation.evaluation_examples(["a", "b", "c"], [0, 1, 2],
            [{"task_id": "01", "class_a": 0, "class_b": 1}], [{"task_id": "02", "class_a": 0, "class_b": 2}])
        def build(budget):
            return [{**r, "method": method, "seed": seed, "correct": 1, "old_panel": 1,
                     "margin": .3 if budget == 98 and method == "full_model" else .2}
                    for method in analysis.METHODS
                    for seed in ([1] if method in ("conditioned", "image_only") else [5] if method == "random" else [-1])
                    for r in examples]
        with self.assertRaises(AssertionError):
            analysis.analyze_rows({49: build(49), 98: build(98), 147: build(147)}, [1], [5], 20, 1)


if __name__ == "__main__":
    unittest.main()
