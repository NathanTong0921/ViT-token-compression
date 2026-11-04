import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    "familiarity_prior", Path(__file__).resolve().parents[1] / "scripts/analyze_familiarity_prior.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class FamiliarityPriorTests(unittest.TestCase):
    def test_folds_keep_all_tasks_of_an_image_together(self):
        ids = np.array(["b", "a", "b", "c", "d", "a", "e"])
        folds, lookup = analysis.assign_image_folds(ids, 3, 2301)
        np.testing.assert_array_equal(folds, analysis.assign_image_folds(ids, 3, 2301)[0])
        for image in np.unique(ids):
            self.assertEqual(len(np.unique(folds[ids == image])), 1)
        self.assertEqual(set(lookup), set(ids))

    def test_held_values_cannot_change_their_fitted_offsets(self):
        tasks = np.array(["a"] * 4 + ["b"] * 4)
        folds = np.array([0, 1, 0, 1] * 2)
        full = np.arange(8) / 10
        q = full + np.array([.2] * 4 + [-.1] * 4)
        for kind in ("global", "task"):
            fitted, records = analysis.crossfit_offsets(q, full, tasks, folds, kind)
            changed = q.copy();changed[folds == 0] += 10
            alternative, _ = analysis.crossfit_offsets(changed, full, tasks, folds, kind)
            np.testing.assert_array_equal(fitted[folds == 0], alternative[folds == 0])
            self.assertTrue(all(r["calibration_rows"] > 0 for r in records))
        fitted, _ = analysis.crossfit_offsets(q, full, tasks, folds, "task")
        np.testing.assert_allclose(q - fitted, full)

    def test_auc_is_invariant_to_task_constants(self):
        truth = np.array([True, True, False, False] * 2)
        tasks = np.array(["a"] * 4 + ["b"] * 4)
        q = np.array([.2, .1, -.1, -.2, .2, -.1, .1, -.2])
        shifted = q + np.where(tasks == "a", 5., -3.)
        np.testing.assert_allclose(analysis.task_auc(q, truth, tasks), analysis.task_auc(shifted, truth, tasks))

    def test_pointwise_role_margin_is_direction_shift_not_separation(self):
        sign = np.array([1, -1, 1, -1])
        effects0, effects1 = np.array([.1, .2, -.1, -.3]), np.array([.5, .4, .1, -.2])
        true_effect0, true_effect1 = sign * effects0, sign * effects1
        np.testing.assert_allclose(sign * (true_effect1 - true_effect0), effects1 - effects0)

    def test_pure_prior_produces_role_effect_without_better_discrimination(self):
        n = 8
        truth = np.array([True, False] * 4)
        sign = np.where(truth, 1., -1.)
        full = .4 * sign
        arms = ["real_text", "permuted_2097", "permuted_2098", "image_only_wide"]
        q = np.broadcast_to(full, (2, 1, 4, n)).copy()
        q[1, 0, 0] += .6
        panel = {"image_ids": np.array([f"x{i}" for i in range(n)]), "weights": np.ones(n) / n,
                 "task_ids": np.array(["a"] * 4 + ["b"] * 4), "true_original_held": truth}
        folds = np.array([0, 1, 0, 1] * 2)
        offsets = {kind: np.empty_like(q) for kind in ("global", "task")}
        for kind in offsets:
            for p in range(2):
                for a in range(4):
                    offsets[kind][p, 0, a], _ = analysis.crossfit_offsets(q[p, 0, a], full, panel["task_ids"], folds, kind)
        result, rows, auc = analysis.summarize(q, offsets, panel, arms, [2041], full,
                                             np.zeros(n, int), np.ones(n, int), 50)
        decomposition = result["decomposition"]["real_minus_identity"]
        self.assertAlmostEqual(decomposition["midpoint_shift"]["estimate"], .6)
        self.assertAlmostEqual(decomposition["half_separation_change"]["estimate"], 0.)
        self.assertAlmostEqual(result["correction"]["raw"]["contrasts"]["accuracy_pp"]["real_minus_identity"]["seen_minus_held"]["estimate"], 50.)
        self.assertAlmostEqual(result["correction"]["task"]["contrasts"]["accuracy_pp"]["real_minus_identity"]["seen_minus_held"]["estimate"], 0.)
        self.assertAlmostEqual(result["offset_prediction"]["real_minus_identity"]["task"]["fraction_MSE_reduction_vs_zero"], 1.)
        np.testing.assert_array_equal(auc, np.ones_like(auc))
        import json
        json.dumps(result, allow_nan=False)

    def test_exact_score_ties_follow_original_class_index_rule(self):
        q = np.zeros(2)
        truth = np.array([True, True])
        correct = analysis.correct_from_q(q, truth, np.array([0, 2]), np.array([1, 1]))
        np.testing.assert_array_equal(correct, [1, 0])


if __name__ == "__main__":
    unittest.main()
