import unittest

import numpy as np

from token_compression.selection_analysis import SHARED_METHODS, cluster_summary, fit_h2, grouped_folds, row_metrics, summarize_single_tasks


class SelectionAnalysisTests(unittest.TestCase):
    def make_row(self):
        row = {"image_id": "a", "base_margin_a": .2, "base_margin_b": .1,
               "angle_degrees": 45, "overlap_fraction_task_ab": .5,
               "selection_feature_distance": .2, "exchange_alignment": .3,
               "exchange_projection_residual": 0.}
        for method in ("task_a", "task_b", "task_search_a", "task_search_b") + SHARED_METHODS:
            for side in ("a", "b"):
                row[f"margin_{method}_{side}"] = .1
                row[f"correct_{method}_{side}"] = 1
        return row

    def test_exchange_uses_matching_and_crossed_selections(self):
        row = self.make_row()
        row.update(margin_task_a_a=.4, margin_task_a_b=.1,
                   margin_task_b_a=-.1, margin_task_b_b=.2,
                   correct_task_a_a=1, correct_task_a_b=1,
                   correct_task_b_a=0, correct_task_b_b=1)
        metrics = row_metrics([row])
        self.assertAlmostEqual(metrics["own_margin"][0], .3)
        self.assertAlmostEqual(metrics["swapped_margin"][0], 0.)
        self.assertAlmostEqual(metrics["exchange_margin_loss"][0], .3)
        self.assertAlmostEqual(metrics["exchange_accuracy_loss"][0], .5)
        self.assertAlmostEqual(metrics["own_minus_shared_proxy_margin"][0], .2)

    def test_repeated_rows_do_not_increase_independent_sample_size(self):
        original = cluster_summary(["a", "b", "c"], [0., 1., 3.], 200, 11)
        repeated = cluster_summary(["a"] * 10 + ["b", "c"], [0.] * 10 + [1., 3.], 200, 11)
        self.assertEqual(repeated["row_count"], 12)
        self.assertEqual(repeated["image_count"], 3)
        for key in ("mean", "ci95_low", "ci95_high"):
            self.assertEqual(original[key], repeated[key])

    def test_grouped_cv_never_splits_an_image(self):
        ids = np.repeat([f"image_{index}" for index in range(13)], [1, 2, 3, 4, 1, 3, 2, 4, 1, 2, 3, 4, 5])
        folds = grouped_folds(ids)
        self.assertEqual(len(np.unique(folds)), 5)
        np.testing.assert_array_equal(folds, grouped_folds(ids))
        for image_id in np.unique(ids):
            self.assertEqual(len(np.unique(folds[ids == image_id])), 1)

    def test_rank_deficient_controls_still_report_identifiable_angle(self):
        rows = []
        for i in range(15):
            row = self.make_row()
            row.update(image_id=f"image_{i}", angle_degrees=10 + 4 * i)
            rows.append(row)
        y = np.asarray([.7 + .2 * row["angle_degrees"] / 90 for row in rows])
        result, _, _ = fit_h2(rows, y, bootstrap_replicates=20)
        self.assertTrue(result["rank_deficient"])
        self.assertTrue(result["angle_identifiable_after_controls"])
        self.assertAlmostEqual(result["angle_coefficient"], .2)
        self.assertLess(result["cv_controls_plus_angle_mse"], 1e-20)
        self.assertGreater(result["cv_mse_reduction_from_angle"], 0.)

    def test_single_task_panel_requires_both_class_sides_for_balanced_metric(self):
        rows = [{"keep_patch_tokens": 49, "method": "task_rank", "image_id": name,
                 "task_id": task, "class_id": label, "correct": correct}
                for name, task, label, correct in
                [("a", "0_1", 0, 1), ("b", "0_1", 0, 1), ("c", "0_1", 1, 0),
                 ("d", "0_2", 0, 1)]]
        result = summarize_single_tasks(rows)[0]
        self.assertEqual(result["two_sided_task_count"], 1)
        self.assertAlmostEqual(result["two_sided_task_macro_balanced_accuracy"], .5)
        self.assertAlmostEqual(result["panel_accuracy_micro"], .75)


if __name__ == "__main__":
    unittest.main()
