import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    "geometry_matched_conflict", Path(__file__).resolve().parents[1] / "scripts/analyze_geometry_matched_conflict.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def pair(key, different, angle, low, high):
    return {"key": key, "different_objects": different, "angle_degrees": angle,
            "margin_min": low, "margin_max": high, "tasks": []}


class GeometryMatchedConflictTests(unittest.TestCase):
    def test_match_enforces_all_three_calipers_and_is_order_invariant(self):
        candidates = [pair("same", False, 60, .03, .08),
                      pair("bad_angle", True, 66, .03, .08),
                      pair("bad_margin", True, 60, .03, .096),
                      pair("valid", True, 64, .04, .09)]
        match, diagnostic = analysis.choose_match(candidates)
        self.assertEqual(match["different"]["key"], "valid")
        self.assertEqual(diagnostic["joint_pass"], 1)
        reversed_match, _ = analysis.choose_match(list(reversed(candidates)))
        self.assertEqual(match, reversed_match)
        for key, threshold in (("angle_degrees", 5), ("margin_min", .015), ("margin_max", .015)):
            self.assertLessEqual(match["absolute_gaps"][key], threshold)

    def test_native_correct_panel_is_selected_without_compressed_scores(self):
        candidates = [pair("same_bad", False, 60, -.01, .03),
                      pair("different_bad", True, 60, -.01, .03),
                      pair("same_good", False, 80, .03, .08),
                      pair("different_good", True, 83, .03, .08)]
        general, _ = analysis.choose_match(candidates)
        correct, _ = analysis.choose_match(candidates, native_correct=True)
        self.assertEqual(general["same"]["key"], "same_bad")
        self.assertEqual(correct["same"]["key"], "same_good")
        no_match, diag = analysis.choose_match([pair("s", False, 40, .03, .08), pair("d", True, 60, .03, .08)])
        self.assertIsNone(no_match)
        self.assertEqual(diag["reason"], "no_angle_overlap")

    def test_candidate_construction_excludes_test_pairs_and_present_distractors(self):
        panel = {"sources": [{"class_id": 0}, {"class_id": 1}]}
        tasks = [{"task_id": f"{a}_{b}", "class_a": a, "class_b": b,
                  "split": "test" if (a, b) == (0, 2) else "val"}
                 for a in range(5) for b in range(a + 1, 5)]
        pairs, _ = analysis.candidates_for_panel(panel, tasks, np.eye(5), np.arange(5) / 10)
        for p in pairs:
            for task in p["tasks"]:
                self.assertNotEqual(task["task_id"], "0_2")
                self.assertIn(task["label"], [0, 1])
                self.assertNotIn(task["distractor"], [0, 1])
                self.assertAlmostEqual(task["native_margin"],
                    (task["label"] - task["distractor"]) / (10 * np.sqrt(2)))

    def test_expanded_pool_adds_train_tasks_without_admitting_test_tasks(self):
        panel = {"sources": [{"class_id": 0}, {"class_id": 1}]}
        tasks = [{"task_id": "0_2", "class_a": 0, "class_b": 2, "split": "train"},
                 {"task_id": "0_3", "class_a": 0, "class_b": 3, "split": "val"},
                 {"task_id": "1_2", "class_a": 1, "class_b": 2, "split": "val"},
                 {"task_id": "1_3", "class_a": 1, "class_b": 3, "split": "test"}]
        val_pairs, val_count = analysis.candidates_for_panel(panel, tasks, np.eye(4), np.arange(4) / 10)
        expanded_pairs, expanded_count = analysis.candidates_for_panel(panel, tasks, np.eye(4), np.arange(4) / 10, "train_val")
        self.assertEqual(val_count, 2)
        self.assertEqual(expanded_count, 3)
        expanded_ids = {task["task_id"] for pair in expanded_pairs for task in pair["tasks"]}
        self.assertIn("0_2", expanded_ids)
        self.assertNotIn("1_3", expanded_ids)
        self.assertFalse(any(t["task_id"] == "0_2" for p in val_pairs for t in p["tasks"]))

    def test_conflict_is_separate_success_without_joint_success(self):
        candidate = {"tasks": [{"label": 0, "distractor": 2, "direction_norm": 1.},
                               {"label": 1, "distractor": 2, "direction_norm": 1.}]}
        scores = np.array([[1., -1., 0], [-1., 1., 0]])
        out = analysis.bank_outcomes(scores, candidate)
        self.assertEqual(out["separately_feasible"], 1)
        self.assertEqual(out["common_feasible"], 0)
        self.assertEqual(out["conflict"], 1)
        out = analysis.bank_outcomes(np.vstack((scores, [.1, .1, 0])), candidate)
        self.assertEqual(out["conflict"], 0)
        self.assertEqual(out["common_feasible"], 1)

    def test_bootstrap_preserves_paired_direction_and_zero_match_result(self):
        rows = []
        for same, different in [(0, 1), (0, 0)]:
            row = {}
            for metric in ("conflict", "common_feasible", "separately_feasible", "mean_margin_gap"):
                row.update({f"same_{metric}": same, f"different_{metric}": different})
            rows.append(row)
        summary = analysis.paired_summary(rows, 100, 2111)
        self.assertEqual(summary["outcomes"]["conflict"]["different_minus_same"], 50.)
        self.assertEqual(summary["outcomes"]["conflict"]["different_only"], 1)
        self.assertEqual(analysis.paired_summary([], 100, 2111), {"composites": 0, "outcomes": {}})


if __name__ == "__main__":
    unittest.main()
