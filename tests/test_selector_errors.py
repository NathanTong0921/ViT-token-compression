import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_selector_errors.py"
SPEC = importlib.util.spec_from_file_location("analyze_selector_errors", SCRIPT)
errors = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(errors)


class SelectorErrorTests(unittest.TestCase):
    def make_rows(self):
        rows = []
        # The first case reverses direction across seeds; the second is always repaired.
        cases = [("image_a", "00_01", .1, 1, [(0, 1), (1, 0)]),
                 ("image_b", "00_02", -.2, 0, [(0, 1), (0, 1)])]
        for split in ("seen", "unseen"):
            for image, task, margin, base_correct, outcomes in cases:
                common = {"task_split": split, "image_id": image, "task_id": task,
                          "class_id": 0, "class_a": 0, "class_b": int(task[-1]),
                          "base_margin": margin, "base_correct": base_correct}
                for method in ("full_model", "attention"):
                    rows.append({**common, "seed": -1, "method": method,
                                 "margin": margin, "correct": base_correct})
                for seed, (shared, conditioned) in zip((1, 2), outcomes):
                    for method, correct in (("image_only", shared), ("conditioned", conditioned)):
                        rows.append({**common, "seed": seed, "method": method,
                                     "margin": .3 if correct else -.1, "correct": correct})
        return rows

    def test_transition_direction_and_all_case_identity_are_preserved(self):
        output, changed = errors.analyze(self.make_rows()[::-1], [.1, .2, .3])
        info = output["task_splits"]["unseen"]["strata"]["all"]
        first, second = info["per_seed"]
        self.assertEqual(first["conditioned_vs_image_only"]["repaired_errors"], 2)
        self.assertEqual(first["conditioned_vs_image_only"]["new_errors"], 0)
        self.assertEqual(second["conditioned_vs_image_only"]["repaired_errors"], 1)
        self.assertEqual(second["conditioned_vs_image_only"]["new_errors"], 1)
        self.assertEqual(second["conditioned_vs_full_model"]["new_errors"], 1)
        self.assertEqual(second["conditioned_vs_full_model"]["repaired_errors"], 1)
        self.assertEqual(len(changed), 8)
        self.assertEqual({(r["image_id"], r["task_id"]) for r in changed},
                         {("image_a", "00_01"), ("image_b", "00_02")})
        patterns = output["task_splits"]["unseen"]["correctness_seed_patterns"]["counts"]
        self.assertEqual(patterns, {"direction_reversal": 1, "persistent_repair": 1})
        np.testing.assert_allclose(info["seed_averaged_conditional_minus_image_only"]["margin"]["mean"], .2)
        json.dumps(output, allow_nan=False)

    def test_fixed_boundaries_and_empty_strata(self):
        edges = [.1, .2, .3]
        self.assertEqual([errors.quartile(x, edges) for x in [-1., .0999, .1, .2, .3]],
                         ["Q1", "Q1", "Q2", "Q3", "Q4"])
        output, changed = errors.analyze(self.make_rows(), edges)
        bins = output["task_splits"]["seen"]["strata"]
        self.assertEqual(bins["Q1"]["example_count"], 1)
        self.assertEqual(bins["Q2"]["example_count"], 1)
        self.assertEqual(bins["Q3"]["example_count"], 0)
        self.assertIsNone(bins["Q3"]["per_seed"][0]["conditioned_minus_image_only_margin"]["mean"])

    def test_pairing_rejects_missing_duplicate_or_inconsistent_keys(self):
        rows = self.make_rows()
        with self.assertRaises(AssertionError):
            errors.analyze(rows[:-1], [.1, .2, .3])
        with self.assertRaises(AssertionError):
            errors.analyze(rows + [rows[-1]], [.1, .2, .3])
        changed = [dict(row) for row in rows]
        changed[-1]["base_margin"] = 99.
        with self.assertRaises(AssertionError):
            errors.analyze(changed, [.1, .2, .3])


if __name__ == "__main__":
    unittest.main()
