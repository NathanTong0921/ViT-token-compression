import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_budget_ranking.py"
SPEC = importlib.util.spec_from_file_location("compare_budget_ranking", SCRIPT)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class BudgetComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.results = self.root / "results"
        self.previous = self.root / "legacy"
        self.paths = {"legacy_block": self.previous,
                      "near_full": self.results / "near_full",
                      "budget_matched": self.results / "budget_matched"}
        self.margins = {"legacy_block": [[.1, .1], [.1, .1]],
                        "near_full": [[.1, -.1], [.2, .2]],
                        "budget_matched": [[.2, .1], [.4, .4]]}
        # The legacy bank supplies the strongest shared candidate. Each arm
        # must be compared to that same union winner, regardless of its bank.
        banks = {
            "legacy_block": [[[.9, .1, .1], [.2, .7, .5]],
                             [[.8, .1, .1], [.2, .6, .5]]],
            "near_full": [[[.7, -.3, .2], [.7, .2, -.3]],
                          [[.6, -.3, .2], [.6, .2, -.3]]],
            "budget_matched": [[[.5, .1, .1], [.4, .0, .3]],
                               [[.4, .1, .1], [.3, .0, .2]]],
        }
        self.singles = {}
        for arm, path in self.paths.items():
            path.mkdir(parents=True)
            rows, exchange = [], []
            for i, image_id in enumerate(("image_a", "image_b")):
                margins = self.margins[arm][i]
                for side in range(2):
                    row = {"keep_patch_tokens": 49, "method": "task_rank",
                           "image_id": image_id, "task_id": f"0_{side + 1}",
                           "margin": margins[side], "correct": int(margins[side] > 0),
                           "base_correct": 1}
                    rows.append(row)
                    rows.append(dict(row, method="attention", margin=.1, correct=1))
                exchange.append({"keep_patch_tokens": 49, "image_id": image_id,
                                 "class_id": 0, "competitor_a": 1, "competitor_b": 2,
                                 "margin_task_a_a": margins[0], "margin_task_b_b": margins[1],
                                 "correct_task_a_a": int(margins[0] > 0),
                                 "correct_task_b_b": int(margins[1] > 0)})
            # CSV order may differ; task keys, rather than row positions, pair observations.
            self.singles[arm] = rows[::-1] if arm == "budget_matched" else rows
            write_rows(path / "single_task_rows.csv", self.singles[arm])
            write_rows(path / "exchange_rows.csv", exchange)
            np.savez(path / "candidate_bank_k49.npz", image_ids=["image_a", "image_b"],
                     cases=[[0, 0, 1, 2], [1, 0, 1, 2]], cosine_scores=banks[arm])
        np.savez(self.results / "evidence.npz", text_features=np.eye(3))
        prediction = []
        for arm, mse, correlation in (("near_full", .04, .3), ("budget_matched", .01, .8)):
            for image_id in ("image_a", "image_b"):
                prediction.append({"keep_patch_tokens": 49, "arm": arm,
                                   "image_id": image_id, "raw_mse": mse,
                                   "centered_mse": mse / 2, "pearson": correlation,
                                   "pearson_valid": 1})
        write_rows(self.results / "holdout_prediction_rows.csv", prediction)

    def tearDown(self):
        self.temp.cleanup()

    def test_paired_contrasts_keep_direction_and_all_metrics_are_finite(self):
        output = comparison.compare(self.results, self.previous)
        # Strict serialization rejects NaN/Inf anywhere, including bootstrap intervals.
        json.dumps(output, allow_nan=False)
        budget = output["budgets"][0]
        contrasts = budget["paired_effects"]
        self.assertAlmostEqual(contrasts["budget_minus_near_full_correct"]["mean"], .25)
        self.assertAlmostEqual(contrasts["budget_minus_near_full_margin"]["mean"], .175)
        self.assertEqual(contrasts["budget_minus_near_full_margin"]["image_count"], 2)
        self.assertEqual(contrasts["budget_minus_near_full_margin"]["row_count"], 4)
        self.assertAlmostEqual(contrasts["budget_minus_legacy_block_margin"]["mean"], .175)
        prediction = budget["independent_mask_prediction"]
        self.assertAlmostEqual(prediction["budget_matched"]["pearson"]["mean"], .8)
        self.assertLess(prediction["budget_matched"]["raw_mse"]["mean"],
                        prediction["near_full"]["raw_mse"]["mean"])

    def test_every_arm_uses_identical_union_shared_reference(self):
        budget = comparison.compare(self.results, self.previous)["budgets"][0]
        self.assertEqual(budget["union_candidate_count"], 6)
        shared = budget["union_shared_reference"]
        for objective in ("mean", "maxmin"):
            self.assertAlmostEqual(shared[f"margin_{objective}"]["mean"], .75 / np.sqrt(2))
            self.assertEqual(shared[f"accuracy_{objective}"]["mean"], 1.)
            for arm in self.paths:
                for metric in ("margin", "accuracy"):
                    own = shared[f"{arm}_own_{metric}"]["mean"]
                    difference = shared[f"{arm}_own_minus_union_shared_{metric}_{objective}"]["mean"]
                    self.assertAlmostEqual(own - difference, shared[f"{metric}_{objective}"]["mean"])
        self.assertGreater(shared["task_search_margin"]["mean"], shared["margin_mean"]["mean"])

    def test_changed_task_or_candidate_case_keys_are_rejected(self):
        single_path = self.paths["budget_matched"] / "single_task_rows.csv"
        modified = [dict(row) for row in self.singles["budget_matched"]]
        next(row for row in modified if row["method"] == "task_rank")["task_id"] = "0_99"
        write_rows(single_path, modified)
        with self.assertRaises(AssertionError):
            comparison.compare(self.results, self.previous)
        write_rows(single_path, self.singles["budget_matched"])

        bank_path = self.paths["budget_matched"] / "candidate_bank_k49.npz"
        with np.load(bank_path) as bank:
            changed_bank = {key: bank[key].copy() for key in bank.files}
        changed_bank["cases"][0, 2:] = [2, 1]
        np.savez(bank_path, **changed_bank)
        with self.assertRaises(AssertionError):
            comparison.compare(self.results, self.previous)


if __name__ == "__main__":
    unittest.main()
