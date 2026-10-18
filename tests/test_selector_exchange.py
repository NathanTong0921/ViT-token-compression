"""Checks for fixed-model descriptor exchange and image-level uncertainty."""

import unittest

import numpy as np

from token_compression.selector_exchange import build_exchange_cases, summarize_exchange


def fixture():
    rows = []
    for image, tasks in (("x", ("01", "02")), ("y", ("01",))):
        for method in ("conditioned", "image_only", "full_model"):
            for task in tasks:
                score = ([.6, .2, .8] if task == "01" else [.6, .9, .1]) if method == "conditioned" else ([.6, .4, .5] if method == "image_only" else [.9, .3, .2])
                pos = ([0, 1, -1] if task == "01" else [1, 2, -1]) if method == "conditioned" else ([0, 2, -1] if method == "image_only" else [0, 1, 2])
                rows.append((image, task, method, score, pos))
    return {"row_id": np.arange(len(rows)), "seed": np.array([-1 if r[2] == "full_model" else 7 for r in rows]),
            "method": np.array([r[2] for r in rows]), "task_split": np.array(["unseen"] * len(rows)),
            "image_ids": np.array([r[0] for r in rows]), "task_ids": np.array([r[1] for r in rows]),
            "labels": np.zeros(len(rows), dtype=int),
            "class_pairs": np.array([[0, int(r[1][1])] for r in rows]),
            "cosine_scores": np.array([r[3] for r in rows]),
            "patch_indices": np.array([r[4] for r in rows]),
            "keep_patch_tokens": np.array([3 if r[2] == "full_model" else 2 for r in rows]),
            "text_features": np.eye(3)}


class SelectorExchangeTests(unittest.TestCase):
    def test_descriptor_task_mapping_and_same_image_pairs(self):
        cases = build_exchange_cases(fixture(), [7], ["unseen"])
        self.assertEqual(cases["image_ids"].tolist(), ["x"])
        scale = np.sqrt(2)
        for suffix, expected in (("aa", .4), ("ba", -.3), ("bb", .5), ("ab", -.2)):
            self.assertAlmostEqual(cases[f"margin_{suffix}"][0], expected / scale)
        self.assertEqual([cases[f"correct_{s}"][0] for s in ("aa", "ba", "bb", "ab")], [1, 0, 1, 0])
        self.assertEqual(cases["overlap_fraction"][0], .5)
        self.assertAlmostEqual(cases["angle_degrees"][0], 60.)

    def test_image_only_exchange_is_exact_null(self):
        cases = build_exchange_cases(fixture(), [7], ["unseen"])
        out = summarize_exchange(cases, [.45, .6, .8], 100, 11)["task_splits"]["unseen"]
        self.assertEqual(out["image_only_null_max_abs_margin_difference"], 0)
        self.assertEqual(out["image_only_null_prediction_disagreements"], 0)
        self.assertAlmostEqual(out["metrics"]["own_minus_swapped_margin"]["mean"], .7 / np.sqrt(2))
        self.assertEqual(out["metrics"]["own_minus_swapped_accuracy"]["mean"], 1.)
        self.assertEqual(out["metrics"]["correct_descriptor_repairs"]["mean"], 1.)
        self.assertEqual(out["metrics"]["correct_descriptor_new_errors"]["mean"], 0.)
        self.assertEqual(out["metrics"]["prediction_change_fraction"]["mean"], 1.)

    def test_fixed_difficulty_strata_and_same_stratum_filter(self):
        cases = build_exchange_cases(fixture(), [7], ["unseen"])
        out = summarize_exchange(cases, [.45, .6, .8], 100, 11)["task_splits"]["unseen"]
        self.assertEqual(out["same_quartile_sensitivity"]["observation_count"], 0)
        self.assertEqual([q["observation_count"] for q in out["directed_difficulty_strata"]], [1, 1, 0, 0])
        cases["base_margin_a"][:] = .45
        cases["base_margin_b"][:] = .5
        out = summarize_exchange(cases, [.45, .6, .8], 100, 11)["task_splits"]["unseen"]
        self.assertEqual(out["same_quartile_sensitivity"]["observation_count"], 1)
        self.assertEqual([q["observation_count"] for q in out["directed_difficulty_strata"]], [0, 2, 0, 0])

    def test_repeating_seeds_does_not_create_independent_images(self):
        one = build_exchange_cases(fixture(), [7], ["unseen"])
        cases = {key: np.tile(value, 2) for key, value in one.items()}
        cases["image_ids"] = np.array(["x", "y"])
        cases["margin_aa"][1] += .4
        repeated = {key: np.tile(value, 3) for key, value in cases.items()}
        repeated["seed"] = np.repeat([7, 8, 9], 2)
        first = summarize_exchange(cases, [.45, .6, .8], 500, 11)["task_splits"]["unseen"]
        second = summarize_exchange(repeated, [.45, .6, .8], 500, 11)["task_splits"]["unseen"]
        self.assertEqual(first["image_count"], second["image_count"])
        np.testing.assert_allclose(first["metrics"]["own_minus_swapped_margin"]["ci95"],
                                   second["metrics"]["own_minus_swapped_margin"]["ci95"])


if __name__ == "__main__":
    unittest.main()
