"""Reserved-test analysis: row alignment, weighting and exchange reconstruction."""

import contextlib
import io
from itertools import combinations
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_final_test import budget_contrasts, load_outcomes, run
from analyze_selector_scale import panel_weights
from evaluate_selectors import binary_outcomes, write_rows


def fixture(folder, budget, shuffle=False):
    rng = np.random.default_rng(32)
    text = rng.normal(size=(5, 9))
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    labels = np.repeat(np.arange(5), np.arange(4, 9))
    image_ids = np.array([f"image_{i:03d}" for i in range(len(labels))])
    full = rng.normal(size=(len(labels), 9)) + 1.1 * text[labels]
    full /= np.linalg.norm(full, axis=1, keepdims=True)
    examples = []
    for a, b in combinations(range(5), 2):
        for ix in np.flatnonzero(np.isin(labels, [a, b])):
            examples.append({"image_index": int(ix), "image_id": image_ids[ix],
                             "class_id": int(labels[ix]), "task_id": f"{a}_{b}",
                             "class_a": a, "class_b": b, "task_split": "unseen",
                             "task_class_weight": 1 / (20 * np.sum(labels == labels[ix]))})
    if shuffle:
        rng.shuffle(examples)
    folder.mkdir()
    write_rows(folder / "examples.csv", examples)
    checkpoints = {m: "unused" for m in ("conditioned", "image_only", "image_only_wide")}
    if budget == 98:
        checkpoints["patch_ranking"] = "unused"
    rows = np.array([r["image_index"] for r in examples])
    y = labels[rows]
    pairs = np.array([[r["class_a"], r["class_b"]] for r in examples])
    competitor = np.where(y == pairs[:, 0], pairs[:, 1], pairs[:, 0])
    methods = {"full_model": [-1], "attention": [-1], "random": [17],
               **{method: [11, 12] for method in checkpoints}}
    for method, seeds in methods.items():
        for seed in seeds:
            z = full[rows].copy()
            if method == "conditioned":
                z += .05 * text[y] - (.04 + .0001 * budget) * text[competitor]
            elif method != "full_model":
                z += .02 * text[y]
            if seed != -1:
                z += .0003 * (seed - 11)
            z /= np.linalg.norm(z, axis=1, keepdims=True)
            scores = z @ text.T
            margin, correct = binary_outcomes(scores, y, pairs, text)
            positions = np.full((len(rows), 196), -1, np.int16)
            k = 196 if method == "full_model" else budget
            positions[:, :k] = np.arange(k)
            target = folder / method / f"seed_{seed}"
            target.mkdir(parents=True)
            np.savez_compressed(target / "predictions.npz", image_features=z, cosine_scores=scores,
                                patch_indices=positions, text_features=text, margin=margin, correct=correct)
    return {"id": f"synthetic_{budget}", "dataset": "synthetic", "model": "ViT-B/16",
            "keep_patch_tokens": budget, "prune_after": 6, "eval_dir": str(folder),
            "checkpoints": checkpoints, "seeds": [11, 12]}, examples


class FinalTestAnalysisTests(unittest.TestCase):
    def test_complete_pipeline_with_shuffled_budget_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            low, rows = fixture(root / "low", 49)
            high, _ = fixture(root / "high", 98, shuffle=True)
            config = {"settings": [low, high], "seeds": [11, 12], "random_seeds": [17],
                      "bootstrap_replicates": 25, "bootstrap_seed": 8}
            with contextlib.redirect_stdout(io.StringIO()):
                result = run(config, root / "analysis")
            self.assertTrue(result["final_test_used"])
            self.assertEqual(result["families"], {"original": 2, "strong": 2, "external": 1})
            self.assertIn("synthetic_98_minus_49", result["budget_changes"])
            weights = panel_weights(rows, "all")
            manual = load_outcomes(low, "conditioned", [11, 12], rows).mean(axis=0)
            table = {r["method"]: r for r in result["settings"][low["id"]]["table"]}
            self.assertAlmostEqual(table["conditioned"]["accuracy"]["mean"], weights @ manual[:, 0])
            for setting in result["settings"].values():
                self.assertEqual(setting["audit"]["image_only_feature_exchange_max_abs_difference"], 0)
                self.assertLess(setting["audit"]["geometry_identity_max_abs_error"], 1e-12)
                self.assertGreater(setting["exchange_own_minus_swapped"]["margin"]["mean"], 0)
                self.assertTrue(setting["h2"]["difficulty_adjusted"]["angle_estimable_beyond_controls"])
                self.assertEqual(setting["exchange_image_count"], 30)
            restored = json.loads((root / "analysis/analysis.json").read_text())
            self.assertEqual(restored["families"], result["families"])

    def test_budget_interaction_is_paired_and_order_invariant(self):
        weights = np.array([.25, .25, .5])
        base = np.array([[1., .1], [0., .2], [1., .3]])
        low = {"keys": [("a", "01"), ("a", "02"), ("b", "01")],
               "ids": np.array(["a", "a", "b"]), "weights": weights,
               "metadata": [(0, 0, 1), (0, 0, 2), (1, 0, 1)],
               "means": {"full_model": base, "conditioned": base + [.1, .02], "image_only": base}}
        order = np.array([2, 0, 1])
        high = {"keys": [low["keys"][i] for i in order], "ids": low["ids"][order],
                "weights": weights[order], "metadata": [low["metadata"][i] for i in order],
                "means": {"full_model": base[order], "conditioned": base[order] + [.15, .01],
                          "image_only": base[order]}}
        result = budget_contrasts(low, high, 80, 3, 2)["conditioned_minus_image_only"]
        self.assertAlmostEqual(result["accuracy"]["mean"], .05)
        np.testing.assert_allclose(result["accuracy"]["ci95"], [.05, .05])
        self.assertAlmostEqual(result["margin"]["mean"], -.01)

    def test_saved_outcomes_must_match_task_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            setting, rows = fixture(Path(temporary) / "source", 49)
            target = Path(setting["eval_dir"]) / "conditioned/seed_11/predictions.npz"
            with np.load(target) as data:
                values = {key: data[key].copy() for key in data.files}
            values["correct"][0] = 1 - values["correct"][0]
            np.savez_compressed(target, **values)
            with self.assertRaises(AssertionError):
                load_outcomes(setting, "conditioned", [11, 12], rows)


if __name__ == "__main__":
    unittest.main()
