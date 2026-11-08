"""Check multiclass reconstruction, paired image inference and task exchange."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_nway import exchange_losses, load_setting, outcomes, run, way_interaction
from analyze_selector_budgets import weighted_estimates


def fixture(root, way, shuffle=False):
    rng = np.random.default_rng(70 + way)
    text = np.eye(7)
    labels = np.repeat(np.arange(7), [2, 3, 2, 4, 3, 2, 4])
    ids = np.array([f"image_{i:02d}" for i in range(len(labels))])
    full = np.random.default_rng(99).uniform(0, .5, (len(labels), 7))
    full[np.arange(len(labels)), labels] += .1
    task = np.sort((np.arange(7)[:, None] + np.arange(way)[None]) % 7, axis=1)
    image_rows, task_rows = np.nonzero((labels[:, None, None] == task[None]).any(axis=-1))
    candidates, y = task[task_rows], labels[image_rows]
    weights = 1 / (len(task) * way * np.bincount(labels)[y])
    panel = dict(image_ids=ids[image_rows], task_ids=np.asarray([f"task_{t}" for t in task_rows]),
                 candidates=candidates, labels=y, weights=weights)
    folder = root / "eval" / f"{way}way"
    folder.mkdir(parents=True)
    for filename, methods, seed in [("references", ["full_model", "attention"], -1),
                                     ("seed_11", ["conditioned", "image_only_wide"], 11),
                                     ("seed_12", ["conditioned", "image_only_wide"], 12)]:
        parts = []
        for method in methods:
            scores = full[image_rows].copy()
            if method == "attention":
                scores[np.arange(len(y)), y] -= .02
            elif method in ("conditioned", "image_only_wide"):
                scores[np.arange(len(y)), y] += .005 * (seed - 10)
                if method == "conditioned":
                    scores[np.arange(len(y))[:, None], candidates] -= .03 * way
                    scores[np.arange(len(y)), y] += .03 * way
            values = outcomes(scores, candidates, y, text)
            part = {**panel, "method": np.full(len(y), method), "seed": np.full(len(y), seed),
                    "cosine_scores": scores, "correct": values[:, 0], "margin": values[:, 1],
                    "geometric_margin": values[:, 2]}
            parts.append(part)
        saved = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
        if shuffle:
            order = rng.permutation(len(saved["method"]))
            saved = {key: value[order] for key, value in saved.items()}
        np.savez_compressed(folder / f"{filename}.npz", **saved, text_features=text)
    return load_setting(folder)


class NWayAnalysisTests(unittest.TestCase):
    def test_multiclass_outcomes_choose_strongest_competitor_and_stable_tie(self):
        text = np.eye(4)
        scores = np.array([[.3, .2, .5, .4], [.5, .5, .2, .1]])
        candidates = np.array([[3, 0, 2], [2, 1, 0]])
        labels = np.array([0, 1])
        values = outcomes(scores, candidates, labels, text)
        np.testing.assert_allclose(values[:, 0], [0, 0])
        np.testing.assert_allclose(values[:, 1], [-.2, 0])
        np.testing.assert_allclose(values[:, 2], [-.2 / np.sqrt(2), 0])
        np.testing.assert_allclose(values, outcomes(scores, candidates[:, ::-1], labels, text))

    def test_exchange_uses_actual_task_and_averages_within_image(self):
        panel = {"image_ids": np.array(["a", "a"]), "task_ids": np.array(["01", "02"]),
                 "labels": np.array([0, 0]), "candidates": np.array([[0, 1], [0, 2]])}
        text = np.eye(3)
        scores = np.array([[1., 0., .9], [1., .9, 0.]])
        data = {"panel": panel, "text": text, "seeds": {"conditioned": [11]},
                "records": {("conditioned", 11): {"scores": scores,
                    "outcomes": outcomes(scores, panel["candidates"], panel["labels"], text)},
                    ("image_only_wide", 11): {"scores": np.tile([1., .2, .4], (2, 1))}}}
        result = exchange_losses(data)
        np.testing.assert_allclose(result["per_seed"], [[[0, .9, .9 / np.sqrt(2)]]])
        self.assertEqual(result["shared_score_exchange_max_difference"], 0)
        self.assertEqual(result["pair_count"], 1)

    def test_image_bootstrap_is_invariant_to_duplicated_task_rows(self):
        ids, weights = np.array(["a", "b", "c"]), np.array([.2, .3, .5])
        values = np.array([[0., .01], [1., .2], [.5, -.1]])
        original = weighted_estimates(ids, weights, values, ["accuracy", "margin"], 100, 12)
        repeated = weighted_estimates(np.repeat(ids, 7), np.repeat(weights / 7, 7),
                                      np.repeat(values, 7, axis=0), ["accuracy", "margin"], 100, 12)
        for key in original:
            self.assertAlmostEqual(original[key]["mean"], repeated[key]["mean"])
            np.testing.assert_allclose(original[key]["ci95"], repeated[key]["ci95"], atol=1e-14)

    def test_full_pipeline_and_cross_way_paired_difference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            low = fixture(root, 2)
            high = fixture(root, 5, shuffle=True)
            interaction = way_interaction(low, high, 80, 15, 1)
            manual = []
            for data in (low, high):
                delta = data["arrays"]["conditioned"].mean(axis=0) - data["arrays"]["image_only_wide"].mean(axis=0)
                manual.append(data["panel"]["weights"] @ delta)
            for j, name in enumerate(("accuracy", "margin", "geometric_margin")):
                self.assertAlmostEqual(interaction[name]["mean"], (manual[1] - manual[0])[j])
            np.testing.assert_allclose(interaction["margin"]["ci95"], [.09, .09], atol=1e-14)
            with contextlib.redirect_stdout(io.StringIO()):
                result = run({"synthetic": root}, root / "analysis", replicates=40, seed=12)
            self.assertFalse(result["final_test_used"])
            self.assertEqual(result["settings"]["synthetic_5way"]["seeds"], [11, 12])
            self.assertGreater(result["settings"]["synthetic_5way"]["exchange_own_minus_swapped"]["margin"]["mean"], 0)
            for setting in result["settings"].values():
                for row in setting["table"]:
                    base = next(r for r in setting["table"] if r["method"] == "full_model")
                    self.assertAlmostEqual(row["accuracy"]["mean"] - base["accuracy"]["mean"],
                                           row["repairs_vs_full"] - row["new_errors_vs_full"])
            for name in ("classification", "contrasts", "way_interactions", "exchange", "per_seed"):
                self.assertTrue((root / "analysis" / f"{name}.csv").exists())

    def test_corrupted_saved_accuracy_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root, 2)
            path = root / "eval/2way/seed_11.npz"
            with np.load(path) as data:
                saved = {key: data[key].copy() for key in data.files}
            saved["correct"][0] = 1 - saved["correct"][0]
            np.savez_compressed(path, **saved)
            with self.assertRaises(AssertionError):
                load_setting(path.parent)


if __name__ == "__main__":
    unittest.main()
