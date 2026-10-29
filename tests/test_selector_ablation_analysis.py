import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("ablation_analysis", SCRIPTS / "analyze_selector_ablations.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def sample_rows(seeds=(1, 2, 3)):
    examples = [(f"a{i}", "01", 0, 0, 1) for i in range(3)]
    examples += [("b0", "01", 1, 0, 1)]
    examples += [(f"a{i}", "02", 0, 0, 2) for i in range(3)]
    examples += [(f"c{i}", "02", 2, 0, 2) for i in range(2)]
    offsets = {"conditioned": .4, "image_only": .1, "image_only_wide": .2,
               "mean_only": .3, "difference_only": .15}
    rows = {s: [] for s in analysis.SOURCE_METHODS}
    for source, methods in analysis.SOURCE_METHODS.items():
        for method in methods:
            for seed in seeds:
                for image, task, label, a, b in examples:
                    rows[source].append({"image_id": image, "task_id": task, "class_id": label,
                        "class_a": a, "class_b": b, "method": method, "seed": seed,
                        "task_split": "unseen", "keep_patch_tokens": 49, "correct": 1,
                        "margin": offsets[method] + label * .1})
    return rows


def synthetic_raw(base, seeds=(1, 2, 3)):
    raw = {}
    for method in analysis.METHODS:
        strength = {"conditioned": .7, "mean_only": .3, "difference_only": .5,
                    "image_only": 0, "image_only_wide": 0}[method]
        scores = np.full((len(seeds), len(base), 3), .9)
        positions = np.tile([0, 1, -1, -1], (len(seeds), len(base), 1))
        for i, row in enumerate(base):
            label, a, b = [int(row[k]) for k in ("class_id", "class_a", "class_b")]
            competitor = b if a == label else a
            scores[:, i, label] = 1
            scores[:, i, competitor] -= strength
            if method not in ("image_only", "image_only_wide") and row["task_id"] == "02":
                positions[:, i, :2] = [1, 2]
        raw[method] = {"scores": scores, "positions": positions,
                       "source_rows": np.arange(len(seeds) * len(base)).reshape(len(seeds), len(base))}
    return raw, np.eye(3)


class AblationAnalysisTest(unittest.TestCase):
    def test_primary_contrasts_and_family_sizes(self):
        summary, cases, packed, base, values = analysis.analyze_classification(sample_rows(), [1, 2, 3], 49, 100, 9)
        expected = {"conditioned_minus_image_only_wide": (.2, .975),
                    "conditioned_minus_mean_only": (.1, .9875),
                    "conditioned_minus_difference_only": (.25, .9875)}
        for name, (point, level) in expected.items():
            estimate = summary["primary_contrasts"][name]["margin"]
            self.assertAlmostEqual(estimate["mean"], point)
            self.assertAlmostEqual(estimate["individual_interval_level"], level)
        self.assertEqual(packed["difference_only_per_seed"].shape, (3, 9, 2))
        self.assertAlmostEqual(cases["task_class_weight"][cases["class_id"] == 0].sum(), .5)
        self.assertAlmostEqual(summary["secondary_contrasts"]["image_only_wide_minus_image_only"]["margin"]["mean"], .1)

    def test_identity_alignment_is_invariant_to_row_order(self):
        rows = sample_rows()
        shuffled = {source: [part[i] for i in np.random.default_rng(8).permutation(len(part))]
                    for source, part in rows.items()}
        a = analysis.analyze_classification(rows, [1, 2, 3], 49, 100, 7)[0]
        b = analysis.analyze_classification(shuffled, [1, 2, 3], 49, 100, 7)[0]
        self.assertEqual(a, b)

    def test_rejects_missing_duplicate_changed_label_and_budget(self):
        for mutation in ("missing", "duplicate", "label", "budget"):
            rows = sample_rows()
            if mutation == "missing":
                rows["ablations"].pop()
            elif mutation == "duplicate":
                rows["ablations"].append(dict(rows["ablations"][0]))
            else:
                rows["ablations"][0]["class_id" if mutation == "label" else "keep_patch_tokens"] = 999
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                analysis.analyze_classification(rows, [1, 2, 3], 49, 20, 7)

    def test_repeating_seeds_does_not_narrow_image_interval(self):
        results = []
        for seeds in ((1, 2, 3), (1, 2, 3, 4, 5, 6)):
            rows = sample_rows(seeds)
            for row in rows["baseline"]:
                if row["method"] == "conditioned":
                    row["margin"] += row["class_id"] / 3
            results.append(analysis.analyze_classification(rows, seeds, 49, 300, 11)[0])
        for field in ("mean", "ci95", "bonferroni_family_ci95"):
            np.testing.assert_allclose(results[0]["primary_contrasts"]["conditioned_minus_mean_only"]["margin"][field],
                                       results[1]["primary_contrasts"]["conditioned_minus_mean_only"]["margin"][field])

    def test_exchange_uses_fixed_task_and_correct_direction(self):
        base, _, _ = analysis.align_rows(sample_rows(), [1, 2, 3], 49)
        raw, text = synthetic_raw(base)
        result, cases, packed = analysis.analyze_exchange(base, raw, text, [1, 2, 3], 2, 100, 7)
        self.assertEqual(result["image_count"], 3)
        self.assertEqual(result["image_task_pairs"], 3)
        full = next(row for row in result["table"] if row["method"] == "conditioned")
        self.assertAlmostEqual(full["own_minus_swapped"]["margin"]["mean"], .7 / np.sqrt(2))
        self.assertAlmostEqual(full["token_overlap"]["mean"], .5)
        self.assertAlmostEqual(result["secondary_effect_contrasts"]["conditioned_minus_mean_only"]["margin"]["mean"], .4 / np.sqrt(2))
        self.assertEqual(packed["conditioned_own_per_seed"].shape, (3, 3, 2))
        for method in ("image_only", "image_only_wide"):
            self.assertEqual(result["image_only_invariance"][method]["max_abs_cross_task_score_difference"], 0)

    def test_task_dependent_wide_scores_or_masks_rejected(self):
        base, _, _ = analysis.align_rows(sample_rows(), [1, 2, 3], 49)
        for field in ("scores", "positions"):
            raw, text = synthetic_raw(base)
            raw["image_only_wide"][field][0, 0, 0] += .1 if field == "scores" else 1
            with self.subTest(field=field), self.assertRaises(AssertionError):
                analysis.analyze_exchange(base, raw, text, [1, 2, 3], 2, 20, 7)

    def test_binary_ties_choose_lower_candidate(self):
        scores = np.ones((1, 2, 3))
        result = analysis.binary_scores(scores, np.array([0, 2]), np.array([1, 1]), np.eye(3))
        np.testing.assert_array_equal(result[0, :, 0], [1, 0])
        np.testing.assert_array_equal(result[0, :, 1], 0)

    def test_raw_npz_alignment_reconstructs_classification(self):
        base, _, _ = analysis.align_rows(sample_rows(), [1, 2, 3], 49)
        raw, text = synthetic_raw(base)
        labels = np.array([int(row["class_id"]) for row in base])
        pairs = np.array([[int(row["class_a"]), int(row["class_b"])] for row in base])
        competitors = np.where(pairs[:, 0] == labels, pairs[:, 1], pairs[:, 0])
        values = {m: analysis.binary_scores(raw[m]["scores"], labels, competitors, text) for m in analysis.METHODS}
        with tempfile.TemporaryDirectory() as temp:
            sources = {}
            for source, methods in analysis.SOURCE_METHODS.items():
                path = Path(temp) / source / "full"
                path.mkdir(parents=True)
                rows = [(m, si, seed, i) for m in methods for si, seed in enumerate((1, 2, 3)) for i in range(len(base))]
                rows.reverse()
                n = len(rows)
                np.savez_compressed(path / "predictions.npz", row_id=np.arange(n),
                    image_ids=[base[i]["image_id"] for m, si, s, i in rows],
                    task_ids=[base[i]["task_id"] for m, si, s, i in rows], task_split=["unseen"] * n,
                    method=[m for m, si, s, i in rows], seed=[s for m, si, s, i in rows],
                    labels=[labels[i] for m, si, s, i in rows], class_pairs=[pairs[i] for m, si, s, i in rows],
                    keep_patch_tokens=[2] * n, cosine_scores=[raw[m]["scores"][si, i] for m, si, s, i in rows],
                    patch_indices=[raw[m]["positions"][si, i] for m, si, s, i in rows], text_features=text)
                sources[source] = path.parent
            aligned, restored_text = analysis.aligned_raw(sources, base, (1, 2, 3), 2, values)
            np.testing.assert_array_equal(restored_text, text)
            for method in analysis.METHODS:
                np.testing.assert_array_equal(aligned[method]["scores"], raw[method]["scores"])
                np.testing.assert_array_equal(aligned[method]["positions"], raw[method]["positions"])

    def test_plot_only_requires_no_raw_source(self):
        args = analysis.argument_parser().parse_args(["--output", "results/example", "--plot-only"])
        self.assertTrue(args.plot_only)
        self.assertEqual(args.seed, 2091)
        self.assertEqual(args.replicates, 2000)


if __name__ == "__main__":
    unittest.main()
