import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("objective_analysis", SCRIPTS / "analyze_selector_objectives.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def example_rows(seeds=(1, 2, 3), rank_cond=.6, rank_shared=.3):
    # Two tasks share three class-0 images, but have different other-side counts.
    examples = [(f"a{i}", "01", 0, 0, 1) for i in range(3)]
    examples += [("b0", "01", 1, 0, 1)]
    examples += [(f"a{i}", "02", 0, 0, 2) for i in range(3)]
    examples += [(f"c{i}", "02", 2, 0, 2) for i in range(2)]
    rows = {"kl": [], "rank": []}
    for objective in rows:
        for arm in analysis.ARMS:
            for seed in seeds:
                for image, task, label, a, b in examples:
                    shift = (.1 if arm == "conditioned" else 0) if objective == "kl" else (
                        rank_cond if arm == "conditioned" else rank_shared)
                    rows[objective].append({"image_id": image, "task_id": task, "class_id": label,
                        "class_a": a, "class_b": b, "seed": seed, "method": arm,
                        "task_split": "unseen", "keep_patch_tokens": 49, "old_panel": 1,
                        "correct": 1, "margin": .2 + label / 10 + shift})
    return rows


class SelectorObjectiveAnalysisTests(unittest.TestCase):
    def test_interaction_is_not_new_conditioned_model_gain(self):
        rows = example_rows(rank_cond=.6, rank_shared=.3)
        result, cases, packed, base, weights = analysis.analyze_budget(rows, [1, 2, 3], 49, 100, 7)
        self.assertAlmostEqual(result["secondary_contrasts"]["conditioned_rank_minus_kl"]["margin"]["mean"], .5)
        self.assertAlmostEqual(result["secondary_contrasts"]["image_only_rank_minus_kl"]["margin"]["mean"], .3)
        primary = result["primary_interaction"]["margin"]
        self.assertAlmostEqual(primary["mean"], .2)
        np.testing.assert_allclose(primary["ci95"], [.2, .2], atol=1e-14)
        self.assertEqual(primary["individual_interval_level"], .975)
        self.assertEqual(result["image_count"], 6)
        self.assertEqual(result["image_task_examples"], 9)
        # A single class-1 image has the same total weight as three class-0
        # images in task01, rather than weighting every row equally.
        task01_a = sum(w for w, r in zip(weights, base) if r["task_id"] == "01" and r["class_id"] == 0)
        task01_b = sum(w for w, r in zip(weights, base) if r["class_id"] == 1)
        self.assertAlmostEqual(task01_a, .25)
        self.assertAlmostEqual(task01_b, .25)
        np.testing.assert_allclose(cases["interaction_margin"], .2, atol=1e-14)
        self.assertEqual(packed["rank_conditioned_per_seed"].shape, (3, 9, 2))

    def test_equal_improvement_of_both_arms_is_zero_interaction(self):
        result, *_ = analysis.analyze_budget(example_rows(rank_cond=.5, rank_shared=.4), [1, 2, 3], 49, 100, 7)
        estimate = result["primary_interaction"]["margin"]
        self.assertAlmostEqual(estimate["mean"], 0)
        np.testing.assert_allclose(estimate["ci95"], [0, 0], atol=1e-14)

    def test_row_order_does_not_change_paired_results(self):
        first = example_rows()
        shuffled = {name: [rows[i] for i in np.random.default_rng(3).permutation(len(rows))]
                    for name, rows in first.items()}
        a = analysis.analyze_budget(first, [1, 2, 3], 49, 100, 8)[0]
        b = analysis.analyze_budget(shuffled, [1, 2, 3], 49, 100, 8)[0]
        self.assertEqual(a, b)

    def test_identity_checks_reject_missing_duplicate_and_changed_label(self):
        for mutation in ("missing", "duplicate", "label", "pair", "budget"):
            rows = example_rows()
            if mutation == "missing":
                rows["rank"].pop()
            elif mutation == "duplicate":
                rows["rank"].append(dict(rows["rank"][0]))
            else:
                key = {"label": "class_id", "pair": "class_b", "budget": "keep_patch_tokens"}[mutation]
                rows["rank"][0][key] = 999
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                analysis.analyze_budget(rows, [1, 2, 3], 49, 20, 8)

    def test_identical_repeated_seed_results_do_not_change_intervals(self):
        first = example_rows(seeds=(1, 2, 3))
        second = example_rows(seeds=(1, 2, 3, 4, 5, 6))
        # A nonconstant intervention creates a nonzero sampling interval.
        for rows in (first, second):
            for r in rows["rank"]:
                if r["method"] == "conditioned":
                    r["margin"] += r["class_id"] / 5
        a = analysis.analyze_budget(first, [1, 2, 3], 49, 500, 8)[0]
        b = analysis.analyze_budget(second, [1, 2, 3, 4, 5, 6], 49, 500, 8)[0]
        for key in ("mean", "ci95", "bonferroni_family_ci95"):
            np.testing.assert_allclose(a["primary_interaction"]["margin"][key],
                                       b["primary_interaction"]["margin"][key], atol=1e-14)
        self.assertGreater(a["primary_interaction"]["margin"]["ci95"][1],
                           a["primary_interaction"]["margin"]["ci95"][0])

    def test_common_teacher_diagnostics_separate_scale_and_selection(self):
        utility = np.array([[3., 2., 1., 0.], [0., 0., 0., 0.]])
        logits = np.array([[3., 2., 1., 0.], [0., 0., 0., 0.]])
        first, counts = analysis.teacher_metrics(logits, utility, 2)
        scaled, _ = analysis.teacher_metrics(logits * 10, utility, 2)
        # Same selection can fit the probability teacher differently.
        self.assertNotAlmostEqual(first[0, 0], scaled[0, 0])
        np.testing.assert_allclose(first[:, 1:3], scaled[:, 1:3])
        self.assertEqual(first[0, 1], 1.)
        self.assertEqual(first[0, 2], 0.)
        self.assertEqual(first[0, 3], 0.)
        self.assertEqual(counts[0], 4)
        self.assertEqual(counts[1], 0)
        self.assertEqual(first[1, 4], 0.)
        self.assertEqual(first[1, 5], 1.)
        reversed_metrics, _ = analysis.teacher_metrics(-logits[:1], utility[:1], 2)
        self.assertEqual(reversed_metrics[0, 1], 0.)
        self.assertEqual(reversed_metrics[0, 2], 4.)
        self.assertEqual(reversed_metrics[0, 3], 1.)

    def test_teacher_boundary_ties_are_symmetric_in_analysis(self):
        utility = np.array([[4., 2., 2., 0.]])
        logits = np.array([[.1, .5, -.4, .7]])
        metrics, counts = analysis.teacher_metrics(logits, utility, 2)
        self.assertEqual(counts[0], 5)
        permutation = [3, 2, 0, 1]
        other, other_counts = analysis.teacher_metrics(logits[:, permutation], utility[:, permutation], 2)
        # Canonical overlap can change under boundary ties, but the objective,
        # regret, valid-pair count and tie-aware overlap are invariant.
        np.testing.assert_allclose(metrics[:, [0, 2, 3, 4, 5, 6]], other[:, [0, 2, 3, 4, 5, 6]], atol=1e-14)
        np.testing.assert_array_equal(counts, other_counts)

    def test_cli_plot_only_does_not_require_original_predictions(self):
        args = analysis.argument_parser().parse_args(["--output", "results/example", "--plot-only"])
        self.assertTrue(args.plot_only)
        self.assertEqual(args.replicates, 2000)
        self.assertEqual(args.seed, 2085)


if __name__ == "__main__":
    unittest.main()
