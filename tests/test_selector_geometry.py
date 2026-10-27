import importlib.util
from pathlib import Path
import unittest
import tempfile
from itertools import combinations

import numpy as np

from token_compression.selector_geometry import (
    adjusted_angle_association, cross_validated_angle_prediction, difficulty_features,
    exchange_geometry, image_fold_assignments, image_weights,
)


class GeometryIdentityTests(unittest.TestCase):
    def test_full_dimensional_identity_and_task_order_invariance(self):
        text = np.eye(4)
        a = np.array([[.8, .1, .4, .3]])
        b = np.array([[.5, .5, .2, .3]])
        full = np.array([[.9, .2, .1, .3]])
        result = exchange_geometry(a, b, full, text, [0], [1], [2])
        swapped = exchange_geometry(b, a, full, text, [0], [2], [1])
        self.assertAlmostEqual(result["angle_degrees"][0], 60)
        expected = .5 * ((a[0, 0] - a[0, 1] - b[0, 0] + b[0, 1]) +
                         (b[0, 0] - b[0, 2] - a[0, 0] + a[0, 2])) / np.sqrt(2)
        self.assertAlmostEqual(result["exchange_loss"][0], expected)
        self.assertLess(result["identity_error"][0], 1e-14)
        for key in ("exchange_loss", "alignment", "displacement_norm", "angle_degrees"):
            np.testing.assert_allclose(result[key], swapped[key])

    def test_zero_displacement_is_zero_loss_but_undefined_alignment(self):
        z = np.array([[.8, .3, .2]])
        result = exchange_geometry(z, z, z, np.eye(3), [0], [1], [2])
        self.assertEqual(result["exchange_loss"][0], 0)
        self.assertEqual(result["identity_error"][0], 0)
        self.assertFalse(result["alignment_valid"][0])
        self.assertTrue(np.isnan(result["alignment"][0]))

    def test_geometry_can_scale_loss_without_changing_alignment(self):
        # Distinct task directions with the feature displacement aligned to
        # their difference yield increasing loss and exactly constant alignment.
        angles = np.radians([20, 50, 80])
        u_a = np.tile([1., 0., 0.], (3, 1))
        u_b = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(3)])
        direction_delta = u_a - u_b
        displacement = .2 * direction_delta / np.linalg.norm(direction_delta, axis=1)[:, None]
        loss = .5 * np.sum(displacement * direction_delta, axis=1)
        np.testing.assert_allclose(loss, .2 * np.sin(angles / 2))
        np.testing.assert_allclose(loss / (.2 * np.sin(angles / 2)), 1)
        self.assertTrue(np.all(np.diff(loss) > 0))


class GeometryInferenceTests(unittest.TestCase):
    @staticmethod
    def synthetic(seed=7, angle_effect=.003):
        rng = np.random.default_rng(seed)
        ids = np.repeat(np.array([f"image_{i:03d}" for i in range(80)]), 6)
        # Continuous difficulty, angle confounding and repeated image noise.
        ma, mb = rng.uniform(-.1, .3, (2, len(ids)))
        angle = 40 + 100 * ma + rng.uniform(-12, 12, len(ids))
        controls = difficulty_features(ma, mb)
        y = controls @ np.array([.13, -.04, .05, .08, -.07])
        y += angle_effect * angle / 10 + rng.normal(0, .0002, len(ids))
        y += np.repeat(rng.normal(0, .0003, 80), 6)
        return {"image_ids": ids, "labels": np.repeat(np.arange(80) % 4, 6),
                "base_margin_a": ma, "base_margin_b": mb,
                "angle_degrees": angle, "exchange_loss": y}

    def test_equal_image_weighting_not_equal_pair_weighting(self):
        np.testing.assert_allclose(image_weights(["a", "a", "b"]), [.25, .25, .5])

    def test_adjustment_recovers_angle_effect_under_margin_confounding(self):
        cases = self.synthetic()
        result = adjusted_angle_association(cases, replicates=150, seed=13)
        self.assertAlmostEqual(result["angle_slope"], .003, delta=.0001)
        self.assertGreater(result["ci_familywise_three_budgets"][0], 0)
        self.assertEqual(result["design_rank"], result["design_columns"])
        fixed = adjusted_angle_association(cases, replicates=150, seed=13, image_fixed_effects=True)
        self.assertAlmostEqual(fixed["angle_slope"], .003, delta=.0001)

    def test_family_size_changes_only_multiplicity_and_preserves_three_budget_aliases(self):
        cases = self.synthetic()
        default = adjusted_angle_association(cases, replicates=300, seed=13)
        three = adjusted_angle_association(cases, replicates=300, seed=13, family_size=3)
        self.assertEqual(default, three)
        self.assertEqual(three["ci_familywise_budgets"], three["ci_familywise_three_budgets"])
        self.assertEqual(three["positive_after_budget_adjustment"], three["positive_after_three_budget_adjustment"])
        two = adjusted_angle_association(cases, replicates=300, seed=13, family_size=2)
        self.assertEqual(two["angle_slope"], three["angle_slope"])
        self.assertEqual(two["ci95"], three["ci95"])
        self.assertEqual(two["familywise_interval_level"], .975)
        self.assertEqual(two["family_size"], 2)
        self.assertNotIn("ci_familywise_three_budgets", two)
        self.assertGreaterEqual(two["ci_familywise_budgets"][0], three["ci_familywise_budgets"][0])
        self.assertLessEqual(two["ci_familywise_budgets"][1], three["ci_familywise_budgets"][1])

    def test_repeated_pairs_do_not_create_new_independent_images(self):
        cases = self.synthetic()
        result = adjusted_angle_association(cases, replicates=120, seed=15)
        repeated = {k: np.repeat(v, 3, axis=0) for k, v in cases.items()}
        other = adjusted_angle_association(repeated, replicates=120, seed=15)
        self.assertEqual(other["image_count"], result["image_count"])
        self.assertAlmostEqual(other["angle_slope"], result["angle_slope"], places=12)
        np.testing.assert_allclose(other["ci95"], result["ci95"], atol=1e-12)

    def test_fixed_effects_remove_image_level_offsets(self):
        cases = self.synthetic()
        shifted = {k: v.copy() for k, v in cases.items()}
        shifted["exchange_loss"] += np.repeat(np.linspace(-5, 5, 80), 6)
        first = adjusted_angle_association(cases, replicates=80, seed=17, image_fixed_effects=True)
        second = adjusted_angle_association(shifted, replicates=80, seed=17, image_fixed_effects=True)
        self.assertAlmostEqual(first["angle_slope"], second["angle_slope"], places=11)
        np.testing.assert_allclose(first["ci95"], second["ci95"], atol=1e-11)

    def test_continuous_margin_controls_remove_spurious_angle_relation(self):
        cases = self.synthetic(angle_effect=0)
        result = adjusted_angle_association(cases, replicates=150, seed=12)
        self.assertLess(result["ci95"][0], 0)
        self.assertGreater(result["ci95"][1], 0)
        self.assertAlmostEqual(result["angle_slope"], 0, delta=.0001)

    def test_image_folds_keep_repeated_pairs_together_and_are_order_invariant(self):
        cases = self.synthetic()
        folds = image_fold_assignments(cases["image_ids"], cases["labels"], seed=19)
        for image_id in np.unique(cases["image_ids"]):
            self.assertEqual(len(set(folds[cases["image_ids"] == image_id])), 1)
        reversed_folds = image_fold_assignments(cases["image_ids"][::-1], cases["labels"][::-1], seed=19)
        np.testing.assert_array_equal(folds, reversed_folds[::-1])

    def test_cross_fitted_angle_improves_prediction_for_known_signal(self):
        cases = self.synthetic()
        result = cross_validated_angle_prediction(cases, replicates=100)
        self.assertGreater(result["mse_improvement"], 0)
        self.assertGreater(result["incremental_out_of_fold_r2_ci95"][0], 0)
        self.assertEqual(len(result["out_of_fold_predictions"]["fold"]), len(cases["image_ids"]))


class GeometryPipelineTests(unittest.TestCase):
    def test_raw_prediction_loading_seed_alignment_and_saved_analysis(self):
        path = Path(__file__).resolve().parents[1] / "scripts/analyze_selector_geometry.py"
        spec = importlib.util.spec_from_file_location("geometry_pipeline", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rng = np.random.default_rng(44)
        text = rng.normal(size=(4, 8))
        text /= np.linalg.norm(text, axis=1, keepdims=True)
        image_labels = np.repeat(np.arange(4), 10)
        full = rng.normal(size=(40, 8)) + 2 * text[image_labels]
        full /= np.linalg.norm(full, axis=1, keepdims=True)
        seeds = [11, 12, 13]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            for group in ("references", "full"):
                records = []
                jobs = [(-1, "full_model")] if group == "references" else [(seed, method) for seed in seeds for method in ("conditioned", "image_only")]
                for seed, method in jobs:
                    for i, label in enumerate(image_labels):
                        for pair in combinations(range(4), 2):
                            if label not in pair:
                                continue
                            competitor = pair[1] if pair[0] == label else pair[0]
                            z = full[i].copy()
                            if method == "conditioned":
                                z += .08 * text[competitor] + (seed - 12) * .001
                            elif method == "image_only":
                                z += .03 * text[label]
                            z /= np.linalg.norm(z)
                            records.append({"seed": seed, "method": method, "task_split": "unseen",
                                            "image_ids": f"image_{i:03d}", "labels": label,
                                            "task_ids": f"{pair[0]}_{pair[1]}", "class_pairs": pair,
                                            "image_features": z, "cosine_scores": z @ text.T,
                                            "patch_indices": np.r_[np.arange(49), np.repeat(-1, 147)],
                                            "keep_patch_tokens": 196 if method == "full_model" else 49})
                packed = {key: np.asarray([r[key] for r in records]) for key in records[0]}
                packed["row_id"] = np.arange(len(records))
                packed["text_features"] = text
                (source / group).mkdir(parents=True)
                np.savez_compressed(source / group / "predictions.npz", **packed)
            output = Path(directory) / "analysis"
            summary = module.run({49: source}, output, seeds, replicates=15)
            result = summary["budgets"]["49"]
            self.assertEqual(result["scope"]["image_count"], 40)
            self.assertEqual(result["scope"]["pair_count"], 120)
            self.assertEqual(result["audit"]["image_only_feature_exchange_max_abs_difference"], 0)
            self.assertLess(result["scope"]["geometry_identity_max_abs_error"], 1e-12)
            self.assertTrue((output / "analysis.json").exists())
            with np.load(output / "budget_49/seed_mean_cases.npz") as cases:
                self.assertEqual(len(cases["image_ids"]), 120)
                self.assertTrue(np.all(cases["seed_count"] == 3))
            second = Path(directory) / "source_98"
            for group in ("references", "full"):
                with np.load(source / group / "predictions.npz") as saved:
                    packed = {key: saved[key].copy() for key in saved.files}
                learned = packed["method"] != "full_model"
                packed["keep_patch_tokens"][learned] = 98
                packed["patch_indices"][learned] = np.r_[np.arange(98), np.repeat(-1, 98)]
                (second / group).mkdir(parents=True)
                np.savez_compressed(second / group / "predictions.npz", **packed)
            two = module.run({49: source, 98: second}, Path(directory) / "analysis_two", seeds, replicates=15)
            self.assertEqual(two["family_size"], 2)
            for value in two["budgets"].values():
                self.assertEqual(value["primary_difficulty_adjusted"]["familywise_interval_level"], .975)
                self.assertEqual(value["image_fixed_effects_sensitivity"]["family_size"], 2)
                self.assertTrue(all(r["difficulty_adjusted"]["family_size"] == 2 for r in value["per_seed"]))
                self.assertTrue(all(r["difficulty_adjusted"]["family_size"] == 2 for r in value["normalized_alignment_diagnostics"]))


class SeedAggregationTests(unittest.TestCase):
    def test_mean_displacement_preserves_mean_exchange_identity(self):
        path = Path(__file__).resolve().parents[1] / "scripts/analyze_selector_geometry.py"
        spec = importlib.util.spec_from_file_location("analyze_selector_geometry", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows = []
        for seed, delta in [(1, [.2, -.1, .0]), (2, [-.1, .3, .0])]:
            z_a = np.array([delta]);z_b = np.zeros((1, 3));z_full = np.array([[.9, .1, .2]])
            geometry = exchange_geometry(z_a, z_b, z_full, np.eye(3), [0], [1], [2])
            rows.append({**geometry, "seed": np.array([seed]), "image_ids": np.array(["a"]),
                         "task_a": np.array(["01"]), "task_b": np.array(["02"]),
                         "labels": np.array([0]), "competitor_a": np.array([1]), "competitor_b": np.array([2]),
                         "keep_patch_tokens": np.array([49]), "overlap_fraction": np.array([.5]),
                         "correctness_exchange_loss": np.array([0.])})
        cases = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
        result = module.average_seeds(cases, [1, 2])
        self.assertAlmostEqual(result["exchange_loss"][0], cases["exchange_loss"].mean())
        self.assertLess(result["identity_error"][0], 1e-14)
        self.assertNotAlmostEqual(result["displacement_norm"][0], cases["displacement_norm"].mean())


if __name__ == "__main__":
    unittest.main()
