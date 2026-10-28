import unittest
import importlib.util
from pathlib import Path

import numpy as np

from token_compression.conditioning_gap import (
    average_seed_cases, cross_validated_gap_prediction, image_means,
    library_potential, predictor_features, probe_gap_association,
    regret_decomposition, summarize_columns,
)
from token_compression.selector_geometry import image_weights


class ConditioningGapTests(unittest.TestCase):
    def test_specialization_can_exist_without_a_learned_advantage(self):
        oracle = library_potential([[1., 0., .6]], [[0., 1., .6]])
        result = regret_decomposition(oracle["separate"], oracle["shared"], [.5], [.6])
        self.assertAlmostEqual(result["augmented_specialization_potential"][0], .4)
        self.assertAlmostEqual(result["regret_difference"][0], .5)
        self.assertAlmostEqual(result["learned_gain"][0], -.1)
        self.assertLess(np.max(np.abs(result["decomposition_residual"])), 1e-12)

    def test_augmented_library_can_mechanically_track_learned_gain(self):
        # Pure probes reveal no specialization. Adding the model's own masks
        # creates an oracle gap exactly equal to its outcome, so that gap is
        # unsuitable as independent predictive evidence about the same model.
        quality = np.array([.1, .2, .3, .4])
        zeros = np.zeros((4, 2))
        pure = library_potential(zeros, zeros)
        augmented = library_potential(np.column_stack([zeros, quality, -quality]),
                                      np.column_stack([zeros, -quality, quality]))
        np.testing.assert_array_equal(pure["gap"], 0)
        np.testing.assert_allclose(augmented["gap"], quality)
        result = regret_decomposition(augmented["separate"], augmented["shared"], quality, np.zeros(4))
        np.testing.assert_allclose(result["learned_gain"], quality)

    def test_shared_oracle_is_not_the_image_only_selector(self):
        oracle = library_potential([[.8, .3]], [[.7, .2]])
        result = regret_decomposition(oracle["separate"], oracle["shared"], [.6], [.25])
        self.assertAlmostEqual(result["augmented_specialization_potential"][0], 0.)
        self.assertGreater(result["learned_gain"][0], 0.)
        self.assertGreater(result["image_only_regret"][0], result["conditioned_regret"][0])

    def test_image_weighting_does_not_count_task_pairs_as_independent_images(self):
        ids = np.array(["a", "b", "b", "b"])
        cases = {"image_ids": ids, "gain": np.array([1., 0., 0., 0.])}
        _, means = image_means(ids, cases["gain"][:, None])
        np.testing.assert_array_equal(means[:, 0], [1., 0.])
        summary = summarize_columns(cases, ["gain"], replicates=40)
        self.assertEqual(summary["gain"]["mean"], .5)

    @staticmethod
    def prediction_fixture():
        rng = np.random.default_rng(22)
        n_images, pairs = 40, 3
        ids = np.repeat([f"i{i:02d}" for i in range(n_images)], pairs)
        n = len(ids)
        cases = {"image_ids": ids, "labels": np.repeat(np.arange(n_images) % 4, pairs),
                 "base_margin_a": rng.normal(.1, .05, n), "base_margin_b": rng.normal(.1, .05, n),
                 "angle_degrees": rng.uniform(10, 75, n), "probe_gap_512": rng.uniform(.001, .03, n)}
        cases["learned_gain"] = .7*cases["probe_gap_512"] + .002*cases["base_margin_a"] + rng.normal(0, .0003, n)
        return cases

    def test_predictive_effect_and_clustered_intervals(self):
        cases = self.prediction_fixture()
        fit = probe_gap_association(cases, replicates=40)
        self.assertTrue(fit["gap_estimable_beyond_controls"])
        self.assertAlmostEqual(fit["primary_gap_coefficient"]["estimate"], .007, delta=.0003)
        self.assertGreater(fit["primary_gap_coefficient"]["ci_familywise_budgets"][0], 0)
        cv = cross_validated_gap_prediction(cases, replicates=40)
        self.assertGreater(cv["incremental_out_of_fold_r2"], .8)
        self.assertGreater(cv["incremental_out_of_fold_r2_ci_familywise_budgets"][0], 0)

    def test_cross_validation_holds_out_images_and_training_only_scaling(self):
        cases = self.prediction_fixture()
        original = cross_validated_gap_prediction(cases, replicates=20)
        fold = original["out_of_fold_predictions"]["fold"]
        for image_id in np.unique(cases["image_ids"]):
            self.assertEqual(len(np.unique(fold[cases["image_ids"] == image_id])), 1)
        held = fold == 0
        changed = {key: value.copy() for key, value in cases.items()}
        changed["learned_gain"][held] += 100
        refit = cross_validated_gap_prediction(changed, replicates=20)
        for name in ("baseline", "gap_augmented", "training_mean"):
            np.testing.assert_allclose(refit["out_of_fold_predictions"][name][held],
                                       original["out_of_fold_predictions"][name][held], atol=1e-12, rtol=0)
        changed["probe_gap_512"][held] += 100
        rescaled = cross_validated_gap_prediction(changed, replicates=20)
        first, after = original["fold_models"][0], rescaled["fold_models"][0]
        np.testing.assert_allclose(first["training_feature_mean"], after["training_feature_mean"])
        np.testing.assert_allclose(first["training_feature_scale"], after["training_feature_scale"])
        features = predictor_features(cases)[~held]
        expected = np.sum(image_weights(cases["image_ids"][~held])[:, None]*features, axis=0)
        np.testing.assert_allclose(first["training_feature_mean"], expected)

    def test_seed_average_aligns_reordered_pairs_before_bootstrap(self):
        # Three pairs, repeated seeds in different row orders.
        pair = np.array([2, 0, 1, 1, 2, 0])
        seeds = np.array([1, 1, 1, 2, 2, 2])
        exchange = {"seed": seeds, "image_ids": np.array([f"image_{i}" for i in pair]),
                    "task_a": np.array(["a"]*6), "task_b": np.array(["b"]*6),
                    "labels": pair, "competitor_a": np.full(6, 3), "competitor_b": np.full(6, 4),
                    "keep_patch_tokens": np.full(6, 49), "angle_degrees": pair.astype(float),
                    "base_margin_a": pair*.1, "base_margin_b": pair*.2}
        for prefix, offset in [("", 0), ("image_only_", -1)]:
            for ending in ("aa", "bb"):
                exchange[f"{prefix}margin_{ending}"] = pair+seeds+offset
                exchange[f"{prefix}correct_{ending}"] = np.ones(6, dtype=np.int8)
        mean = average_seed_cases(exchange, [1, 2])
        np.testing.assert_allclose(mean["conditioned_margin"], [1.5, 2.5, 3.5])
        np.testing.assert_allclose(mean["image_only_margin"], [.5, 1.5, 2.5])
        np.testing.assert_array_equal(mean["image_ids"], ["image_0", "image_1", "image_2"])

    def test_probe_loader_alignment_and_halves_exclude_augmented_choices(self):
        path = Path(__file__).resolve().parents[1]/"scripts/analyze_conditioning_gap.py"
        spec = importlib.util.spec_from_file_location("analyze_conditioning_gap", path)
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        cases = {"image_ids": np.array(["b", "a"]), "task_a": np.array(["t0", "t0"]),
                 "task_b": np.array(["t1", "t1"]), "labels": np.array([1, 0]),
                 "competitor_a": np.array([2, 2]), "competitor_b": np.array([3, 3]),
                 "conditioned_margin": np.array([.1, .1]), "image_only_margin": np.zeros(2)}
        shared = {key: cases[key][::-1] for key in ("image_ids", "task_a", "task_b", "labels", "competitor_a", "competitor_b")}
        shared.update(separate_margin=np.array([.2, .3]), shared_margin=np.array([.15, .2]))
        evidence = {"image_ids": np.array(["a", "b"]), "labels": np.array([0, 1]), "text_features": np.eye(4)}
        responses = np.zeros((2, 512, 4))
        responses[0, :, 0] = .05
        responses[1, :, 1] = .1
        indices = script.attach_library_cases(cases, shared, evidence, responses)
        np.testing.assert_allclose(cases["augmented_separate_margin"], [.3, .2])
        np.testing.assert_allclose(cases["probe_separate_512"], [.1/np.sqrt(2), .05/np.sqrt(2)])
        np.testing.assert_array_equal(cases["probe_gap_512"], 0)
        np.testing.assert_array_equal(cases["probe_gap_half_a"], 0)
        np.testing.assert_array_equal(cases["probe_gap_half_b"], 0)
        self.assertFalse(set(indices["half_a"]) & set(indices["half_b"]))
        self.assertEqual(set(indices["half_a"]) | set(indices["half_b"]), set(range(512)))


if __name__ == "__main__":
    unittest.main()
