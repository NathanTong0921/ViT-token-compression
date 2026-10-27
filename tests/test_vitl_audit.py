import importlib.util
from pathlib import Path
import unittest

import numpy as np

from token_compression.selector_geometry import (adjusted_angle_association,
                                                 cross_validated_angle_prediction)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_vitl_validation.py"
SPEC = importlib.util.spec_from_file_location("audit_vitl_validation", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class ViTLScientificAuditTests(unittest.TestCase):
    def test_task_class_weights_do_not_assume_equal_image_or_task_counts(self):
        tasks = np.array(["a", "a", "a", "a", "b", "b", "b"])
        labels = np.array([0, 1, 1, 1, 0, 2, 2])
        weights = audit.task_class_weights(tasks, labels)
        np.testing.assert_allclose(weights, [.25, 1/12, 1/12, 1/12, .25, .125, .125])
        self.assertAlmostEqual(weights.sum(), 1.)

    def test_weighted_cluster_bootstrap_matches_direct_repeated_observations(self):
        ids = np.array(["a", "b", "c", "d", "a", "e", "f"])
        weights = np.array([.25, 1/12, 1/12, 1/12, .25, .125, .125])
        values = np.column_stack([np.arange(7), np.arange(7)**2])
        point, ci, family_ci = audit.paired_bootstrap(ids, weights, values, 100, 7)
        unique, rng = np.unique(ids), np.random.default_rng(7)
        direct = []
        for _ in range(100):
            rows = np.concatenate([np.flatnonzero(ids == image) for image in
                                   unique[rng.integers(len(unique), size=len(unique))]])
            direct.append(np.average(values[rows], axis=0, weights=weights[rows]))
        np.testing.assert_allclose(point, np.average(values, axis=0, weights=weights))
        np.testing.assert_allclose(ci, np.quantile(direct, [.025, .975], axis=0))
        np.testing.assert_array_equal(ci, family_ci)

    def test_geometry_uses_all_768_dimensions_and_checks_image_only_null(self):
        basis = np.eye(768)[[0, 1, 767]]
        learned = {"text_features": basis,
                   "image_ids": np.array(["image"] * 4), "task_ids": np.array(["a", "b", "a", "b"]),
                   "seed": np.ones(4, int), "method": np.array(["conditioned"] * 2 + ["image_only"] * 2),
                   "labels": np.zeros(4, int), "class_pairs": np.array([[0, 1], [0, 2]] * 2),
                   "image_features": basis[[0, 2, 0, 0]]}
        reference = {"image_ids": np.array(["image"] * 2), "task_ids": np.array(["a", "b"]),
                     "method": np.array(["full_model"] * 2), "image_features": basis[[0, 0]]}
        cases = {"image_ids": np.array(["image"]), "task_a": np.array(["a"]), "task_b": np.array(["b"]),
                 "seed": np.array([1]), "labels": np.array([0]), "competitor_a": np.array([1]),
                 "competitor_b": np.array([2]), "exchange_loss": np.array([-.5 / np.sqrt(2)]),
                 "exchange_loss_from_scores": np.array([-.5 / np.sqrt(2)]),
                 "base_margin_a": np.array([1 / np.sqrt(2)]), "base_margin_b": np.array([1 / np.sqrt(2)]),
                 "angle_degrees": np.array([60.]), "displacement_norm": np.array([np.sqrt(2)])}
        result = audit.geometry_from_predictions(learned, reference, cases, 768)
        self.assertLess(result["identity_max_abs_error"], 1e-14)
        learned["image_features"][3] = basis[2]
        with self.assertRaises(AssertionError):
            audit.geometry_from_predictions(learned, reference, cases, 768)

    def test_h2_and_held_image_refit_with_unequal_pairs_per_image(self):
        rng = np.random.default_rng(13)
        ids = np.repeat([f"i{i:02d}" for i in range(40)], [2, 5] * 20)
        labels = np.repeat(np.arange(40) % 4, [2, 5] * 20)
        n = len(ids)
        cases = {"image_ids": ids, "labels": labels, "base_margin_a": rng.normal(size=n),
                 "base_margin_b": rng.normal(size=n), "angle_degrees": rng.uniform(20, 80, n)}
        x = audit.regression_features(cases)
        cases["exchange_loss"] = x @ np.array([.2, -.1, .01, .03, -.02, .04]) + rng.normal(0, .03, n)
        for fixed in (False, True):
            reference = adjusted_angle_association(cases, replicates=10, image_fixed_effects=fixed, family_size=1)
            independent = audit.angle_fit(cases, fixed)
            self.assertAlmostEqual(independent["angle_slope"], reference["angle_slope"], places=12)
            self.assertEqual(independent["design_rank"], reference["design_rank"])
        cv = cross_validated_angle_prediction(cases, replicates=10)
        predictions = cv.pop("out_of_fold_predictions")
        independent = audit.held_image_audit(cases, predictions, cv, 2060)
        self.assertTrue(independent["independent_refit_matches_saved_predictions"])
        predictions["difficulty_plus_angle"][0] += .001
        with self.assertRaises(AssertionError):
            audit.held_image_audit(cases, predictions, cv, 2060)


if __name__ == "__main__":
    unittest.main()
