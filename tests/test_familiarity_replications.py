"""Prospective partition support and shared-image aggregation checks."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_familiarity_switch import make_plan
from semantic_identity_protocol import class_partition
from analyze_familiarity_replications import equal_partition_cluster_summary


class FamiliarityReplicationTests(unittest.TestCase):
    def test_new_partitions_are_fixed_fresh_fits_with_shared_valid_codes(self):
        for class_count in (37, 102):
            observed = []
            for seed in (2100, 2101):
                plan = make_plan(class_count, {}, partition_seed=seed)
                self.assertEqual(plan["new_fits"], 24)
                self.assertIsNone(plan["original_references"])
                self.assertEqual(plan["partition_seed"], seed)
                h0, h1, rest = [np.asarray(plan[key]) for key in ("H0", "H1", "always_seen")]
                np.testing.assert_array_equal(h0, class_partition(class_count, seed)[1])
                self.assertFalse(set(h0) & set(h1))
                observed.append(h0)
                for mapping in plan["mappings"].values():
                    mapping = np.asarray(mapping)
                    for group in (h0, h1, rest, np.r_[h0, rest], np.r_[h1, rest]):
                        np.testing.assert_array_equal(np.sort(mapping[group]), np.sort(group))
            self.assertFalse(np.array_equal(*observed))

    def test_historical_reuse_call_preserves_original_plan(self):
        plan = make_plan(37, {}, Path("previous"))
        self.assertEqual(plan["partition_seed"], 2099)
        self.assertEqual(plan["new_fits"], 18)
        self.assertEqual(plan["original_references"], "previous")

    def test_duplicate_partition_does_not_manufacture_independent_images(self):
        case = dict(image_ids=np.array(["a", "b", "c", "d"]), weights=np.array([1., 2., 3., 4.]),
                    values=np.array([[-1., 2.], [1., 3.], [2., -1.], [4., 6.]]))
        single = equal_partition_cluster_summary([case], replicates=200)
        duplicated = equal_partition_cluster_summary([case, case], replicates=200)
        np.testing.assert_allclose(single["estimates"], duplicated["estimates"], atol=1e-12)
        np.testing.assert_allclose(single["ci95"], duplicated["ci95"], atol=1e-12)
        self.assertEqual(duplicated["images"], 4)

    def test_partitions_have_equal_weight_despite_different_total_row_mass(self):
        first = dict(image_ids=np.array(["a", "b"]), weights=np.array([1., 1.]),
                     values=np.array([[0.], [2.]]))
        second = dict(image_ids=np.array(["a", "b"]), weights=np.array([20., 20.]),
                      values=np.array([[4.], [6.]]))
        result = equal_partition_cluster_summary([first, second], replicates=100)
        self.assertAlmostEqual(result["estimates"][0], 3.)


if __name__ == "__main__":
    unittest.main()
