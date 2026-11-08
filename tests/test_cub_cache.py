"""Check CUB cache task-weighting and compatibility with existing mask designs."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_cub_cache import mask_bank, reference_summary
from token_compression.budget_ranking import sample_masks


class CUBCacheTests(unittest.TestCase):
    def test_references_balance_tasks_and_classes_not_image_frequency(self):
        evidence = {
            "image_ids": np.array(["a", "b", "c", "d", "e"]),
            "labels": np.array([0, 0, 0, 1, 2]),
            "text_features": np.eye(3),
            "full_scores": np.array([[.9, .1, .2], [.8, .1, .2], [.7, .1, .2],
                                     [.6, .5, .4], [.3, .7, .9]]),
            "attention_full_scores": np.zeros((5, 3)),
        }
        manifest = {"ways": {"2": {"val": [
            {"task_id": "a", "candidates": [0, 1]},
            {"task_id": "b", "candidates": [1, 2]},
        ]}}}
        rows = reference_summary(evidence, manifest, "val")
        self.assertEqual(rows[0]["method"], "full_model")
        self.assertAlmostEqual(rows[0]["accuracy"], .75)
        self.assertAlmostEqual(rows[1]["accuracy"], .5)
        self.assertEqual(rows[0]["task_examples"], 6)
        self.assertEqual(rows[0]["images"], 5)
        self.assertGreater(rows[0]["geometric_margin"], rows[1]["geometric_margin"])

    def test_mask_bank_matches_existing_budget_protocol(self):
        config = {"mask_seed": 2031, "keep_patch_tokens": 98, "probe_count": 512}
        masks, retained, seed = mask_bank(config, 196)
        old_seed = int(np.random.SeedSequence([2031, 98, 0]).generate_state(1)[0])
        self.assertEqual(seed, old_seed)
        np.testing.assert_array_equal(masks, sample_masks(196, 98, 512, old_seed))
        self.assertEqual(retained.shape, (512, 98))
        self.assertTrue(np.all(np.diff(retained, axis=1) > 0))
        rebuilt = np.zeros_like(masks)
        np.put_along_axis(rebuilt, retained, True, axis=1)
        np.testing.assert_array_equal(masks, rebuilt)


if __name__ == "__main__":
    unittest.main()
