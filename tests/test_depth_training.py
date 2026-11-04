"""Scientific controls for the depth-intervention preparation wrapper."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from run_depth_training import check_full_reference, check_training_cache, raw_utility_scales
from token_compression.budget_ranking import sample_masks


class DepthTrainingTests(unittest.TestCase):
    def test_training_rejects_wrong_depth_or_mask_design(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, features = root / "supervision", root / "features"
            config = {"cache_dir": str(cache), "feature_cache_dir": str(features),
                      "model": "ViT-B/16", "prune_after": 3, "keep_patch_tokens": 2,
                      "probe_count": 4, "mask_seed": 2031, "ridge": .01,
                      "prompt_template": "a {class_name}"}
            seed = int(np.random.SeedSequence([2031, 2, 0]).generate_state(1)[0])
            masks = sample_masks(4, 2, 4, seed)
            for folder in (cache, features):
                folder.mkdir()
                (folder / "prep_summary.json").write_text(json.dumps({"config": config}))
            for split in ("train", "val"):
                (cache / split).mkdir()
                (features / split).mkdir()
                np.save(features / split / "prefix_tokens.npy", np.zeros((2, 5, 3)))
                (cache / split / "prefix_tokens.npy").symlink_to(features / split / "prefix_tokens.npy")
                np.savez(cache / split / "teacher.npz", masks=masks)
            check_training_cache(config)
            (cache / "prep_summary.json").write_text(json.dumps({"config": {**config, "prune_after": 6}}))
            with self.assertRaises(AssertionError):
                check_training_cache(config)
            (cache / "prep_summary.json").write_text(json.dumps({"config": config}))
            np.savez(cache / "val" / "teacher.npz", masks=np.zeros_like(masks))
            with self.assertRaises(AssertionError):
                check_training_cache(config)

    def test_raw_scale_has_expected_geometry_and_class_offset_invariance(self):
        vector = np.array([-3., -1., 1., 3.], np.float32)
        coefficients = np.stack([vector, np.zeros(4), -vector], axis=-1)[None].repeat(2, axis=0)
        labels = np.array([0, 1])
        pairs = np.array([[0, 1], [1, 2]])
        text = np.array([[1., 0.], [0., 1.], [-1., 0.]], np.float32)
        scales, images, tasks = raw_utility_scales(coefficients, labels, pairs, text)
        np.testing.assert_array_equal(images, [0, 1, 1])
        np.testing.assert_array_equal(tasks, [0, 0, 1])
        np.testing.assert_allclose(scales, np.sqrt(5 / 2), rtol=1e-6)
        shifted = coefficients + np.array([2., -7., 11.])[None, None, :]
        np.testing.assert_allclose(raw_utility_scales(shifted, labels, pairs, text)[0], scales)
        np.testing.assert_allclose(raw_utility_scales(coefficients * 2, labels, pairs, text)[0], 2 * scales)
        np.testing.assert_array_equal(raw_utility_scales(np.ones_like(coefficients), labels, pairs, text)[0], 0)

    def test_full_reference_rejects_changed_images_or_task_partition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            current, reference = root / "new", root / "old"
            config = {"feature_cache_dir": str(current), "reference_cache_dir": str(reference),
                      "model": "ViT-B/16", "prompt_template": "a {class_name}",
                      "image_batch_size": 64, "prune_after": 3, "identity_atol": 1e-6}
            evidence = dict(image_ids=np.array(["image_a", "image_b"]), labels=np.array([0, 1]),
                            text_features=np.eye(2), full_features=np.eye(2), full_scores=np.eye(2))
            for cache in (current, reference):
                cache.mkdir()
                for split in ("train", "val"):
                    folder = cache / split
                    folder.mkdir()
                    np.savez(folder / "evidence.npz", **evidence)
                    (cache / f"{split}_tasks.csv").write_text("task_id,class_a,class_b\npair_a,0,1\n")
            (reference / "prep_summary.json").write_text(json.dumps({"config": {**config, "prune_after": 6}}))
            result = check_full_reference(config)
            self.assertEqual(result["splits"]["train"]["images"], 2)
            self.assertEqual(result["splits"]["val"]["multiclass_prediction_disagreements"], 0)
            np.savez(current / "val" / "evidence.npz", **{**evidence, "image_ids": evidence["image_ids"][::-1]})
            with self.assertRaises(AssertionError):
                check_full_reference(config)
            np.savez(current / "val" / "evidence.npz", **evidence)
            (current / "val_tasks.csv").write_text("task_id,class_a,class_b\npair_b,0,1\n")
            with self.assertRaises(AssertionError):
                check_full_reference(config)


if __name__ == "__main__":
    unittest.main()
