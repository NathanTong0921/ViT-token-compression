"""Check the reserved-cache boundary and final-batch handling with synthetic data."""

import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


spec = importlib.util.spec_from_file_location(
    "prepare_final_test_cache", Path(__file__).parents[1] / "scripts" / "prepare_final_test_cache.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class FinalTestCacheTests(unittest.TestCase):
    def manifests(self):
        images = [{"image_id": f"{split}_{i}", "class_id": i, "split": split}
                  for split in ("train", "val", "test") for i in range(3)]
        tasks = [{"task_id": name, "class_a": a, "class_b": b, "split": split}
                 for name, a, b, split in (("01", 0, 1, "train"),
                                           ("12", 1, 2, "val"), ("02", 0, 2, "test"))]
        classes = [{"class_id": i, "class_name": str(i)} for i in range(3)]
        return images, classes, tasks

    def test_rejects_image_overlap_across_splits(self):
        images, _, tasks = self.manifests()
        images[-1]["image_id"] = images[0]["image_id"]
        with self.assertRaises(AssertionError):
            prepare.test_partition(images, tasks)

    def test_rejects_reversed_pair_overlap(self):
        images, _, tasks = self.manifests()
        tasks[-1].update(class_a=1, class_b=0)
        with self.assertRaises(AssertionError):
            prepare.test_partition(images, tasks)

    def test_only_test_images_cached_and_padding_is_discarded(self):
        images, classes, tasks = self.manifests()
        dataset_rows, batches = [], []

        class Images(torch.utils.data.Dataset):
            def __init__(self, root, rows, preprocess):
                dataset_rows.extend(rows)
                self.rows = rows

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                row = self.rows[index]
                values = torch.tensor([1., 2., 3.]) + row["class_id"]
                return values[:, None, None], row["class_id"], row["image_id"]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = torch.nn.Module()
                self.visual.input_resolution = 1
                self.visual.conv1 = torch.nn.Conv2d(3, 3, 1)

            def encode_text(self, tokens):
                return torch.eye(3)

            def encode_image(self, pixels):
                batches.append(len(pixels))
                return pixels[:, :, 0, 0]

        class Pruner:
            def __init__(self, visual, prune_after):
                self.num_patches, self.blocks = 4, [None, None]

            def prefix(self, pixels):
                values = pixels[:, :, 0, 0]
                return values[:, None].expand(-1, 5, -1), torch.ones(len(pixels), 4)

            def finish(self, tokens, indices):
                return tokens[:, 0]

        fake_clip = SimpleNamespace(load=lambda *a, **kw: (Model(), "synthetic"),
                                    tokenize=lambda prompts: torch.arange(len(prompts)))
        config = {"model": "synthetic", "model_cache": "unused", "data_root": "unused",
                  "manifest_dir": "unused", "prompt_template": "a {class_name}", "seed": 2026,
                  "device": "cpu", "prune_after": 1, "image_batch_size": 2,
                  "num_workers": 0, "identity_atol": 1e-6}
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, {"clip": fake_clip}):
            output = Path(temporary) / "cache"
            with patch("token_compression.baseline.PetsImages", Images), \
                 patch("token_compression.pets.load_manifests", return_value=(images, classes, tasks)), \
                 patch("token_compression.pruning.SingleStagePruner", Pruner):
                summary = prepare.run(config, output)
            self.assertEqual([r["split"] for r in dataset_rows], ["test"] * 3)
            self.assertEqual(batches, [2, 2])
            self.assertEqual(summary["batching"]["physical_images_per_forward_path"], 4)
            self.assertEqual(summary["identity"]["test"]["images_checked"], 3)
            self.assertTrue(summary["test_images_used"])
            self.assertFalse((output / "val").exists())
            self.assertFalse((output / "train").exists())
            self.assertEqual(np.load(output / "test" / "prefix_tokens.npy").shape, (3, 5, 3))
            with np.load(output / "test" / "evidence.npz") as evidence:
                np.testing.assert_array_equal(evidence["image_ids"], [f"test_{i}" for i in range(3)])
                np.testing.assert_allclose(evidence["full_scores"], evidence["full_features"])
            with (output / "test_tasks.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0], {k: str(v) for k, v in tasks[-1].items()})


if __name__ == "__main__":
    unittest.main()
