"""Check that budget sweeps reuse only correctly aligned frozen features."""

import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location(
    "prepare_selector_budget", Path(__file__).parents[1] / "scripts" / "prepare_selector_budget.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class SelectorBudgetPreparationTests(unittest.TestCase):
    def create_source(self, root):
        source, output = root / "source", root / "budget"
        panels = {"train": [{"image_id": "train_a", "class_id": 0}],
                  "val": [{"image_id": "val_a", "class_id": 1}]}
        tasks = {"train": [{"task_id": "01", "class_a": 0, "class_b": 1}],
                 "val": [{"task_id": "12", "class_a": 1, "class_b": 2}]}
        for split, panel in panels.items():
            folder = source / split
            folder.mkdir(parents=True)
            np.save(folder / "prefix_tokens.npy", np.ones((1, 5, 8), dtype=np.float16))
            np.savez(folder / "evidence.npz", image_ids=[panel[0]["image_id"]],
                     labels=[panel[0]["class_id"]], attention_scores=np.ones((1, 4)))
            (folder / "panel_images.csv").write_text("image_id,class_id\n")
            with (source / f"{split}_tasks.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(tasks[split][0]))
                writer.writeheader()
                writer.writerows(tasks[split])
        (source / "panel_images.csv").write_text("image_id,class_id\n")
        config = {"model": "ViT-B/16", "prune_after": 6, "image_batch_size": 64,
                  "prompt_template": "a {class_name}", "keep_patch_tokens": 3}
        source_config = {**config, "keep_patch_tokens": 1}
        return source, output, panels, tasks, config, {"config": source_config}

    def test_new_budget_reuses_same_features_without_copying_tensors(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.create_source(Path(folder))
            records = prepare.link_feature_cache(*args)
            source, output = args[:2]
            for split in ("train", "val"):
                for name in ("prefix_tokens.npy", "evidence.npz"):
                    linked = output / split / name
                    self.assertTrue(linked.is_symlink())
                    self.assertEqual(linked.resolve(), (source / split / name).resolve())
                self.assertEqual(records[split]["images"], 1)
                self.assertFalse((output / split / "teacher.npz").exists())

    def test_different_label_assignment_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.create_source(Path(folder))
            args[2]["train"][0]["class_id"] = 2
            with self.assertRaises(AssertionError):
                prepare.link_feature_cache(*args)

    def test_different_task_partition_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.create_source(Path(folder))
            args[3]["train"][0]["class_b"] = 2
            with self.assertRaises(AssertionError):
                prepare.link_feature_cache(*args)

    def test_different_pruning_boundary_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.create_source(Path(folder))
            args[4]["prune_after"] = 7
            with self.assertRaises(AssertionError):
                prepare.link_feature_cache(*args)


if __name__ == "__main__":
    unittest.main()
