import contextlib
import csv
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from token_compression.selector_scale import (balanced_example_weights,
                                               balanced_training_stream,
                                               map_stream_images,
                                               validate_image_partition)


class SelectorScaleTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([0, 1, 1, 1, 2, 2])
        self.ids = np.array(["a", "d", "b", "c", "f", "e"])
        self.pairs = np.array([[0, 1], [1, 2]])

    def test_exact_task_side_counts_and_reproducibility(self):
        stream = balanced_training_stream(self.pairs, 4000, 2041)
        groups = 2 * stream["pair_indices"] + stream["sides"]
        np.testing.assert_array_equal(np.bincount(groups), [1000] * 4)
        repeated = balanced_training_stream(self.pairs, 4000, 2041)
        for key in stream:
            np.testing.assert_array_equal(stream[key], repeated[key])

    def test_unequal_image_counts_do_not_change_task_or_side_distribution(self):
        stream = balanced_training_stream(self.pairs, 40000, 2)
        selected = map_stream_images(self.labels, self.ids, stream, self.pairs)
        draw_labels = self.pairs[stream["pair_indices"], stream["sides"]]
        np.testing.assert_array_equal(self.labels[selected], draw_labels)
        np.testing.assert_array_equal(np.bincount(draw_labels), [10000, 20000, 10000])
        counts = np.bincount(selected, minlength=6)
        # Class 1 has three available images and occurs in both tasks.
        self.assertLess(np.max(np.abs(counts[1:4] / 20000 - 1 / 3)), 0.02)

    def test_nested_coverage_preserves_shared_task_class_stream(self):
        stream = balanced_training_stream(self.pairs, 400, 7)
        small = ["a", "b", "e"]
        full_images = map_stream_images(self.labels, self.ids, stream, self.pairs)
        small_images = map_stream_images(self.labels, self.ids, stream, self.pairs, small)
        np.testing.assert_array_equal(self.labels[full_images], self.labels[small_images])
        self.assertEqual(set(self.ids[small_images]), set(small))
        # Within-class ordering is by ID, not by the cache's array order.
        fixed = {"pair_indices": np.array([0, 0]), "sides": np.array([1, 1]),
                 "image_quantiles": np.array([0.01, 0.99])}
        np.testing.assert_array_equal(
            self.ids[map_stream_images(self.labels, self.ids, fixed, self.pairs)], ["b", "d"])

    def test_evaluation_weights_balance_each_task_and_class(self):
        images = np.array([0, 1, 2, 3, 1, 2, 3, 4, 5])
        tasks = np.array([0] * 4 + [1] * 5)
        weights = balanced_example_weights(self.labels, self.pairs, images, tasks)
        self.assertAlmostEqual(weights.sum(), 1.)
        for task, pair in enumerate(self.pairs):
            for label in pair:
                mask = (tasks == task) & (self.labels[images] == label)
                self.assertAlmostEqual(weights[mask].sum(), .25)

    def test_image_holdout_and_nested_panel_are_checked(self):
        validate_image_partition(self.ids, ["held_out"], ["a", "b", "e"])
        with self.assertRaises(AssertionError):
            validate_image_partition(self.ids, ["a"], ["a"])
        with self.assertRaises(AssertionError):
            validate_image_partition(self.ids, ["held_out"], ["missing"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Training smoke requires PyTorch")
    def test_cpu_training_keeps_update_budget_and_shared_initialization(self):
        import torch

        script = Path(__file__).resolve().parents[1] / "scripts" / "train_selector_scale.py"
        spec = importlib.util.spec_from_file_location("train_selector_scale_smoke", script)
        training = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(training)
        rng = np.random.default_rng(3)
        text = rng.normal(size=(3, 512)).astype(np.float32)
        text /= np.linalg.norm(text, axis=1, keepdims=True)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as folder:
                folder = Path(folder)
                cache, old = folder / "cache", folder / "old"
                for split, labels in (("train", [0, 0, 1, 1, 1, 2, 2]), ("val", [0, 0, 2])):
                    target = cache / split
                    target.mkdir(parents=True)
                    ids = np.array([f"{split}_{i}" for i in range(len(labels))])
                    np.savez(target / "evidence.npz", image_ids=ids, labels=np.array(labels), text_features=text)
                    np.save(target / "prefix_tokens.npy", rng.normal(size=(len(labels), 5, 768)).astype(np.float32))
                    np.savez(target / "teacher.npz", image_ids=ids,
                             coefficients=rng.normal(size=(len(labels), 4, 3)).astype(np.float32))
                (old / "train").mkdir(parents=True)
                np.savez(old / "train" / "evidence.npz", image_ids=np.array(["train_0", "train_2", "train_5"]))
                for split, pairs in (("train", [[0, 1], [1, 2]]), ("val", [[0, 2]])):
                    with (cache / f"{split}_tasks.csv").open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=["task_id", "class_a", "class_b"])
                        writer.writeheader()
                        writer.writerows({"task_id": f"{a}_{b}", "class_a": a, "class_b": b} for a, b in pairs)
                (old / "train_tasks.csv").write_text((cache / "train_tasks.csv").read_text())
                config = {"cache_dir": str(cache), "source_cache_dir": str(old), "device": "cpu",
                          "batch_size": 2, "temperature": 1., "training_steps": 4, "checkpoint_interval": 2,
                          "hidden_dim": 4, "training_seeds": [11], "learning_rate": .0003,
                          "weight_decay": .01, "coverages": ["small", "full"]}
                original_step = torch.optim.AdamW.step
                with mock.patch.object(torch.optim.AdamW, "step", autospec=True, side_effect=original_step) as steps:
                    with contextlib.redirect_stdout(io.StringIO()):
                        summary = training.train(config, folder / "output")
                    self.assertEqual(steps.call_count, 4 * 4)
                runs = {(row["coverage"], row["arm"]): row for row in summary["runs"]}
                self.assertEqual(summary["train_image_counts"], {"small": 3, "full": 7})
                self.assertEqual(summary["val_image_count"], 3)
                for arm in ("conditioned", "image_only"):
                    self.assertEqual(runs[("small", arm)]["initial_val_kl"], runs[("full", arm)]["initial_val_kl"])
                    for coverage in ("small", "full"):
                        row = runs[(coverage, arm)]
                        self.assertEqual(row["training_steps"], 4)
                        self.assertTrue(np.isfinite(row["final_val_kl"]))
                        checkpoint = torch.load(folder / "output" / row["checkpoint"], weights_only=True)
                        self.assertEqual(checkpoint["step"], 4)
                        self.assertEqual(checkpoint["conditioned"], arm == "conditioned")
                # A budget sweep trains only the full panel. Removing the small
                # arm must not change its initialization, draws or final weights.
                full_config = {**config, "coverages": ["full"], "experiment_name": "budget_smoke"}
                with contextlib.redirect_stdout(io.StringIO()):
                    full_summary = training.train(full_config, folder / "full_only")
                self.assertEqual(full_summary["train_image_counts"], {"full": 7})
                self.assertEqual(full_summary["experiment_name"], "budget_smoke")
                self.assertEqual(len(full_summary["runs"]), 2)
                for row in full_summary["runs"]:
                    old = torch.load(folder / "output" / row["checkpoint"], weights_only=True)
                    new = torch.load(folder / "full_only" / row["checkpoint"], weights_only=True)
                    for key in old["state_dict"]:
                        torch.testing.assert_close(old["state_dict"][key], new["state_dict"][key], rtol=0, atol=0)
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
