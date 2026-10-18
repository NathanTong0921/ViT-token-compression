"""Checks for preserving pilot supervision while expanding image coverage."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location(
    "prepare_selector_scale", Path(__file__).parents[1] / "scripts" / "prepare_selector_scale.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class SelectorScalePreparationTests(unittest.TestCase):
    def test_full_panels_preserve_unequal_class_coverage_and_split_order(self):
        images = [{"image_id": image_id, "class_id": label, "split": split}
                  for image_id, label, split in (
                      ("a", 0, "train"), ("b", 0, "train"), ("c", 1, "train"),
                      ("d", 0, "val"), ("e", 2, "val"), ("f", 1, "test"))]
        tasks = [{"task_id": task, "class_a": a, "class_b": b, "split": split}
                 for task, a, b, split in (("01", 0, 1, "train"),
                                         ("02", 0, 2, "val"), ("12", 1, 2, "test"))]
        panels, selected_tasks = prepare.full_panels(images, tasks)
        self.assertEqual([row["image_id"] for row in panels["train"]], ["a", "b", "c"])
        self.assertEqual([row["image_id"] for row in panels["val"]], ["d", "e"])
        self.assertEqual([row["task_id"] for row in selected_tasks["train"]], ["01"])
        self.assertEqual([row["task_id"] for row in selected_tasks["val"]], ["02"])

    def test_source_teachers_are_mapped_by_image_id_and_remain_exact(self):
        masks = np.array([[1, 0, 1], [0, 1, 1]], dtype=bool)
        source = {"image_ids": np.array(["c", "a"]), "masks": masks,
                  "coefficients": np.arange(12, dtype=np.float64).reshape(2, 3, 2),
                  "intercepts": np.arange(4, dtype=np.float64).reshape(2, 2),
                  "responses": np.arange(8, dtype=np.float32).reshape(2, 2, 2)}
        teacher = prepare.initialize_teacher(np.array(["a", "b", "c"]), source, masks)
        np.testing.assert_array_equal(teacher["coefficients"][[2, 0]], source["coefficients"])
        np.testing.assert_array_equal(teacher["responses"][[2, 0]], source["responses"])
        np.testing.assert_array_equal(teacher["intercepts"][[2, 0]], source["intercepts"])
        self.assertEqual(teacher["teacher_reused"].tolist(), [True, False, True])
        self.assertEqual(teacher["response_available"].tolist(), [True, False, True])

    def test_coefficient_only_validation_cache_does_not_fabricate_responses(self):
        masks = np.eye(3, dtype=bool)
        source = {"image_ids": np.array(["b"]), "coefficients": np.ones((1, 3, 2)),
                  "intercepts": np.ones((1, 2))}
        teacher = prepare.initialize_teacher(np.array(["a", "b"]), source, masks)
        self.assertEqual(teacher["teacher_reused"].tolist(), [False, True])
        self.assertEqual(teacher["response_available"].tolist(), [False, False])
        np.testing.assert_array_equal(teacher["coefficients"][1], source["coefficients"][0])
        self.assertEqual(np.count_nonzero(teacher["responses"]), 0)

    def test_different_mask_design_is_not_reused(self):
        masks = np.eye(3, dtype=bool)
        source = {"image_ids": np.array(["a"]), "coefficients": np.ones((1, 3, 2)),
                  "intercepts": np.ones((1, 2)), "masks": np.roll(masks, 1, axis=0)}
        with self.assertRaises(AssertionError):
            prepare.initialize_teacher(np.array(["a"]), source, masks)


if __name__ == "__main__":
    unittest.main()
