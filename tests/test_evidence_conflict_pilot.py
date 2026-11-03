import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

spec = importlib.util.spec_from_file_location(
    "evidence_conflict_pilot", Path(__file__).resolve().parents[1] / "scripts/run_evidence_conflict_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class EvidenceConflictPilotTests(unittest.TestCase):
    def test_panel_excludes_test_images_pairs_and_other_present_label(self):
        images = [{"image_id": f"c{c}_{split}", "class_id": c, "split": split,
                   "official_split": split} for c in range(5) for split in ("train", "val", "test")]
        tasks = [{"task_id": f"{a}_{b}", "class_a": a, "class_b": b,
                  "split": "test" if (a, b) == (0, 1) else "val"}
                 for a in range(5) for b in range(a + 1, 5)]
        panels, _ = pilot.build_panel(images, tasks, np.eye(5), 20, 2081)
        again, _ = pilot.build_panel(images, tasks, np.eye(5), 20, 2081)
        self.assertEqual(panels, again)
        for panel in panels:
            self.assertEqual(len(panel["tasks"]), 4)
            present = {source["class_id"] for source in panel["sources"]}
            self.assertEqual(len(present), 2)
            self.assertTrue(all(source["split"] == "val" for source in panel["sources"]))
            for task in panel["tasks"]:
                self.assertNotEqual(task["task_id"], "0_1")
                self.assertEqual({task["label"], task["distractor"]} & present, {task["label"]})

    def test_spatial_masks_unique_and_tiles_are_local(self):
        panels = [{"layout": "horizontal"}, {"layout": "vertical"}]
        attention = np.tile(np.arange(196), (2, 1))
        masks, names = pilot.mask_bank(attention, panels, 49, 3, 2081)
        self.assertEqual(masks.shape, (2, 7, 49))
        for mask in masks.reshape(-1, 49):
            self.assertEqual(len(set(mask)), 49)
            self.assertTrue(np.all((mask >= 0) & (mask < 196)))
        self.assertTrue(np.all(masks[0, -2] % 14 < 7))
        self.assertTrue(np.all(masks[0, -1] % 14 >= 7))
        self.assertTrue(np.all(masks[1, -2] // 14 < 7))
        self.assertTrue(np.all(masks[1, -1] // 14 >= 7))
        np.testing.assert_array_equal(masks[0, :3], masks[1, :3])

    def test_conflict_requires_individual_feasibility_without_common_mask(self):
        panel = {"panel_id": "p", "layout": "horizontal",
                 "sources": [{"image_id": "a"}, {"image_id": "b"}],
                 "tasks": [{"object_slot": slot, "label": slot, "distractor": 2 + j}
                           for slot in (0, 1) for j in (0, 1)]}
        margins = np.array([[[1., 1., -1., -1.], [-1., -1., 1., 1.]]])
        rows = pilot.pair_diagnostics(margins, np.ones((1, 4)), [panel], 3, 16, np.eye(4))
        self.assertEqual(sum(r["different_objects"] for r in rows), 4)
        for row in rows:
            self.assertEqual(row["conflict"], row["different_objects"])
            self.assertEqual(row["separate_feasible"], 1)
            self.assertGreaterEqual(row["mean_margin_gap"], 0)
        # Adding a jointly successful mask removes conflict, even if another mask
        # still gives a higher task-specific margin.
        margins = np.concatenate((margins, np.full((1, 1, 4), .1)), axis=1)
        rows = pilot.pair_diagnostics(margins, np.ones((1, 4)), [panel], 3, 16, np.eye(4))
        self.assertTrue(all(row["conflict"] == 0 for row in rows))

    def test_cls_only_retains_contextual_cls_then_native_suffix(self):
        torch.manual_seed(4)
        blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)])
        visual = SimpleNamespace(ln_post=torch.nn.LayerNorm(4), proj=torch.randn(4, 3))
        pruner = SimpleNamespace(blocks=blocks, prune_after=1, visual=visual)
        tokens = torch.randn(2, 5, 4)
        expected = visual.ln_post(blocks[1](tokens[:, 0])) @ visual.proj
        actual = pilot.finish_with_cls_only(pruner, tokens)
        torch.testing.assert_close(actual, expected)
        tokens[:, 1:] += 100
        torch.testing.assert_close(actual, pilot.finish_with_cls_only(pruner, tokens))


if __name__ == "__main__":
    unittest.main()
