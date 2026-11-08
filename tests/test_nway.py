import unittest

import numpy as np
import torch
from torch.nn import functional as F

from token_compression.budget_ranking import fit_responses, ridge_projector, sample_masks
from token_compression.nway import (build_panel, make_selector, margin_responses,
                                    normalized_targets, set_descriptor, task_manifest, training_order)
from token_compression.selector import TokenSelector, pair_descriptor, utility_targets


class NWayTests(unittest.TestCase):
    def test_descriptor_binary_equivalence_and_set_permutation(self):
        torch.manual_seed(31)
        text = torch.randn(6, 5, 12)
        torch.testing.assert_close(set_descriptor(text[:, :2]), pair_descriptor(text[:, :2]), atol=2e-7, rtol=2e-7)
        torch.testing.assert_close(set_descriptor(text), set_descriptor(text[:, [3, 1, 4, 0, 2]]))
        model = make_selector("conditioned", 8, 12, 7).eval()
        old = TokenSelector(torch.zeros(24), True, hidden_dim=7, token_dim=8).eval()
        old.load_state_dict(model.state_dict())
        tokens = torch.randn(6, 9, 8)
        torch.testing.assert_close(model(tokens, text[:, :2]), old(tokens, text[:, :2]))
        torch.testing.assert_close(model(tokens, text), model(tokens, text[:, [3, 1, 4, 0, 2]]))
        image_only = make_selector("image_only_wide", 8, 12, 9).eval()
        torch.testing.assert_close(image_only(tokens, text), image_only(tokens, text[:, :2]), rtol=0, atol=0)

    def test_binary_teacher_equals_previous_coefficient_teacher(self):
        rng = np.random.default_rng(7)
        responses = rng.normal(size=(3, 55, 8)).astype(np.float32)
        masks = sample_masks(12, 6, 55, 6)
        projector = ridge_projector(masks)
        coefficients = np.stack([fit_responses(projector, row)[0] for row in responses])
        torch.manual_seed(4)
        text = F.normalize(torch.randn(8, 11), dim=1)
        labels = torch.tensor([0, 2, 5])
        candidates = torch.tensor([[0, 4], [2, 6], [1, 5]])
        margin = margin_responses(torch.tensor(responses, dtype=torch.float64), labels, candidates, text.double())
        new = normalized_targets(margin @ torch.tensor(projector["projection"]).T)
        previous = utility_targets(torch.tensor(coefficients), labels, candidates, text)
        torch.testing.assert_close(new, previous, atol=1e-7, rtol=2e-6)

    def test_multiway_margin_tracks_changing_nearest_competitor(self):
        text = torch.eye(5, dtype=torch.float64)
        responses = torch.tensor([[[1., .2, .8, .1, .4], [1., 1.2, .2, .9, .4]]], dtype=torch.float64)
        result = margin_responses(responses, torch.tensor([0]), torch.arange(5)[None], text)
        expected = torch.tensor([[.2, -.2]], dtype=torch.float64) / np.sqrt(2)
        torch.testing.assert_close(result, expected)
        shuffled = margin_responses(responses, torch.tensor([0]), torch.tensor([[4, 3, 2, 1, 0]]), text)
        torch.testing.assert_close(shuffled, result)

    def test_real_class_counts_have_balanced_disjoint_tasks(self):
        for count in (37, 102):
            manifest = task_manifest(count)
            for way in (2, 5):
                seen = set()
                for split, rounds in (("train", 4), ("val", 2)):
                    rows = manifest["ways"][str(way)][split]
                    self.assertEqual(len(rows), count * rounds)
                    occurrences = np.zeros(count, int)
                    for row in rows:
                        classes = row["candidates"]
                        self.assertEqual(len(set(classes)), way)
                        self.assertNotIn(tuple(classes), seen)
                        seen.add(tuple(classes))
                        occurrences[classes] += 1
                    np.testing.assert_array_equal(occurrences, rounds * way)

    def test_panel_weights_and_balanced_sampling_include_all_candidates(self):
        labels = np.repeat(np.arange(5), [2, 3, 4, 5, 6])
        ids = np.array([f"i{i}" for i in range(len(labels))])
        tasks = [{"task_id": "a", "candidates": [0, 1, 2]}, {"task_id": "b", "candidates": [2, 3, 4]}]
        panel = build_panel(ids, labels, tasks)
        self.assertAlmostEqual(panel["weights"].sum(), 1.)
        for t, task in enumerate(tasks):
            for c in task["candidates"]:
                rows = (panel["task_indices"] == t) & (panel["labels"] == c)
                self.assertAlmostEqual(panel["weights"][rows].sum(), 1 / 6)
        order = training_order(panel, 1003, 13)
        np.testing.assert_array_equal(order, training_order(panel, 1003, 13))
        _, counts = np.unique(np.column_stack((panel["task_indices"][order], panel["labels"][order])), axis=0, return_counts=True)
        self.assertLessEqual(counts.max() - counts.min(), 1)


if __name__ == "__main__":
    unittest.main()
