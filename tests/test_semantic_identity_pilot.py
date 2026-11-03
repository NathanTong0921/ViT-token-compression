"""Scientific controls for the semantic/identity development experiment."""

import csv
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import io
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from semantic_identity_protocol import (class_partition, class_permutation, fixed_step_stream,
                                        mapped_candidate_text, pair_strata, training_membership,
                                        paired_stratum_interaction)
from run_semantic_identity_pilot import load_data, evaluate_model, losses, make_model, run
from token_compression.selector import shared_descriptor
from token_compression.selector_scale import balanced_training_stream


class SemanticIdentityProtocolTests(unittest.TestCase):
    def test_partition_is_fixed_and_permutation_preserves_each_bank(self):
        seen, held = class_partition(37)
        self.assertEqual(len(held), 10)
        np.testing.assert_array_equal(np.sort(np.r_[seen, held]), np.arange(37))
        np.testing.assert_array_equal(held, class_partition(37)[1])
        mappings = [class_permutation(37, seed, (seen, held)) for seed in (2097, 2098)]
        for mapping in mappings:
            np.testing.assert_array_equal(np.sort(mapping[seen]), seen)
            np.testing.assert_array_equal(np.sort(mapping[held]), held)
        self.assertFalse(np.array_equal(*mappings))

    def test_holdout_removes_both_images_and_candidate_classes(self):
        labels = np.array([0, 1, 2, 3, 1])
        pairs = np.array([[0, 1], [1, 2], [2, 3], [0, 3]])
        image_mask, pair_mask = training_membership(labels, pairs, [0, 1])
        np.testing.assert_array_equal(labels[image_mask], [0, 1, 1])
        np.testing.assert_array_equal(pairs[pair_mask], [[0, 1]])
        np.testing.assert_array_equal(pair_strata(pairs, [2, 3]),
                                      ["known_known", "known_new", "new_new", "known_new"])

    def test_known_regime_reuses_exact_original_stream(self):
        pairs = np.array([[0, 1], [0, 2], [1, 2]])
        original = balanced_training_stream(pairs, 600, 2041)
        new = fixed_step_stream(pairs, 600, 2041)
        for key in original:
            np.testing.assert_array_equal(original[key], new[key])
        partial = fixed_step_stream(pairs, 607, 2041)
        counts = np.bincount(partial["pair_indices"] * 2 + partial["sides"], minlength=6)
        self.assertLessEqual(counts.max() - counts.min(), 1)
        self.assertEqual(counts.sum(), 607)

    def test_mapping_changes_candidate_inputs_without_mutating_dictionary(self):
        text = torch.eye(4)
        original = text.clone()
        pairs = torch.tensor([[0, 1], [2, 3]])
        mapping = torch.tensor([1, 0, 3, 2])
        torch.testing.assert_close(mapped_candidate_text(text, pairs, mapping), text[torch.tensor([[1, 0], [3, 2]])])
        torch.testing.assert_close(text, original)
        torch.testing.assert_close(pairs, torch.tensor([[0, 1], [2, 3]]))

    def test_interaction_uses_difference_of_effects(self):
        ids = np.array(["a", "b", "a", "b"])
        strata = np.array(["known_known", "known_known", "new_new", "new_new"])
        stats = paired_stratum_interaction(ids, np.ones(4), np.array([1., 1., 3., 3.]),
                                          strata, "new_new", "known_known", replicates=30)
        self.assertEqual(stats["estimate"], 2.)
        np.testing.assert_allclose(stats["ci95"], [2., 2.])


class SemanticIdentityTrainingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.cache = Path(self.directory.name)
        rng = np.random.default_rng(12)
        for split in ("train", "val"):
            folder = self.cache / split
            folder.mkdir()
            ids = np.array([f"{split}{i}" for i in range(4)])
            np.savez(folder / "evidence.npz", image_ids=ids, labels=np.arange(4),
                     text_features=np.eye(4, dtype=np.float32), full_scores=np.eye(4, dtype=np.float32))
            tokens = rng.normal(size=(4, 5, 4)).astype(np.float32)
            tokens[:, 0] = np.eye(4)
            np.save(folder / "prefix_tokens.npy", tokens)
            np.savez(folder / "teacher.npz", image_ids=ids,
                     coefficients=rng.normal(size=(4, 4, 4)).astype(np.float32))
            with (self.cache / f"{split}_tasks.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["task_id", "class_a", "class_b"])
                writer.writeheader()
                writer.writerows([dict(task_id=f"{split}{i}", class_a=a, class_b=b)
                                  for i, (a, b) in enumerate([(0, 1), (0, 2), (2, 3)])])

    def tearDown(self):
        self.directory.cleanup()

    def test_loader_never_uses_heldout_training_rows(self):
        data = load_data(self.cache, "train", torch.device("cpu"), 2, 1., np.array([0, 1]))
        np.testing.assert_array_equal(data["labels"], [0, 1])
        np.testing.assert_array_equal(data["pairs_array"], [[0, 1]])
        self.assertEqual(len(data["tokens"]), 2)
        self.assertEqual(len(data["targets"]), 2)
        self.assertEqual(data["text"].shape, (4, 4))

    def test_short_training_is_paired_and_teacher_targets_stay_fixed(self):
        data = load_data(self.cache, "train", torch.device("cpu"), 2, 1.)
        prior = shared_descriptor(data["text"], data["pairs"])
        config = {"hidden_dim": 8}
        starts, finishes = [], []
        original_targets, original_text = data["targets"].clone(), data["text"].clone()
        for mapping in (torch.arange(4), torch.tensor([2, 3, 0, 1])):
            torch.manual_seed(2041)
            model = make_model(prior, "real_text", config, 4, torch.device("cpu"))
            starts.append(torch.cat([p.detach().flatten().clone() for p in model.parameters()]))
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            for step in range(2):
                optimizer.zero_grad()
                value = losses(model, data, torch.arange(2), mapping, 1.).mean()
                value.backward()
                optimizer.step()
            finishes.append(torch.cat([p.detach().flatten().clone() for p in model.parameters()]))
        torch.testing.assert_close(starts[0], starts[1], rtol=0, atol=0)
        self.assertFalse(torch.equal(finishes[0], finishes[1]))
        torch.testing.assert_close(data["targets"], original_targets, rtol=0, atol=0)
        torch.testing.assert_close(data["text"], original_text, rtol=0, atol=0)

    def test_final_classifier_uses_correct_text_even_with_permuted_selector(self):
        data = load_data(self.cache, "val", torch.device("cpu"), 2, 1.)

        class CaptureSelector(torch.nn.Module):
            conditioned = True
            def __init__(self):
                super().__init__()
                self.candidates = []
            def forward(self, tokens, candidates):
                self.candidates.append(candidates.clone())
                return torch.ones((len(tokens), 4))

        class IdentitySuffix:
            def finish(self, tokens, indices):
                return tokens[:, 0]

        model = CaptureSelector()
        mapping = torch.tensor([2, 3, 0, 1])
        outcome = evaluate_model(model, data, mapping, IdentitySuffix(), 2, 4,
                                 self.cache / "predictions.npz")
        np.testing.assert_array_equal(outcome["correct"], np.ones(len(data["weights"])))
        torch.testing.assert_close(model.candidates[0][0], data["text"][mapping[data["examples"]["class_pairs"][0]]])

    def test_complete_tiny_run_writes_both_regimes_and_all_arms(self):
        """Exercise training, saved checkpoints, suffix scoring and aggregate output."""
        class Backbone(torch.nn.Module):
            visual = None
        class IdentitySuffix:
            def __init__(self, visual, block):
                pass
            def finish(self, tokens, indices):
                return tokens[:, 0]
        fake_clip = types.SimpleNamespace(load=lambda *args, **kwargs: (Backbone(), None))
        config = dict(device="cpu", cache_dir=str(self.cache), batch_size=2, temperature=1.,
                      model="dummy", model_cache="dummy", prune_after=1, training_seeds=[2041],
                      hidden_dim=8, wide_hidden_dim=8, learning_rate=.001, weight_decay=.01,
                      training_steps=2, keep_patch_tokens=2, image_batch_size=4)
        output = self.cache / "run"
        previous_clip = sys.modules.get("clip")
        sys.modules["clip"] = fake_clip
        try:
            with patch("token_compression.pruning.SingleStagePruner", IdentitySuffix), redirect_stdout(io.StringIO()):
                run(config, output, ["known_classes", "class_holdout"], [2097, 2098], 2099, 20)
        finally:
            if previous_clip is None:
                del sys.modules["clip"]
            else:
                sys.modules["clip"] = previous_clip
        import json
        summary = json.loads((output / "summary.json").read_text())
        self.assertFalse(summary["final_test_used"])
        self.assertEqual(set(summary["regimes"]), {"known_classes", "class_holdout"})
        for regime in summary["regimes"]:
            self.assertEqual(len(list((output / regime).glob("seed_*.pt"))), 4)
            self.assertEqual(len(list((output / regime).glob("seed_*.npz"))), 4)
            for group in summary["regimes"][regime]["strata"].values():
                self.assertEqual(group["full_model_accuracy_percent"], 100.)
                self.assertEqual(group["metrics"]["correct"]["contrasts"]["real_minus_permuted_mean"]["estimate"], 0.)


if __name__ == "__main__":
    unittest.main()
