import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from token_compression.selector import ImageOnlyWideSelector, pair_descriptor
from token_compression.shared_residual import ResidualBranch, SharedResidualSelector, shared_features

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_shared_residual as runner
from evaluate_selectors import write_rows


class TinyPruner:
    def __init__(self, visual, prune_after):
        pass

    def finish(self, tokens, selected):
        return tokens[torch.arange(len(tokens))[:, None], selected + 1, :4].sum(1) + tokens[:, 0, :4]


class SharedResidualTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_exact_zero_start_and_actual_parameter_budgets(self):
        torch.manual_seed(19)
        shared = ImageOnlyWideSelector()
        tokens, candidates = torch.randn(2, 6, 768), torch.randn(2, 2, 512)
        baseline = shared(tokens, candidates)
        for conditioned, expected in ((True, (9801, 357200)), (False, (9850, 357249))):
            model = SharedResidualSelector(shared, conditioned)
            counts = model.parameter_counts()
            self.assertEqual(counts["trainable_correction"], expected[0])
            self.assertEqual(counts["total"], expected[1])
            torch.testing.assert_close(model(tokens, candidates), baseline, rtol=0, atol=0)
            self.assertTrue(all(not p.requires_grad for p in model.shared.parameters()))

    def test_split_projection_matches_literal_concat_and_candidate_order_is_symmetric(self):
        torch.manual_seed(20)
        branch = ResidualBranch(shared_dim=11, text_dim=4, conditioned=True)
        with torch.no_grad():
            branch.output_projection.weight.normal_()
        hidden, candidate = torch.randn(3, 5, 11), torch.randn(3, 2, 4)
        context = branch.context_norm(pair_descriptor(candidate))[:, None].expand(-1, 5, -1)
        direct = branch.output_projection(F.gelu(branch.input_projection(
            torch.cat((branch.hidden_norm(hidden), context), dim=-1)))).squeeze(-1)
        torch.testing.assert_close(branch(hidden, candidate), direct, atol=2e-7, rtol=1e-6)
        torch.testing.assert_close(branch(hidden, candidate.flip(1)), branch(hidden, candidate), rtol=0, atol=0)

    def test_cached_training_updates_branch_but_not_shared_parameters(self):
        torch.manual_seed(21)
        tokens, candidate = torch.randn(3, 6, 6), torch.randn(3, 2, 4)
        shared = ImageOnlyWideSelector(hidden_dim=11, token_dim=6)
        model = SharedResidualSelector(shared, True, text_dim=4)
        original = {key: value.detach().clone() for key, value in shared.state_dict().items()}
        cached = runner.cache_shared(shared, tokens, 2)
        torch.testing.assert_close(model.score_cached(*cached, candidate), model(tokens, candidate), rtol=0, atol=0)
        targets = F.softmax(torch.randn(3, 5), dim=-1)
        optimizer = torch.optim.AdamW(model.branch.parameters(), lr=.01)
        for _ in range(3):
            optimizer.zero_grad()
            loss = F.kl_div(F.log_softmax(model.score_cached(*cached, candidate), dim=-1), targets, reduction="batchmean")
            loss.backward()
            optimizer.step()
        self.assertGreater(model.branch.output_projection.weight.abs().sum().item(), 0)
        self.assertFalse(torch.equal(model(tokens, candidate), shared(tokens, candidate)))
        for key, value in shared.state_dict().items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in shared.parameters()))
        torch.testing.assert_close(model.score_cached(*cached, candidate), model(tokens, candidate), rtol=1e-6, atol=1e-7)

    def test_task_swap_changes_only_conditioned_branch(self):
        torch.manual_seed(22)
        tokens = torch.randn(1, 6, 6).expand(2, -1, -1).clone()
        candidate = torch.randn(2, 2, 4)
        for conditioned in (True, False):
            model = SharedResidualSelector(ImageOnlyWideSelector(11, 6), conditioned, text_dim=4)
            with torch.no_grad():
                model.branch.output_projection.weight.normal_()
            scores = model(tokens, candidate)
            swapped = model(tokens, candidate.flip(0))
            if conditioned:
                self.assertFalse(torch.equal(scores, swapped))
            else:
                torch.testing.assert_close(scores, swapped, rtol=0, atol=0)

    def test_tiny_two_seed_pipeline_saves_paired_models_and_online_predictions(self):
        rng = np.random.default_rng(23)
        labels = np.array([0, 1, 2, 2])
        text = rng.normal(size=(3, 4)).astype(np.float32)
        text /= np.linalg.norm(text, axis=1, keepdims=True)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache, shared_dir, output = root / "cache", root / "shared", root / "output"
            for split in ("train", "val"):
                directory = cache / split
                directory.mkdir(parents=True)
                ids = np.asarray([f"{split}_{i}" for i in range(4)])
                tokens = rng.normal(size=(4, 6, 6)).astype(np.float32)
                features = tokens[:, 0, :4].copy()
                features /= np.linalg.norm(features, axis=1, keepdims=True)
                np.save(directory / "prefix_tokens.npy", tokens)
                np.savez(directory / "evidence.npz", image_ids=ids, labels=labels,
                         text_features=text, full_scores=features @ text.T)
                np.savez(directory / "teacher.npz", image_ids=ids,
                         coefficients=rng.normal(size=(4, 5, 3)).astype(np.float32))
                pairs = [(0, 1)] if split == "train" else [(0, 2), (1, 2)]
                write_rows(cache / f"{split}_tasks.csv", [{"task_id": f"{a}_{b}", "class_a": a, "class_b": b} for a, b in pairs])
            for seed in (101, 102):
                folder = shared_dir / "full" / f"seed_{seed}"
                folder.mkdir(parents=True)
                torch.manual_seed(seed)
                shared = ImageOnlyWideSelector(hidden_dim=11, token_dim=6)
                torch.save({"state_dict": shared.state_dict(), "hidden_dim": 11,
                            "selector_variant": "image_only_wide"}, folder / "image_only_wide.pt")
            config = {"device": "cpu", "cache_dir": str(cache), "batch_size": 2, "temperature": 1.,
                      "model": "tiny", "model_cache": str(root), "prune_after": 1, "training_seeds": [101, 102],
                      "learning_rate": .01, "weight_decay": .01, "training_steps": 4,
                      "checkpoint_interval": 2, "keep_patch_tokens": 2, "image_batch_size": 2, "bootstrap_replicates": 40}
            backbone = torch.nn.Module()
            backbone.visual = types.SimpleNamespace()
            fake_clip = types.SimpleNamespace(load=lambda *a, **k: (backbone, None))
            # Resolve pruning before taking the sys.modules snapshot for isolation.
            with patch("token_compression.pruning.SingleStagePruner", TinyPruner), patch.dict(sys.modules, {"clip": fake_clip}):
                result = runner.run(config, shared_dir, output, "fixture")
            self.assertEqual(result["primary_family_size"], 3)
            self.assertEqual(result["classification_image_task_count"], 6)
            self.assertEqual(len(result["training_records"]), 4)
            for seed in (101, 102):
                records = [r for r in result["training_records"] if r["seed"] == seed]
                self.assertEqual(records[0]["initial_val_teacher_kl"], records[1]["initial_val_teacher_kl"])
                raw = np.load(output / f"seed_{seed}.npz")
                self.assertEqual(raw["changed_tokens_vs_shared"][0].sum(), 0)
                self.assertEqual(raw["correction_patch_std"][0].sum(), 0)
                self.assertTrue(np.all(np.diff(raw["patch_indices"], axis=-1) > 0))
            for method in ("shared", "image_residual"):
                value = next(r for r in result["exchange"]["table"] if r["method"] == method)
                self.assertEqual(value["margin"]["mean"], 0)
            audit = json.loads((output / "sampling_audit.json").read_text())
            self.assertTrue(all(r["task_side_min"] == r["task_side_max"] for r in audit))


if __name__ == "__main__":
    unittest.main()
