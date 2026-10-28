import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch.nn import functional as F

from token_compression.selector import TokenSelector
from token_compression.selector_objectives import topk_rank_losses, teacher_topk_diagnostics


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_selector_scale.py"
spec = importlib.util.spec_from_file_location("train_selector_objective_test", SCRIPT)
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


class SelectorObjectiveTests(unittest.TestCase):
    def test_unique_boundary_agrees_with_all_cross_set_preferences(self):
        scores = torch.tensor([[.1, .5, -.4, .7]], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[.4, .3, .2, .1]], dtype=torch.float64)
        measured = topk_rank_losses(scores, target, 2)
        expected = torch.stack([F.softplus(scores[0, j] - scores[0, i])
                                for i in [0, 1] for j in [2, 3]]).mean()
        torch.testing.assert_close(measured[0], expected, rtol=0, atol=0)
        grad = torch.autograd.grad(measured.sum(), scores)[0]
        self.assertTrue(torch.all(grad[0, :2] < 0))
        self.assertTrue(torch.all(grad[0, 2:] > 0))
        self.assertAlmostEqual(float(grad.sum()), 0.)
        improved = topk_rank_losses((scores - .5 * grad).detach(), target, 2)
        self.assertLess(float(improved), float(measured.detach()))

    def test_uniform_teacher_has_zero_loss_and_gradient(self):
        scores = torch.tensor([[1., -2., .4, .8]], requires_grad=True)
        target = torch.full_like(scores, .25)
        loss = topk_rank_losses(scores, target, 2)
        self.assertEqual(float(loss.detach()), 0.)
        grad = torch.autograd.grad(loss.sum(), scores)[0]
        torch.testing.assert_close(grad, torch.zeros_like(grad), rtol=0, atol=0)
        diagnostics = teacher_topk_diagnostics(scores, target, 2)
        self.assertEqual(float(diagnostics["topk_tie_aware_overlap"]), 1.)
        self.assertEqual(float(diagnostics["teacher_boundary_tie_fraction"]), 1.)

    def test_boundary_ties_are_permutation_equivariant(self):
        # Either middle token is a valid retained choice. Both must receive the
        # same supervision relationship to strict upper and lower groups.
        scores = torch.tensor([[.1, .5, -.4, .7]], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[.4, .25, .25, .1]], dtype=torch.float64)
        loss = topk_rank_losses(scores, target, 2)
        wanted_pairs = [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)]
        expected = torch.stack([F.softplus(scores[0, j] - scores[0, i])
                                for i, j in wanted_pairs]).mean()
        torch.testing.assert_close(loss[0], expected, rtol=0, atol=0)
        grad = torch.autograd.grad(loss.sum(), scores)[0]
        permutation = torch.tensor([3, 2, 0, 1])
        transformed = scores.detach()[:, permutation].requires_grad_()
        other = topk_rank_losses(transformed, target[:, permutation], 2)
        other_grad = torch.autograd.grad(other.sum(), transformed)[0]
        torch.testing.assert_close(other, loss.detach(), rtol=0, atol=1e-15)
        torch.testing.assert_close(other_grad, grad[:, permutation], rtol=0, atol=1e-15)
        alternate = torch.tensor([[3., 0., 2., -1.]])
        metrics = teacher_topk_diagnostics(alternate, target.float(), 2)
        self.assertEqual(float(metrics["topk_overlap"]), .5)
        self.assertEqual(float(metrics["topk_tie_aware_overlap"]), 1.)

    def test_ranking_ignores_common_logit_shift_but_not_order(self):
        scores = torch.tensor([[.1, .5, -.4, .7]], dtype=torch.float64)
        target = torch.tensor([[.4, .3, .2, .1]], dtype=torch.float64)
        torch.testing.assert_close(topk_rank_losses(scores, target, 2),
                                   topk_rank_losses(scores + 12, target, 2), rtol=0, atol=1e-14)
        self.assertNotEqual(float(topk_rank_losses(scores, target, 2)),
                            float(topk_rank_losses(-scores, target, 2)))

    def test_default_kl_matches_original_loss_and_gradients_exactly(self):
        torch.manual_seed(9)
        model = TokenSelector(torch.randn(4), True, hidden_dim=4, token_dim=4)
        data = self.small_data()
        indices = torch.arange(4)
        current = training.example_losses(model, data, indices, 1.)
        logits = model(data["tokens"][data["examples"]["image_indices"]],
                       data["text"][data["examples"]["class_pairs"]])
        original = F.kl_div(F.log_softmax(logits, dim=1), data["targets"], reduction="none").sum(dim=1)
        torch.testing.assert_close(current, original, rtol=0, atol=0)
        new_grad = torch.autograd.grad(current.mean(), tuple(model.parameters()))
        old_grad = torch.autograd.grad(original.mean(), tuple(model.parameters()))
        for a, b in zip(new_grad, old_grad):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    @staticmethod
    def small_data():
        return {"tokens": torch.randn(3, 5, 4), "text": torch.randn(3, 2),
                "examples": {"image_indices": torch.tensor([0, 1, 1, 2]),
                             "class_pairs": torch.tensor([[0, 1], [0, 1], [1, 2], [1, 2]])},
                "targets": torch.softmax(torch.randn(4, 4), dim=1)}

    def test_initial_gradient_measurement_does_not_train_or_consume_rng(self):
        torch.manual_seed(12)
        model = TokenSelector(torch.randn(4), True, hidden_dim=4, token_dim=4)
        model.train()
        data = self.small_data()
        state = {name: value.clone() for name, value in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        diagnostics = training.initial_gradient_diagnostics(model, data, torch.arange(4), 2, 1., 2)
        self.assertEqual(diagnostics["batch_count"], 2)
        self.assertGreater(diagnostics["mean_kl_gradient_l2"], 0)
        self.assertGreater(diagnostics["mean_topk_rank_gradient_l2"], 0)
        self.assertTrue(model.training)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_rank_training_records_correct_loss_names_and_common_endpoints(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        rng = np.random.default_rng(8)
        try:
            with tempfile.TemporaryDirectory() as folder:
                folder = Path(folder)
                cache, old = folder / "cache", folder / "old"
                text = rng.normal(size=(3, 2)).astype(np.float32)
                text /= np.linalg.norm(text, axis=1, keepdims=True)
                for split, labels in (("train", [0, 0, 1, 1, 2, 2]), ("val", [0, 2])):
                    target = cache / split
                    target.mkdir(parents=True)
                    ids = np.array([f"{split}_{i}" for i in range(len(labels))])
                    np.savez(target / "evidence.npz", image_ids=ids, labels=np.array(labels), text_features=text)
                    np.save(target / "prefix_tokens.npy", rng.normal(size=(len(labels), 5, 4)).astype(np.float32))
                    np.savez(target / "teacher.npz", image_ids=ids,
                             coefficients=rng.normal(size=(len(labels), 4, 3)).astype(np.float32))
                (old / "train").mkdir(parents=True)
                np.savez(old / "train/evidence.npz", image_ids=np.array(["train_0", "train_2", "train_4"]))
                for split, pairs in (("train", [(0, 1), (1, 2)]), ("val", [(0, 2)])):
                    with (cache / f"{split}_tasks.csv").open("w") as handle:
                        writer = csv.DictWriter(handle, fieldnames=["task_id", "class_a", "class_b"])
                        writer.writeheader()
                        writer.writerows({"task_id": f"{a}_{b}", "class_a": a, "class_b": b} for a, b in pairs)
                (old / "train_tasks.csv").write_text((cache / "train_tasks.csv").read_text())
                config = {"cache_dir": str(cache), "source_cache_dir": str(old), "device": "cpu",
                          "batch_size": 2, "temperature": 1., "training_steps": 4, "checkpoint_interval": 2,
                          "hidden_dim": 4, "training_seeds": [11], "learning_rate": .0003,
                          "weight_decay": .01, "coverages": ["full"], "keep_patch_tokens": 2,
                          "objective": "topk_rank"}
                original_step = torch.optim.AdamW.step
                with mock.patch.object(torch.optim.AdamW, "step", autospec=True, side_effect=original_step) as steps:
                    with contextlib.redirect_stdout(io.StringIO()):
                        summary = training.train(config, folder / "output")
                    self.assertEqual(steps.call_count, 8)
                self.assertEqual(summary["objective"], "topk_rank")
                self.assertEqual(len(summary["runs"]), 2)
                for row in summary["runs"]:
                    self.assertEqual(row["initial_gradient_diagnostics"]["batch_count"], 4)
                    self.assertIn("final_train_objective_loss", row)
                    self.assertAlmostEqual(row["final_val_kl"], row["final_teacher_diagnostics"]["val"]["kl"])
                    checkpoint = torch.load(folder / "output" / row["checkpoint"], weights_only=True)
                    self.assertEqual(checkpoint["objective"], "topk_rank")
                    self.assertEqual(checkpoint["step"], 4)
                with (folder / "output/training_history.csv").open() as handle:
                    history = list(csv.DictReader(handle))
                self.assertTrue(all("train_objective_loss" in row for row in history))
                self.assertTrue(all(row["train_kl"] == "" for row in history if row["step"] == "2"))
                self.assertTrue(all(row["train_kl"] != "" for row in history if row["step"] in ("0", "4")))
        finally:
            torch.set_num_threads(previous_threads)

    def test_rank_configs_preserve_matched_training_recipe(self):
        root = Path(__file__).resolve().parents[1]
        for dataset in ("pets", "flowers"):
            for k in (49, 98):
                stem = f"{dataset}_selector_budget_{k}" if (dataset, k) != ("pets", 49) else "pets_selector_scale"
                old = json.loads((root / "configs" / f"{stem}.json").read_text())
                new = json.loads((root / "configs" / f"{dataset}_selector_rank_{k}.json").read_text())
                self.assertEqual(new["objective"], "topk_rank")
                self.assertEqual(new["coverages"], ["full"])
                for key in set(old) - {"experiment_name", "coverages"}:
                    self.assertEqual(old[key], new[key], key)


if __name__ == "__main__":
    unittest.main()
