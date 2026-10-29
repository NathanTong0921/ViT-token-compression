import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from token_compression.selector import TokenSelector, make_selector, pair_descriptor


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_selector_ablations_test", ROOT / "scripts/train_selector_scale.py")
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


class SelectorAblationTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(43)
        self.prior = torch.randn(8)
        self.tokens = torch.randn(3, 5, 6)
        self.text = torch.randn(3, 2, 4)

    def tearDown(self):
        torch.set_num_threads(self.old_threads)

    def test_legacy_factory_preserves_state_outputs_and_gradients(self):
        for arm, conditioned in (("conditioned", True), ("image_only", False)):
            torch.manual_seed(17)
            legacy = TokenSelector(self.prior, conditioned, hidden_dim=7, token_dim=6)
            torch.manual_seed(17)
            current = make_selector(self.prior, arm, hidden_dim=7, token_dim=6)
            self.assertEqual(set(legacy.state_dict()), set(current.state_dict()))
            for key, value in legacy.state_dict().items():
                torch.testing.assert_close(value, current.state_dict()[key], rtol=0, atol=0)
            # The pre-ablation forward formula, including the old fixed-prior path.
            normalized = legacy.image_norm(self.tokens.float())
            patches, cls = normalized[:, 1:], normalized[:, :1].expand(-1, 4, -1)
            descriptor = pair_descriptor(self.text) if conditioned else self.prior.expand(3, -1)
            context = legacy.context_norm(descriptor).unsqueeze(1).expand(-1, 4, -1)
            expected = legacy.scorer(torch.cat((patches, cls, context), dim=-1)).squeeze(-1)
            observed = current(self.tokens, self.text)
            torch.testing.assert_close(expected, observed, rtol=0, atol=0)
            old_grad = torch.autograd.grad(expected.square().sum(), tuple(legacy.parameters()))
            new_grad = torch.autograd.grad(observed.square().sum(), tuple(current.parameters()))
            for a, b in zip(old_grad, new_grad):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_missing_half_uses_training_prior_before_joint_normalization(self):
        descriptor = pair_descriptor(self.text)
        for arm, missing in (("mean_only", slice(4, 8)), ("difference_only", slice(0, 4))):
            model = make_selector(self.prior, arm, hidden_dim=7, token_dim=6)
            seen = []
            hook = model.context_norm.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
            output = model(self.tokens, self.text)
            hook.remove()
            expected = descriptor.clone()
            expected[:, missing] = self.prior[missing]
            torch.testing.assert_close(seen[0], expected, rtol=0, atol=0)
            torch.testing.assert_close(output, model(self.tokens, self.text.flip(1)), rtol=0, atol=0)

    def test_component_variants_ignore_the_removed_descriptor(self):
        # Exactly represented values isolate each part independently of rounding.
        mean = torch.tensor([[1., 2., 4., 8.]]).expand(3, -1)
        delta = torch.tensor([[.25, .5, 1., 2.]]).expand(3, -1)
        pair = torch.stack((mean + delta, mean - delta), dim=1)
        changed_difference = torch.stack((mean + 2 * delta, mean - 2 * delta), dim=1)
        changed_mean = pair + 1.
        model_mean = make_selector(self.prior, "mean_only", hidden_dim=7, token_dim=6)
        model_diff = make_selector(self.prior, "difference_only", hidden_dim=7, token_dim=6)
        torch.testing.assert_close(model_mean(self.tokens, pair), model_mean(self.tokens, changed_difference), rtol=0, atol=0)
        torch.testing.assert_close(model_diff(self.tokens, pair), model_diff(self.tokens, changed_mean), rtol=0, atol=0)

    def test_wide_image_baseline_spends_parameters_on_image_processing(self):
        wide = make_selector(torch.zeros(1024), "image_only_wide", hidden_dim=199, token_dim=768)
        full = make_selector(torch.zeros(1024), "conditioned", hidden_dim=128, token_dim=768)
        self.assertEqual(sum(p.numel() for p in wide.parameters()), 347399)
        self.assertEqual(sum(p.numel() for p in full.parameters()), 348033)
        self.assertNotIn("shared_descriptor", wide.state_dict())
        self.assertFalse(any(key.startswith("context_norm") for key in wide.state_dict()))
        tiny = make_selector(self.prior, "image_only_wide", hidden_dim=9, token_dim=6)
        a = tiny(self.tokens, self.text)
        b = tiny(self.tokens, torch.randn_like(self.text))
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in tiny.parameters()))

    def test_ablation_configs_reuse_full_coverage_kl_recipe(self):
        for dataset in ("pets", "flowers"):
            for budget in (49, 98):
                old_stem = "pets_selector_scale" if (dataset, budget) == ("pets", 49) else f"{dataset}_selector_budget_{budget}"
                old = json.loads((ROOT / "configs" / f"{old_stem}.json").read_text())
                new = json.loads((ROOT / "configs" / f"{dataset}_selector_ablations_{budget}.json").read_text())
                for key in set(old) - {"coverages", "random_seeds", "experiment_name"}:
                    self.assertEqual(new[key], old[key], key)
                self.assertEqual(new["objective"], "kl")
                self.assertEqual(new["selector_arms"], ["mean_only", "difference_only", "image_only_wide"])
                self.assertEqual(new["coverages"], ["full"])
                self.assertEqual(new["wide_hidden_dim"], 199)
                self.assertFalse(new["include_references"])
                self.assertEqual(new["random_seeds"], [])

    @staticmethod
    def make_cache(folder):
        rng = np.random.default_rng(8)
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
        return {"cache_dir": str(cache), "source_cache_dir": str(old), "device": "cpu",
                "batch_size": 2, "temperature": 1., "training_steps": 4, "checkpoint_interval": 2,
                "hidden_dim": 4, "training_seeds": [11], "learning_rate": .0003,
                "weight_decay": .01, "coverages": ["full"], "keep_patch_tokens": 2}

    def test_training_uses_all_variants_and_roundtrips_actual_architectures(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            config = self.make_cache(folder)
            config.update(selector_arms=["mean_only", "difference_only", "image_only_wide"], wide_hidden_dim=5)
            with contextlib.redirect_stdout(io.StringIO()):
                summary = training.train(config, folder / "output")
            self.assertEqual(len(summary["runs"]), 3)
            self.assertEqual(summary["selector_arms"], config["selector_arms"])
            self.assertEqual(summary["run_type"], "fixed_update_selector_component_ablations")
            for row in summary["runs"]:
                self.assertEqual(row["training_steps"], 4)
                self.assertEqual(row["training_draws"], 8)
                self.assertTrue(row["image_draws_reused_between_arms"])
                self.assertEqual(row["initial_state_exactly_matches_shared_seed_state"], row["arm"] != "image_only_wide")
                self.assertTrue(row["initial_state_exactly_matches_own_architecture_seed_state"])
                checkpoint = torch.load(folder / "output" / row["checkpoint"], weights_only=True)
                self.assertEqual(checkpoint["selector_variant"], row["arm"])
                self.assertEqual(checkpoint["hidden_dim"], 5 if row["arm"] == "image_only_wide" else 4)
                prior = torch.zeros(4)
                model = make_selector(prior, row["arm"], checkpoint["hidden_dim"], checkpoint["token_dim"])
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                output = model(torch.zeros(2, 5, 4), torch.zeros(2, 2, 2))
                self.assertEqual(output.shape, (2, 4))
                self.assertTrue(torch.isfinite(output).all())
                self.assertEqual(sum(p.numel() for p in model.parameters()), row["parameter_count"])


if __name__ == "__main__":
    unittest.main()
