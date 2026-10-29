import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from token_compression.selector import TokenSelector, make_selector, shared_descriptor


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("evaluate_selector_scale", SCRIPTS / "evaluate_selector_scale.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class TinyPruner:
    """Small deterministic suffix used to check saved inference routing."""

    num_patches = 5

    def __init__(self, visual, prune_after):
        pass

    def finish(self, tokens, selected):
        rows = torch.arange(len(tokens))[:, None]
        return tokens[rows, selected + 1, :4].sum(dim=1) + tokens[:, 0, :4]


class SelectorAblationEvaluationTests(unittest.TestCase):
    def test_job_groups_preserve_legacy_and_run_only_requested_new_models(self):
        config = {"training_seeds": [1, 2], "coverages": ["full"]}
        groups, random_seeds = evaluation.evaluation_groups(config)
        self.assertEqual(groups, [
            ("references", [(-1, "full_model"), (-1, "attention"), (-1, "label_informed_teacher")]),
            ("full", [(1, "conditioned"), (1, "image_only"), (2, "conditioned"), (2, "image_only")]),
        ])
        self.assertEqual(random_seeds, [])
        config.update(selector_arms=["mean_only", "difference_only", "image_only_wide"],
                      include_references=False, random_seeds=[])
        groups, _ = evaluation.evaluation_groups(config)
        self.assertEqual(groups, [("full", [(seed, method) for seed in [1, 2]
                                             for method in config["selector_arms"]])])
        groups, seeds = evaluation.evaluation_groups({"training_seeds": [1], "coverages": ["full"]}, True)
        self.assertEqual(groups, [("random", [(seed, "random") for seed in [2055, 2056, 2057]])])
        self.assertEqual(seeds, [2055, 2056, 2057])

    def test_original_checkpoint_forward_is_unchanged(self):
        torch.manual_seed(18)
        prior, tokens, text = torch.randn(8), torch.randn(3, 6, 6), torch.randn(3, 2, 4)
        for method in ("conditioned", "image_only"):
            original = TokenSelector(prior, method == "conditioned", hidden_dim=7, token_dim=6)
            checkpoint = {"conditioned": method == "conditioned", "hidden_dim": 7,
                          "state_dict": original.state_dict()}
            loaded = evaluation.selector_from_checkpoint(checkpoint, prior, method, 6, "cpu")
            torch.testing.assert_close(loaded(tokens, text), original(tokens, text), rtol=0, atol=0)

    def test_variant_checkpoint_dispatch_uses_recorded_model_and_rejects_wrong_arm(self):
        torch.manual_seed(19)
        prior, tokens, text = torch.randn(8), torch.randn(3, 6, 6), torch.randn(3, 2, 4)
        for method, width in (("mean_only", 7), ("difference_only", 7), ("image_only_wide", 11)):
            model = make_selector(prior, method, hidden_dim=width, token_dim=6)
            checkpoint = {"selector_variant": method, "hidden_dim": width, "state_dict": model.state_dict()}
            loaded = evaluation.selector_from_checkpoint(checkpoint, prior, method, 6, "cpu")
            torch.testing.assert_close(loaded(tokens, text), model(tokens, text), rtol=0, atol=0)
            self.assertFalse(loaded.training)
            self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
            with self.assertRaisesRegex(ValueError, "does not match"):
                evaluation.selector_from_checkpoint(checkpoint, prior, "conditioned", 6, "cpu")

    def test_saved_predictions_match_direct_per_task_scoring_and_wide_image_reuse(self):
        rng = np.random.default_rng(29)
        ids, labels = np.array(["a", "b", "c"]), np.arange(3)
        text = rng.normal(size=(3, 4)).astype(np.float32)
        text /= np.linalg.norm(text, axis=1, keepdims=True)
        prefix = rng.normal(size=(3, 6, 6)).astype(np.float32)
        full_features = prefix[:, 0, :4].copy()
        full_features /= np.linalg.norm(full_features, axis=1, keepdims=True)
        seen = [{"task_id": "0_1", "class_a": 0, "class_b": 1}]
        unseen = [{"task_id": "0_2", "class_a": 0, "class_b": 2},
                  {"task_id": "1_2", "class_a": 1, "class_b": 2}]
        methods = ["mean_only", "difference_only", "image_only_wide"]
        prior = shared_descriptor(torch.tensor(text), torch.tensor([[0, 1]]))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache, models_dir, output = root / "cache", root / "models", root / "evaluation"
            (cache / "val").mkdir(parents=True)
            np.save(cache / "val" / "prefix_tokens.npy", prefix)
            np.savez(cache / "val" / "evidence.npz", image_ids=ids, labels=labels,
                     text_features=text, full_features=full_features, full_scores=full_features @ text.T,
                     attention_scores=rng.normal(size=(3, 5)))
            np.savez(cache / "val" / "teacher.npz", image_ids=ids,
                     coefficients=rng.normal(size=(3, 5, 3)))
            for split, tasks in (("train", seen), ("val", unseen)):
                evaluation.write_rows(cache / f"{split}_tasks.csv", tasks)
            checkpoint_folder = models_dir / "full" / "seed_101"
            checkpoint_folder.mkdir(parents=True)
            models = {}
            for method in methods:
                torch.manual_seed(101)
                width = 11 if method == "image_only_wide" else 7
                models[method] = make_selector(prior, method, hidden_dim=width, token_dim=6)
                torch.save({"selector_variant": method, "hidden_dim": width,
                            "state_dict": models[method].state_dict()}, checkpoint_folder / f"{method}.pt")
            config = {"cache_dir": str(cache), "source_cache_dir": str(cache),
                      "model": "tiny", "model_cache": str(root), "device": "cpu", "prune_after": 1,
                      "image_batch_size": 2, "keep_patch_tokens": 2, "training_seeds": [101],
                      "coverages": ["full"], "selector_arms": methods, "include_references": False,
                      "random_seeds": [], "identity_atol": 1e-6}
            backbone = torch.nn.Module()
            backbone.visual = types.SimpleNamespace()
            fake_clip = types.SimpleNamespace(load=lambda *args, **kwargs: (backbone, None))
            with patch.dict(sys.modules, {"clip": fake_clip}), patch(
                    "token_compression.pruning.SingleStagePruner", TinyPruner):
                summary = evaluation.run(config, models_dir, output)
            self.assertFalse((output / "references").exists())
            self.assertFalse((output / "random").exists())
            self.assertEqual(summary["selector_arms"], methods)
            self.assertFalse(summary["include_references"])
            with (output / "full" / "predictions.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            with np.load(output / "full" / "predictions.npz") as saved:
                values = {key: saved[key] for key in saved.files}
            self.assertEqual(len(rows), 18)
            np.testing.assert_array_equal(values["row_id"], np.arange(18))
            for i, row in enumerate(rows):
                self.assertEqual(row["method"], values["method"][i])
                self.assertEqual(row["image_id"], values["image_ids"][i])
            examples = evaluation.evaluation_examples(ids, labels, seen, unseen)
            image_ix = np.asarray([row["image_index"] for row in examples])
            pairs = np.asarray([[row["class_a"], row["class_b"]] for row in examples])
            for method in methods:
                selected_rows = values["method"] == method
                with torch.inference_mode():
                    logits = models[method](torch.tensor(prefix[image_ix]), torch.tensor(text[pairs])).numpy()
                    positions = np.sort(np.argsort(-logits, axis=1, kind="stable")[:, :2], axis=1)
                    features = TinyPruner(None, None).finish(torch.tensor(prefix[image_ix]),
                                                            torch.tensor(positions)).numpy()
                features /= np.linalg.norm(features, axis=1, keepdims=True)
                np.testing.assert_allclose(values["selection_logits"][selected_rows], logits, atol=1e-6)
                np.testing.assert_array_equal(values["patch_indices"][selected_rows, :2], positions)
                np.testing.assert_allclose(values["image_features"][selected_rows], features, atol=1e-6)
                np.testing.assert_allclose(values["cosine_scores"][selected_rows], features @ text.T, atol=1e-6)
                margin, correct = evaluation.binary_outcomes(features @ text.T, labels[image_ix], pairs, text)
                np.testing.assert_allclose(values["margin"][selected_rows], margin, atol=1e-6)
                np.testing.assert_array_equal(values["correct"][selected_rows], correct)
            costs = {cost["method"]: cost for cost in summary["cost"]["methods"]}
            self.assertEqual(costs["image_only_wide"]["logical_suffix_examples"], 3)
            self.assertEqual(costs["image_only_wide"]["physical_suffix_examples"], 4)
            self.assertTrue(costs["image_only_wide"]["unique_image_reuse"])
            for method in ("mean_only", "difference_only"):
                self.assertEqual(costs[method]["logical_suffix_examples"], 6)
                self.assertEqual(costs[method]["physical_suffix_examples"], 6)
                self.assertFalse(costs[method]["unique_image_reuse"])
            wide = values["method"] == "image_only_wide"
            for image_id in ids:
                image_rows = np.flatnonzero(wide & (values["image_ids"] == image_id))
                for key in ("cosine_scores", "image_features", "patch_indices", "selection_logits"):
                    np.testing.assert_array_equal(values[key][image_rows[0]], values[key][image_rows[1]])
            self.assertEqual(json.loads((output / "eval_summary.json").read_text())["examples_per_method"], 6)


if __name__ == "__main__":
    unittest.main()
