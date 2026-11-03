import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from token_compression.selector import make_selector, shared_descriptor


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_core_protection as evaluation
import analyze_core_protection as analysis


class TinyPruner:
    num_patches = 5

    def __init__(self, visual, prune_after):
        pass

    def finish(self, tokens, selected):
        rows = torch.arange(len(tokens))[:, None]
        return tokens[rows, selected + 1, :4].sum(dim=1) + tokens[:, 0, :4]


class CoreProtectionTests(unittest.TestCase):
    def test_protected_core_changes_competing_ranks_and_endpoints_are_exact(self):
        conditioned = np.array([8., 7., 6., 5., 4., 3.])
        np.testing.assert_array_equal(evaluation.protected_selection(conditioned, [4, 5], 4), [0, 1, 4, 5])
        np.testing.assert_array_equal(evaluation.protected_selection(conditioned, [], 4), [0, 1, 2, 3])
        np.testing.assert_array_equal(evaluation.protected_selection(conditioned, [5, 4, 3, 2], 4), [2, 3, 4, 5])
        np.testing.assert_array_equal(evaluation.protected_selection(np.ones(6), [5], 4), [0, 1, 2, 5])

    def test_random_core_is_task_independent_nested_and_mask_reuse_is_image_specific(self):
        c = np.array([[9, 8, 7, 6, 5, 4], [1, 2, 3, 4, 5, 6], [9, 8, 7, 6, 5, 4]], float)
        u = np.array([[6, 5, 4, 3, 2, 1], [6, 5, 4, 3, 2, 1]], float)
        ix = np.array([0, 0, 1])
        variants, masks = evaluation.build_masks(c, u, ix, np.array(["a", "b"]), 4, "fixture")
        np.testing.assert_array_equal(masks[variants.index("wide"), 0], masks[variants.index("wide"), 1])
        order = evaluation.random_core_order("a", "fixture", 2211, 4)
        for percentage in (50, 75):
            protected = order[:4 * percentage // 100]
            for row in (0, 1):
                self.assertTrue(set(protected) <= set(masks[variants.index(f"random_{percentage}_2211"), row]))
        unique, inverse = evaluation.unique_image_masks(ix, masks)
        wide = variants.index("wide")
        self.assertEqual(inverse[wide, 0], inverse[wide, 1])
        self.assertNotEqual(inverse[wide, 0], inverse[wide, 2])
        np.testing.assert_array_equal(unique[inverse][..., 1:], masks)

    def test_exchange_evaluates_other_selection_on_current_task(self):
        text = np.eye(3)
        # Same image (class 0), tasks 0-v-1 and 0-v-2. Each feature solves its
        # own task but fails the other, despite identical ground-truth labels.
        scores = np.array([[[[.6, .5, .7], [.6, .7, .5]]]])
        own, swapped = analysis.exchange_outcomes(scores, np.array([0, 0]),
                                                  np.array([1, 2]), text, np.array([0]), np.array([1]))
        self.assertEqual(own[0, 0, 0, 0], 1.)
        self.assertEqual(swapped[0, 0, 0, 0], 0.)
        self.assertAlmostEqual((own - swapped)[0, 0, 0, 1], .2 / np.sqrt(2))

    def test_random_seeds_are_outcome_average_not_prediction_ensemble(self):
        variants = ["conditioned", "wide", "top_50", "top_75", "random_50_1", "random_50_2",
                    "random_75_1", "random_75_2"]
        outcomes = np.zeros((1, 8, 1, 2))
        outcomes[0, 4, 0] = [1, .1]
        outcomes[0, 5, 0] = [0, -.8]
        value = analysis.grouped_variants(outcomes, variants)["random_50"]
        np.testing.assert_allclose(value[0, 0], [.5, -.35])

    def test_mocked_suffix_replay_and_analysis_match_direct_fixed_mask_inference(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(29)
        ids, labels = np.array(["a", "b", "c", "d"]), np.array([0, 1, 2, 2])
        text = rng.normal(size=(3, 4)).astype(np.float32)
        text /= np.linalg.norm(text, axis=1, keepdims=True)
        prefix = rng.normal(size=(4, 6, 6)).astype(np.float32)
        full_features = prefix[:, 0, :4].copy()
        full_features /= np.linalg.norm(full_features, axis=1, keepdims=True)
        seen = [{"task_id": "0_1", "class_a": 0, "class_b": 1}]
        unseen = [{"task_id": "0_2", "class_a": 0, "class_b": 2},
                  {"task_id": "1_2", "class_a": 1, "class_b": 2}]
        prior = shared_descriptor(torch.tensor(text), torch.tensor([[0, 1]]))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache, models, output = root / "cache", root / "models", root / "evaluation"
            (cache / "val").mkdir(parents=True)
            np.save(cache / "val/prefix_tokens.npy", prefix)
            np.savez(cache / "val/evidence.npz", image_ids=ids, labels=labels, text_features=text,
                     full_scores=full_features @ text.T)
            for split, tasks in (("train", seen), ("val", unseen)):
                evaluation.write_rows(cache / f"{split}_tasks.csv", tasks)
            for seed in (101, 102):
                folder = models / "full" / f"seed_{seed}"
                folder.mkdir(parents=True)
                for variant, width in (("conditioned", 7), ("image_only_wide", 11)):
                    torch.manual_seed(seed)
                    model = make_selector(prior, variant, hidden_dim=width, token_dim=6)
                    torch.save({"selector_variant": variant, "hidden_dim": width,
                                "state_dict": model.state_dict()}, folder / f"{variant}.pt")
            config = {"cache_dir": str(cache), "model": "tiny", "model_cache": str(root), "device": "cpu",
                      "prune_after": 1, "image_batch_size": 2, "keep_patch_tokens": 4,
                      "training_seeds": [101, 102]}
            backbone = torch.nn.Module()
            backbone.visual = types.SimpleNamespace()
            fake_clip = types.SimpleNamespace(load=lambda *args, **kwargs: (backbone, None))
            with patch.dict(sys.modules, {"clip": fake_clip}), patch("token_compression.pruning.SingleStagePruner", TinyPruner):
                summary = evaluation.run(config, models, models, output, "fixture")
            self.assertEqual(summary["image_task_count"], 6)
            base = analysis.read_rows(output / "examples.csv")
            indices = np.array([int(r["image_index"]) for r in base])
            for seed in (101, 102):
                with np.load(output / f"seed_{seed}.npz") as saved:
                    for v in range(len(saved["variants"])):
                        selected = torch.tensor(saved["patch_indices"][v].astype(int))
                        direct = TinyPruner(None, 1).finish(torch.tensor(prefix[indices]), selected)
                        direct = direct / direct.norm(dim=-1, keepdim=True)
                        np.testing.assert_allclose(saved["cosine_scores"][v], direct.numpy() @ text.T,
                                                   rtol=1e-6, atol=1e-7)
                    np.testing.assert_array_equal(saved["patch_indices"], saved["unique_keys"][saved["feature_index"]][..., 1:])
            results = analysis.analyze(output, root / "analysis", replicates=40)
            self.assertEqual(results["classification_image_task_count"], 6)
            self.assertEqual(results["exchange"]["image_count"], 2)
            self.assertEqual(results["exchange"]["task_pair_count"], 2)
            wide = next(r for r in results["exchange"]["table"] if r["method"] == "wide")
            self.assertEqual(wide["accuracy"]["mean"], 0)
            self.assertEqual(wide["margin"]["mean"], 0)
            c_tokens = next(r for r in results["token_diagnostics"]["classification_replacements"]
                            if r["method"] == "conditioned")
            self.assertEqual(c_tokens["replaced_tokens_vs_conditioned"]["mean"], 0)
            u_tokens = next(r for r in results["token_diagnostics"]["exchange_replacements"]
                            if r["method"] == "wide")
            self.assertEqual(u_tokens["changed_tokens_between_tasks"]["mean"], 0)
            with np.load(root / "analysis/classification_cases.npz") as saved:
                # Each task gets one half; each class side gets one quarter.
                np.testing.assert_allclose(saved["weight"], [.25, .125, .125, .25, .125, .125])
            self.assertTrue((root / "analysis/exchange_images.csv").exists())
            self.assertEqual(json.loads((root / "analysis/summary.json").read_text())["setting"], "fixture")


if __name__ == "__main__":
    unittest.main()
