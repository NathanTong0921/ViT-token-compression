import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_matched_replacement as analysis
import evaluate_matched_replacement as evaluation
from evaluate_core_protection import build_masks, unique_image_masks
from evaluate_selector_scale import evaluation_examples
from evaluate_selectors import write_rows


class TinyPruner:
    def __init__(self, visual, prune_after):
        pass

    def finish(self, tokens, selected):
        return tokens[torch.arange(len(tokens))[:, None], selected + 1, :4].sum(1) + tokens[:, 0, :4]


class MatchedReplacementTests(unittest.TestCase):
    def test_hand_computed_disagreement_replacements_have_same_drops(self):
        # C = [0,1,2,3,4,5], U = [0,1,6,7,8,9]. Common tokens 0,1 survive.
        c = np.array([[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]], float)
        u = np.array([[10, 9, 2, 1, 0, -1, 8, 7, 6, 5]], float)
        data = evaluation.build_interventions(c, u, [0], ["image"], ["0_1"], 6, "fixture")
        names = list(data["variants"])
        np.testing.assert_array_equal(data["replacement_counts"][:, 0], [0, 2, 3, 4])
        np.testing.assert_array_equal(data["patch_indices"][names.index("top_50"), 0], [0, 1, 2, 3, 6, 7])
        np.testing.assert_array_equal(data["patch_indices"][names.index("top_75"), 0], [0, 1, 2, 6, 7, 8])
        np.testing.assert_array_equal(data["patch_indices"][names.index("wide"), 0], [0, 1, 6, 7, 8, 9])
        for percentage, dropped in [(50, {4, 5}), (75, {3, 4, 5})]:
            variants = [i for i, name in enumerate(names) if f"_{percentage}" in name]
            for i in variants:
                selected = set(data["patch_indices"][i, 0])
                self.assertEqual(set(range(6)) - selected, dropped)
                self.assertTrue({0, 1} <= selected)
                self.assertEqual(len(selected), 6)
        self.assertTrue(analysis.validate_interventions(data))

    def test_no_disagreement_and_floor_zero_retain_identical_endpoints(self):
        c = np.array([[5., 4., 3., 2., 1.], [5., 4., 3., 2., 1.]])
        u = np.array([[5., 4., 3., 2., 1.], [5., 4., 2., 3., 1.]])
        data = evaluation.build_interventions(c, u, [0, 1], ["a", "b"], ["0_1", "0_1"], 3, "fixture")
        np.testing.assert_array_equal(data["replacement_counts"], [[0, 0], [0, 0], [0, 0], [0, 1]])
        for i, name in enumerate(data["variants"]):
            np.testing.assert_array_equal(data["patch_indices"][i, 0], [0, 1, 2])
            if name != "wide":
                np.testing.assert_array_equal(data["patch_indices"][i, 1], [0, 1, 2])
        self.assertTrue(analysis.validate_interventions(data))

    def test_random_additions_nested_and_reproducible(self):
        c = np.arange(12, 0, -1, dtype=float)[None]
        u = np.concatenate([np.arange(6, 0, -1), np.arange(12, 6, -1)])[None].astype(float)
        first = evaluation.build_interventions(c, u, [0], ["a"], ["1_2"], 6, "fixture")
        second = evaluation.build_interventions(c, u, [0], ["a"], ["1_2"], 6, "fixture")
        np.testing.assert_array_equal(first["added_indices"], second["added_indices"])
        names = list(first["variants"])
        for seed in evaluation.RANDOM_SEEDS:
            small = first["added_indices"][names.index(f"random_50_{seed}"), 0]
            large = first["added_indices"][names.index(f"random_75_{seed}"), 0]
            self.assertTrue(set(small[small >= 0]) <= set(large[large >= 0]))

    def test_saved_intervention_audit_detects_equal_count_but_different_drop(self):
        c = np.array([[6, 5, 4, 3, 2, 1]], float)
        u = np.array([[2, 1, 6, 5, 4, 3]], float)
        data = evaluation.build_interventions(c, u, [0], ["a"], ["1_2"], 4, "fixture")
        vi = list(data["variants"]).index("random_50_2211")
        selected = data["patch_indices"][vi, 0].copy()
        # C\U = {0,1}; the lowest-C removal must be 1. Swap the removal to 0.
        selected[selected == 0] = 1
        data["patch_indices"][vi, 0] = np.sort(selected)
        with self.assertRaises(AssertionError):
            analysis.validate_interventions(data)

    def test_mocked_replay_reuses_endpoints_and_analysis_preserves_matching(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(53)
        image_ids, labels = np.array(["a", "b", "c", "d"]), np.array([0, 1, 2, 2])
        text = rng.normal(size=(3, 4)).astype(np.float32)
        text /= np.linalg.norm(text, axis=1, keepdims=True)
        prefixes = rng.normal(size=(4, 10, 6)).astype(np.float32)
        full = prefixes[:, 0, :4].copy()
        full /= np.linalg.norm(full, axis=1, keepdims=True)
        tasks = [{"task_id": "0_2", "class_a": 0, "class_b": 2},
                 {"task_id": "1_2", "class_a": 1, "class_b": 2}]
        examples = evaluation_examples(image_ids, labels, [], tasks)
        ix = np.array([r["image_index"] for r in examples])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache, source, output = root / "cache", root / "source", root / "output"
            (cache / "val").mkdir(parents=True)
            source.mkdir()
            np.save(cache / "val/prefix_tokens.npy", prefixes)
            np.savez(source / "evidence.npz", image_ids=image_ids, labels=labels,
                     text_features=text, full_scores=full @ text.T)
            write_rows(source / "examples.csv", examples)
            config = {"cache_dir": str(cache), "keep_patch_tokens": 4, "image_batch_size": 2,
                      "model": "tiny", "model_cache": str(root), "prune_after": 1, "device": "cpu"}
            (source / "summary.json").write_text(json.dumps({"config": config, "setting": "fixture",
                "git_commit": "fixture", "training_seeds": [101, 102]}))
            for seed in [101, 102]:
                gen = np.random.default_rng(seed)
                c, u = gen.normal(size=(len(examples), 9)), gen.normal(size=(4, 9))
                variants, masks = build_masks(c, u, ix, image_ids, 4, "fixture")
                keys, inverse = unique_image_masks(ix, masks)
                values = TinyPruner(None, 1).finish(torch.tensor(prefixes[keys[:, 0]]), torch.tensor(keys[:, 1:]))
                values = (values / values.norm(dim=-1, keepdim=True)).numpy()
                np.savez(source / f"seed_{seed}.npz", variants=variants, patch_indices=masks,
                         conditioned_logits=c, wide_logits=u, unique_keys=keys, unique_features=values,
                         cosine_scores=(values @ text.T)[inverse])
            backbone = torch.nn.Module()
            backbone.visual = types.SimpleNamespace()
            fake_clip = types.SimpleNamespace(load=lambda *args, **kwargs: (backbone, None))
            # Import the patched module before snapshotting sys.modules; otherwise
            # patch.dict removes its new entry but leaves the package attribute.
            with patch("token_compression.pruning.SingleStagePruner", TinyPruner), patch.dict(sys.modules, {"clip": fake_clip}):
                summary = evaluation.run(source, output)
            self.assertTrue(all(r["reused_image_masks"] > 0 for r in summary["runs"]))
            self.assertTrue(all(r["new_suffix_image_masks"] > 0 for r in summary["runs"]))
            for seed in [101, 102]:
                with np.load(output / f"seed_{seed}.npz") as new, np.load(source / f"seed_{seed}.npz") as old:
                    for vi in [0, 1]:
                        np.testing.assert_allclose(new["cosine_scores"][vi], old["cosine_scores"][vi], atol=1e-7, rtol=1e-6)
                    for vi in range(len(new["variants"])):
                        direct = TinyPruner(None, 1).finish(torch.tensor(prefixes[ix]), torch.tensor(new["patch_indices"][vi].astype(int)))
                        direct = (direct / direct.norm(dim=-1, keepdim=True)).numpy() @ text.T
                        np.testing.assert_allclose(new["cosine_scores"][vi], direct, atol=1e-7, rtol=1e-6)
            result = analysis.analyze(output, root / "analysis", replicates=40)
            self.assertEqual(result["primary_family_size"], 8)
            self.assertEqual(result["classification_image_task_count"], 6)
            primary = result["classification"]["contrasts"]["top_50_minus_random_50"]
            self.assertAlmostEqual(primary["margin"]["individual_interval_level"], .99375)
            wide = next(r for r in result["exchange"]["table"] if r["method"] == "wide")
            self.assertEqual(wide["margin"]["mean"], 0)
            self.assertTrue((root / "analysis/actual_replacement_counts.csv").exists())


if __name__ == "__main__":
    unittest.main()
