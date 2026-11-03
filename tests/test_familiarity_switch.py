"""Paired roles, unchanged identity codes, and exact reference reuse."""

import io
import json
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from semantic_identity_protocol import class_partition
from train_familiarity_switch import make_plan
from evaluate_familiarity_switch import switch_tasks
from analyze_familiarity_switch import summarize_switch
from run_semantic_identity_pilot import run


class FamiliaritySwitchTests(unittest.TestCase):
    def test_original_partition_is_unchanged_and_second_block_is_disjoint(self):
        for count in (37, 102):
            size = int(np.ceil(count / 4))
            order = np.random.default_rng(2099).permutation(count)
            h0 = class_partition(count)[1]
            h1 = class_partition(count, holdout_block=1)[1]
            np.testing.assert_array_equal(h0, np.sort(order[:size]))
            np.testing.assert_array_equal(h1, np.sort(order[size:2 * size]))
            self.assertFalse(set(h0) & set(h1))

    def test_same_codes_preserve_vector_banks_under_both_roles(self):
        plan = make_plan(37, {}, Path("old"))
        h0, h1, remaining = [np.asarray(plan[key]) for key in ("H0", "H1", "always_seen")]
        for values in plan["mappings"].values():
            mapping = np.asarray(values)
            for group in (h0, h1, remaining, np.r_[h1, remaining], np.r_[h0, remaining]):
                np.testing.assert_array_equal(np.sort(mapping[group]), np.sort(group))
            np.testing.assert_array_equal(np.sort(mapping), np.arange(37))

    def test_shared_tasks_are_excluded_from_both_fits(self):
        def row(name, a, b):
            return dict(task_id=name, class_a=a, class_b=b)
        train = [row("cross1", 0, 2), row("h0only", 0, 4), row("h1only", 3, 4), row("neither", 4, 5)]
        val = [row("cross2", 3, 1), row("sameblock", 0, 1)]
        chosen = switch_tasks(train, val, [0, 1], [2, 3])
        self.assertEqual([r["task_id"] for r in chosen], ["cross1", "cross2"])
        for row in chosen:
            pair = {row["class_a"], row["class_b"]}
            self.assertTrue(pair & {0, 1})
            self.assertTrue(pair & {2, 3})

    def test_oriented_effect_and_did_have_correct_sign_and_factor(self):
        n = 40
        original_held = np.tile([True, False], n // 2)
        panel = dict(weights=np.ones(n) / n, image_ids=np.array([f"im{i}" for i in range(n)]),
                     task_ids=np.repeat(["a", "b"], n // 2), true_original_held=original_held,
                     full_correct=np.ones(n), full_margin=np.ones(n))
        arms = ["real_text", "permuted_2097", "permuted_2098", "image_only_wide"]
        predictions = {}
        for phase in ("original", "switched"):
            seen = ~original_held if phase == "original" else original_held
            for arm in arms:
                predictions[(phase, 2041, arm)] = dict(correct=seen.astype(float) if arm == "real_text" else np.zeros(n),
                    margin=.1 + .2 * seen if arm == "real_text" else np.full(n, .1))
        summary = summarize_switch(panel, predictions, arms, [2041], 30)
        accuracy = summary["metrics"]["correct"]["contrasts"]["real_minus_permuted_mean"]
        self.assertAlmostEqual(accuracy["seen_minus_held"]["estimate"], 100.)
        self.assertAlmostEqual(accuracy["difference_in_differences"]["estimate"], 200.)
        margin = summary["metrics"]["margin"]["contrasts"]["real_minus_permuted_mean"]
        self.assertAlmostEqual(margin["seen_minus_held"]["estimate"], .2)
        self.assertAlmostEqual(margin["difference_in_differences"]["estimate"], .4)

    def test_reused_real_and_wide_weights_are_exact_and_permutations_retrained(self):
        from test_semantic_identity_pilot import SemanticIdentityTrainingTests
        fixture = SemanticIdentityTrainingTests("test_loader_never_uses_heldout_training_rows")
        fixture.setUp()
        class Backbone(torch.nn.Module):
            visual = None
        class IdentitySuffix:
            def __init__(self, visual, block):
                pass
            def finish(self, tokens, indices):
                return tokens[:, 0]
        config = dict(device="cpu", cache_dir=str(fixture.cache), batch_size=2, temperature=1.,
                      model="dummy", model_cache="dummy", prune_after=1, training_seeds=[2041],
                      hidden_dim=8, wide_hidden_dim=8, learning_rate=.001, weight_decay=.01,
                      training_steps=2, keep_patch_tokens=2, image_batch_size=4)
        old, refresh = fixture.cache / "old", fixture.cache / "refresh"
        previous = sys.modules.get("clip")
        sys.modules["clip"] = types.SimpleNamespace(load=lambda *a, **k: (Backbone(), None))
        try:
            with patch("token_compression.pruning.SingleStagePruner", IdentitySuffix), redirect_stdout(io.StringIO()):
                run(config, old, ["class_holdout"], [2097, 2098], 2099, 10)
                mappings = make_plan(4, config, old)["mappings"]
                run(config, refresh, ["class_holdout"], [2097, 2098], 2099, 10,
                    fixed_mappings=mappings, reuse_references=old)
            for arm in ("real_text", "image_only_wide"):
                initial = torch.load(old / "class_holdout" / f"seed_2041_{arm}.pt", weights_only=True)
                saved = torch.load(refresh / "class_holdout" / f"seed_2041_{arm}.pt", weights_only=True)
                self.assertFalse(saved["trained_in_this_run"])
                for key in initial["state_dict"]:
                    torch.testing.assert_close(initial["state_dict"][key], saved["state_dict"][key], rtol=0, atol=0)
            records = json.loads((refresh / "class_holdout" / "training_records.json").read_text())
            self.assertEqual([r["arm"] for r in records if r["trained_in_this_run"]], ["permuted_2097", "permuted_2098"])
        finally:
            if previous is None:
                del sys.modules["clip"]
            else:
                sys.modules["clip"] = previous
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main()
