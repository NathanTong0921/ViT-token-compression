import unittest

import torch
from clip.model import VisionTransformer

from token_compression.pruning import SingleStagePruner


class SingleStagePruningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.visual = VisionTransformer(input_resolution=16, patch_size=4,
                                        width=32, layers=4, heads=4, output_dim=16)
        self.visual.eval().requires_grad_(False)
        self.pruner = SingleStagePruner(self.visual, prune_after=2)
        self.images = torch.randn(3, 3, 16, 16)

    def test_keep_all_matches_native_forward(self):
        with torch.inference_mode():
            expected = self.visual(self.images)
            tokens, scores = self.pruner.prefix(self.images)
            indices = torch.arange(16).expand(3, -1)
            actual = self.pruner.finish(tokens, indices)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(tokens.shape, (3, 17, 32))
        self.assertEqual(scores.shape, (3, 16))
        self.assertEqual(scores.dtype, torch.float32)

    def test_cls_attention_matches_native_multihead_attention(self):
        captured = []
        block = self.visual.transformer.resblocks[1]
        hook = block.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0].clone()))
        with torch.inference_mode():
            _, scores = self.pruner.prefix(self.images)
            normalized = block.ln_1(captured[0])
            _, weights = block.attn(normalized, normalized, normalized,
                                    need_weights=True, average_attn_weights=False)
        hook.remove()
        expected = weights[:, :, 0, 1:].mean(dim=1)
        torch.testing.assert_close(scores, expected, rtol=1e-6, atol=1e-7)
        self.assertTrue(torch.all(scores.sum(dim=1) < 1))

    def test_gather_preserves_cls_and_original_spatial_order(self):
        captured = []
        block = self.visual.transformer.resblocks[2]
        hook = block.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0].clone()))
        tokens = torch.arange(3 * 17 * 32, dtype=torch.float32).reshape(3, 17, 32)
        indices = torch.tensor([[8, 1, 13], [4, 0, 7], [15, 2, 9]])
        with torch.inference_mode():
            embedding = self.pruner.finish(tokens, indices)
        hook.remove()
        expected = torch.stack([
            tokens[0, [0, 2, 9, 14]],
            tokens[1, [0, 1, 5, 8]],
            tokens[2, [0, 3, 10, 16]],
        ])
        torch.testing.assert_close(captured[0].permute(1, 0, 2), expected,
                                   rtol=0, atol=0)
        self.assertEqual(embedding.shape, (3, 16))

    def test_scoring_does_not_change_prefix_states(self):
        with torch.inference_mode():
            scored, scores = self.pruner.prefix(self.images, return_attention=True)
            plain, no_scores = self.pruner.prefix(self.images, return_attention=False)
        torch.testing.assert_close(scored, plain, rtol=0, atol=0)
        self.assertIsNotNone(scores)
        self.assertIsNone(no_scores)


if __name__ == "__main__":
    unittest.main()
