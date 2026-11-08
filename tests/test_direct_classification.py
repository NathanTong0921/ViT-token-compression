import unittest

import torch
from torch import nn
from torch.nn import functional as F

from token_compression.direct_classification import (
    DifferentiablePruner, soft_topk, straight_through_topk,
)


class TinyBlock(nn.Module):
    def __init__(self, width=12, heads=3):
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=0.)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))

    def forward(self, tokens):
        normalized = self.ln_1(tokens)
        tokens = tokens + self.attn(normalized, normalized, normalized, need_weights=False)[0]
        return tokens + self.mlp(self.ln_2(tokens))


class TinyVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(10, 12))
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.Sequential(*(TinyBlock() for _ in range(4)))
        self.ln_post = nn.LayerNorm(12)
        self.proj = nn.Parameter(torch.randn(12, 7) / 12 ** .5)


class DirectClassificationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(72)
        self.visual = TinyVisual().double().eval().requires_grad_(False)
        self.pruner = DifferentiablePruner(self.visual, prune_after=1)
        self.tokens = torch.randn(3, 10, 12, dtype=torch.double)

    def test_policy_forward_matches_native_hard_deletion_and_keep_all(self):
        scores = torch.randn(3, 9, dtype=torch.double)
        for keep in (1, 4, 9):
            policy, indices, _ = straight_through_topk(scores, keep)
            torch.testing.assert_close(policy.sum(-1), torch.full((3,), keep, dtype=torch.double), rtol=0, atol=0)
            native = self.pruner.finish(self.tokens, indices)
            dense = self.pruner.finish_policy(self.tokens, policy)
            torch.testing.assert_close(dense, native, rtol=1e-11, atol=1e-11)

    def test_continuous_soft_topk_gradient_and_mass(self):
        scores = torch.randn(2, 6, dtype=torch.double, requires_grad=True)
        for keep in (1, 3, 5):
            soft = soft_topk(scores, keep, .7)
            torch.testing.assert_close(soft.sum(-1), torch.full((2,), keep, dtype=torch.double), rtol=0, atol=1e-11)
            self.assertTrue(torch.autograd.gradcheck(lambda value: soft_topk(value, keep, .7),
                                                    (scores,), eps=1e-5, atol=1e-6, rtol=1e-4))

    def test_classification_gradient_reaches_selector_not_frozen_backbone(self):
        selector = nn.Linear(12, 1).double()
        scores = selector(self.tokens[:, 1:]).squeeze(-1)
        scores.retain_grad()
        policy, indices, _ = straight_through_topk(scores, 4)
        features = F.normalize(self.pruner.finish_policy(self.tokens, policy), dim=-1)
        text = F.normalize(torch.randn(3, 2, 7, dtype=torch.double), dim=-1)
        logits = 10. * torch.einsum("bd,bcd->bc", features, text)
        F.cross_entropy(logits, torch.tensor([0, 1, 0])).backward()
        self.assertTrue(torch.isfinite(selector.weight.grad).all())
        self.assertGreater(float(selector.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(scores.grad[policy.detach() == 0].abs().sum()), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in self.visual.parameters()))
        torch.testing.assert_close(policy.detach().gather(1, indices), torch.ones(3, 4, dtype=torch.double))

    def test_score_shift_and_positive_scale_preserve_policy_and_soft_mass(self):
        scores = torch.randn(3, 9, dtype=torch.double, requires_grad=True)
        original = straight_through_topk(scores, 4, .5)
        changed = straight_through_topk(2.5 * scores + 17., 4, .5)
        for left, right in zip(original, changed):
            torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-11)
        self.assertTrue(torch.all((original[0] == 0) | (original[0] == 1)))


if __name__ == "__main__":
    unittest.main()
