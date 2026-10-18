import unittest

import torch
from torch.nn import functional as F

from token_compression.selector import (TokenSelector, build_examples, pair_descriptor,
                                        shared_descriptor, utility_targets)


class TokenSelectorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.text = F.normalize(torch.randn(4, 512), dim=-1)
        self.pairs = torch.tensor([[0, 1], [0, 2], [1, 3]])
        self.prior = shared_descriptor(self.text, self.pairs)
        self.tokens = torch.randn(2, 6, 768)
        self.candidates = self.text[torch.tensor([[0, 1], [1, 3]])]

    def test_candidate_order_cannot_change_conditioned_prediction(self):
        model = TokenSelector(self.prior, conditioned=True, hidden_dim=8).eval()
        with torch.no_grad():
            original = model(self.tokens, self.candidates)
            reversed_pair = model(self.tokens, self.candidates.flip(1))
        torch.testing.assert_close(original, reversed_pair, rtol=0, atol=0)
        self.assertEqual(original.shape, (2, 5))
        torch.testing.assert_close(pair_descriptor(self.candidates),
                                   pair_descriptor(self.candidates.flip(1)), rtol=0, atol=0)

    def test_image_only_ignores_candidates_with_matched_state_and_parameter_count(self):
        conditioned = TokenSelector(self.prior, conditioned=True, hidden_dim=8).eval()
        image_only = TokenSelector(self.prior, conditioned=False, hidden_dim=8).eval()
        image_only.load_state_dict(conditioned.state_dict())
        self.assertEqual(sum(p.numel() for p in conditioned.parameters()),
                         sum(p.numel() for p in image_only.parameters()))
        for key, value in conditioned.state_dict().items():
            torch.testing.assert_close(value, image_only.state_dict()[key], rtol=0, atol=0)
        changed = self.text[torch.tensor([[2, 3], [0, 2]])]
        with torch.no_grad():
            torch.testing.assert_close(image_only(self.tokens, self.candidates),
                                       image_only(self.tokens, changed), rtol=0, atol=0)
            self.assertFalse(torch.allclose(conditioned(self.tokens, self.candidates),
                                            conditioned(self.tokens, changed)))

    def test_teacher_targets_follow_true_class_and_preserve_unordered_pairs(self):
        coefficients = torch.tensor([[[2., 0.], [0., 1.], [-1., 2.]],
                                     [[2., 0.], [0., 1.], [-1., 2.]]])
        labels = torch.tensor([0, 1])
        pairs = torch.tensor([[0, 1], [0, 1]])
        targets = utility_targets(coefficients, labels, pairs, torch.eye(2))
        self.assertEqual(targets.argmax(dim=1).tolist(), [0, 2])
        torch.testing.assert_close(targets.sum(dim=1), torch.ones(2))
        torch.testing.assert_close(targets,
                                   utility_targets(coefficients, labels, pairs.flip(1), torch.eye(2)))
        torch.testing.assert_close(utility_targets(torch.zeros_like(coefficients), labels, pairs,
                                                   torch.eye(2)), torch.full((2, 3), 1 / 3))

    def test_examples_include_only_pairs_containing_each_image_label(self):
        examples = build_examples(torch.tensor([0, 1, 3]), self.pairs)
        self.assertEqual(examples["image_indices"].tolist(), [0, 0, 1, 1, 2])
        self.assertEqual(examples["pair_indices"].tolist(), [0, 1, 0, 2, 2])
        torch.testing.assert_close(examples["class_pairs"], self.pairs[examples["pair_indices"]])

    def test_small_selector_can_reduce_soft_target_loss(self):
        model = TokenSelector(self.prior, conditioned=True, hidden_dim=8)
        target = torch.tensor([[.65, .1, .1, .1, .05], [.05, .1, .1, .1, .65]])
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        with torch.no_grad():
            initial = F.kl_div(model(self.tokens, self.candidates).log_softmax(dim=1),
                               target, reduction="batchmean").item()
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            loss = F.kl_div(model(self.tokens, self.candidates).log_softmax(dim=1),
                            target, reduction="batchmean")
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            final = F.kl_div(model(self.tokens, self.candidates).log_softmax(dim=1),
                             target, reduction="batchmean").item()
        self.assertLess(final, initial * .3)


if __name__ == "__main__":
    unittest.main()
