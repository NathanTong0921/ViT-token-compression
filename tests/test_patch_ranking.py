import unittest

import torch
from torch.nn import functional as F

from token_compression.patch_ranking import (PatchRankingPredictor, overlap_at,
                                            rank_scores, ranking_kl,
                                            ranking_probabilities)


class PatchRankingTests(unittest.TestCase):
    def test_target_matches_released_loop_for_permuted_rankings(self):
        torch.manual_seed(41)
        rankings = torch.stack([torch.randperm(196) for _ in range(3)])
        reference = []
        for ranking in rankings:
            scores = torch.zeros(196)
            for index, patch in enumerate(ranking):
                scores[patch] = -5 - index * 0.1 if index >= 146 else 0
            reference.append(scores.softmax(dim=0))
        torch.testing.assert_close(ranking_probabilities(rankings), torch.stack(reference),
                                   rtol=1e-6, atol=1e-12)
        ordered = ranking_probabilities(rankings).gather(1, rankings)
        torch.testing.assert_close(ordered[:, :146], torch.full((3, 146), 1/146))
        self.assertTrue(torch.all(ordered[:, 146:] < 1e-9))

    def test_model_shape_pooling_and_backward(self):
        torch.manual_seed(42)
        model = PatchRankingPredictor()
        tokens = torch.randn(2, 197, 768)
        scores = model(tokens)
        self.assertEqual(scores.shape, (2, 196))
        torch.testing.assert_close(scores, model(F.adaptive_avg_pool1d(tokens, 64)),
                                   rtol=0, atol=0)
        target = ranking_probabilities(torch.stack([torch.randperm(196) for _ in range(2)]))
        ranking_kl(scores, target).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        self.assertGreater(sum(float(p.grad.square().sum()) for p in model.parameters()), 0)

    def test_kl_matches_public_per_image_reduction(self):
        torch.manual_seed(43)
        scores = torch.randn(3, 196)
        targets = ranking_probabilities(torch.stack([torch.randperm(196) for _ in range(3)]))
        expected = torch.stack([F.kl_div(s.softmax(0).log(), t, reduction="sum")
                                for s, t in zip(scores, targets)]).mean()
        torch.testing.assert_close(ranking_kl(scores, targets), expected)

    def test_overlap_distinguishes_retained_and_removed_boundaries(self):
        teacher = torch.arange(196).unsqueeze(0)
        prediction = teacher.clone()
        prediction[0, 97], prediction[0, 146] = 146, 97
        torch.testing.assert_close(overlap_at(prediction, teacher, 98), torch.tensor([97/98]))
        torch.testing.assert_close(overlap_at(prediction, teacher, 100), torch.tensor([99/100]))
        torch.testing.assert_close(overlap_at(prediction, teacher, 50, deleted=True), torch.tensor([49/50]))
        torch.testing.assert_close(rank_scores(torch.ones(1, 196)), teacher)


if __name__ == "__main__":
    unittest.main()
