"""Preservation predictor adapted from CEWu/PatchRanking's public training entry.

Source: https://github.com/CEWu/PatchRanking/blob/main/train_predictor.py
Upstream permission notice: third_party/PatchRanking.LICENSE.
The predictor uses timm's MixerBlock directly. The released KL target suppresses
the bottom 50 patches and assigns equal probability to the remaining patches.
"""

import torch
from torch import nn
from torch.nn import functional as F
from timm.models.mlp_mixer import MixerBlock


class PatchRankingPredictor(nn.Module):
    """Channel pooling, one token/channel Mixer block, and patch scores."""

    def __init__(self, pooled_dim=64, sequence_length=197):
        super().__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool1d(pooled_dim)
        self.in_conv = nn.Sequential(MixerBlock(pooled_dim, sequence_length))
        self.fc = nn.Linear(pooled_dim, 1)

    def forward(self, tokens):
        values = self.adaptive_pool(tokens.float())
        values = self.in_conv(values)
        return self.fc(values[:, 1:]).squeeze(-1)


def ranking_probabilities(rankings):
    """Vectorize the public ranking_to_prob_distribution function exactly.

    Rankings list patch indices from greatest to least preservation importance.
    At 196 patches, ranks 0..145 have zero target logit, while ranks 146..195
    receive -5 - 0.1 * rank. Softmax maps these logits back to patch positions.
    """
    ranks = torch.arange(rankings.shape[-1], device=rankings.device)
    ranked_scores = torch.where(ranks >= rankings.shape[-1] - 50,
                                -5.0 - ranks.float() * 0.1, 0.0)
    scores = torch.zeros_like(rankings, dtype=torch.float32)
    scores.scatter_(-1, rankings.long(), ranked_scores.expand_as(scores))
    return scores.softmax(dim=-1)


def ranking_kl(scores, targets):
    """Mean per-image KL, using log_softmax for the predicted log probabilities."""
    return F.kl_div(scores.log_softmax(dim=-1), targets, reduction="batchmean")


def rank_scores(scores):
    return torch.argsort(scores, dim=-1, descending=True, stable=True)


def overlap_at(predicted_rankings, teacher_rankings, count, deleted=False):
    """Per-image agreement fraction for retained or lowest-ranked patches."""
    selected = slice(-count, None) if deleted else slice(None, count)
    predicted = torch.zeros_like(predicted_rankings, dtype=torch.bool)
    teacher = torch.zeros_like(teacher_rankings, dtype=torch.bool)
    predicted.scatter_(1, predicted_rankings[:, selected], True)
    teacher.scatter_(1, teacher_rankings[:, selected], True)
    return (predicted & teacher).sum(dim=1).float() / count
