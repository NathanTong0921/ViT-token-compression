"""Small task or image corrections around an existing frozen shared selector."""

import torch
from torch import nn
from torch.nn import functional as F

from token_compression.selector import pair_descriptor


def shared_features(shared, tokens):
    """Expose the unchanged wide selector's penultimate patch features and scores."""
    normalized = shared.image_norm(tokens.float())
    patches = normalized[:, 1:]
    cls = normalized[:, :1].expand(-1, patches.shape[1], -1)
    hidden = shared.scorer[:-1](torch.cat((patches, cls), dim=-1))
    scores = shared.scorer[-1](hidden).squeeze(-1)
    return hidden, scores


class ResidualBranch(nn.Module):
    """A zero-output correction with matched small trainable parameter budgets."""

    def __init__(self, shared_dim=199, text_dim=512, conditioned=True, hidden_dim=None):
        super().__init__()
        self.conditioned = bool(conditioned)
        self.shared_dim = shared_dim
        self.context_dim = 2 * text_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else (8 if conditioned else 49)
        self.hidden_norm = nn.LayerNorm(shared_dim, elementwise_affine=False)
        self.context_norm = nn.LayerNorm(self.context_dim, elementwise_affine=False)
        inputs = shared_dim + (self.context_dim if conditioned else 0)
        self.input_projection = nn.Linear(inputs, self.hidden_dim)
        self.output_projection = nn.Linear(self.hidden_dim, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, hidden, candidate_text):
        hidden = self.hidden_norm(hidden.float())
        if self.conditioned:
            context = self.context_norm(pair_descriptor(candidate_text))
            weight = self.input_projection.weight
            # The split equals one concatenated linear layer, avoiding a repeated
            # 1024-dimensional context allocation for every patch.
            value = F.linear(hidden, weight[:, :self.shared_dim], self.input_projection.bias)
            value = value + F.linear(context, weight[:, self.shared_dim:])[:, None]
        else:
            value = self.input_projection(hidden)
        return self.output_projection(F.gelu(value)).squeeze(-1)


class SharedResidualSelector(nn.Module):
    """Online inference recomputes the frozen base features, then adds a correction."""

    def __init__(self, shared, conditioned=True, text_dim=512, hidden_dim=None):
        super().__init__()
        self.shared = shared.eval().requires_grad_(False)
        self.conditioned = bool(conditioned)
        self.branch = ResidualBranch(shared.hidden_dim, text_dim, conditioned, hidden_dim)

    def train(self, mode=True):
        super().train(mode)
        self.shared.eval()
        return self

    def score_cached(self, hidden, base_scores, candidate_text):
        return base_scores + self.branch(hidden, candidate_text)

    def forward(self, tokens, candidate_text):
        hidden, scores = shared_features(self.shared, tokens)
        return self.score_cached(hidden, scores, candidate_text)

    def parameter_counts(self):
        frozen = sum(p.numel() for p in self.shared.parameters())
        trainable = sum(p.numel() for p in self.branch.parameters())
        return {"frozen_shared": frozen, "trainable_correction": trainable,
                "total": frozen + trainable}
