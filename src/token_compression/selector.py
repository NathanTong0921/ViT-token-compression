"""Matched lightweight selectors with unordered candidate-class conditioning."""

import torch
from torch import nn


def pair_descriptor(candidate_text):
    """Describe two candidate classes without exposing their presentation order."""
    candidate_text = candidate_text.float()
    return torch.cat((candidate_text.mean(dim=1),
                      (candidate_text[:, 0] - candidate_text[:, 1]).abs()), dim=-1)


def shared_descriptor(text_features, class_pairs):
    """Use the mean descriptor of training pairs as the image-only fixed prior."""
    return pair_descriptor(text_features[class_pairs]).mean(dim=0)


class TokenSelector(nn.Module):
    """Score patches from frozen patch/CLS tokens and candidate text embeddings.

    Both arms have the same trainable architecture. The image-only arm replaces
    the current pair descriptor with a fixed training-pair average. Labels are
    used to form training targets, never as selector inputs.
    """

    def __init__(self, shared_descriptor, conditioned, hidden_dim=128, token_dim=768,
                 descriptor_mode=None):
        super().__init__()
        self.conditioned = bool(conditioned)
        self.descriptor_mode = descriptor_mode or ("full" if conditioned else "fixed")
        if self.descriptor_mode not in ("full", "fixed", "mean_only", "difference_only"):
            raise ValueError(f"Unknown descriptor mode: {self.descriptor_mode}")
        self.hidden_dim = hidden_dim
        self.token_dim = int(token_dim)
        self.context_dim = shared_descriptor.numel()
        self.register_buffer("shared_descriptor", shared_descriptor.detach().float().clone())
        self.image_norm = nn.LayerNorm(self.token_dim)
        self.context_norm = nn.LayerNorm(self.context_dim)
        self.scorer = nn.Sequential(
            nn.Linear(2 * self.token_dim + self.context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens, candidate_text):
        tokens = self.image_norm(tokens.float())
        patches = tokens[:, 1:]
        cls = tokens[:, :1].expand(-1, patches.shape[1], -1)
        if self.conditioned:
            context = pair_descriptor(candidate_text)
            half = self.context_dim // 2
            if self.descriptor_mode == "mean_only":
                context = torch.cat((context[:, :half], self.shared_descriptor[half:].expand(len(tokens), -1)), dim=-1)
            elif self.descriptor_mode == "difference_only":
                context = torch.cat((self.shared_descriptor[:half].expand(len(tokens), -1), context[:, half:]), dim=-1)
        else:
            context = self.shared_descriptor.unsqueeze(0).expand(len(tokens), -1)
        context = self.context_norm(context).unsqueeze(1).expand(-1, patches.shape[1], -1)
        return self.scorer(torch.cat((patches, cls, context), dim=-1)).squeeze(-1)


class ImageOnlyWideSelector(nn.Module):
    """Allocate the selector budget to patch/CLS processing without a text branch."""

    def __init__(self, hidden_dim=199, token_dim=768):
        super().__init__()
        self.conditioned = False
        self.hidden_dim = hidden_dim
        self.token_dim = int(token_dim)
        self.image_norm = nn.LayerNorm(self.token_dim)
        self.scorer = nn.Sequential(
            nn.Linear(2 * self.token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens, candidate_text):
        tokens = self.image_norm(tokens.float())
        patches = tokens[:, 1:]
        cls = tokens[:, :1].expand(-1, patches.shape[1], -1)
        return self.scorer(torch.cat((patches, cls), dim=-1)).squeeze(-1)


def make_selector(shared_descriptor, variant, hidden_dim=128, token_dim=768):
    """Construct a named arm, with unchanged state keys for the original arms."""
    if variant == "image_only_wide":
        return ImageOnlyWideSelector(hidden_dim=hidden_dim, token_dim=token_dim)
    modes = {"conditioned": "full", "image_only": "fixed",
             "mean_only": "mean_only", "difference_only": "difference_only"}
    if variant not in modes:
        raise ValueError(f"Unknown selector variant: {variant}")
    return TokenSelector(shared_descriptor, conditioned=variant != "image_only",
                         hidden_dim=hidden_dim, token_dim=token_dim, descriptor_mode=modes[variant])


def build_examples(labels, class_pairs):
    """Enumerate image/pair examples when the true class belongs to the pair.

    Rows are image-major, then follow the supplied class-pair order. Pair indices
    can be used to recover task IDs from the corresponding task manifest.
    """
    labels = torch.as_tensor(labels, dtype=torch.long)
    class_pairs = torch.as_tensor(class_pairs, dtype=torch.long, device=labels.device)
    matches = (labels[:, None, None] == class_pairs[None]).any(dim=-1).nonzero()
    return {"image_indices": matches[:, 0], "pair_indices": matches[:, 1],
            "class_pairs": class_pairs[matches[:, 1]]}


def utility_targets(coefficients, labels, class_pairs, text_features, temperature=1.0):
    """Soft patch targets from standardized, true-class-oriented teacher utility.

    Coefficients have shape (examples, patches, classes). A constant teacher
    vector becomes the uniform target; no arbitrary hard mask is imposed.
    """
    coefficients = coefficients.float()
    labels = labels.long()
    class_pairs = class_pairs.long()
    assert torch.all((class_pairs == labels[:, None]).any(dim=1))
    competitors = torch.where(class_pairs[:, 0] == labels, class_pairs[:, 1], class_pairs[:, 0])
    rows = torch.arange(len(labels), device=coefficients.device)
    utility = coefficients[rows, :, labels] - coefficients[rows, :, competitors]
    norm = (text_features[labels] - text_features[competitors]).float().norm(dim=-1)
    utility = utility / norm[:, None]
    utility = utility - utility.mean(dim=1, keepdim=True)
    utility = utility / utility.std(dim=1, correction=0, keepdim=True).clamp_min(1e-6)
    return (utility / temperature).softmax(dim=1)
