"""Scale-invariant structured supervision for one fixed-cardinality token set."""

import torch


STD_FLOOR = 1e-6


def normalized_scores(scores):
    """Standardize each row; positive affine changes preserve non-floor rows."""
    centered = scores - scores.mean(dim=-1, keepdim=True)
    std = scores.std(dim=-1, keepdim=True, unbiased=False)
    return centered / std.clamp_min(STD_FLOOR), std.squeeze(-1)


def set_losses(scores, target_mask):
    """Cost-augmented top-k hinge, divided by the fixed target cardinality.

    The augmentation is one for every non-target token. Comparing its best
    k-set with the target set upper-bounds the hard missing-target fraction.
    """
    z, _ = normalized_scores(scores)
    selected = torch.zeros_like(z, dtype=torch.bool)
    selected.scatter_(1, target_mask, True)
    k = target_mask.shape[1]
    best = torch.topk(z + (~selected).to(z.dtype), k, dim=-1).values.sum(-1)
    target = z.gather(1, target_mask).sum(-1)
    return ((best - target) / k).clamp_min(0.)
