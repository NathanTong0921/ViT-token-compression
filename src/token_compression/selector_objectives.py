"""Budget-specific ranking supervision and common teacher diagnostics."""

import torch
from torch.nn import functional as F


def teacher_boundary(targets, keep_count):
    """Identify the teacher's top-k cutoff and genuinely ambiguous boundary ties."""
    if not 0 < keep_count < targets.shape[1]:
        raise ValueError("Ranking requires a nonempty retained and discarded set.")
    order = torch.argsort(targets, dim=1, descending=True, stable=True)
    sorted_targets = targets.gather(1, order)
    cutoff = sorted_targets[:, keep_count - 1:keep_count]
    tied_boundary = sorted_targets[:, keep_count - 1] == sorted_targets[:, keep_count]
    return order, cutoff, tied_boundary


def topk_rank_losses(logits, targets, keep_count):
    """Mean softplus(s_out-s_in) over strict teacher preferences across top-k.

    Positive soft targets preserve the teacher utility ordering. With a unique
    boundary, the pairs are exactly top-k versus its complement. If a tie spans
    the boundary, all tied tokens are treated symmetrically: strict-upper tokens
    rank above the tied group, and that group ranks above strict-lower tokens.
    Equal teacher values never impose an ordering. Uniform targets therefore
    produce zero loss and zero gradient, rather than index-based supervision.
    """
    _, cutoff, tied_boundary = teacher_boundary(targets, keep_count)
    possible_in = targets >= cutoff
    possible_out = (targets < cutoff) | (tied_boundary[:, None] & (targets == cutoff))
    valid = (possible_in[:, :, None] & possible_out[:, None, :]
             & (targets[:, :, None] > targets[:, None, :]))
    losses = F.softplus(logits[:, None, :] - logits[:, :, None])
    counts = valid.sum(dim=(1, 2)).clamp_min(1)
    return (losses * valid).sum(dim=(1, 2)) / counts


def teacher_topk_diagnostics(logits, targets, keep_count):
    """Per-example overlap, including a version accepting any valid boundary tie."""
    order, cutoff, tied_boundary = teacher_boundary(targets, keep_count)
    predicted = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :keep_count]
    teacher_selected = torch.zeros_like(targets, dtype=torch.bool)
    teacher_selected.scatter_(1, order[:, :keep_count], True)
    overlap = teacher_selected.gather(1, predicted).sum(dim=1) / keep_count
    selected_targets = targets.gather(1, predicted)
    strict_hits = (selected_targets > cutoff).sum(dim=1)
    tied_hits = (selected_targets == cutoff).sum(dim=1)
    remaining = keep_count - (targets > cutoff).sum(dim=1)
    tie_aware = (strict_hits + torch.minimum(tied_hits, remaining)) / keep_count
    return {"topk_overlap": overlap, "topk_tie_aware_overlap": tie_aware,
            "teacher_boundary_tie_fraction": tied_boundary.float()}
