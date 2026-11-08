"""Fixed-budget hard selection with a differentiable classification path.

The forward pass masks keys in every frozen suffix block. Kept-token states
therefore match physical token deletion. The backward pass uses a mass-constrained
sigmoid relaxation of the selector scores; it is a straight-through estimator.
"""

import torch
from torch.nn import functional as F

from token_compression.pruning import SingleStagePruner


class _MassConstrainedSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, keep_count, temperature):
        # The threshold is the unique root of sum(sigmoid((scores-t)/tau))=k.
        # Solve outside autograd and differentiate this constraint implicitly.
        lower = scores.amin(dim=-1, keepdim=True) - 30. * temperature
        upper = scores.amax(dim=-1, keepdim=True) + 30. * temperature
        for _ in range(48):
            threshold = (lower + upper) * .5
            mass = torch.sigmoid((scores - threshold) / temperature).sum(-1, keepdim=True)
            lower = torch.where(mass > keep_count, threshold, lower)
            upper = torch.where(mass > keep_count, upper, threshold)
        soft = torch.sigmoid((scores - (lower + upper) * .5) / temperature)
        ctx.save_for_backward(soft)
        ctx.temperature = temperature
        return soft

    @staticmethod
    def backward(ctx, gradient):
        (soft,) = ctx.saved_tensors
        q = soft * (1. - soft)
        mass = q.sum(dim=-1, keepdim=True)
        average = (gradient * q).sum(dim=-1, keepdim=True) / mass.clamp_min(torch.finfo(q.dtype).tiny)
        return q * (gradient - average) / ctx.temperature, None, None


def soft_topk(scores, keep_count, temperature=.5):
    """Continuous probabilities with per-row mass ``keep_count``.

This function receives unstandardized scores so its continuous derivative can
also be checked directly. The training wrapper below standardizes score rows.
"""
    if not 0 < keep_count <= scores.shape[-1] or temperature <= 0:
        raise ValueError("Require 0 < keep_count <= patch count and positive temperature")
    if keep_count == scores.shape[-1]:
        return scores * 0. + 1.
    return _MassConstrainedSigmoid.apply(scores, int(keep_count), float(temperature))


def straight_through_topk(scores, keep_count, temperature=.5):
    """Return hard-forward policy, sorted patch indices and relaxed probabilities.

The hard mask has exactly k ones. Its backward derivative is that of the
fixed-mass relaxation of row-standardized scores. Positive score scaling or
translation leaves the selection unchanged; normalization keeps the relaxation
temperature comparable between the C and U selectors.
"""
    centered = scores - scores.mean(dim=-1, keepdim=True)
    standardized = centered / scores.std(dim=-1, correction=0, keepdim=True).clamp_min(1e-6)
    soft = soft_topk(standardized, keep_count, temperature)
    indices = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :keep_count]
    indices = indices.sort(dim=-1).values
    hard = torch.zeros_like(scores).scatter(-1, indices, 1.)
    policy = hard + (soft - soft.detach())
    return policy, indices, soft


def policy_attention(block, tokens, key_policy):
    """CLIP multi-head attention with batch-first tokens and key policies.

No probability floor or smoothing is added: a hard-zero key has no forward
contribution. The retained CLS key makes every attention row nonempty.
"""
    batch, length, width = tokens.shape
    heads = block.attn.num_heads
    head_width = width // heads
    qkv = F.linear(tokens, block.attn.in_proj_weight, block.attn.in_proj_bias)
    query, key, value = [part.reshape(batch, length, heads, head_width).transpose(1, 2)
                         for part in qkv.chunk(3, dim=-1)]
    logits = (query @ key.transpose(-2, -1)) * (head_width ** -.5)
    exponent = (logits - logits.amax(dim=-1, keepdim=True).detach()).exp()
    weights = exponent * key_policy[:, None, None, :]
    weights = weights / weights.sum(dim=-1, keepdim=True)
    attended = (weights @ value).transpose(1, 2).reshape(batch, length, width)
    return F.linear(attended, block.attn.out_proj.weight, block.attn.out_proj.bias)


class DifferentiablePruner(SingleStagePruner):
    """Use ``finish_policy`` for direct training and native ``finish`` for eval.

The caller freezes the CLIP parameters and keeps the backbone in evaluation
mode. Production direct-classification training uses FP32 for the suffix and
cached prefixes. Dropped queries are computed densely during training, but
cannot influence a kept query while their key policy is zero.
"""

    def finish_policy(self, tokens, patch_policy):
        key_policy = torch.cat((torch.ones_like(patch_policy[:, :1]), patch_policy), dim=1)
        for block in self.blocks[self.prune_after:]:
            tokens = tokens + policy_attention(block, block.ln_1(tokens), key_policy)
            tokens = tokens + block.mlp(block.ln_2(tokens))
        features = self.visual.ln_post(tokens[:, 0])
        if self.visual.proj is not None:
            features = features @ self.visual.proj
        return features
