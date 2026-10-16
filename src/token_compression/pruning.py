"""Single-stage patch deletion for the original OpenAI CLIP vision transformer."""

import torch
import torch.nn.functional as F


class SingleStagePruner:
    """Split a frozen CLIP ViT immediately after ``prune_after`` full blocks.

    Patch indices are zero-based positions in the original image grid. The CLS
    token is always retained, and retained patches keep their existing features
    and positional information. No token merging or parameter changes are made.
    """

    attention_score_precision = (
        "Layer normalization uses the original CLIP LayerNorm; Q/K projection uses the backbone dtype; "
        "QK products, softmax, and head averaging use float32."
    )

    def __init__(self, visual, prune_after=6):
        self.visual = visual
        self.prune_after = prune_after
        self.blocks = visual.transformer.resblocks
        self.num_patches = visual.positional_embedding.shape[0] - 1
        if not 1 <= prune_after <= len(self.blocks):
            raise ValueError("prune_after must identify a completed transformer block")

    @staticmethod
    def _cls_attention_scores(block, tokens):
        """Recompute CLS attention for scoring without changing hidden states.

        ``tokens`` are the sequence-first inputs to the selected native block.
        Softmax includes the CLS key; only the returned scores exclude CLS.
        """
        normalized = block.ln_1(tokens)
        qkv = F.linear(normalized, block.attn.in_proj_weight,
                       block.attn.in_proj_bias)
        query, key, _ = qkv.chunk(3, dim=-1)
        length, batch, width = tokens.shape
        heads = block.attn.num_heads
        head_width = width // heads
        query = query[0].reshape(batch, heads, head_width).float()
        key = key.reshape(length, batch, heads, head_width).permute(1, 2, 0, 3).float()
        logits = torch.einsum("bhd,bhld->bhl", query, key) * (head_width ** -0.5)
        weights = logits.softmax(dim=-1)
        return weights[:, :, 1:].mean(dim=1)

    def prefix(self, images, return_attention=True):
        """Return batch-first hidden tokens and optional pre-pruning CLS scores.

        Hidden states always use the original block forward functions. The
        separate attention calculation is only for selecting patch positions.
        """
        visual = self.visual
        tokens = visual.conv1(images.to(dtype=visual.conv1.weight.dtype))
        tokens = tokens.reshape(tokens.shape[0], tokens.shape[1], -1)
        tokens = tokens.permute(0, 2, 1)
        cls = visual.class_embedding.to(tokens.dtype) + torch.zeros(
            tokens.shape[0], 1, tokens.shape[-1], dtype=tokens.dtype,
            device=tokens.device)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + visual.positional_embedding.to(tokens.dtype)
        tokens = visual.ln_pre(tokens).permute(1, 0, 2)
        scores = None
        for index in range(self.prune_after):
            block = self.blocks[index]
            if return_attention and index == self.prune_after - 1:
                scores = self._cls_attention_scores(block, tokens)
            tokens = block(tokens)
        return tokens.permute(1, 0, 2), scores

    def finish(self, tokens, patch_indices):
        """Gather unique original-grid positions, then run the native suffix.

        ``tokens`` has shape (batch, original patches + 1, width), and
        ``patch_indices`` has shape (batch, retained patches). Indices are sorted
        internally so patches remain in spatial order. Keep-all follows this
        same gather and suffix path rather than calling the original encoder.
        Returned image embeddings are not normalized.
        """
        if tokens.ndim != 3 or tokens.shape[1] != self.num_patches + 1:
            raise ValueError("tokens must contain CLS and the full original patch grid")
        if (patch_indices.ndim != 2 or patch_indices.shape[0] != tokens.shape[0]
                or not 1 <= patch_indices.shape[1] <= self.num_patches):
            raise ValueError("patch_indices must contain 1 to N positions per image")
        patch_indices = patch_indices.sort(dim=1).values
        cls_indices = torch.zeros((tokens.shape[0], 1), dtype=torch.long,
                                  device=patch_indices.device)
        indices = torch.cat([cls_indices, patch_indices + 1], dim=1)
        tokens = tokens.gather(1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        tokens = tokens.permute(1, 0, 2)
        for index in range(self.prune_after, len(self.blocks)):
            tokens = self.blocks[index](tokens)
        tokens = self.visual.ln_post(tokens.permute(1, 0, 2)[:, 0, :])
        if self.visual.proj is not None:
            tokens = tokens @ self.visual.proj
        return tokens
