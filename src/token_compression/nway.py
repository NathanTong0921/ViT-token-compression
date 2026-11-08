"""Candidate-set selection with a common two- and five-way representation."""

from math import gcd

import numpy as np
import torch

from .selector import ImageOnlyWideSelector, TokenSelector


def set_descriptor(candidate_text):
    """Mean and twice population spread; at K=2 spread is absolute difference."""
    values = candidate_text.float()
    return torch.cat((values.mean(dim=1), 2 * values.std(dim=1, correction=0)), dim=-1)


class SetTokenSelector(TokenSelector):
    def __init__(self, token_dim=768, text_dim=512, hidden_dim=128):
        super().__init__(torch.zeros(2 * text_dim), conditioned=True,
                         token_dim=token_dim, hidden_dim=hidden_dim)

    def forward(self, tokens, candidate_text):
        normalized = self.image_norm(tokens.float())
        patches = normalized[:, 1:]
        cls = normalized[:, :1].expand(-1, patches.shape[1], -1)
        context = self.context_norm(set_descriptor(candidate_text))
        context = context[:, None].expand(-1, patches.shape[1], -1)
        return self.scorer(torch.cat((patches, cls, context), dim=-1)).squeeze(-1)


def make_selector(variant, token_dim=768, text_dim=512, hidden_dim=128):
    if variant == "conditioned":
        return SetTokenSelector(token_dim, text_dim, hidden_dim)
    if variant == "image_only_wide":
        return ImageOnlyWideSelector(hidden_dim=hidden_dim, token_dim=token_dim)
    raise ValueError(f"Unknown selector variant: {variant}")


def task_manifest(class_count, ways=(2, 5), seed=2611, train_rounds=4, val_rounds=2):
    """Regular cyclic task sets: each class participates equally in each split."""
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(class_count)
    offsets = [d for d in range(1, (class_count - 1) // 2 + 1)
               if class_count // gcd(class_count, d) >= max(ways)]
    chosen = rng.permutation(offsets)[:train_rounds + val_rounds]
    if len(chosen) != train_rounds + val_rounds:
        raise ValueError("Not enough distinct offsets for the requested task rounds")
    manifest = {"seed": seed, "class_permutation": permutation.tolist(),
                "offsets": chosen.tolist(), "ways": {}}
    for way in ways:
        seen = set()
        splits = {"train": [], "val": []}
        for round_index, offset in enumerate(chosen):
            split = "train" if round_index < train_rounds else "val"
            for anchor in range(class_count):
                candidates = tuple(sorted(permutation[(anchor + np.arange(way) * offset) % class_count].tolist()))
                if candidates in seen:
                    raise ValueError("Cyclic construction produced a repeated candidate set")
                seen.add(candidates)
                splits[split].append({"task_id": f"{way}way_{split}_r{round_index:02d}_a{anchor:03d}",
                                      "candidates": list(candidates), "round": round_index,
                                      "anchor": anchor, "offset": int(offset)})
        manifest["ways"][str(way)] = splits
    return manifest


def build_panel(image_ids, labels, tasks):
    """All images for every true candidate; equal task and within-task class mass."""
    image_ids, labels = np.asarray(image_ids), np.asarray(labels)
    images, task_indices, candidates, weights, task_ids = [], [], [], [], []
    for task_index, task in enumerate(tasks):
        classes = task["candidates"]
        for label in classes:
            indices = np.flatnonzero(labels == label)
            if not len(indices):
                raise ValueError(f"Class {label} has no images in this split")
            images.extend(indices.tolist())
            task_indices.extend([task_index] * len(indices))
            candidates.extend([classes] * len(indices))
            weights.extend([1 / (len(tasks) * len(classes) * len(indices))] * len(indices))
            task_ids.extend([task["task_id"]] * len(indices))
    images = np.asarray(images, dtype=np.int64)
    return {"image_indices": images, "task_indices": np.asarray(task_indices, np.int64),
            "candidates": np.asarray(candidates, np.int64), "labels": labels[images],
            "image_ids": image_ids[images], "task_ids": np.asarray(task_ids),
            "weights": np.asarray(weights, np.float64)}


def margin_responses(responses, labels, candidates, text):
    """Nearest class-separating geometric margin for each sampled mask.

    responses: (examples, masks, all classes). Labels only construct supervision.
    """
    rows = torch.arange(len(labels), device=responses.device)
    true_scores = responses[rows, :, labels]
    candidate_scores = responses.gather(2, candidates[:, None].expand(-1, responses.shape[1], -1))
    distances = (text[labels, None] - text[candidates]).norm(dim=-1)
    is_true = candidates == labels[:, None]
    distances = distances.masked_fill(is_true, 1)
    margins = (true_scores[:, :, None] - candidate_scores) / distances[:, None]
    return margins.masked_fill(is_true[:, None], torch.inf).min(dim=-1).values


def normalized_targets(utilities, temperature=1.0):
    utilities = utilities.float()
    centered = utilities - utilities.mean(dim=1, keepdim=True)
    standardized = centered / utilities.std(dim=1, correction=0, keepdim=True).clamp_min(1e-6)
    return (standardized / temperature).softmax(dim=1)


def training_order(panel, draws, seed):
    """Balance task/class draws, then sample images uniformly in each cell."""
    tasks, labels = panel["task_indices"], panel["labels"]
    cells, inverse = np.unique(np.column_stack((tasks, labels)), axis=0, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(cells))]
    rng = np.random.default_rng(seed)
    repeats, remainder = divmod(draws, len(cells))
    order = np.concatenate((np.tile(np.arange(len(cells)), repeats),
                            rng.choice(len(cells), remainder, replace=False)))
    rng.shuffle(order)
    quantiles = rng.random(draws)
    indices = np.empty(draws, dtype=np.int64)
    for cell, choices in enumerate(members):
        take = np.flatnonzero(order == cell)
        indices[take] = choices[(quantiles[take] * len(choices)).astype(np.int64)]
    return indices
