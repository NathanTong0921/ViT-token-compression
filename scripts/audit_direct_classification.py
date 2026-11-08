"""Compare dense hard-policy CLIP with native token gathering on real prefixes."""

import argparse
import csv
import json
from pathlib import Path

import clip
import numpy as np
import torch
from torch.nn import functional as F

from evaluate_selector_scale import selector_from_checkpoint
from token_compression.direct_classification import DifferentiablePruner, straight_through_topk
from token_compression.selector import build_examples, shared_descriptor


def main(args):
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    cache = Path("runs/flowers-selector-budget-49-supervision-v1")
    with (cache / "train_tasks.csv").open() as handle:
        pairs = np.asarray([[int(r["class_a"]), int(r["class_b"])] for r in csv.DictReader(handle)])
    with np.load(cache / "train/evidence.npz") as a:
        labels = torch.as_tensor(a["labels"], device=device)
        text = torch.as_tensor(a["text_features"], device=device).float()
    pair_tensor = torch.as_tensor(pairs, device=device)
    examples = build_examples(labels, pair_tensor)
    rows = torch.linspace(0, len(examples["image_indices"])-1, 8, device=device).long()
    image_ix = examples["image_indices"][rows]
    selected_pairs = examples["class_pairs"][rows]
    prefixes = np.load(cache / "train/prefix_tokens.npy", mmap_mode="r")
    tokens = torch.as_tensor(prefixes[image_ix.cpu().numpy()].copy(), device=device).float()
    candidates = text[selected_pairs]
    targets = (selected_pairs[:, 1] == labels[image_ix]).long()
    backbone, _ = clip.load("ViT-B/16", device=device, jit=False, download_root="cache/clip")
    backbone.float().eval().requires_grad_(False)
    pruner = DifferentiablePruner(backbone.visual, 6)
    scale = backbone.logit_scale.exp().detach()
    prior = shared_descriptor(text, pair_tensor)
    sources = {
        "conditioned": "runs/flowers-selector-budget-49-training-v1/full/seed_2041/conditioned.pt",
        "image_only_wide": "runs/flowers-selector-ablations-49-training-v1/full/seed_2041/image_only_wide.pt",
    }
    models = {}
    for arm, path in sources.items():
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        models[arm] = selector_from_checkpoint(checkpoint, prior, arm, 768, device).requires_grad_(True)
    records = []
    generator = torch.Generator(device=device).manual_seed(2705)
    for name in ("conditioned", "image_only_wide", "random49", "random98", "all196"):
        model = models.get(name)
        k = 49 if model is not None else int(name.replace("random", "").replace("all", ""))
        scores = model(tokens, candidates) if model is not None else torch.randn(8, 196, generator=generator, device=device)
        policy, indices, soft = straight_through_topk(scores, k, .5)
        assert torch.equal(policy.detach().sum(1), torch.full((8,), k, device=device))
        dense = F.normalize(pruner.finish_policy(tokens, policy), dim=-1)
        with torch.no_grad():
            native = F.normalize(pruner.finish(tokens, indices), dim=-1)
        dense_logits = scale * torch.einsum("bd,bcd->bc", dense, candidates)
        native_logits = scale * torch.einsum("bd,bcd->bc", native, candidates)
        dense_ce = F.cross_entropy(dense_logits, targets, reduction="none")
        native_ce = F.cross_entropy(native_logits, targets, reduction="none")
        record = dict(name=name, patches=k,
                      normalized_feature_error=float((dense.detach()-native).abs().max()),
                      logit_error=float((dense_logits.detach()-native_logits).abs().max()),
                      ce_error=float((dense_ce.detach()-native_ce).abs().max()),
                      soft_mass_error=float((soft.sum(1)-k).abs().max()))
        torch.testing.assert_close(dense.detach(), native, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(dense_ce.detach(), native_ce, atol=2e-4, rtol=1e-5)
        if model is not None:
            model.zero_grad(set_to_none=True)
            scores.retain_grad()
            dense_ce.mean().backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            norm = torch.sqrt(sum(g.double().square().sum() for g in grads))
            assert norm > 0 and all(p.grad is None for p in backbone.parameters())
            kept = policy.detach().bool()
            record.update(selector_gradient_l2=float(norm),
                          selected_score_gradient_mean=float(scores.grad[kept].abs().mean()),
                          dropped_score_gradient_mean=float(scores.grad[~kept].abs().mean()),
                          backbone_gradients_absent=True)
        records.append(record)
        print(json.dumps(record), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(passed=True, device=args.device, images=8,
        dtype="float32", tf32=False, examples_source="original training images/tasks", checks=records), indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
