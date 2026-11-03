"""Replay frozen CLIP with a protected image-only core and task-specific fill."""

import argparse
import json
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

import numpy as np

from evaluate_selector_scale import evaluation_examples, selector_from_checkpoint
from evaluate_selectors import read_tasks, write_rows


SETTINGS = {
    "pets_b16_49": {
        "config": "configs/pets_selector_ablations_49.json",
        "conditioned_dir": "runs/pets-selector-scale-training-v1",
        "wide_dir": "runs/pets-selector-ablations-49-training-v1",
    },
    "flowers_b16_98": {
        "config": "configs/flowers_selector_ablations_98.json",
        "conditioned_dir": "runs/flowers-selector-budget-98-training-v1",
        "wide_dir": "runs/flowers-selector-ablations-98-training-v1",
    },
    "pets_b16_98": {
        "config": "configs/pets_selector_ablations_98.json",
        "conditioned_dir": "runs/pets-selector-budget-98-training-v1",
        "wide_dir": "runs/pets-selector-ablations-98-training-v1",
    },
    "flowers_b16_49": {
        "config": "configs/flowers_selector_ablations_49.json",
        "conditioned_dir": "runs/flowers-selector-budget-49-training-v1",
        "wide_dir": "runs/flowers-selector-ablations-49-training-v1",
    },
}
RANDOM_SEEDS = (2211, 2212, 2213)
PERCENTAGES = (50, 75)


def random_core_order(image_id, setting, seed, k):
    """A stable image-local permutation, shared by tasks and nested core sizes."""
    entropy = [int(seed), *f"{setting}:{image_id}".encode("utf-8")]
    return np.random.default_rng(np.random.SeedSequence(entropy)).permutation(k)


def protected_selection(conditioned_scores, protected, k):
    """Retain protected positions; fill from the conditional rank without repeats."""
    protected = np.asarray(protected, dtype=int)
    order = np.argsort(-np.asarray(conditioned_scores), kind="stable")
    fill = order[~np.isin(order, protected)][:k - len(protected)]
    return np.sort(np.concatenate((protected, fill))).astype(np.int16)


def build_masks(conditioned_scores, wide_scores, image_indices, image_ids, k,
                setting, random_seeds=RANDOM_SEEDS):
    """The same wide core is used for all candidate tasks of a given image."""
    wide_rank = np.argsort(-wide_scores, axis=1, kind="stable")[:, :k]
    variants = ["conditioned", "wide"]
    masks = [np.sort(np.argsort(-conditioned_scores, axis=1, kind="stable")[:, :k], axis=1),
             np.sort(wide_rank[image_indices], axis=1)]
    for percentage in PERCENTAGES:
        q = k * percentage // 100
        variants.append(f"top_{percentage}")
        masks.append(np.stack([protected_selection(score, wide_rank[i, :q], k)
                               for score, i in zip(conditioned_scores, image_indices)]))
        for seed in random_seeds:
            cores = [rank[random_core_order(str(image_ids[i]), setting, seed, k)[:q]]
                     for i, rank in enumerate(wide_rank)]
            variants.append(f"random_{percentage}_{seed}")
            masks.append(np.stack([protected_selection(score, cores[i], k)
                                   for score, i in zip(conditioned_scores, image_indices)]))
    return variants, np.stack(masks).astype(np.int16)


def unique_image_masks(image_indices, masks):
    """Identical image/mask pairs share one suffix result across all variants."""
    indices = np.broadcast_to(image_indices, masks.shape[:2])
    keys = np.concatenate((indices[..., None], masks), axis=-1).reshape(-1, masks.shape[-1] + 1)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    return unique, inverse.reshape(masks.shape[:2])


def padded_batches(count, batch_size):
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        rows = np.arange(start, end)
        yield start, end, np.pad(rows, (0, batch_size - len(rows)), mode="edge")


def run(config, conditioned_dir, wide_dir, output, setting, random_seeds=RANDOM_SEEDS):
    import clip
    import torch

    from token_compression.pruning import SingleStagePruner
    from token_compression.selector import shared_descriptor

    started, timer = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(config["cache_dir"])
    with np.load(cache / "val/evidence.npz") as archive:
        ids, labels, text = [archive[key].copy() for key in ("image_ids", "labels", "text_features")]
        full_scores = archive["full_scores"].copy()
    train_tasks, val_tasks = [read_tasks(cache / f"{split}_tasks.csv") for split in ("train", "val")]
    examples = [r for r in evaluation_examples(ids, labels, train_tasks, val_tasks)
                if r["task_split"] == "unseen"]
    write_rows(output / "examples.csv", examples)
    ix = np.asarray([r["image_index"] for r in examples])
    pairs = np.asarray([[r["class_a"], r["class_b"]] for r in examples])
    np.savez_compressed(output / "evidence.npz", image_ids=ids, labels=labels,
                        text_features=text, full_scores=full_scores)
    prefixes = np.load(cache / "val/prefix_tokens.npy", mmap_mode="r")
    device, batch_size, k = config["device"], config["image_batch_size"], config["keep_patch_tokens"]
    backbone, _ = clip.load(config["model"], device=device, jit=False, download_root=config["model_cache"])
    backbone.eval().requires_grad_(False)
    pruner = SingleStagePruner(backbone.visual, config["prune_after"])
    text_tensor = torch.as_tensor(text, device=device)
    prior = shared_descriptor(text_tensor, torch.tensor(
        [[t["class_a"], t["class_b"]] for t in train_tasks], device=device))

    def sync():
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)

    def token_batch(indices):
        return torch.as_tensor(np.asarray(prefixes[indices]), device=device)

    records = []
    with torch.inference_mode():
        for seed in config["training_seeds"]:
            paths = {"conditioned": Path(conditioned_dir) / "full" / f"seed_{seed}/conditioned.pt",
                     "image_only_wide": Path(wide_dir) / "full" / f"seed_{seed}/image_only_wide.pt"}
            logits, parameter_counts = {}, {}
            sync()
            before = time.perf_counter()
            for method, path in paths.items():
                checkpoint = torch.load(path, map_location=device, weights_only=True)
                model = selector_from_checkpoint(checkpoint, prior, method, prefixes.shape[-1], device)
                parameter_counts[method] = sum(p.numel() for p in model.parameters())
                indices = ix if method == "conditioned" else np.arange(len(ids))
                logits[method] = np.empty((len(indices), pruner.num_patches), np.float32)
                for start, end, rows in padded_batches(len(indices), batch_size):
                    candidate_pairs = pairs[rows] if method == "conditioned" else np.tile(pairs[0], (len(rows), 1))
                    result = model(token_batch(indices[rows]), text_tensor[torch.as_tensor(candidate_pairs, device=device)])
                    logits[method][start:end] = result[:end - start].float().cpu().numpy()
                del model
            sync()
            selector_seconds = time.perf_counter() - before
            variants, masks = build_masks(logits["conditioned"], logits["image_only_wide"],
                                           ix, ids, k, setting, random_seeds)
            unique, inverse = unique_image_masks(ix, masks)
            features = np.empty((len(unique), text.shape[1]), np.float32)
            sync()
            before = time.perf_counter()
            for start, end, rows in padded_batches(len(unique), batch_size):
                keys = unique[rows]
                result = pruner.finish(token_batch(keys[:, 0]),
                                       torch.as_tensor(keys[:, 1:], dtype=torch.long, device=device)).float()
                result = result / result.norm(dim=-1, keepdim=True)
                features[start:end] = result[:end - start].cpu().numpy()
            sync()
            finish_seconds = time.perf_counter() - before
            # Compute each unique vector's scores once as well as its suffix.
            scores = (features @ text.T)[inverse]
            assert np.isfinite(scores).all()
            np.savez_compressed(output / f"seed_{seed}.npz", seed=seed, variants=np.asarray(variants),
                                cosine_scores=scores, patch_indices=masks, unique_keys=unique,
                                unique_features=features, feature_index=inverse,
                                conditioned_logits=logits["conditioned"], wide_logits=logits["image_only_wide"])
            record = {"seed": seed, "checkpoint_paths": {m: str(p.resolve()) for m, p in paths.items()},
                      "parameters": parameter_counts, "selector_seconds": selector_seconds,
                      "suffix_seconds": finish_seconds, "unique_image_masks": len(unique),
                      "raw_image_masks": int(np.prod(masks.shape[:2]))}
            records.append(record)
            print(json.dumps({"setting": setting, **record}), flush=True)
    summary = {"setting": setting, "started_utc": started, "elapsed_seconds": time.perf_counter() - timer,
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "config": config, "cache_dir": str(cache.resolve()), "training_seeds": config["training_seeds"],
               "random_core_seeds": list(random_seeds), "variants": variants,
               "protected_counts": {str(p): k * p // 100 for p in (0, 50, 75, 100)},
               "scope": "All existing validation images on unseen validation class pairs; no reserved test data.",
               "image_count": len(set(ix)), "image_task_count": len(examples),
               "task_count": len({r["task_id"] for r in examples}), "runs": records,
               "implementation": "Frozen selectors. Fixed spatially sorted k. Rank ties use patch index. Random cores are nested image-local subsets of the wide top-k. Identical image/mask pairs reuse one fixed-batch suffix result.",
               "cost_scope": "Cached-prefix replay only; this two-selector diagnostic does not establish deployment speed."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=SETTINGS, required=True)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = SETTINGS[args.setting]
    config = json.loads(Path(spec["config"]).read_text())
    if args.device:
        config["device"] = args.device
    run(config, spec["conditioned_dir"], spec["wide_dir"], args.output, args.setting)


if __name__ == "__main__":
    main()
