"""Evaluate matched selectors and fixed hybrid rules at one pruning depth."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from evaluate_core_protection import SETTINGS, padded_batches, unique_image_masks
from evaluate_hybrid_specificity import hybrid_mask, seed_pairs
from evaluate_selector_scale import evaluation_examples, selector_from_checkpoint
from evaluate_selectors import read_tasks, write_rows

METHODS = ("conditioned", "shared_recipient", "shared_donor", "cu_half", "uu_half")


def pair_masks(c, u, donor, image_indices, k):
    """Apply the previously fixed half-exclusive rule to both recipients."""
    masks = np.empty((5, len(image_indices), k), np.int16)
    changes = np.zeros((2, len(image_indices)), np.int16)
    for row, image in enumerate(image_indices):
        for mi, scores in enumerate((c[row], u[image], donor[image])):
            masks[mi, row] = np.sort(np.argsort(-scores, kind="stable")[:k])
        for offset, scores in enumerate((c[row], u[image])):
            mask, _, count, _ = hybrid_mask(scores, donor[image], k)
            masks[3 + offset, row], changes[offset, row] = mask, count
    return masks, changes


def import_middle(dataset, output):
    """Reuse the unchanged block-6 evaluated panel, without new CLIP forwards."""
    setting = f"{dataset}_b16_49"
    source = Path("runs/hybrid-specificity-v1") / setting
    output.mkdir(parents=True, exist_ok=True)
    prior = json.loads((source / "protocol.json").read_text())
    assert prior["config"]["prune_after"] == 6
    assert prior["config"]["keep_patch_tokens"] == 49
    for name in ("examples.csv", "evidence.npz"):
        shutil.copyfile(source / name, output / name)
    for s, t in prior["primary_seed_pairs"]:
        with np.load(source / f"pair_{s}_{t}.npz") as saved:
            indices = [saved["methods"].tolist().index(m) for m in METHODS]
            np.savez_compressed(output / f"pair_{s}_{t}.npz", methods=METHODS,
                recipient_seed=s, donor_seed=t,
                cosine_scores=saved["cosine_scores"][indices],
                patch_indices=saved["patch_indices"][indices],
                changed_tokens=saved["changed_tokens_vs_recipient"][[3, 4]])
    prior.update(setting=f"{dataset}_d6", source=str(source.resolve()),
                 reuse_original_predictions=True, methods=list(METHODS),
                 import_git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    (output / "protocol.json").write_text(json.dumps(prior, indent=2) + "\n")
    (output / "execution.json").write_text(json.dumps(dict(new_selector_fits=0,
        new_suffix_forwards=0, original_costs=str(source / "execution.json")), indent=2) + "\n")


def run(config, models_dir, output, dataset):
    import clip
    import torch
    from token_compression.pruning import SingleStagePruner
    from token_compression.selector import shared_descriptor

    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(config["cache_dir"])
    prep = json.loads((cache / "prep_summary.json").read_text())
    for field in ("model", "prune_after", "keep_patch_tokens", "probe_count", "mask_seed"):
        assert prep["config"][field] == config[field], field
    with np.load(cache / "val/evidence.npz") as saved:
        evidence = {key: saved[key].copy() for key in
                    ("image_ids", "labels", "text_features", "full_scores")}
    ids, labels, text = [evidence[key] for key in ("image_ids", "labels", "text_features")]
    tasks = {split: read_tasks(cache / f"{split}_tasks.csv") for split in ("train", "val")}
    rows = [r for r in evaluation_examples(ids, labels, tasks["train"], tasks["val"])
            if r["task_split"] == "unseen"]
    write_rows(output / "examples.csv", rows)
    np.savez_compressed(output / "evidence.npz", **evidence)
    ix = np.array([r["image_index"] for r in rows])
    pairs = np.array([[r["class_a"], r["class_b"]] for r in rows])
    prefixes = np.load(cache / "val/prefix_tokens.npy", mmap_mode="r")
    device, batch, k = config["device"], config["image_batch_size"], config["keep_patch_tokens"]
    model, _ = clip.load(config["model"], device=device, jit=False, download_root=config["model_cache"])
    model.eval().requires_grad_(False)
    pruner = SingleStagePruner(model.visual, config["prune_after"])
    text_tensor = torch.as_tensor(text, device=device)
    prior = shared_descriptor(text_tensor, torch.tensor(
        [[r["class_a"], r["class_b"]] for r in tasks["train"]], device=device))
    seeds = config["training_seeds"]
    protocol = dict(setting=f"{dataset}_d{config['prune_after']}", config=config,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        training_seeds=seeds, primary_seed_pairs=seed_pairs(seeds), methods=list(METHODS),
        final_test_used=False, reuse_original_predictions=False,
        training_models_dir=str(models_dir.resolve()),
        source_scope="Existing validation images and validation candidate pairs only.")
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    logits, records, feature_cache = {}, [], {}

    def token_batch(indices):
        return torch.as_tensor(np.asarray(prefixes[indices]), device=device)

    with torch.inference_mode():
        for seed in seeds:
            values = {}
            for arm in ("conditioned", "image_only_wide"):
                path = models_dir / "full" / f"seed_{seed}" / f"{arm}.pt"
                checkpoint = torch.load(path, map_location=device, weights_only=True)
                for field in ("model", "prune_after", "keep_patch_tokens", "training_steps"):
                    assert checkpoint["config"][field] == config[field], (arm, field)
                assert checkpoint["seed"] == seed
                selector = selector_from_checkpoint(checkpoint, prior, arm, prefixes.shape[-1], device)
                image_rows = ix if arm == "conditioned" else np.arange(len(ids))
                scores = np.empty((len(image_rows), pruner.num_patches), np.float32)
                for start, end, positions in padded_batches(len(image_rows), batch):
                    candidates = pairs[positions] if arm == "conditioned" else np.tile(pairs[0], (len(positions), 1))
                    result = selector(token_batch(image_rows[positions]), text_tensor[
                        torch.as_tensor(candidates, device=device)])
                    scores[start:end] = result[:end-start].cpu().numpy()
                values[arm] = scores
                del selector
            logits[seed] = values
            np.savez_compressed(output / f"logits_{seed}.npz", **values)
        torch.cuda.synchronize(device)
        selector_seconds = time.perf_counter() - started
        for s, t in seed_pairs(seeds):
            masks, changed = pair_masks(logits[s]["conditioned"], logits[s]["image_only_wide"],
                                        logits[t]["image_only_wide"], ix, k)
            keys, inverse = unique_image_masks(ix, masks)
            keys = keys.astype(np.int64)
            features = np.empty((len(keys), text.shape[1]), np.float32)
            pending = []
            for i, key in enumerate(keys):
                cached = feature_cache.get(key.tobytes())
                if cached is None:
                    pending.append(i)
                else:
                    features[i] = cached
            pending = np.asarray(pending, dtype=int)
            torch.cuda.synchronize(device)
            before = time.perf_counter()
            for start, end, positions in padded_batches(len(pending), batch):
                current = keys[pending[positions]]
                features_t = pruner.finish(token_batch(current[:, 0]), torch.as_tensor(
                    current[:, 1:], dtype=torch.long, device=device)).float()
                features[pending[start:end]] = (features_t / features_t.norm(dim=-1, keepdim=True))[:end-start].cpu().numpy()
            torch.cuda.synchronize(device)
            for i in pending:
                feature_cache[keys[i].tobytes()] = features[i].copy()
            scores = (features @ text.T)[inverse]
            assert np.isfinite(scores).all()
            np.savez_compressed(output / f"pair_{s}_{t}.npz", methods=METHODS,
                recipient_seed=s, donor_seed=t, cosine_scores=scores,
                patch_indices=masks, changed_tokens=changed)
            record = dict(recipient_seed=s, donor_seed=t, unique_masks=len(keys),
                          new_masks=len(pending), suffix_seconds=time.perf_counter()-before)
            records.append(record)
            print(json.dumps(record), flush=True)
    (output / "execution.json").write_text(json.dumps(dict(
        total_seconds=time.perf_counter()-started, setup_and_selector_seconds=selector_seconds,
        pairs=records, cache_scope="Current dataset/depth only; never shared across depths."), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("pets", "flowers"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--import-middle", action="store_true")
    args = parser.parse_args()
    if args.import_middle:
        import_middle(args.dataset, args.output)
    else:
        config = json.loads(args.config.read_text())
        if args.device:
            config["device"] = args.device
        run(config, args.models, args.output, args.dataset)
