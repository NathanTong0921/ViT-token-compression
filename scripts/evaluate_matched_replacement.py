"""Compare shared-ranking additions with exactly matched token replacements."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from analyze_selector_scale import read_rows
from evaluate_core_protection import SETTINGS, padded_batches, unique_image_masks


RANDOM_SEEDS = (2211, 2212, 2213)
PERCENTAGES = (0, 50, 75, 100)


def random_addition_order(setting, image_id, task_id, seed, count):
    """A fixed image/task order, nested across replacement fractions."""
    entropy = [int(seed), *f"matched:{setting}:{image_id}:{task_id}".encode("utf-8")]
    return np.random.default_rng(np.random.SeedSequence(entropy)).permutation(count)


def build_interventions(conditioned_scores, wide_scores, image_indices, image_ids,
                        task_ids, k, setting, random_seeds=RANDOM_SEEDS):
    c_rank = np.argsort(-conditioned_scores, axis=1, kind="stable")[:, :k]
    u_rank = np.argsort(-wide_scores, axis=1, kind="stable")[:, :k]
    variants = ["conditioned", "wide"]
    variants += [name for p in (50, 75) for name in
                 [f"top_{p}", *[f"random_{p}_{s}" for s in random_seeds]]]
    masks = np.empty((len(variants), len(image_indices), k), dtype=np.int16)
    added = np.full_like(masks, -1)
    removed = np.full((len(PERCENTAGES), len(image_indices), k), -1, dtype=np.int16)
    counts = np.zeros((len(PERCENTAGES), len(image_indices)), dtype=np.int16)
    for row, (c, image_index) in enumerate(zip(c_rank, image_indices)):
        u = u_rank[image_index]
        candidates = u[~np.isin(u, c)]
        removable = c[~np.isin(c, u)]
        assert len(candidates) == len(removable)
        masks[0, row], masks[1, row] = np.sort(c), np.sort(u)
        added[1, row, :len(candidates)] = candidates
        random_orders = {seed: random_addition_order(setting, image_ids[image_index], task_ids[row],
                                                     seed, len(candidates)) for seed in random_seeds}
        for pi, percentage in enumerate(PERCENTAGES):
            d = len(candidates) * percentage // 100
            counts[pi, row] = d
            # Removable follows descending C rank; drop its same lowest-ranked d.
            drop = removable[len(removable) - d:] if d else removable[:0]
            removed[pi, row, :d] = drop
            if percentage in (0, 100):
                continue
            retained = c[~np.isin(c, drop)]
            additions = {f"top_{percentage}": candidates[:d]}
            additions.update({f"random_{percentage}_{seed}": candidates[order[:d]]
                              for seed, order in random_orders.items()})
            for name, insert in additions.items():
                vi = variants.index(name)
                masks[vi, row] = np.sort(np.concatenate((retained, insert)))
                added[vi, row, :d] = insert
    return {"variants": np.asarray(variants), "patch_indices": masks,
            "replacement_counts": counts, "removed_indices": removed, "added_indices": added,
            "percentages": np.asarray(PERCENTAGES)}


def cached_feature_indices(keys, old_keys):
    """Locate exact image/mask matches in the previous frozen suffix replay."""
    lookup = {row.tobytes(): i for i, row in enumerate(old_keys.astype(keys.dtype, copy=False))}
    return np.asarray([lookup.get(row.tobytes(), -1) for row in keys])


def run(source, output, device=None):
    import clip
    import torch

    from token_compression.pruning import SingleStagePruner

    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads((source / "summary.json").read_text())
    config, setting = dict(original["config"]), original["setting"]
    if device is not None:
        config["device"] = device
    device, k, batch_size = config["device"], config["keep_patch_tokens"], config["image_batch_size"]
    started, timer = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    for name in ("examples.csv", "evidence.npz"):
        shutil.copyfile(source / name, output / name)
    base = read_rows(source / "examples.csv")
    ix = np.asarray([int(r["image_index"]) for r in base])
    task_ids = np.asarray([r["task_id"] for r in base])
    with np.load(source / "evidence.npz") as saved:
        image_ids, text = saved["image_ids"].copy(), saved["text_features"].copy()
    prefixes = np.load(Path(config["cache_dir"]) / "val/prefix_tokens.npy", mmap_mode="r")
    backbone, _ = clip.load(config["model"], device=device, jit=False, download_root=config["model_cache"])
    backbone.eval().requires_grad_(False)
    pruner = SingleStagePruner(backbone.visual, config["prune_after"])

    def sync():
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)

    records = []
    with torch.inference_mode():
        for seed in original["training_seeds"]:
            old_path = source / f"seed_{seed}.npz"
            with np.load(old_path) as saved:
                interventions = build_interventions(saved["conditioned_logits"], saved["wide_logits"],
                    ix, image_ids, task_ids, k, setting)
                keys, inverse = unique_image_masks(ix, interventions["patch_indices"])
                previous = cached_feature_indices(keys, saved["unique_keys"])
                features = np.empty((len(keys), text.shape[1]), np.float32)
                reused = previous >= 0
                features[reused] = saved["unique_features"][previous[reused]]
                original_variants = list(saved["variants"])
                for index, method in enumerate(("conditioned", "wide")):
                    np.testing.assert_array_equal(interventions["patch_indices"][index],
                                                  saved["patch_indices"][original_variants.index(method)])
            pending = np.flatnonzero(~reused)
            sync()
            before = time.perf_counter()
            for start, end, rows in padded_batches(len(pending), batch_size):
                indices = pending[rows]
                current = keys[indices]
                tokens = torch.as_tensor(np.asarray(prefixes[current[:, 0]]), device=device)
                selected = torch.as_tensor(current[:, 1:], dtype=torch.long, device=device)
                result = pruner.finish(tokens, selected).float()
                result = (result / result.norm(dim=-1, keepdim=True)).cpu().numpy()
                features[pending[start:end]] = result[:end - start]
            sync()
            seconds = time.perf_counter() - before
            scores = (features @ text.T)[inverse]
            assert np.isfinite(scores).all()
            np.savez_compressed(output / f"seed_{seed}.npz", seed=seed, **interventions,
                                cosine_scores=scores, unique_keys=keys, unique_features=features,
                                feature_index=inverse, previous_feature_index=previous)
            record = {"seed": seed, "source_predictions": str(old_path.resolve()),
                      "raw_image_masks": int(np.prod(interventions["patch_indices"].shape[:2])),
                      "unique_image_masks": len(keys), "reused_image_masks": int(reused.sum()),
                      "new_suffix_image_masks": len(pending), "new_suffix_seconds": seconds,
                      "endpoint_masks_exact": True}
            records.append(record)
            print(json.dumps({"setting": setting, **record}), flush=True)
    summary = {"setting": setting, "config": config, "source": str(source.resolve()),
               "source_git_commit": original["git_commit"],
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "started_utc": started, "elapsed_seconds": time.perf_counter() - timer,
               "training_seeds": original["training_seeds"], "random_addition_seeds": list(RANDOM_SEEDS),
               "variants": interventions["variants"].tolist(), "replacement_percentages": list(PERCENTAGES),
               "scope": "Existing validation images and unseen validation class pairs. Frozen saved rankings, no training or reserved test use.",
               "intervention": "Keep C intersection U. Replace floor(fraction*|U minus C|) tokens: remove the identical lowest-C-ranked positions in C minus U, then add highest-U-ranked or random positions in U minus C. Fixed total k. Random order is fixed per image/task and nested across fractions.",
               "runs": records, "cost_scope": "Cached-prefix diagnostic replay; reused exact image/mask features. No deployment latency measured."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=SETTINGS, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    run(args.source or Path("runs/core-protection-v1") / args.setting, args.output, args.device)
