"""Generate feature-preservation patch rankings from frozen CLIP prefixes."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import clip
import numpy as np
import torch

from token_compression.pruning import SingleStagePruner
from token_compression.selection import deletion_windows, window_to_patch_scores


@torch.inference_mode()
def prepare(config, output, limit=None):
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config["device"])
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    start = time.perf_counter()
    model, _ = clip.load(config["model"], device=device, jit=False,
                         download_root=config["model_cache"])
    model.eval().requires_grad_(False)
    pruner = SingleStagePruner(model.visual, config["prune_after"])
    n_patches = pruner.num_patches
    windows = deletion_windows(int(n_patches ** .5), config["window_size"])
    complements = np.stack([np.setdiff1d(np.arange(n_patches), window) for window in windows])
    batch_size = config["teacher_batch_size"]
    retained = torch.as_tensor(complements, device=device, dtype=torch.long)
    summary = {"config": config, "code_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
        "teacher": "Mean cosine drop over stride-one deletion windows covering each patch",
        "window_count": len(windows), "splits": {}}
    for split in ("train", "val"):
        split_start = time.perf_counter()
        source = Path(config["feature_cache_dir"]) / split
        folder = output / split
        folder.mkdir(exist_ok=True)
        with np.load(source / "evidence.npz") as saved:
            image_ids = saved["image_ids"].copy()
            full = saved["full_features"].copy()
            text = saved["text_features"].copy()
        if limit is not None:
            image_ids, full = image_ids[:limit], full[:limit]
        prefixes = np.load(source / "prefix_tokens.npy", mmap_mode="r")
        tokens = torch.as_tensor(np.array(prefixes[:len(image_ids)]), device=device)
        full_tensor = torch.as_tensor(full, device=device)
        text_tensor = torch.as_tensor(text, device=device)
        window_drop = np.empty((len(image_ids), len(windows)), np.float32)
        importance = np.empty((len(image_ids), n_patches), np.float32)
        # Class scores support later task-preservation analysis without another
        # round of window deletion. They are not used by this shared teacher.
        deleted_scores = np.lib.format.open_memmap(
            folder / "deleted_class_scores.npy", mode="w+", dtype=np.float32,
            shape=(len(image_ids), len(windows), len(text)))
        for image_index in range(len(image_ids)):
            parts = []
            for begin in range(0, len(windows), batch_size):
                indices = retained[begin:begin + batch_size]
                valid = len(indices)
                if valid < batch_size:
                    indices = torch.cat((indices, indices[-1:].expand(batch_size-valid, -1)))
                values = pruner.finish(tokens[image_index:image_index+1].expand(len(indices), -1, -1), indices).float()
                parts.append(torch.nn.functional.normalize(values, dim=-1)[:valid])
            features = torch.cat(parts)
            drop = (1 - (features * full_tensor[image_index]).sum(dim=-1)).cpu().numpy()
            window_drop[image_index] = drop
            importance[image_index] = window_to_patch_scores(drop, windows, n_patches)
            deleted_scores[image_index] = (features @ text_tensor.T).cpu().numpy()
            if (image_index + 1) % 128 == 0 or image_index + 1 == len(image_ids):
                print(json.dumps({"split": split, "images": image_index+1,
                                  "total_images": len(image_ids),
                                  "seconds": time.perf_counter()-split_start}), flush=True)
        deleted_scores.flush()
        rankings = np.argsort(-importance, axis=-1, kind="stable")
        np.savez_compressed(folder / "ranking.npz", image_ids=image_ids,
                            importance=importance, rankings=rankings,
                            windows=windows, window_cosine_drop=window_drop)
        torch.cuda.synchronize(device)
        summary["splits"][split] = {"images": len(image_ids), "logical_suffix_examples": len(image_ids)*len(windows),
                                    "seconds": time.perf_counter()-split_start,
                                    "mean_window_cosine_drop": float(window_drop.mean())}
        (output / "preparation_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        del tokens, prefixes, deleted_scores
    summary["total_seconds"] = time.perf_counter()-start
    summary["gpu"] = torch.cuda.get_device_name(device)
    (output / "preparation_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    prepare(config, args.output or Path(config["preservation_dir"]), args.limit)
