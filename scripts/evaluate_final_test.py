"""Evaluate fixed selectors on the reserved image and candidate-pair split."""

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from evaluate_selectors import binary_outcomes, read_tasks, write_rows
from evaluate_selector_scale import evaluation_examples, random_rankings, selector_from_checkpoint


def selected_positions(values, keep):
    return np.sort(np.argsort(-values, axis=1, kind="stable")[:, :keep], axis=1)


def run(config, setting):
    import clip
    import torch

    from token_compression.patch_ranking import PatchRankingPredictor
    from token_compression.pruning import SingleStagePruner
    from token_compression.selector import shared_descriptor

    begin = time.perf_counter()
    output = Path(setting["eval_dir"])
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(setting["cache_dir"])
    image_split = config.get("image_split", "test")
    task_split = config.get("task_split", "test")
    with np.load(cache / image_split / "evidence.npz") as saved:
        evidence = {key: saved[key].copy() for key in saved.files}
    ids, labels, text = (evidence[key] for key in ("image_ids", "labels", "text_features"))
    train_tasks = read_tasks(cache / "train_tasks.csv")
    test_tasks = read_tasks(cache / f"{task_split}_tasks.csv")
    # Pass no seen tasks to avoid materializing their much larger example panel.
    train_pairs = {tuple(sorted((r["class_a"], r["class_b"]))) for r in train_tasks}
    assert train_pairs.isdisjoint(tuple(sorted((r["class_a"], r["class_b"]))) for r in test_tasks)
    examples = evaluation_examples(ids, labels, [], test_tasks)
    write_rows(output / "examples.csv", examples)
    np.savez_compressed(output / "evidence.npz", **evidence)
    ix = np.asarray([r["image_index"] for r in examples])
    pairs = np.asarray([[r["class_a"], r["class_b"]] for r in examples])
    y = labels[ix]
    prefixes = np.load(cache / image_split / "prefix_tokens.npy", mmap_mode="r")
    device = torch.device(config["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
    backbone, _ = clip.load(setting["model"], device=device, jit=False,
                            download_root=config["model_cache"])
    backbone.eval().requires_grad_(False)
    pruner = SingleStagePruner(backbone.visual, setting["prune_after"])
    text_tensor = torch.as_tensor(text, device=device)
    training_pairs = torch.tensor([[r["class_a"], r["class_b"]] for r in train_tasks], device=device)
    descriptor = shared_descriptor(text_tensor, training_pairs)
    batch_size, keep = config["image_batch_size"], setting["keep_patch_tokens"]
    jobs = [("full_model", -1, None), ("attention", -1, None)]
    jobs += [("random", seed, None) for seed in config["random_seeds"]]
    jobs += [(method, seed, template.format(seed=seed))
             for method, template in setting["checkpoints"].items() for seed in config["seeds"]]

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    costs = []
    with torch.inference_mode():
        for method, seed, path in jobs:
            before = time.perf_counter()
            model = None
            if path:
                checkpoint = torch.load(path, map_location=device, weights_only=True)
                if method == "patch_ranking":
                    model = PatchRankingPredictor().to(device)
                    model.load_state_dict(checkpoint["model_state"])
                    model.eval().requires_grad_(False)
                else:
                    model = selector_from_checkpoint(checkpoint, descriptor, method, prefixes.shape[-1], device)
            unique = method != "conditioned"
            indices = np.arange(len(ids)) if unique else ix
            count = len(indices)
            features = np.empty((count, text.shape[1]), np.float32)
            logits = np.zeros((count, pruner.num_patches), np.float32)
            positions = np.full((count, pruner.num_patches), -1, np.int16)
            random_values = (random_rankings(len(ids), pruner.num_patches, seed)
                             if method == "random" else None)
            selector_seconds = suffix_seconds = 0.0
            if method == "full_model":
                features[:] = evidence["full_features"]
                positions[:] = np.arange(pruner.num_patches)
            else:
                for start in range(0, count, batch_size):
                    take = np.arange(start, min(start + batch_size, count))
                    valid = len(take)
                    if valid < batch_size:
                        take = np.pad(take, (0, batch_size - valid), mode="edge")
                    image_indices = indices[take]
                    tokens = torch.from_numpy(np.array(prefixes[image_indices])).to(device)
                    sync()
                    tick = time.perf_counter()
                    if method == "patch_ranking":
                        values = model(tokens).cpu().numpy()
                    elif model is not None:
                        candidate_pairs = np.tile(pairs[0], (len(take), 1)) if unique else pairs[take]
                        candidate_text = text_tensor[torch.as_tensor(candidate_pairs, device=device)]
                        values = model(tokens.float(), candidate_text).cpu().numpy()
                    elif method == "attention":
                        values = evidence["attention_scores"][image_indices]
                    else:
                        values = random_values[image_indices]
                    selected = selected_positions(values, keep)
                    retained = torch.as_tensor(selected, dtype=torch.long, device=device)
                    sync()
                    selector_seconds += time.perf_counter() - tick
                    tick = time.perf_counter()
                    z = pruner.finish(tokens, retained).float()
                    z = (z / z.norm(dim=-1, keepdim=True)).cpu().numpy()
                    sync()
                    suffix_seconds += time.perf_counter() - tick
                    features[start:start + valid] = z[:valid]
                    logits[start:start + valid] = values[:valid]
                    positions[start:start + valid, :keep] = selected[:valid]
            if unique:
                features, logits, positions = features[ix], logits[ix], positions[ix]
            scores = features @ text.T
            margin, correct = binary_outcomes(scores, y, pairs, text)
            folder = output / method / f"seed_{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / "predictions.npz", image_features=features,
                                cosine_scores=scores, patch_indices=positions,
                                selection_logits=logits, margin=margin, correct=correct,
                                text_features=text)
            row = {"method": method, "seed": seed, "checkpoint": path,
                   "parameter_count": sum(p.numel() for p in model.parameters()) if model else 0,
                   "selector_seconds": selector_seconds, "suffix_seconds": suffix_seconds,
                   "method_wall_seconds": time.perf_counter() - before,
                   "unique_image_reuse": unique, "logical_suffix_examples": count if method != "full_model" else 0}
            costs.append(row)
            print(json.dumps({"setting": setting["id"], **row}), flush=True)
    summary = {"setting": setting, "image_split": image_split, "task_split": task_split,
               "image_count": len(ids), "evaluated_image_count": len(set(ids[ix])),
               "task_count": len(test_tasks), "example_count": len(examples), "runs": costs,
               "seeds": config["seeds"], "batch_size": batch_size,
               "aggregation": "Equal candidate tasks and class sides; average trained seeds.",
               "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "total_wall_seconds": time.perf_counter() - begin,
               "cost_scope": "Cached-prefix evaluation and output; not deployment latency.",
               "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}
    (output / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--setting", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    setting = next(s for s in config["settings"] if s["id"] == args.setting)
    run(config, setting)
