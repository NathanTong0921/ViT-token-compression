"""Evaluate preservation rankings and their predictors on frozen CLIP features."""

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from evaluate_selectors import binary_outcomes, read_tasks, write_rows
from evaluate_selector_scale import evaluation_examples, random_rankings


def selected_positions(scores, k):
    return np.sort(np.argsort(-scores, axis=1, kind="stable")[:, :k], axis=1)


def set_overlap(a, b, patch_count=196):
    mask = np.zeros((len(a), patch_count), dtype=bool)
    np.put_along_axis(mask, a, True, axis=1)
    return np.take_along_axis(mask, b, axis=1).mean(axis=1)


def run(config, models_dir, output):
    import clip
    import torch
    from token_compression.pruning import SingleStagePruner

    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(config["feature_cache_dir"])
    with np.load(cache / "val/evidence.npz") as archive:
        evidence = {key: archive[key] for key in archive.files}
    ids, labels, text = (evidence[key] for key in ("image_ids", "labels", "text_features"))
    with np.load(Path(config["preservation_dir"]) / "val/ranking.npz") as archive:
        np.testing.assert_array_equal(archive["image_ids"], ids)
        teacher_ranks = archive["rankings"].copy()
    k = config["keep_patch_tokens"]
    teacher_mask = np.sort(teacher_ranks[:, :k], axis=1)
    tasks = [read_tasks(cache / f"{split}_tasks.csv") for split in ("train", "val")]
    examples = [row for row in evaluation_examples(ids, labels, *tasks)
                if row["task_split"] == "unseen"]
    write_rows(output / "examples.csv", examples)
    ix = np.array([row["image_index"] for row in examples])
    y = labels[ix]
    pairs = np.array([[row["class_a"], row["class_b"]] for row in examples])
    prefixes = np.load(cache / "val/prefix_tokens.npy", mmap_mode="r")
    device = torch.device(config["device"])
    backbone, _ = clip.load(config["model"], device=device, jit=False,
                            download_root=config["model_cache"])
    backbone.eval().requires_grad_(False)
    pruner = SingleStagePruner(backbone.visual, config["prune_after"])
    batch_size = config.get("image_batch_size", 64)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def suffix(mask):
        values = []
        for start in range(0, len(ids), batch_size):
            take = np.arange(start, min(start + batch_size, len(ids)))
            valid = len(take)
            if valid < batch_size:
                take = np.pad(take, (0, batch_size - valid), mode="edge")
            tokens = torch.from_numpy(np.array(prefixes[take])).to(device)
            indices = torch.as_tensor(mask[take], device=device, dtype=torch.long)
            feature = pruner.finish(tokens, indices).float()
            values.append((feature / feature.norm(dim=-1, keepdim=True))[:valid].cpu().numpy())
        return np.concatenate(values)

    jobs = [("full_model", -1, np.tile(np.arange(pruner.num_patches), (len(ids), 1)), None),
            ("preservation_teacher", -1, teacher_mask, teacher_ranks),
            ("attention", -1, selected_positions(evidence["attention_scores"], k), None)]
    for seed in config.get("random_seeds", [2055, 2056, 2057]):
        value = random_rankings(len(ids), pruner.num_patches, seed)
        jobs.append(("random", seed, selected_positions(value, k), None))
    for seed in config["seeds"]:
        path = models_dir / f"seed_{seed}/predictions_val.npz"
        with np.load(path) as archive:
            np.testing.assert_array_equal(archive["image_ids"], ids)
            ranks = np.argsort(-archive["predicted_scores"], axis=1, kind="stable")
            masks = archive["selected_indices"].copy()
            np.testing.assert_array_equal(np.sort(ranks[:, :k], axis=1), masks)
        jobs.append(("patch_ranking", seed, masks, ranks))

    records, run_metadata, features, masks, all_scores, overlaps = [], [], [], [], [], []
    with torch.inference_mode():
        for run_index, (method, seed, selected, ranking) in enumerate(jobs):
            begin = time.perf_counter()
            feature = evidence["full_features"] if method == "full_model" else suffix(selected)
            scores = feature @ text.T
            margins, correct = binary_outcomes(scores[ix], y, pairs, text)
            padded = np.full((len(ids), pruner.num_patches), -1, dtype=np.int16)
            padded[:, :selected.shape[1]] = selected
            overlap = (set_overlap(selected, teacher_mask, pruner.num_patches)
                       if method != "full_model" else np.full(len(ids), np.nan))
            row = {"method": method, "seed": seed, "run_index": run_index,
                   "suffix_seconds": time.perf_counter() - begin,
                   "multiclass_accuracy": float(np.mean(scores.argmax(1) == labels))}
            if method != "full_model":
                row["teacher_overlap_at_k"] = float(overlap.mean())
            if ranking is not None:
                row["teacher_overlap_at_100"] = float(set_overlap(
                    ranking[:, :100], teacher_ranks[:, :100], pruner.num_patches).mean())
                row["teacher_overlap_bottom_50"] = float(set_overlap(
                    ranking[:, -50:], teacher_ranks[:, -50:], pruner.num_patches).mean())
            for j, example in enumerate(examples):
                records.append({**example, "run_index": run_index, "method": method,
                                "seed": seed, "keep_patch_tokens": selected.shape[1],
                                "correct": int(correct[j]), "margin": float(margins[j])})
            run_metadata.append(row)
            features.append(feature)
            masks.append(padded)
            all_scores.append(scores)
            overlaps.append(overlap)
            print(json.dumps(row), flush=True)
    write_rows(output / "predictions.csv", records)
    np.savez_compressed(output / "predictions.npz", image_ids=ids, labels=labels,
                        text_features=text, image_features=np.stack(features),
                        cosine_scores=np.stack(all_scores), patch_indices=np.stack(masks),
                        teacher_overlap=np.stack(overlaps),
                        method=np.array([r["method"] for r in run_metadata]),
                        seed=np.array([r["seed"] for r in run_metadata]))
    summary = {"config": config, "source_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(), "runs": run_metadata,
        "image_count": len(ids), "binary_image_count": len(set(ids[ix])),
        "task_count": len({r["task_id"] for r in examples}), "image_task_count": len(examples),
        "models_dir": str(models_dir), "total_wall_seconds": time.perf_counter() - started,
        "cost_scope": "Cached-prefix suffix evaluation and file output; not deployment latency.",
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}
    (output / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.models_dir, args.output)
