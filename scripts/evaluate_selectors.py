"""Evaluate learned token selectors on seen and unseen class combinations."""

import argparse
import csv
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def read_tasks(path):
    with Path(path).open() as handle:
        rows = list(csv.DictReader(handle))
    return [{**row, "class_a": int(row["class_a"]),
             "class_b": int(row["class_b"])} for row in rows]


def evaluation_examples(image_ids, labels, train_tasks, val_tasks):
    """Use the same image panel for both task splits, with both classes present."""
    labels = np.asarray(labels)
    present = set(labels.tolist())
    rows = []
    seen_pairs = {(min(t["class_a"], t["class_b"]), max(t["class_a"], t["class_b"]))
                  for t in train_tasks}
    unseen_pairs = {(min(t["class_a"], t["class_b"]), max(t["class_a"], t["class_b"]))
                    for t in val_tasks}
    assert not seen_pairs & unseen_pairs
    for split, tasks in (("seen", train_tasks), ("unseen", val_tasks)):
        for task in sorted(tasks, key=lambda t: t["task_id"]):
            a, b = sorted((task["class_a"], task["class_b"]))
            if a not in present or b not in present:
                continue
            assert np.sum(labels == a) == np.sum(labels == b)
            for index in np.flatnonzero((labels == a) | (labels == b)):
                rows.append({"image_index": int(index), "image_id": str(image_ids[index]),
                             "class_id": int(labels[index]), "task_id": task["task_id"],
                             "class_a": a, "class_b": b, "task_split": split})
    return rows


def binary_outcomes(scores, labels, class_pairs, text_features):
    """Signed geometric margins and class-index-consistent tie handling."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    pairs = np.asarray(class_pairs)
    index = np.arange(len(scores))
    competitors = np.where(labels == pairs[:, 0], pairs[:, 1], pairs[:, 0])
    norms = np.linalg.norm(text_features[labels] - text_features[competitors], axis=1)
    margins = (scores[index, labels] - scores[index, competitors]) / norms
    predicted = np.where(scores[index, pairs[:, 0]] >= scores[index, pairs[:, 1]],
                         pairs[:, 0], pairs[:, 1])
    return margins, (predicted == labels).astype(np.int8)


def utility_correlations(logits, utilities):
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(utilities, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    valid = denominator > 1e-14
    values = np.zeros(len(x))
    values[valid] = np.sum(x[valid] * y[valid], axis=1) / denominator[valid]
    return values, valid


def write_rows(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(config, models_dir, output):
    import clip
    import torch

    from token_compression.pruning import SingleStagePruner
    from token_compression.selector import TokenSelector, shared_descriptor

    start_wall = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(config["cache_dir"])
    with np.load(cache / "val" / "evidence.npz") as evidence:
        image_ids, labels = evidence["image_ids"].copy(), evidence["labels"].copy()
        text = evidence["text_features"].copy()
        full_scores = evidence["full_scores"].copy()
        attention = evidence["attention_scores"].copy()
    with np.load(cache / "val" / "teacher.npz") as teacher:
        assert np.array_equal(teacher["image_ids"], image_ids)
        coefficients = teacher["coefficients"].copy()
    train_tasks = read_tasks(cache / "train_tasks.csv")
    val_tasks = read_tasks(cache / "val_tasks.csv")
    examples = evaluation_examples(image_ids, labels, train_tasks, val_tasks)
    write_rows(output / "examples.csv", examples)
    image_indices = np.array([row["image_index"] for row in examples])
    example_labels = labels[image_indices]
    pairs = np.array([[row["class_a"], row["class_b"]] for row in examples])
    competitors = np.where(example_labels == pairs[:, 0], pairs[:, 1], pairs[:, 0])
    norms = np.linalg.norm(text[example_labels] - text[competitors], axis=1)
    utilities = np.stack([(coefficients[i, :, y] - coefficients[i, :, c]) / n
                          for i, y, c, n in zip(image_indices, example_labels, competitors, norms)])
    baseline_scores = full_scores[image_indices]
    base_margin, base_correct = binary_outcomes(baseline_scores, example_labels, pairs, text)
    count, k = len(examples), config["keep_patch_tokens"]
    batch_size = config["image_batch_size"]
    device = torch.device(config["device"])
    backbone, _ = clip.load(config["model"], device=device, jit=False,
                            download_root=config["model_cache"])
    backbone.eval().requires_grad_(False)
    pruner = SingleStagePruner(backbone.visual, config["prune_after"])
    prefixes = torch.as_tensor(np.load(cache / "val" / "prefix_tokens.npy"), device=device)
    text_tensor = torch.as_tensor(text, device=device)
    train_pairs = torch.tensor([[row["class_a"], row["class_b"]] for row in train_tasks],
                               dtype=torch.long, device=device)
    descriptor = shared_descriptor(text_tensor, train_pairs)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    all_rows, all_scores, all_positions, all_logits, all_available = [], [], [], [], []
    costs = []
    jobs = [(-1, "full_model"), (-1, "attention"), (-1, "label_informed_teacher")]
    jobs += [(seed, method) for seed in config["training_seeds"]
             for method in ("conditioned", "image_only")]
    with torch.inference_mode():
        for seed, method in jobs:
            before_method = time.perf_counter()
            model = None
            checkpoint_path = None
            if seed != -1:
                checkpoint_path = models_dir / f"seed_{seed}" / f"{method}.pt"
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
                model = TokenSelector(descriptor, conditioned=method == "conditioned",
                                      hidden_dim=checkpoint["hidden_dim"], token_dim=prefixes.shape[-1]).to(device)
                model.load_state_dict(checkpoint["state_dict"])
                model.eval().requires_grad_(False)
            scores = baseline_scores.copy() if method == "full_model" else np.empty((count, len(text)), np.float32)
            retained_count = pruner.num_patches if method == "full_model" else k
            positions = np.full((count, pruner.num_patches), -1, dtype=np.int16)
            logits = np.zeros((count, pruner.num_patches), np.float64)
            available = np.full(count, method != "full_model", dtype=bool)
            selector_seconds, finish_seconds, physical_examples = 0.0, 0.0, 0
            if method == "full_model":
                positions[:] = np.arange(pruner.num_patches)
            else:
                for start in range(0, count, batch_size):
                    end = min(start + batch_size, count)
                    valid = end - start
                    example_idx = np.arange(start, end)
                    if valid < batch_size:
                        example_idx = np.concatenate([example_idx, np.repeat(example_idx[-1], batch_size - valid)])
                    tokens = prefixes[torch.as_tensor(image_indices[example_idx], device=device)]
                    if model is not None:
                        candidate_text = text_tensor[torch.as_tensor(pairs[example_idx], device=device)]
                        synchronize()
                        before = time.perf_counter()
                        # Candidate classes are sorted, and the target label is never a model input.
                        value = model(tokens.float(), candidate_text).cpu().numpy()
                        synchronize()
                        selector_seconds += time.perf_counter() - before
                    elif method == "attention":
                        value = attention[image_indices[example_idx]]
                    else:
                        value = utilities[example_idx]
                    selected = np.sort(np.argsort(-value, axis=1, kind="stable")[:, :k], axis=1)
                    retained = torch.as_tensor(selected, dtype=torch.long, device=device)
                    synchronize()
                    before = time.perf_counter()
                    features = pruner.finish(tokens, retained).float()
                    features = (features / features.norm(dim=-1, keepdim=True)).cpu().numpy()
                    synchronize()
                    finish_seconds += time.perf_counter() - before
                    scores[start:end] = (features @ text.T)[:valid]
                    positions[start:end, :k] = selected[:valid]
                    logits[start:end] = value[:valid]
                    physical_examples += len(example_idx)
            margins, correct = binary_outcomes(scores, example_labels, pairs, text)
            correlations, correlation_valid = utility_correlations(logits, utilities)
            correlation_valid &= available
            offset = len(all_rows)
            for index, example in enumerate(examples):
                all_rows.append({"row_id": offset + index, "seed": seed, "method": method,
                                 **example, "keep_patch_tokens": retained_count,
                                 "base_margin": float(base_margin[index]), "base_correct": int(base_correct[index]),
                                 "margin": float(margins[index]), "correct": int(correct[index]),
                                 "teacher_utility_correlation": float(correlations[index]) if correlation_valid[index] else "",
                                 "teacher_utility_correlation_valid": int(correlation_valid[index])})
            all_scores.append(scores)
            all_positions.append(positions)
            all_logits.append(logits)
            all_available.append(available)
            costs.append({"seed": seed, "method": method,
                          "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
                          "selector_seconds": selector_seconds,
                          "finish_and_normalize_seconds": finish_seconds,
                          "total_method_seconds": time.perf_counter() - before_method,
                          "logical_suffix_examples": 0 if method == "full_model" else count,
                          "physical_suffix_examples": physical_examples})
            print(f"{method} seed={seed}: " + ", ".join(
                f"{split} {correct[np.array([row['task_split'] == split for row in examples])].mean():.4f}"
                for split in ("seen", "unseen")), flush=True)
            del model
    write_rows(output / "predictions.csv", all_rows)
    np.savez_compressed(output / "predictions.npz",
                        row_id=np.arange(len(all_rows)),
                        seed=np.array([row["seed"] for row in all_rows]),
                        method=np.array([row["method"] for row in all_rows]),
                        task_split=np.array([row["task_split"] for row in all_rows]),
                        image_index=np.array([row["image_index"] for row in all_rows]),
                        image_ids=np.array([row["image_id"] for row in all_rows]),
                        labels=np.array([row["class_id"] for row in all_rows]),
                        task_ids=np.array([row["task_id"] for row in all_rows]),
                        class_pairs=np.array([[row["class_a"], row["class_b"]] for row in all_rows]),
                        keep_patch_tokens=np.array([row["keep_patch_tokens"] for row in all_rows]),
                        cosine_scores=np.concatenate(all_scores),
                        patch_indices=np.concatenate(all_positions),
                        selection_logits=np.concatenate(all_logits),
                        selection_logits_available=np.concatenate(all_available),
                        text_features=text)
    metrics = []
    for seed, method in jobs:
        for split in ("seen", "unseen"):
            rows = [row for row in all_rows if row["seed"] == seed and row["method"] == method and row["task_split"] == split]
            metrics.append({"seed": seed, "method": method, "task_split": split,
                            "image_task_examples": len(rows),
                            "task_count": len({row["task_id"] for row in rows}),
                            "image_count": len({row["image_id"] for row in rows}),
                            "accuracy": float(np.mean([row["correct"] for row in rows])),
                            "mean_margin": float(np.mean([row["margin"] for row in rows]))})
    summary = {"started_utc": started, "code_commit": source_commit, "config": config,
               "cache_dir": str(cache), "models_dir": str(models_dir),
               "image_count": len(image_ids), "examples_per_method": count,
               "evaluation": metrics,
               "cost": {"methods": costs, "total_wall_seconds": time.perf_counter() - start_wall,
                        "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0},
               "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
               "artifact_schema": {"row_alignment": "CSV row_id equals every per-row NPZ array index.",
                                   "patch_indices": "Original 0-based patch positions; unused columns are -1. Full-model rows retain all patches.",
                                   "selection_logits": "Raw ranking scores, with an availability flag. Full-model zero placeholders are not predictions.",
                                   "seed": "Shared baselines use -1 and are stored once, not repeated per training seed."},
               "scope": "Validation images only. Seen pairs appeared in selector training; unseen validation pairs did not. Official test images and pairs are unused. Label-informed teacher uses target labels only as an offline diagnostic.",
               "cost_scope": "Cached-prefix, fixed padded batch evaluation costs. Selector timing includes score transfer; finish timing includes gathering, suffix, normalization and feature transfer. Neither is deployment latency; prefix extraction, teacher generation and training are separate."}
    (output / "eval_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pets_selector_pilot.json"))
    parser.add_argument("--models", type=Path, default=Path("runs/pets-selector-training-v1"))
    parser.add_argument("--output", type=Path, default=Path("runs/pets-selector-eval-v1"))
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.models, args.output)
