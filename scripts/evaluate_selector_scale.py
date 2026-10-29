"""Evaluate fixed-step selectors on every development image and save CLIP features."""

import argparse
import csv
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from evaluate_selectors import binary_outcomes, read_tasks, utility_correlations, write_rows


SELECTOR_METHODS = ("conditioned", "image_only", "mean_only", "difference_only", "image_only_wide")
TASK_INDEPENDENT_METHODS = ("full_model", "attention", "image_only", "image_only_wide", "random")


def evaluation_groups(config, random_only=False):
    """Choose fixed saved models and optional references without rerunning old arms."""
    arms = config.get("selector_arms", ["conditioned", "image_only"])
    if not arms or len(set(arms)) != len(arms) or any(arm not in SELECTOR_METHODS for arm in arms):
        raise ValueError("selector_arms must name distinct supported selector variants")
    groups = []
    if not random_only:
        if config.get("include_references", True):
            groups.append(("references", [(-1, x) for x in ("full_model", "attention", "label_informed_teacher")]))
        groups += [(coverage, [(seed, arm) for seed in config["training_seeds"] for arm in arms])
                   for coverage in config["coverages"]]
    random_seeds = config.get("random_seeds", [2055, 2056, 2057] if random_only else [])
    if random_seeds:
        groups.append(("random", [(seed, "random") for seed in random_seeds]))
    return groups, random_seeds


def selector_from_checkpoint(checkpoint, descriptor, method, token_dim, device):
    """Restore the recorded variant; legacy checkpoints retain their original arm."""
    from token_compression.selector import make_selector

    variant = checkpoint.get("selector_variant")
    if variant is None:
        variant = "conditioned" if checkpoint["conditioned"] else "image_only"
    if variant != method:
        raise ValueError(f"Checkpoint variant {variant} does not match requested method {method}")
    model = make_selector(descriptor, variant, hidden_dim=checkpoint["hidden_dim"],
                          token_dim=token_dim).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval().requires_grad_(False)


def evaluation_examples(image_ids, labels, train_tasks, val_tasks):
    """Enumerate complete tasks without assuming equal image counts by class."""
    labels = np.asarray(labels)
    present = set(labels.tolist())
    seen = {tuple(sorted((r["class_a"], r["class_b"]))) for r in train_tasks}
    unseen = {tuple(sorted((r["class_a"], r["class_b"]))) for r in val_tasks}
    assert not seen & unseen
    rows = []
    for split, tasks in (("seen", train_tasks), ("unseen", val_tasks)):
        complete = [t for t in tasks if {t["class_a"], t["class_b"]} <= present]
        for task in sorted(complete, key=lambda t: t["task_id"]):
            a, b = sorted((task["class_a"], task["class_b"]))
            for i in np.flatnonzero((labels == a) | (labels == b)):
                rows.append({"image_index": int(i), "image_id": str(image_ids[i]),
                             "class_id": int(labels[i]), "task_id": task["task_id"],
                             "class_a": a, "class_b": b, "task_split": split,
                             "task_class_weight": 1.0 / (2 * len(complete) * np.sum(labels == labels[i]))})
    return rows


def random_rankings(image_count, patch_count, seed):
    """One task-independent random ordering per image, reused across budgets."""
    return np.random.default_rng(seed).random((image_count, patch_count), dtype=np.float32)


def run(config, models_dir, output, random_only=False):
    import clip
    import torch

    from token_compression.pruning import SingleStagePruner
    from token_compression.selector import shared_descriptor

    started = datetime.now(timezone.utc).isoformat()
    before_run = time.perf_counter()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(config["cache_dir"])
    with np.load(cache / "val" / "evidence.npz") as saved:
        evidence = {key: saved[key].copy() for key in saved.files}
    ids, labels, text = [evidence[k] for k in ("image_ids", "labels", "text_features")]
    with np.load(cache / "val" / "teacher.npz") as teacher:
        np.testing.assert_array_equal(teacher["image_ids"], ids)
        coefficients = teacher["coefficients"].copy()
    with np.load(Path(config["source_cache_dir"]) / "val" / "evidence.npz") as old:
        old_ids = old["image_ids"].copy()
    assert set(old_ids) <= set(ids)
    train_tasks, val_tasks = [read_tasks(cache / f"{split}_tasks.csv") for split in ("train", "val")]
    examples = evaluation_examples(ids, labels, train_tasks, val_tasks)
    write_rows(output / "examples.csv", examples)
    ix = np.asarray([r["image_index"] for r in examples])
    y = labels[ix]
    pairs = np.asarray([[r["class_a"], r["class_b"]] for r in examples])
    competitors = np.where(y == pairs[:, 0], pairs[:, 1], pairs[:, 0])
    norms = np.linalg.norm(text[y] - text[competitors], axis=1)
    utilities = np.stack([(coefficients[i, :, a] - coefficients[i, :, b]) / n
                          for i, a, b, n in zip(ix, y, competitors, norms)])
    base_margin, base_correct = binary_outcomes(evidence["full_scores"][ix], y, pairs, text)
    count, batch_size, k = len(examples), config["image_batch_size"], config["keep_patch_tokens"]
    device = torch.device(config["device"])
    backbone, _ = clip.load(config["model"], device=device, jit=False, download_root=config["model_cache"])
    backbone.eval().requires_grad_(False)
    pruner = SingleStagePruner(backbone.visual, config["prune_after"])
    prefixes = torch.as_tensor(np.load(cache / "val" / "prefix_tokens.npy"), device=device)
    text_tensor = torch.as_tensor(text, device=device)
    train_pairs = torch.tensor([[r["class_a"], r["class_b"]] for r in train_tasks], device=device)
    descriptor = shared_descriptor(text_tensor, train_pairs)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def predict(seed, method, coverage):
        before = time.perf_counter()
        model, checkpoint_path = None, None
        if method in SELECTOR_METHODS:
            checkpoint_path = models_dir / coverage / f"seed_{seed}" / f"{method}.pt"
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model = selector_from_checkpoint(checkpoint, descriptor, method, prefixes.shape[-1], device)
        # These methods are task-independent: run each image once and re-score
        # its identical feature vector for every relevant candidate pair.
        unique = method in TASK_INDEPENDENT_METHODS
        image_indices = np.arange(len(ids)) if unique else ix
        n = len(image_indices)
        features = np.empty((n, text.shape[1]), np.float32)
        logits = np.zeros((n, pruner.num_patches),
                          np.float64 if method == "label_informed_teacher" else np.float32)
        positions = np.full((n, pruner.num_patches), -1, np.int16)
        selector_seconds = finish_seconds = 0.0
        physical = 0
        random_scores = random_rankings(len(ids), pruner.num_patches, seed) if method == "random" else None
        if method == "full_model":
            features[:] = evidence["full_features"]
            positions[:] = np.arange(pruner.num_patches)
        else:
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                valid = end - start
                rows = np.arange(start, end)
                if valid < batch_size:
                    rows = np.concatenate([rows, np.repeat(rows[-1], batch_size - valid)])
                tokens = prefixes[torch.as_tensor(image_indices[rows], device=device)]
                if model is not None:
                    candidate_pairs = (np.tile(pairs[0], (len(rows), 1)) if unique else pairs[rows])
                    candidate_text = text_tensor[torch.as_tensor(candidate_pairs, device=device)]
                    sync()
                    before_select = time.perf_counter()
                    value = model(tokens.float(), candidate_text).cpu().numpy()
                    sync()
                    selector_seconds += time.perf_counter() - before_select
                elif method == "attention":
                    value = evidence["attention_scores"][image_indices[rows]]
                elif method == "random":
                    value = random_scores[image_indices[rows]]
                else:
                    value = utilities[rows]
                selected = np.sort(np.argsort(-value, axis=1, kind="stable")[:, :k], axis=1)
                retained = torch.as_tensor(selected, dtype=torch.long, device=device)
                sync()
                before_finish = time.perf_counter()
                result = pruner.finish(tokens, retained).float()
                result = (result / result.norm(dim=-1, keepdim=True)).cpu().numpy()
                sync()
                finish_seconds += time.perf_counter() - before_finish
                features[start:end] = result[:valid]
                logits[start:end] = value[:valid]
                positions[start:end, :k] = selected[:valid]
                physical += len(rows)
        if unique:
            features, logits, positions = features[ix], logits[ix], positions[ix]
        scores = features @ text.T
        if method == "full_model":
            np.testing.assert_allclose(scores, evidence["full_scores"][ix], atol=config["identity_atol"], rtol=0)
        margins, correct = binary_outcomes(scores, y, pairs, text)
        corr, valid_corr = utility_correlations(logits, utilities)
        valid_corr &= method != "full_model"
        arrays = {"cosine_scores": scores, "image_features": features,
                  "patch_indices": positions, "selection_logits": logits,
                  "selection_logits_available": np.full(count, method != "full_model", bool),
                  "margin": margins, "correct": correct,
                  "teacher_utility_correlation": corr, "teacher_utility_correlation_valid": valid_corr}
        cost = {"coverage": coverage, "seed": seed, "method": method,
                "checkpoint": str(checkpoint_path) if checkpoint_path else None,
                "selector_seconds": selector_seconds, "finish_and_normalize_seconds": finish_seconds,
                "total_method_seconds": time.perf_counter() - before,
                "logical_task_examples": count, "unique_image_reuse": unique,
                "logical_suffix_examples": 0 if method == "full_model" else n,
                "physical_suffix_examples": physical}
        print(json.dumps({"coverage": coverage, "seed": seed, "method": method,
                          "seconds": cost["total_method_seconds"]}), flush=True)
        return arrays, cost

    costs = []
    groups, random_seeds = evaluation_groups(config, random_only)
    with torch.inference_mode():
        for coverage, jobs in groups:
            folder = output / coverage
            folder.mkdir(exist_ok=True)
            parts, records = [], []
            for seed, method in jobs:
                arrays, cost = predict(seed, method, coverage)
                costs.append(cost)
                parts.append(arrays)
                retained = pruner.num_patches if method == "full_model" else k
                offset = len(records)
                for j, example in enumerate(examples):
                    records.append({"row_id": offset + j, "coverage": coverage, "seed": seed,
                                    "method": method, **example, "old_panel": int(example["image_id"] in old_ids),
                                    "keep_patch_tokens": retained, "base_margin": float(base_margin[j]),
                                    "base_correct": int(base_correct[j]), "margin": float(arrays["margin"][j]),
                                    "correct": int(arrays["correct"][j]),
                                    "teacher_utility_correlation": float(arrays["teacher_utility_correlation"][j])
                                    if arrays["teacher_utility_correlation_valid"][j] else ""})
            write_rows(folder / "predictions.csv", records)
            packed = {name: np.concatenate([p[name] for p in parts]) for name in parts[0]}
            packed.update(row_id=np.arange(len(records)),
                          **{name: np.asarray([r[name] for r in records]) for name in
                             ("seed", "method", "task_split", "image_index", "keep_patch_tokens", "old_panel")},
                          image_ids=np.asarray([r["image_id"] for r in records]),
                          labels=np.asarray([r["class_id"] for r in records]),
                          task_ids=np.asarray([r["task_id"] for r in records]),
                          class_pairs=np.asarray([[r["class_a"], r["class_b"]] for r in records]),
                          text_features=text)
            np.savez_compressed(folder / "predictions.npz", **packed)
            del parts, records, packed
    summary = {"started_utc": started, "code_commit": commit, "config": config,
               "models_dir": str(models_dir), "random_only": random_only,
               "selector_arms": config.get("selector_arms", ["conditioned", "image_only"]) if not random_only else [],
               "include_references": config.get("include_references", True) and not random_only,
               "random_seeds": random_seeds, "image_count": len(ids), "old_panel_image_count": len(old_ids),
               "examples_per_method": count, "cost": {"methods": costs,
                    "total_wall_seconds": time.perf_counter() - before_run,
                    "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0},
               "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
               "scope": "All development images; seen training pairs and unseen validation pairs. Official test remains unused.",
               "schema": {"row_alignment": "CSV row_id equals per-row NPZ index within each coverage folder.",
                          "image_features": f"Normalized full {text.shape[1]}-dimensional float32 CLIP output, retained for later geometric analysis.",
                          "references": ("Full model, attention and label-informed teacher stored once globally; labels only enter the offline teacher."
                                         if config.get("include_references", True) and not random_only
                                         else "Reference predictions omitted in this run; previously saved reference results are reused."),
                          "task_class_weight": "Within each task split, each task has equal weight and each side contributes one half.",
                          "image_only": "Both image_only and image_only_wide ignore candidate classes. One image forward is exactly reused across candidate pairs.",
                          "descriptor_ablations": "mean_only and difference_only run separately for every image/task; their missing descriptor component is fixed by the training prior saved in the checkpoint.",
                          "random": "Each random seed gives one fixed ordering per image; same image order and seed give nested retained sets across budgets. Seeds index random masks, not trained models."},
               "cost_scope": "Cached-prefix evaluation with fixed padded suffix batches; includes host transfers, excludes prefix extraction. These are experiment costs, not deployment latency."}
    (output / "eval_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--models", type=Path, default=Path("runs/pets-selector-scale-training-v1"))
    parser.add_argument("--output", type=Path, default=Path("runs/pets-selector-scale-eval-v1"))
    parser.add_argument("--random-only", action="store_true", help="Add random baselines while reusing an earlier budget's existing model/reference evaluation.")
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.models, args.output, args.random_only)
