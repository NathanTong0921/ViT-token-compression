"""Measure a strong shared offline reference on the expanded exchange panel."""

import argparse
import json
import subprocess
import time
from itertools import combinations
from pathlib import Path

import numpy as np

from token_compression.budget_ranking import ridge_projector, fit_responses
from token_compression.selection_analysis import cluster_summary
from token_compression.shared_budget import search_common_library


def fill_missing_probes(config, cache, teacher, evidence, output):
    """Recover old missing responses separately; keep the source teachers intact."""
    missing = np.flatnonzero(~teacher["response_available"])
    if not len(missing):
        return {"images": 0, "seconds": 0., "suffix_examples": 0}
    import clip
    import torch
    from token_compression.pruning import SingleStagePruner

    before = time.perf_counter()
    backbone, _ = clip.load(config["model"], device=config["device"], jit=False,
                             download_root=config["model_cache"])
    backbone.eval().requires_grad_(False)
    pruner = SingleStagePruner(backbone.visual, config["prune_after"])
    prefix = np.load(cache / "val/prefix_tokens.npy", mmap_mode="r")
    masks = teacher["masks"]
    k = config["keep_patch_tokens"]
    positions = np.flatnonzero(masks).reshape(len(masks), k) % masks.shape[1]
    projector = ridge_projector(masks, config["ridge"])
    responses = np.empty((len(missing), len(masks), evidence["text_features"].shape[0]), np.float32)
    errors, intercept_errors = [], []
    with torch.inference_mode():
        for n, ix in enumerate(missing):
            tokens = torch.from_numpy(np.array(prefix[ix:ix + 1])).to(config["device"])
            parts = []
            for start in range(0, len(positions), config["teacher_batch_size"]):
                selected = torch.as_tensor(positions[start:start + config["teacher_batch_size"]],
                                           device=config["device"], dtype=torch.long)
                features = pruner.finish(tokens.expand(len(selected), -1, -1), selected).float()
                parts.append((features / features.norm(dim=-1, keepdim=True)).cpu().numpy())
            responses[n] = np.concatenate(parts) @ evidence["text_features"].T
            coeff, intercept = fit_responses(projector, responses[n])
            errors.append(float(np.max(np.abs(coeff - teacher["coefficients"][ix]))))
            intercept_errors.append(float(np.max(np.abs(intercept - teacher["intercepts"][ix]))))
    np.savez_compressed(output / "missing_probe_responses.npz", image_ids=evidence["image_ids"][missing],
                        source_indices=missing, masks=masks, responses=responses,
                        coefficient_max_abs_error=np.asarray(errors),
                        intercept_max_abs_error=np.asarray(intercept_errors))
    assert max(errors) < 1e-6, max(errors)
    assert max(intercept_errors) < 1e-6, max(intercept_errors)
    teacher["responses"][missing] = responses
    teacher["response_available"][missing] = True
    return {"images": len(missing), "seconds": time.perf_counter() - before,
            "suffix_examples": len(missing) * len(masks), "coefficient_max_abs_error": max(errors),
            "intercept_max_abs_error": max(intercept_errors),
            "source_teacher_modified": False}


def prediction_subset(path):
    with np.load(path) as saved:
        take = saved["task_split"] == "unseen"
        return {key: saved[key][take] for key in ("method", "seed", "image_ids", "task_ids",
                "labels", "class_pairs", "cosine_scores", "patch_indices")}


def run(config, source, output):
    before = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(config["cache_dir"])
    with np.load(cache / "val/evidence.npz") as saved:
        evidence = {key: saved[key].copy() for key in saved.files}
    with np.load(cache / "val/teacher.npz") as saved:
        teacher = {key: saved[key].copy() for key in saved.files}
    np.testing.assert_array_equal(teacher["image_ids"], evidence["image_ids"])
    supplement = fill_missing_probes(config, cache, teacher, evidence, output)
    refs, learned = [prediction_subset(source / group / "predictions.npz") for group in ("references", "full")]
    k, seeds = config["keep_patch_tokens"], config["training_seeds"]
    text, labels = evidence["text_features"], evidence["labels"]
    probe_positions = np.flatnonzero(teacher["masks"]).reshape(len(teacher["masks"]), k) % teacher["masks"].shape[1]
    lookup = {}
    for group, data in (("ref", refs), ("learned", learned)):
        lookup[group] = {(str(im), str(task), str(method), int(seed)): i for i, (im, task, method, seed)
                         in enumerate(zip(data["image_ids"], data["task_ids"], data["method"], data["seed"]))}
    fields = {name: [] for name in ("image_ids", "task_a", "task_b", "labels", "competitor_a", "competitor_b",
             "selected_candidate_indices", "selected_patch_indices", "selected_class_scores",
             "separate_margin", "shared_margin", "separate_minus_shared_margin", "separate_accuracy",
             "shared_accuracy", "shared_both_correct", "any_shared_both_correct", "teacher_margin",
             "teacher_accuracy", "teacher_minus_shared_margin", "teacher_minus_shared_accuracy")}
    for ix, (image_id, label) in enumerate(zip(evidence["image_ids"], labels)):
        image_id, label = str(image_id), int(label)
        tasks = sorted(set(refs["task_ids"][(refs["image_ids"] == image_id) & (refs["method"] == "full_model")]))
        for task_a, task_b in combinations(tasks, 2):
            scores, positions = [teacher["responses"][ix]], [probe_positions]
            extras = [("ref", task_a, "label_informed_teacher", -1),
                      ("ref", task_b, "label_informed_teacher", -1),
                      ("ref", task_a, "attention", -1)]
            extras += [("learned", task_a, "image_only", seed) for seed in seeds]
            extras += [("learned", task, "conditioned", seed) for seed in seeds for task in (task_a, task_b)]
            for group, task, method, seed in extras:
                data = refs if group == "ref" else learned
                row = lookup[group][image_id, str(task), method, seed]
                scores.append(data["cosine_scores"][row:row + 1])
                positions.append(data["patch_indices"][row:row + 1, :k])
            scores, positions = np.concatenate(scores), np.concatenate(positions)
            competitors = []
            for task in (task_a, task_b):
                row = lookup["ref"][image_id, str(task), "full_model", -1]
                competitors.append(int(next(c for c in refs["class_pairs"][row] if c != label)))
            search = search_common_library(scores, label, competitors, text)
            slots = np.arange(2) + len(probe_positions)
            teacher_margin = float(search["margins"][slots, np.arange(2)].mean())
            teacher_accuracy = float(search["correct"][slots, np.arange(2)].mean())
            chosen = search["selected_candidate_indices"]
            case = {"image_ids": image_id, "task_a": task_a, "task_b": task_b, "labels": label,
                    "competitor_a": competitors[0], "competitor_b": competitors[1],
                    "selected_patch_indices": positions[chosen], "selected_class_scores": scores[chosen],
                    "teacher_margin": teacher_margin, "teacher_accuracy": teacher_accuracy,
                    "teacher_minus_shared_margin": teacher_margin - search["shared_margin"],
                    "teacher_minus_shared_accuracy": teacher_accuracy - search["shared_accuracy"]}
            for key in fields:
                fields[key].append(case[key] if key in case else search[key])
    arrays = {key: np.asarray(value) for key, value in fields.items()}
    np.savez_compressed(output / "shared_search_cases.npz", **arrays, text_features=text,
                        probe_masks=teacher["masks"], keep_patch_tokens=k)
    metric_names = [name for name in fields if name.endswith(("margin", "accuracy", "correct"))]
    stats = cluster_summary(arrays["image_ids"], np.column_stack([arrays[key] for key in metric_names]),
                            config["bootstrap_replicates"], config["bootstrap_seed"])
    result = {"code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "budget": k, "evaluation_source": str(source), "teacher_cache": str(cache),
              "image_count": stats["image_count"], "task_pair_count": len(arrays["image_ids"]),
              "aggregation": "Equal image weights, then equal unordered task-pair weights within image; image-cluster intervals.",
              "candidate_library": {"probe_masks": len(probe_positions),
                                    "extras_order": ["task_A_teacher", "task_B_teacher", "attention"] +
                                        [f"image_only_seed_{s}" for s in seeds] +
                                        [f"conditioned_task_{t}_seed_{s}" for s in seeds for t in ("A", "B")],
                                    "size_with_possible_duplicates": len(probe_positions) + len(extras),
                                    "same_library_for_shared_and_separate": True},
              "scope": "Label-informed measured-library diagnostic. Separate mean-margin advantage is nonnegative by construction, not a deployable improvement or a global optimum.",
              "metrics": {key: {"mean": float(stats["mean"][j]),
                          "ci95": [float(stats["ci95_low"][j]), float(stats["ci95_high"][j])]}
                          for j, key in enumerate(metric_names)},
              "probe_supplement": supplement, "total_wall_seconds": time.perf_counter() - before,
              "cost_scope": "Only missing-probe recovery uses GPU; search/statistics are CPU. Not deployment latency."}
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"budget": k, "images": result["image_count"], "cases": result["task_pair_count"],
                      "metrics": result["metrics"], "supplement": supplement}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.source, args.output)
