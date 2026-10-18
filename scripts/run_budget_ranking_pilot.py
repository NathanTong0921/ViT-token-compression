"""Compare near-full and budget-matched additive token rankings on a fixed panel."""

import argparse
import json
import platform
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import clip
import numpy as np
import torch
from torch.utils.data import DataLoader

from token_compression.baseline import PetsImages, write_csv
from token_compression.budget_ranking import (sample_masks, ridge_projector,
                                             fit_responses, predict_masks,
                                             task_coefficients)
from token_compression.metrics import direction_angle
from token_compression.pets import load_manifests
from token_compression.pruning import SingleStagePruner
from token_compression.selection import make_candidate_bank, select_panel
from run_selection_pilot import normalized, topk, correct


@torch.inference_mode()
def run(config, output):
    started = datetime.now(timezone.utc).isoformat()
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    before_run = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config["seed"])
    images, classes, tasks = load_manifests(config["manifest_dir"])
    panel, panel_tasks = select_panel(images, tasks, config["images_per_class"],
                                     seed=config["balance_seed"],
                                     image_split=config["image_split"],
                                     task_split=config["task_split"])
    write_csv(output / "panel_images.csv", panel)
    write_csv(output / "panel_tasks.csv", panel_tasks)
    by_class = defaultdict(list)
    for task in panel_tasks:
        by_class[task["class_a"]].append((task["class_b"], task["task_id"]))
        by_class[task["class_b"]].append((task["class_a"], task["task_id"]))
    for label in by_class:
        by_class[label].sort()
    image_ids = np.array([row["image_id"] for row in panel])
    labels = np.array([row["class_id"] for row in panel])
    with np.load(Path(config["baseline_dir"]) / "embeddings.npz") as saved:
        lookup = {image_id: i for i, image_id in enumerate(saved["image_ids"])}
        indices = [lookup[image_id] for image_id in image_ids]
        reference_features = saved["image_features"][indices].copy()
        reference_scores = saved["cosine_scores"][indices].copy()
        text = saved["text_features"].copy()
        assert np.array_equal(labels, saved["labels"][indices])

    device = torch.device(config["device"])
    model, preprocess = clip.load(config["model"], device=device, jit=False,
                                  download_root=config["model_cache"])
    model.eval().requires_grad_(False)
    pruner = SingleStagePruner(model.visual, config["prune_after"])
    prompts = [config["prompt_template"].format(class_name=row["class_name"])
               for row in classes]
    measured_text = normalized(model.encode_text(clip.tokenize(prompts).to(device)))
    np.testing.assert_allclose(measured_text, text, atol=config["identity_atol"], rtol=0)
    # Preserve the original encoder batch shapes before sampling the panel:
    # FP16 kernels can otherwise change even unpruned embeddings slightly.
    full_image_rows = [row for row in images if row["split"] == config["image_split"]]
    loader = DataLoader(PetsImages(config["data_root"], full_image_rows, preprocess),
                        batch_size=config["batch_size"], shuffle=False,
                        num_workers=config["num_workers"], pin_memory=True)
    torch.cuda.reset_peak_memory_stats(device)
    prefixes, attentions, keep_all, native, visited_ids = [], [], [], [], []
    selected_ids = set(image_ids.tolist())
    for pixels, _, batch_ids in loader:
        pixels = pixels.to(device)
        tokens, attention = pruner.prefix(pixels)
        full = torch.arange(pruner.num_patches, device=device).expand(len(pixels), -1)
        batch_keep = normalized(pruner.finish(tokens, full))
        batch_native = normalized(model.encode_image(pixels))
        selected = [i for i, image_id in enumerate(batch_ids) if image_id in selected_ids]
        if selected:
            keep_all.append(batch_keep[selected])
            native.append(batch_native[selected])
            prefixes.append(tokens[selected].cpu())
            attentions.append(attention[selected].cpu().numpy())
            visited_ids.extend(batch_ids[i] for i in selected)
    assert visited_ids == image_ids.tolist()
    prefixes = torch.cat(prefixes)
    attentions = np.concatenate(attentions)
    keep_all = np.concatenate(keep_all)
    native = np.concatenate(native)
    max_error = float(np.max(np.abs(keep_all - reference_features)))
    assert max_error <= config["identity_atol"], max_error
    keep_scores = keep_all @ text.T
    identity = {"max_feature_error": max_error,
                "max_feature_error_vs_native": float(np.max(np.abs(keep_all - native))),
                "max_score_error": float(np.max(np.abs(keep_scores - reference_scores))),
                "prediction_disagreements_37class": int(np.sum(keep_scores.argmax(1) != reference_scores.argmax(1))),
                "prediction_disagreements_binary": sum(
                    correct(keep_scores[i], int(label), competitor) != correct(reference_scores[i], int(label), competitor)
                    for i, label in enumerate(labels) for competitor, _ in by_class[int(label)])}
    assert identity["max_score_error"] <= config["identity_atol"], identity
    assert identity["max_feature_error_vs_native"] <= config["identity_atol"], identity
    assert identity["prediction_disagreements_37class"] == 0, identity
    assert identity["prediction_disagreements_binary"] == 0, identity
    print(f"Keep-all verified on {len(panel)} images: {json.dumps(identity)}", flush=True)

    identity_seconds = time.perf_counter() - before_run
    contexts = [config["near_full_context"]] + config["keep_patch_tokens"]
    designs, projectors, evidence = {}, {}, {}
    context_seconds = {context: {"probe_seconds": 0.0, "fit_seconds": 0.0,
                                  "holdout_seconds": 0.0} for context in contexts}
    for context in contexts:
        train_seed = int(np.random.SeedSequence([config["mask_seed"], context, 0]).generate_state(1)[0])
        holdout_seed = int(np.random.SeedSequence([config["mask_seed"], context, 1]).generate_state(1)[0])
        masks = sample_masks(pruner.num_patches, context, config["probe_count"], train_seed)
        designs[context] = {"masks": masks, "indices": np.flatnonzero(masks).reshape(len(masks), context) % pruner.num_patches,
                            "train_seed": train_seed, "holdout_seed": holdout_seed}
        projectors[context] = ridge_projector(masks, ridge=config["ridge"])
        evidence[context] = {"responses": [], "coefficients": [], "intercepts": []}
        if context in config["keep_patch_tokens"]:
            held = sample_masks(pruner.num_patches, context, config["holdout_count"],
                                holdout_seed, exclude=masks)
            designs[context]["holdout_masks"] = held
            designs[context]["holdout_indices"] = np.flatnonzero(held).reshape(len(held), context) % pruner.num_patches
            evidence[context]["holdout_responses"] = []
    np.savez_compressed(output / "evidence.npz", image_ids=image_ids, labels=labels,
                        text_features=text, full_features=reference_features,
                        full_scores=reference_scores, attention_scores=attentions)
    arms = {}
    for name in ("near_full", "budget_matched"):
        arms[name] = {"exchange_rows": [], "single_rows": [], "positions": [],
                      "banks": {k: {"indices": [], "scores": [], "features": [], "cases": []}
                                for k in config["keep_patch_tokens"]}}
    holdout_rows = []
    selection_seconds = {name: 0.0 for name in arms}
    suffix_examples = 0
    suffix_calls = 0
    logical_suffix_examples = 0

    def finish_one(tokens, retained):
        nonlocal suffix_examples, suffix_calls, logical_suffix_examples
        parts = []
        for start in range(0, len(retained), config["suffix_batch_size"]):
            idx = torch.as_tensor(retained[start:start + config["suffix_batch_size"]],
                                  device=device, dtype=torch.long)
            valid = len(idx)
            if valid < config["suffix_batch_size"]:
                idx = torch.cat([idx, idx[-1:].expand(config["suffix_batch_size"] - valid, -1)])
            parts.append(normalized(pruner.finish(tokens.expand(len(idx), -1, -1), idx))[:valid])
            suffix_examples += len(idx)
            suffix_calls += 1
        logical_suffix_examples += len(retained)
        return np.concatenate(parts)

    def responses_from_features(features, image_index):
        return np.column_stack([features @ text.T,
                                features @ reference_features[image_index]])

    for image_index, image_row in enumerate(panel):
        image_id, label = image_row["image_id"], image_row["class_id"]
        tokens = prefixes[image_index:image_index + 1].to(device)
        fits, held_responses = {}, {}
        for context in contexts:
            before = time.perf_counter()
            responses = responses_from_features(finish_one(tokens, designs[context]["indices"]), image_index)
            context_seconds[context]["probe_seconds"] += time.perf_counter() - before
            before = time.perf_counter()
            coefficients, intercept = fit_responses(projectors[context], responses)
            context_seconds[context]["fit_seconds"] += time.perf_counter() - before
            fits[context] = coefficients, intercept
            evidence[context]["responses"].append(responses)
            evidence[context]["coefficients"].append(coefficients)
            evidence[context]["intercepts"].append(intercept)
            if context in config["keep_patch_tokens"]:
                before = time.perf_counter()
                held = responses_from_features(finish_one(tokens, designs[context]["holdout_indices"]), image_index)
                context_seconds[context]["holdout_seconds"] += time.perf_counter() - before
                held_responses[context] = held
                evidence[context]["holdout_responses"].append(held)
        competitors = [entry[0] for entry in by_class[label]]
        task_ids = [entry[1] for entry in by_class[label]]
        for arm_name, arm in arms.items():
            before = time.perf_counter()
            exchange_rows, single_rows = arm["exchange_rows"], arm["single_rows"]
            selection_indices, banks = arm["positions"], arm["banks"]
            for k in config["keep_patch_tokens"]:
                context = config["near_full_context"] if arm_name == "near_full" else k
                coefficients, intercept = fits[context]
                utilities = task_coefficients(coefficients[:, :len(text)], text, label, competitors)
                representation = coefficients[:, len(text)]
                predicted = predict_masks(designs[k]["holdout_masks"], coefficients, intercept)
                actual = held_responses[k]
                for j, competitor in enumerate(competitors):
                    norm = np.linalg.norm(text[label] - text[competitor])
                    truth = (actual[:, label] - actual[:, competitor]) / norm
                    prediction = (predicted[:, label] - predicted[:, competitor]) / norm
                    centered_truth = truth - truth.mean()
                    centered_prediction = prediction - prediction.mean()
                    denominator = np.linalg.norm(centered_truth) * np.linalg.norm(centered_prediction)
                    valid = denominator > 1e-14
                    holdout_rows.append({"image_id": image_id, "class_id": label,
                                         "competitor": competitor, "task_id": task_ids[j],
                                         "keep_patch_tokens": k, "arm": arm_name,
                                         "true_margin_mean": float(truth.mean()),
                                         "predicted_margin_mean": float(prediction.mean()),
                                         "raw_mse": float(np.mean((truth - prediction) ** 2)),
                                         "centered_mse": float(np.mean((centered_truth - centered_prediction) ** 2)),
                                         "pearson": float(np.dot(centered_truth, centered_prediction) / denominator) if valid else 0.0,
                                         "pearson_valid": int(valid)})
                task_sets = [topk(utilities[:, j], k) for j in range(len(competitors))]
                common_sets = [topk(attentions[image_index], k), topk(representation, k)]
                rng = np.random.default_rng(np.random.SeedSequence([config["seed"], image_index, k]))
                common_sets.append(np.sort(rng.permutation(196)[:k]))
                single_indices = np.stack(task_sets + common_sets)
                single_scores = finish_one(tokens, single_indices) @ text.T
                for j, competitor in enumerate(competitors):
                    norm = np.linalg.norm(text[label] - text[competitor])
                    base = (reference_scores[image_index, label] - reference_scores[image_index, competitor]) / norm
                    for name, slot in (("task_rank", j), ("attention", len(competitors)),
                                       ("representation", len(competitors) + 1),
                                       ("random", len(competitors) + 2)):
                        scores = single_scores[slot]
                        single_rows.append({"image_id": image_id, "class_id": label,
                                            "competitor": competitor, "task_id": task_ids[j],
                                            "keep_patch_tokens": k, "method": name,
                                            "base_margin": float(base),
                                            "margin": float((scores[label] - scores[competitor]) / norm),
                                            "correct": correct(scores, label, competitor),
                                            "base_correct": correct(reference_scores[image_index], label, competitor)})
                        selection_indices.append(single_indices[slot].copy())
                for ja, jb in combinations(range(len(competitors)), 2):
                    ca, cb = competitors[ja], competitors[jb]
                    seed = np.random.SeedSequence([config["seed"], image_index, ca, cb, k])
                    retained, names = make_candidate_bank(utilities[:, ja], utilities[:, jb],
                                                           attentions[image_index], representation,
                                                           k, seed, config["candidate_count"])
                    features = finish_one(tokens, retained)
                    scores = features @ text.T
                    norms = np.linalg.norm(text[label] - text[[ca, cb]], axis=1)
                    margins = (scores[:, label, None] - scores[:, [ca, cb]]) / norms
                    base_margins = (reference_scores[image_index, label] - reference_scores[image_index, [ca, cb]]) / norms
                    chosen = {name: names.index(name) for name in
                              ("task_a", "task_b", "shared_proxy", "attention", "representation", "random_0")}
                    chosen.update(shared_search_mean=int(np.argmax(margins.mean(axis=1))),
                                  shared_search_maxmin=int(np.argmax(margins.min(axis=1))),
                                  task_search_a=int(np.argmax(margins[:, 0])),
                                  task_search_b=int(np.argmax(margins[:, 1])))
                    row = {"image_id": image_id, "class_id": label,
                           "competitor_a": ca, "competitor_b": cb,
                           "task_a": task_ids[ja], "task_b": task_ids[jb],
                           "keep_patch_tokens": k,
                           "angle_degrees": direction_angle(text, label, ca, cb),
                           "base_margin_a": float(base_margins[0]), "base_margin_b": float(base_margins[1]),
                           "overlap_fraction_task_ab": float(len(np.intersect1d(retained[chosen["task_a"]], retained[chosen["task_b"]])) / k)}
                    dz = features[chosen["task_a"]] - features[chosen["task_b"]]
                    directions = (text[label] - text[[ca, cb]]) / norms[:, None]
                    loss = 0.5 * (margins[chosen["task_a"], 0] - margins[chosen["task_b"], 0]
                                  + margins[chosen["task_b"], 1] - margins[chosen["task_a"], 1])
                    scale = float(np.linalg.norm(dz) * np.linalg.norm(directions[0] - directions[1]) / 2)
                    row["selection_feature_distance"] = float(np.linalg.norm(dz))
                    row["exchange_alignment"] = float(loss / scale) if scale > 1e-12 else 0.0
                    row["exchange_projection_residual"] = float(loss - 0.5 * np.dot(dz, directions[0] - directions[1]))
                    for name, slot in chosen.items():
                        row[f"margin_{name}_a"] = float(margins[slot, 0])
                        row[f"margin_{name}_b"] = float(margins[slot, 1])
                        row[f"correct_{name}_a"] = correct(scores[slot], label, ca)
                        row[f"correct_{name}_b"] = correct(scores[slot], label, cb)
                        row[f"candidate_{name}"] = slot
                    exchange_rows.append(row)
                    bank = banks[k]
                    bank["indices"].append(retained.astype(np.int16))
                    bank["scores"].append(scores)
                    bank["features"].append(features[[chosen["task_a"], chosen["task_b"]]])
                    bank["cases"].append([image_index, label, ca, cb])
            selection_seconds[arm_name] += time.perf_counter() - before
        if (image_index + 1) % 6 == 0 or image_index + 1 == len(panel):
            print(f"Images {image_index + 1}/{len(panel)}; wall {time.perf_counter() - before_run:.1f}s", flush=True)

    before_save = time.perf_counter()
    write_csv(output / "holdout_prediction_rows.csv", holdout_rows)
    for context in contexts:
        arrays = {key: np.stack(values) for key, values in evidence[context].items()}
        arrays.update(masks=designs[context]["masks"], image_ids=image_ids, labels=labels)
        if "holdout_masks" in designs[context]:
            arrays["holdout_masks"] = designs[context]["holdout_masks"]
        np.savez_compressed(output / f"probes_k{context}.npz", **arrays)
    for arm_name, arm in arms.items():
        arm_output = output / arm_name
        arm_output.mkdir(exist_ok=True)
        write_csv(arm_output / "panel_images.csv", panel)
        write_csv(arm_output / "panel_tasks.csv", panel_tasks)
        write_csv(arm_output / "exchange_rows.csv", arm["exchange_rows"])
        write_csv(arm_output / "single_task_rows.csv", arm["single_rows"])
        padded = np.full((len(arm["positions"]), max(config["keep_patch_tokens"])), -1, np.int16)
        for i, indices in enumerate(arm["positions"]):
            padded[i, :len(indices)] = indices
        np.savez_compressed(arm_output / "single_task_positions.npz", patch_indices=padded)
        np.savez_compressed(arm_output / "deletion_evidence.npz", image_ids=image_ids,
                            labels=labels, text_features=text)
        for k, bank in arm["banks"].items():
            np.savez_compressed(arm_output / f"candidate_bank_k{k}.npz", image_ids=image_ids,
                                cases=np.asarray(bank["cases"]),
                                patch_indices=np.stack(bank["indices"]),
                                cosine_scores=np.stack(bank["scores"]),
                                task_image_features=np.stack(bank["features"]))
        arm_summary = {"run_type": "budget_context_additive_ranking", "arm": arm_name,
                       "started_utc": started, "code_commit": source_commit,
                       "config": config, "image_count": len(panel), "task_count": len(panel_tasks),
                       "exchange_row_count": len(arm["exchange_rows"]),
                       "single_task_row_count": len(arm["single_rows"]), "identity": identity,
                       "ranking_context": config["near_full_context"] if arm_name == "near_full" else "target budget",
                       "interpretation_scope": "Label-informed development diagnostics; candidate-bank searches are optimistic bounded references, not learned selectors."}
        (arm_output / "run_summary.json").write_text(json.dumps(arm_summary, indent=2) + "\n")
    save_seconds = time.perf_counter() - before_save
    summary = {
        "run_type": "budget_context_additive_ranking_comparison",
        "started_utc": started, "code_commit": source_commit, "config": config,
        "image_count": len(panel), "task_count": len(panel_tasks), "identity": identity,
        "exchange_row_count_per_arm": len(arms["near_full"]["exchange_rows"]),
        "probe_design": {str(context): {"train_seed": designs[context]["train_seed"],
                                        "holdout_seed": designs[context]["holdout_seed"] if context in config["keep_patch_tokens"] else None,
                                        "probe_count": config["probe_count"],
                                        "holdout_count": config["holdout_count"] if context in config["keep_patch_tokens"] else 0}
                         for context in contexts},
        "ranking_objective": "Ridge regression of masked-image cosine scores on centered, standardized fixed-cardinality masks; coefficients mapped to signed geometric task margins.",
        "representation_objective": "The 38th fitted response is cosine similarity to the original uncompressed image feature.",
        "mask_sharing": "Each context uses the same fixed design across all images; target-context holdouts are disjoint from that context's training masks and shared across prediction arms.",
        "candidate_bank": "Identical generator, candidate count, seed and direct CLIP evaluation in both arms; bank contents differ where model-based rankings differ.",
        "budget_scope": "Equal probe counts, not equal GPU time: retained sequence lengths differ by context. The 192-token fit is reused across all target budgets.",
        "interpretation_scope": "Label-informed development experiment. No learned selector, official test evaluation, global optimization or deployment claim.",
        "cost": {"identity_and_prefix_seconds": identity_seconds,
                 "context_seconds": context_seconds,
                 "selection_seconds": selection_seconds,
                 "save_seconds": save_seconds,
                 "suffix_forward_batches": suffix_calls,
                 "suffix_forward_examples": suffix_examples,
                 "unique_requested_suffix_examples": logical_suffix_examples,
                 "suffix_counter_scope": "Probe, holdout and candidate evaluation only; excludes native and keep-all identity passes.",
                 "total_wall_seconds": time.perf_counter() - before_run,
                 "training_seconds": 0,
                 "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(device)},
        "gpu": torch.cuda.get_device_name(device),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["cost"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.output)
