"""Cache frozen CLIP prefixes and budget-matched supervision for a selector pilot."""

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import clip
import numpy as np
import torch
from torch.utils.data import DataLoader

from token_compression.baseline import PetsImages, write_csv
from token_compression.budget_ranking import sample_masks, ridge_projector, fit_responses
from token_compression.pets import load_manifests
from token_compression.pruning import SingleStagePruner
from token_compression.selection import select_panel
from run_selection_pilot import normalized


@torch.inference_mode()
def run(config, output):
    started = datetime.now(timezone.utc).isoformat()
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    before_run = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config["seed"])
    images, classes, tasks = load_manifests(config["manifest_dir"])
    panels, selected_tasks = {}, {}
    for split in ("train", "val"):
        panels[split], selected_tasks[split] = select_panel(
            images, tasks, config[f"{split}_images_per_class"],
            config["balance_seed"], split, split)
        write_csv(output / f"{split}_tasks.csv", selected_tasks[split])
    train_ids = {row["image_id"] for row in panels["train"]}
    val_ids = {row["image_id"] for row in panels["val"]}
    assert train_ids.isdisjoint(val_ids)
    assert {row["task_id"] for row in selected_tasks["train"]}.isdisjoint(
        row["task_id"] for row in selected_tasks["val"])
    with np.load(Path(config["baseline_dir"]) / "embeddings.npz") as saved:
        baseline_ids = saved["image_ids"].copy()
        baseline_labels = saved["labels"].copy()
        baseline_features = saved["image_features"].copy()
        baseline_scores = saved["cosine_scores"].copy()
        text = saved["text_features"].copy()
    baseline_lookup = {image_id: i for i, image_id in enumerate(baseline_ids)}

    device = torch.device(config["device"])
    model, preprocess = clip.load(config["model"], device=device, jit=False,
                                  download_root=config["model_cache"])
    model.eval().requires_grad_(False)
    pruner = SingleStagePruner(model.visual, config["prune_after"])
    prompts = [config["prompt_template"].format(class_name=row["class_name"])
               for row in classes]
    measured_text = normalized(model.encode_text(clip.tokenize(prompts).to(device)))
    np.testing.assert_allclose(measured_text, text, atol=config["identity_atol"], rtol=0)
    torch.cuda.reset_peak_memory_stats(device)
    model_setup_seconds = time.perf_counter() - before_run
    identities, counts, costs = {}, {}, {}
    teacher_seed = int(np.random.SeedSequence(
        [config["mask_seed"], config["keep_patch_tokens"], 0]).generate_state(1)[0])
    masks = sample_masks(pruner.num_patches, config["keep_patch_tokens"],
                        config["probe_count"], teacher_seed)
    retained = (np.flatnonzero(masks).reshape(len(masks), config["keep_patch_tokens"])
                % pruner.num_patches)
    projector = ridge_projector(masks, ridge=config["ridge"])
    suffix_batches, padded_examples = 0, 0

    for split in ("train", "val"):
        split_output = output / split
        split_output.mkdir(exist_ok=True)
        panel = panels[split]
        write_csv(split_output / "panel_images.csv", panel)
        panel_ids = np.asarray([row["image_id"] for row in panel])
        panel_labels = np.asarray([row["class_id"] for row in panel])
        selected_ids = set(panel_ids.tolist())
        all_rows = [row for row in images if row["split"] == split]
        # The full split fixes FP16 kernel batch shapes before panel selection.
        loader = DataLoader(PetsImages(config["data_root"], all_rows, preprocess),
                            batch_size=config["image_batch_size"], shuffle=False,
                            num_workers=config["num_workers"], pin_memory=True)
        prefix_parts, feature_parts, attention_parts, visited_ids = [], [], [], []
        identity = {"images_checked": 0, "max_feature_error_vs_native": 0.0,
                    "prediction_disagreements_37class_vs_native": 0}
        if split == "val":
            identity.update(max_feature_error_vs_saved_baseline=0.0,
                            max_score_error_vs_saved_baseline=0.0,
                            prediction_disagreements_37class_vs_saved_baseline=0)
        before_features = time.perf_counter()
        for pixels, labels, batch_ids in loader:
            pixels = pixels.to(device)
            tokens, attention = pruner.prefix(pixels)
            all_patches = torch.arange(pruner.num_patches, device=device).expand(len(pixels), -1)
            features = normalized(pruner.finish(tokens, all_patches))
            native = normalized(model.encode_image(pixels))
            scores = features @ text.T
            error = float(np.max(np.abs(features - native)))
            identity["max_feature_error_vs_native"] = max(identity["max_feature_error_vs_native"], error)
            identity["prediction_disagreements_37class_vs_native"] += int(np.sum(
                scores.argmax(1) != (native @ text.T).argmax(1)))
            identity["images_checked"] += len(pixels)
            if split == "val":
                indices = [baseline_lookup[image_id] for image_id in batch_ids]
                assert np.array_equal(labels.numpy(), baseline_labels[indices])
                identity["max_feature_error_vs_saved_baseline"] = max(
                    identity["max_feature_error_vs_saved_baseline"],
                    float(np.max(np.abs(features - baseline_features[indices]))))
                identity["max_score_error_vs_saved_baseline"] = max(
                    identity["max_score_error_vs_saved_baseline"],
                    float(np.max(np.abs(scores - baseline_scores[indices]))))
                identity["prediction_disagreements_37class_vs_saved_baseline"] += int(np.sum(
                    scores.argmax(1) != baseline_scores[indices].argmax(1)))
            selected = [i for i, image_id in enumerate(batch_ids) if image_id in selected_ids]
            if selected:
                prefix_parts.append(tokens[selected].cpu().numpy())
                feature_parts.append(features[selected])
                attention_parts.append(attention[selected].cpu().numpy())
                visited_ids.extend(batch_ids[i] for i in selected)
        assert visited_ids == panel_ids.tolist()
        assert identity["max_feature_error_vs_native"] <= config["identity_atol"], identity
        assert identity["prediction_disagreements_37class_vs_native"] == 0, identity
        if split == "val":
            assert identity["max_feature_error_vs_saved_baseline"] <= config["identity_atol"], identity
            assert identity["max_score_error_vs_saved_baseline"] <= config["identity_atol"], identity
            assert identity["prediction_disagreements_37class_vs_saved_baseline"] == 0, identity
        prefixes = np.concatenate(prefix_parts).astype(np.float16, copy=False)
        features = np.concatenate(feature_parts)
        attention = np.concatenate(attention_parts)
        del prefix_parts, feature_parts, attention_parts
        np.save(split_output / "prefix_tokens.npy", prefixes)
        np.savez_compressed(split_output / "evidence.npz", image_ids=panel_ids,
                            labels=panel_labels, text_features=text,
                            full_features=features, full_scores=features @ text.T,
                            attention_scores=attention)
        costs[f"{split}_feature_seconds"] = time.perf_counter() - before_features
        counts[split] = {"images": len(panel), "classes": len(np.unique(panel_labels)),
                         "tasks": len(selected_tasks[split]),
                         "full_split_images_encoded": len(all_rows),
                         "prefix_bytes": prefixes.nbytes}
        identities[split] = identity
        print(f"{split} features verified: {json.dumps(identity)}", flush=True)

        before_teacher = time.perf_counter()
        if split == "train":
            response_parts, coefficient_parts, intercept_parts = [], [], []
            for image_index in range(len(panel)):
                tokens = torch.from_numpy(prefixes[image_index:image_index + 1]).to(device)
                feature_batches = []
                for start in range(0, len(retained), config["teacher_batch_size"]):
                    indices = torch.as_tensor(retained[start:start + config["teacher_batch_size"]],
                                              dtype=torch.long, device=device)
                    valid = len(indices)
                    if valid < config["teacher_batch_size"]:
                        indices = torch.cat([indices, indices[-1:].expand(
                            config["teacher_batch_size"] - valid, -1)])
                    feature_batches.append(normalized(pruner.finish(
                        tokens.expand(len(indices), -1, -1), indices))[:valid])
                    suffix_batches += 1
                    padded_examples += len(indices)
                responses = np.concatenate(feature_batches) @ text.T
                coefficients, intercept = fit_responses(projector, responses)
                response_parts.append(responses)
                coefficient_parts.append(coefficients)
                intercept_parts.append(intercept)
                if (image_index + 1) % 32 == 0 or image_index + 1 == len(panel):
                    print(f"Train teacher {image_index + 1}/{len(panel)}; "
                          f"wall {time.perf_counter() - before_run:.1f}s", flush=True)
            np.savez_compressed(split_output / "teacher.npz", masks=masks,
                                responses=np.stack(response_parts),
                                coefficients=np.stack(coefficient_parts),
                                intercepts=np.stack(intercept_parts), image_ids=panel_ids)
            del response_parts, coefficient_parts, intercept_parts
        else:
            source = Path(config["teacher_source_dir"])
            with np.load(source / "evidence.npz") as saved:
                assert np.array_equal(saved["image_ids"], panel_ids)
                assert np.array_equal(saved["labels"], panel_labels)
                np.testing.assert_array_equal(saved["text_features"], text)
                np.testing.assert_allclose(saved["full_features"], features,
                                           atol=config["identity_atol"], rtol=0)
            with np.load(source / f"probes_k{config['keep_patch_tokens']}.npz") as saved:
                assert np.array_equal(saved["image_ids"], panel_ids)
                assert np.array_equal(saved["masks"], masks)
                # The source fit includes one extra representation response;
                # only class-score coefficients are needed for task supervision.
                np.savez_compressed(split_output / "teacher.npz",
                                    coefficients=saved["coefficients"][:, :, :len(text)],
                                    intercepts=saved["intercepts"][:, :len(text)],
                                    image_ids=panel_ids)
        costs[f"{split}_teacher_seconds"] = time.perf_counter() - before_teacher
        del prefixes, features, attention

    summary = {
        "run_type": "frozen_prefix_and_budget_matched_selector_supervision",
        "started_utc": started, "code_commit": source_commit, "config": config,
        "counts": counts, "identity": identities,
        "train_val_images_disjoint": train_ids.isdisjoint(val_ids),
        "train_val_tasks_disjoint": True,
        "teacher": {"context_patch_tokens": config["keep_patch_tokens"],
                    "probe_count_per_training_image": config["probe_count"],
                    "mask_design_seed": teacher_seed, "ridge": config["ridge"],
                    "response_columns": "Class cosine scores in manifest class order.",
                    "training_supervision": "Only train images and train task pairs are supplied to selector optimization.",
                    "validation_source": str(Path(config["teacher_source_dir"]) /
                                             f"probes_k{config['keep_patch_tokens']}.npz"),
                    "validation_use": "Reused label-informed offline teacher for evaluation only; never an optimization target during training."},
        "cost": {**costs, "model_setup_seconds": model_setup_seconds,
                 "teacher_suffix_forward_batches": suffix_batches,
                 "teacher_suffix_forward_examples": padded_examples,
                 "teacher_requested_suffix_examples": len(panels["train"]) * config["probe_count"],
                 "total_wall_seconds": time.perf_counter() - before_run,
                 "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(device)},
        "storage": "Selected prefix caches stay on the server; no full-split prefix cache is written.",
        "gpu": torch.cuda.get_device_name(device),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__},
    }
    (output / "prep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["cost"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path,
                        help="Defaults to cache_dir in the configuration.")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    run(config, args.output or Path(config["cache_dir"]))
