"""Fit budget-specific teachers on the existing full-panel frozen CLIP cache."""

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from token_compression.budget_ranking import sample_masks, ridge_projector, fit_responses
from token_compression.pets import load_manifests


def link_feature_cache(source, output, panels, selected_tasks, config, source_summary):
    """Reuse frozen features only after checking their configuration and image order."""
    for field in ("model", "prune_after", "image_batch_size", "prompt_template"):
        assert config[field] == source_summary["config"][field], field
    records = {}
    for split, panel in panels.items():
        folder, old_folder = output / split, source / split
        with np.load(old_folder / "evidence.npz") as evidence:
            np.testing.assert_array_equal(evidence["image_ids"], [r["image_id"] for r in panel])
            np.testing.assert_array_equal(evidence["labels"], [r["class_id"] for r in panel])
            shape = np.load(old_folder / "prefix_tokens.npy", mmap_mode="r").shape
            assert shape[0] == len(panel)
            assert shape[1] == evidence["attention_scores"].shape[1] + 1
        # Tasks also remain unchanged; no held-out pair becomes a training pair.
        with (source / f"{split}_tasks.csv").open() as handle:
            saved_tasks = list(csv.DictReader(handle))
        expected = [{key: str(value) for key, value in row.items()}
                    for row in selected_tasks[split]]
        assert saved_tasks == expected
        folder.mkdir(parents=True)
        for name in ("prefix_tokens.npy", "evidence.npz"):
            target = (old_folder / name).resolve()
            (folder / name).symlink_to(os.path.relpath(target, folder.resolve()))
        shutil.copy2(old_folder / "panel_images.csv", folder / "panel_images.csv")
        shutil.copy2(source / f"{split}_tasks.csv", output / f"{split}_tasks.csv")
        records[split] = {"images": len(panel), "prefix_shape": list(shape),
                          "source": str(old_folder), "storage": "relative symbolic links"}
    shutil.copy2(source / "panel_images.csv", output / "panel_images.csv")
    return records


def run(config, output):
    import clip
    import torch

    from token_compression.pruning import SingleStagePruner

    def normalized(features):
        features = features.float()
        return (features / features.norm(dim=-1, keepdim=True)).cpu().numpy()

    started = datetime.now(timezone.utc).isoformat()
    before_run = time.perf_counter()
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source = Path(config["feature_cache_dir"])
    source_summary = json.loads((source / "prep_summary.json").read_text())
    images, classes, tasks = load_manifests(config["manifest_dir"])
    panels = {split: [row for row in images if row["split"] == split]
              for split in ("train", "val")}
    selected_tasks = {split: [row for row in tasks if row["split"] == split]
                      for split in ("train", "val")}
    output.mkdir(parents=True, exist_ok=True)
    feature_reuse = link_feature_cache(source, output, panels, selected_tasks,
                                      config, source_summary)
    device = torch.device(config["device"])
    torch.manual_seed(config["seed"])
    model, _ = clip.load(config["model"], device=device, jit=False,
                         download_root=config["model_cache"])
    model.eval().requires_grad_(False)
    pruner = SingleStagePruner(model.visual, config["prune_after"])
    prompts = [config["prompt_template"].format(class_name=row["class_name"])
               for row in classes]
    with torch.inference_mode():
        text = normalized(model.encode_text(clip.tokenize(prompts).to(device)))
    torch.cuda.reset_peak_memory_stats(device)
    model_setup_seconds = time.perf_counter() - before_run
    mask_seed = int(np.random.SeedSequence(
        [config["mask_seed"], config["keep_patch_tokens"], 0]).generate_state(1)[0])
    masks = sample_masks(pruner.num_patches, config["keep_patch_tokens"],
                        config["probe_count"], mask_seed)
    retained = (np.flatnonzero(masks).reshape(len(masks), config["keep_patch_tokens"])
                % pruner.num_patches)
    projector = ridge_projector(masks, ridge=config["ridge"])
    identities, counts, costs = {}, {}, {}
    suffix_batches, suffix_examples = 0, 0
    with torch.inference_mode():
        for split, panel in panels.items():
            folder = output / split
            prefixes = np.load(folder / "prefix_tokens.npy", mmap_mode="r")
            with np.load(folder / "evidence.npz") as evidence:
                panel_ids, labels = evidence["image_ids"].copy(), evidence["labels"].copy()
                full_features = evidence["full_features"].copy()
                np.testing.assert_array_equal(evidence["text_features"], text)
            before_identity = time.perf_counter()
            identity = {"images_checked": 0, "max_feature_error_vs_source": 0.,
                        "prediction_disagreements_37class_vs_source": 0}
            for start in range(0, len(panel), config["image_batch_size"]):
                tokens = torch.from_numpy(np.array(prefixes[start:start + config["image_batch_size"]])).to(device)
                all_patches = torch.arange(pruner.num_patches, device=device).expand(len(tokens), -1)
                features = normalized(pruner.finish(tokens, all_patches))
                reference = full_features[start:start + len(tokens)]
                identity["images_checked"] += len(tokens)
                identity["max_feature_error_vs_source"] = max(
                    identity["max_feature_error_vs_source"], float(np.max(np.abs(features - reference))))
                identity["prediction_disagreements_37class_vs_source"] += int(np.sum(
                    (features @ text.T).argmax(1) != (reference @ text.T).argmax(1)))
            assert identity["max_feature_error_vs_source"] <= config["identity_atol"], identity
            assert identity["prediction_disagreements_37class_vs_source"] == 0, identity
            identities[split] = identity
            costs[f"{split}_identity_seconds"] = time.perf_counter() - before_identity
            print(f"{split} reused prefixes verified: {json.dumps(identity)}", flush=True)

            before_teacher = time.perf_counter()
            responses = np.empty((len(panel), len(masks), len(classes)), dtype=np.float32)
            coefficients = np.empty((len(panel), pruner.num_patches, len(classes)), dtype=np.float64)
            intercepts = np.empty((len(panel), len(classes)), dtype=np.float64)
            split_batches, split_examples = 0, 0
            for image_index in range(len(panel)):
                tokens = torch.from_numpy(np.array(prefixes[image_index:image_index + 1])).to(device)
                response_parts = []
                for start in range(0, len(retained), config["teacher_batch_size"]):
                    selection = torch.as_tensor(retained[start:start + config["teacher_batch_size"]],
                                                dtype=torch.long, device=device)
                    valid = len(selection)
                    if valid < config["teacher_batch_size"]:
                        selection = torch.cat([selection, selection[-1:].expand(
                            config["teacher_batch_size"] - valid, -1)])
                    response_parts.append(normalized(pruner.finish(
                        tokens.expand(len(selection), -1, -1), selection))[:valid])
                    split_batches += 1
                    split_examples += len(selection)
                response = np.concatenate(response_parts) @ text.T
                coefficients[image_index], intercepts[image_index] = fit_responses(projector, response)
                responses[image_index] = response
                if (image_index + 1) % 64 == 0 or image_index + 1 == len(panel):
                    print(f"budget={config['keep_patch_tokens']} {split} teachers "
                          f"{image_index + 1}/{len(panel)}; wall {time.perf_counter() - before_run:.1f}s",
                          flush=True)
            np.savez_compressed(folder / "teacher.npz", image_ids=panel_ids, masks=masks,
                                responses=responses, response_available=np.ones(len(panel), dtype=bool),
                                coefficients=coefficients, intercepts=intercepts,
                                teacher_reused=np.zeros(len(panel), dtype=bool))
            costs[f"{split}_teacher_seconds"] = time.perf_counter() - before_teacher
            counts[split] = {"images": len(panel), "classes": len(np.unique(labels)),
                             "tasks": len(selected_tasks[split]), "prefix_bytes": int(prefixes.nbytes),
                             "prefix_images_reused": len(panel), "full_split_images_encoded": 0,
                             "teacher_images_reused": 0, "teacher_images_generated": len(panel),
                             "response_images_available": len(panel), "response_images_unavailable": 0,
                             "teacher_suffix_forward_batches": split_batches,
                             "teacher_suffix_forward_examples": split_examples,
                             "teacher_requested_suffix_examples": len(panel) * len(masks)}
            suffix_batches += split_batches
            suffix_examples += split_examples
            del prefixes, responses, coefficients, intercepts, full_features
    summary = {
        "run_type": "budget_matched_full_panel_selector_supervision",
        "experiment_name": config["experiment_name"], "started_utc": started,
        "code_commit": source_commit, "config": config, "counts": counts,
        "identity": identities, "feature_reuse": feature_reuse,
        "source_cache": {"path": str(source), "code_commit": source_summary["code_commit"],
                         "started_utc": source_summary["started_utc"]},
        "teacher": {"context_patch_tokens": config["keep_patch_tokens"],
                    "probe_count_per_image": config["probe_count"], "mask_design_seed": mask_seed,
                    "ridge": config["ridge"], "response_columns": "Class cosine scores in manifest class order.",
                    "response_availability": "Every image has newly computed responses at the requested budget.",
                    "optimization_scope": "Training images and training pairs only; validation teachers are evaluation only.",
                    "test_data_used": False},
        "cost": {**costs, "model_setup_seconds": model_setup_seconds,
                 "teacher_suffix_forward_batches": suffix_batches,
                 "teacher_suffix_forward_examples": suffix_examples,
                 "teacher_requested_suffix_examples": sum(len(p) for p in panels.values()) * len(masks),
                 "total_wall_seconds": time.perf_counter() - before_run,
                 "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(device)},
        "storage": "Prefix tensors and full-model evidence are relative symlinks to the verified full-panel cache; teachers are budget-specific.",
        "gpu": torch.cuda.get_device_name(device),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__},
    }
    (output / "prep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["cost"]), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Defaults to cache_dir in the configuration.")
    args = parser.parse_args()
    configuration = json.loads(args.config.read_text())
    run(configuration, args.output or Path(configuration["cache_dir"]))
