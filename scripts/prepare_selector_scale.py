"""Extend selector caches to all training/development images, reusing old teachers."""

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from token_compression.budget_ranking import sample_masks, ridge_projector, fit_responses
from token_compression.pets import load_manifests


def full_panels(images, tasks):
    """Keep the established image and class-pair splits; include every image."""
    panels = {split: [row for row in images if row["split"] == split]
              for split in ("train", "val")}
    selected_tasks = {split: [row for row in tasks if row["split"] == split]
                      for split in ("train", "val")}
    assert {row["image_id"] for row in panels["train"]}.isdisjoint(
        row["image_id"] for row in panels["val"])
    assert {row["task_id"] for row in selected_tasks["train"]}.isdisjoint(
        row["task_id"] for row in selected_tasks["val"])
    return panels, selected_tasks


def initialize_teacher(panel_ids, source, masks):
    """Map the old panel into the expanded panel without inventing responses.

    The pilot validation cache saved fitted coefficients only. Missing raw
    responses occupy zero-filled rows marked by ``response_available=False``;
    the fitted coefficients remain available and unchanged for those images.
    """
    lookup = {str(image_id): i for i, image_id in enumerate(panel_ids)}
    source_indices = np.asarray([lookup[str(image_id)] for image_id in source["image_ids"]])
    assert len(set(source_indices.tolist())) == len(source_indices)
    if "masks" in source:
        np.testing.assert_array_equal(source["masks"], masks)
    old_coefficients = source["coefficients"]
    old_intercepts = source["intercepts"]
    n, classes = len(panel_ids), old_coefficients.shape[-1]
    coefficients = np.zeros((n,) + old_coefficients.shape[1:], dtype=old_coefficients.dtype)
    intercepts = np.zeros((n, classes), dtype=old_intercepts.dtype)
    response_dtype = source["responses"].dtype if "responses" in source else np.float32
    responses = np.zeros((n, len(masks), classes), dtype=response_dtype)
    response_available = np.zeros(n, dtype=bool)
    reused = np.zeros(n, dtype=bool)
    coefficients[source_indices] = old_coefficients
    intercepts[source_indices] = old_intercepts
    reused[source_indices] = True
    if "responses" in source:
        available = source.get("response_available", np.ones(len(source_indices), dtype=bool))
        responses[source_indices[available]] = source["responses"][available]
        response_available[source_indices[available]] = True
    return {"image_ids": np.asarray(panel_ids), "masks": masks,
            "responses": responses, "response_available": response_available,
            "coefficients": coefficients, "intercepts": intercepts,
            "teacher_reused": reused}


def run(config, output):
    # The small panel-mapping helpers above can be checked without a GPU stack.
    import clip
    import torch
    from torch.utils.data import DataLoader

    from token_compression.baseline import PetsImages, write_csv
    from token_compression.pruning import SingleStagePruner

    def normalized(features):
        features = features.float()
        return (features / features.norm(dim=-1, keepdim=True)).cpu().numpy()

    started = datetime.now(timezone.utc).isoformat()
    before_run = time.perf_counter()
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source_dir = Path(config["source_cache_dir"])
    old_summary = json.loads((source_dir / "prep_summary.json").read_text())
    identity_fields = ("model", "prune_after", "keep_patch_tokens", "probe_count", "mask_seed",
                       "ridge", "image_batch_size", "teacher_batch_size", "prompt_template")
    for field in identity_fields:
        assert config[field] == old_summary["config"][field], field
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config["seed"])
    images, classes, tasks = load_manifests(config["manifest_dir"])
    panels, selected_tasks = full_panels(images, tasks)
    write_csv(output / "panel_images.csv", panels["train"] + panels["val"])
    for split in panels:
        write_csv(output / f"{split}_tasks.csv", selected_tasks[split])

    with np.load(Path(config["baseline_dir"]) / "embeddings.npz") as saved:
        baseline_ids = saved["image_ids"].copy()
        baseline_labels = saved["labels"].copy()
        baseline_features = saved["image_features"].copy()
        baseline_scores = saved["cosine_scores"].copy()
        text = saved["text_features"].copy()
    baseline_lookup = {str(image_id): i for i, image_id in enumerate(baseline_ids)}
    device = torch.device(config["device"])
    model, preprocess = clip.load(config["model"], device=device, jit=False,
                                  download_root=config["model_cache"])
    model.eval().requires_grad_(False)
    pruner = SingleStagePruner(model.visual, config["prune_after"])
    prompts = [config["prompt_template"].format(class_name=row["class_name"])
               for row in classes]
    with torch.inference_mode():
        measured_text = normalized(model.encode_text(clip.tokenize(prompts).to(device)))
    np.testing.assert_array_equal(measured_text, text)
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
    suffix_batches, padded_examples, generated_images = 0, 0, 0

    with torch.inference_mode():
        for split, panel in panels.items():
            folder = output / split
            folder.mkdir(exist_ok=True)
            write_csv(folder / "panel_images.csv", panel)
            panel_ids = np.asarray([row["image_id"] for row in panel])
            panel_labels = np.asarray([row["class_id"] for row in panel])
            old_folder = source_dir / split
            with np.load(old_folder / "evidence.npz") as saved:
                old_evidence = {key: saved[key].copy() for key in saved.files}
            old_prefix = np.load(old_folder / "prefix_tokens.npy", mmap_mode="r")
            old_lookup = {str(image_id): i for i, image_id in enumerate(old_evidence["image_ids"])}
            assert set(old_lookup).issubset(set(panel_ids.tolist()))
            np.testing.assert_array_equal(old_evidence["text_features"], text)
            loader = DataLoader(PetsImages(config["data_root"], panel, preprocess),
                                batch_size=config["image_batch_size"], shuffle=False,
                                num_workers=config["num_workers"], pin_memory=True)
            prefixes = None
            features = np.empty((len(panel), text.shape[1]), dtype=np.float32)
            attention = np.empty((len(panel), pruner.num_patches), dtype=np.float32)
            identity = {"images_checked": 0, "max_feature_error_vs_native": 0.0,
                        "prediction_disagreements_37class_vs_native": 0,
                        "old_prefixes_checked": 0, "max_prefix_error_vs_old_cache": 0.0}
            if split == "val":
                identity.update(max_feature_error_vs_saved_baseline=0.0,
                                max_score_error_vs_saved_baseline=0.0,
                                prediction_disagreements_37class_vs_saved_baseline=0)
            before_features = time.perf_counter()
            offset = 0
            for pixels, labels, batch_ids in loader:
                pixels = pixels.to(device)
                tokens, batch_attention = pruner.prefix(pixels)
                indices = torch.arange(pruner.num_patches, device=device).expand(len(pixels), -1)
                batch_features = normalized(pruner.finish(tokens, indices))
                native = normalized(model.encode_image(pixels))
                scores = batch_features @ text.T
                prefix_values = tokens.cpu().numpy()
                attention_values = batch_attention.cpu().numpy()
                if prefixes is None:
                    prefixes = np.lib.format.open_memmap(
                        folder / "prefix_tokens.npy", mode="w+", dtype=prefix_values.dtype,
                        shape=(len(panel),) + prefix_values.shape[1:])
                np.testing.assert_array_equal(panel_ids[offset:offset + len(pixels)], batch_ids)
                np.testing.assert_array_equal(panel_labels[offset:offset + len(pixels)], labels.numpy())
                prefixes[offset:offset + len(pixels)] = prefix_values
                features[offset:offset + len(pixels)] = batch_features
                attention[offset:offset + len(pixels)] = attention_values
                identity["max_feature_error_vs_native"] = max(
                    identity["max_feature_error_vs_native"], float(np.max(np.abs(batch_features - native))))
                identity["prediction_disagreements_37class_vs_native"] += int(np.sum(
                    scores.argmax(1) != (native @ text.T).argmax(1)))
                identity["images_checked"] += len(pixels)
                old_batch = [i for i, image_id in enumerate(batch_ids) if image_id in old_lookup]
                if old_batch:
                    old_indices = [old_lookup[batch_ids[i]] for i in old_batch]
                    np.testing.assert_array_equal(prefix_values[old_batch], old_prefix[old_indices])
                    np.testing.assert_array_equal(batch_features[old_batch], old_evidence["full_features"][old_indices])
                    np.testing.assert_array_equal(attention_values[old_batch], old_evidence["attention_scores"][old_indices])
                    np.testing.assert_array_equal(labels.numpy()[old_batch], old_evidence["labels"][old_indices])
                    identity["old_prefixes_checked"] += len(old_batch)
                if split == "val":
                    old_indices = [baseline_lookup[image_id] for image_id in batch_ids]
                    np.testing.assert_array_equal(labels.numpy(), baseline_labels[old_indices])
                    identity["max_feature_error_vs_saved_baseline"] = max(
                        identity["max_feature_error_vs_saved_baseline"],
                        float(np.max(np.abs(batch_features - baseline_features[old_indices]))))
                    identity["max_score_error_vs_saved_baseline"] = max(
                        identity["max_score_error_vs_saved_baseline"],
                        float(np.max(np.abs(scores - baseline_scores[old_indices]))))
                    identity["prediction_disagreements_37class_vs_saved_baseline"] += int(np.sum(
                        scores.argmax(1) != baseline_scores[old_indices].argmax(1)))
                offset += len(pixels)
            assert offset == len(panel)
            assert identity["old_prefixes_checked"] == len(old_lookup)
            assert identity["max_feature_error_vs_native"] <= config["identity_atol"], identity
            assert identity["prediction_disagreements_37class_vs_native"] == 0, identity
            if split == "val":
                assert identity["max_feature_error_vs_saved_baseline"] <= config["identity_atol"], identity
                assert identity["max_score_error_vs_saved_baseline"] <= config["identity_atol"], identity
                assert identity["prediction_disagreements_37class_vs_saved_baseline"] == 0, identity
            prefixes.flush()
            np.savez_compressed(folder / "evidence.npz", image_ids=panel_ids,
                                labels=panel_labels, text_features=text, full_features=features,
                                full_scores=features @ text.T, attention_scores=attention)
            costs[f"{split}_feature_seconds"] = time.perf_counter() - before_features
            identities[split] = identity
            print(f"{split} full-panel prefixes verified: {json.dumps(identity)}", flush=True)

            before_teacher = time.perf_counter()
            with np.load(old_folder / "teacher.npz") as saved:
                old_teacher = {key: saved[key].copy() for key in saved.files}
            np.testing.assert_array_equal(old_teacher["image_ids"], old_evidence["image_ids"])
            teacher = initialize_teacher(panel_ids, old_teacher, masks)
            new_indices = np.flatnonzero(~teacher["teacher_reused"])
            split_batches, split_padded = 0, 0
            for number, image_index in enumerate(new_indices, start=1):
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
                    split_padded += len(selection)
                responses = np.concatenate(response_parts) @ text.T
                coefficients, intercept = fit_responses(projector, responses)
                teacher["responses"][image_index] = responses
                teacher["response_available"][image_index] = True
                teacher["coefficients"][image_index] = coefficients
                teacher["intercepts"][image_index] = intercept
                if number % 32 == 0 or number == len(new_indices):
                    print(f"{split} new teachers {number}/{len(new_indices)}; "
                          f"total wall {time.perf_counter() - before_run:.1f}s", flush=True)
            np.savez_compressed(folder / "teacher.npz", **teacher)
            costs[f"{split}_teacher_seconds"] = time.perf_counter() - before_teacher
            counts[split] = {
                "images": len(panel), "classes": len(np.unique(panel_labels)),
                "tasks": len(selected_tasks[split]), "prefix_bytes": int(prefixes.nbytes),
                "full_split_images_encoded": len(panel), "old_prefixes_identical": len(old_lookup),
                "teacher_images_reused": int(np.sum(teacher["teacher_reused"])),
                "teacher_images_generated": len(new_indices),
                "response_images_available": int(np.sum(teacher["response_available"])),
                "response_images_unavailable": int(np.sum(~teacher["response_available"])),
                "teacher_suffix_forward_batches": split_batches,
                "teacher_suffix_forward_examples": split_padded,
                "teacher_requested_suffix_examples": len(new_indices) * len(masks),
            }
            suffix_batches += split_batches
            padded_examples += split_padded
            generated_images += len(new_indices)
            del prefixes, old_prefix, old_evidence, old_teacher, teacher, features, attention

    summary = {
        "run_type": "full_training_and_development_selector_supervision",
        "started_utc": started, "code_commit": source_commit, "config": config,
        "counts": counts, "identity": identities,
        "source_cache": {"path": str(source_dir), "code_commit": old_summary["code_commit"],
                         "started_utc": old_summary["started_utc"]},
        "teacher": {
            "context_patch_tokens": config["keep_patch_tokens"],
            "probe_count_per_image": config["probe_count"], "mask_design_seed": mask_seed,
            "ridge": config["ridge"], "response_columns": "Class cosine scores in manifest class order.",
            "response_availability": "Old validation rows retain their exact coefficients but have no stored raw responses; response_available marks zero-filled unavailable rows.",
            "optimization_scope": "Only training images and training class pairs; development teachers are evaluation only.",
            "test_data_used": False,
        },
        "cost": {**costs, "model_setup_seconds": model_setup_seconds,
                 "teacher_suffix_forward_batches": suffix_batches,
                 "teacher_suffix_forward_examples": padded_examples,
                 "teacher_requested_suffix_examples": generated_images * len(masks),
                 "teacher_suffix_examples_avoided_by_reuse": sum(
                     item["teacher_images_reused"] for item in counts.values()) * len(masks),
                 "total_wall_seconds": time.perf_counter() - before_run,
                 "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(device)},
        "storage": "Full training and development prefixes and teacher arrays stay on the server.",
        "gpu": torch.cuda.get_device_name(device),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__},
    }
    (output / "prep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["cost"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Defaults to cache_dir in the configuration.")
    args = parser.parse_args()
    configuration = json.loads(args.config.read_text())
    run(configuration, args.output or Path(configuration["cache_dir"]))
