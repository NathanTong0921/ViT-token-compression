"""Cache CUB features/references, then fixed-budget mask responses for n-way training."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from evaluate_nway_selectors import classification_outcomes
from token_compression.budget_ranking import sample_masks
from token_compression.nway import build_panel, task_manifest


def normalized(features):
    features = features.float()
    return (features / features.norm(dim=-1, keepdim=True)).cpu().numpy()


def reference_summary(evidence, manifest, split):
    """Use the same task/class weighting and margins as learned-selector evaluation."""
    text = evidence["text_features"]
    records = []
    for way, tasks in manifest["ways"].items():
        panel = build_panel(evidence["image_ids"], evidence["labels"], tasks[split])
        for method, key in (("full_model", "full_scores"),
                            ("attention", "attention_full_scores")):
            values = classification_outcomes(evidence[key][panel["image_indices"]],
                                             panel["labels"], panel["candidates"], text)
            records.append({"way": int(way), "method": method,
                            "images": len(evidence["image_ids"]), "tasks": len(tasks[split]),
                            "task_examples": len(panel["labels"]),
                            **{name: float(np.dot(panel["weights"], values[key]))
                               for name, key in (("accuracy", "correct"),
                                                 ("margin", "margin"),
                                                 ("geometric_margin", "geometric_margin"))}})
    return records


def mask_bank(config, num_patches):
    seed = int(np.random.SeedSequence(
        [config["mask_seed"], config["keep_patch_tokens"], 0]).generate_state(1)[0])
    masks = sample_masks(num_patches, config["keep_patch_tokens"], config["probe_count"], seed)
    retained = np.stack([np.flatnonzero(mask) for mask in masks])
    return masks, retained, seed


def prepare_features(config, output, split, model, preprocess, pruner, device, commit):
    import torch
    from torch.utils.data import DataLoader
    from token_compression.baseline import write_csv
    from token_compression.cub import CUBImages, load_manifests
    import clip

    began = time.perf_counter()
    images, classes = load_manifests(config["manifest_dir"])
    panel = [row for row in images if row["split"] == split]
    folder = output / split
    folder.mkdir(parents=True, exist_ok=True)
    write_csv(folder / "panel_images.csv", panel)
    prompts = [row["class_text"] for row in classes]
    text = normalized(model.encode_text(clip.tokenize(prompts).to(device)))
    ids = np.asarray([row["image_id"] for row in panel])
    labels = np.asarray([row["class_id"] for row in panel], dtype=np.int64)
    manifest = task_manifest(len(classes), config["ways"], config["task_seed"],
                             config["train_task_rounds"], config["val_task_rounds"])
    loader = DataLoader(CUBImages(config["data_root"], panel, preprocess),
                        batch_size=config["image_batch_size"], shuffle=False,
                        num_workers=config["num_workers"], pin_memory=device.type == "cuda")
    features = np.empty((len(panel), text.shape[1]), np.float32)
    attention_features = np.empty_like(features)
    attention = np.empty((len(panel), pruner.num_patches), np.float32)
    selected = np.empty((len(panel), config["keep_patch_tokens"]), np.int16)
    prefixes = None
    identity = {"images_checked": 0, "max_feature_error_vs_native": 0.,
                "prediction_disagreements_vs_native": 0}
    offset = 0
    for pixels, batch_labels, batch_ids in loader:
        count = len(pixels)
        np.testing.assert_array_equal(ids[offset:offset + count], batch_ids)
        np.testing.assert_array_equal(labels[offset:offset + count], batch_labels.numpy())
        if count < config["image_batch_size"]:
            pixels = torch.cat([pixels, pixels[-1:].expand(
                config["image_batch_size"] - count, -1, -1, -1)])
        pixels = pixels.to(device)
        tokens, scores = pruner.prefix(pixels)
        all_indices = torch.arange(pruner.num_patches, device=device).expand(len(tokens), -1)
        full = normalized(pruner.finish(tokens, all_indices))[:count]
        native = normalized(model.encode_image(pixels))[:count]
        scores = scores.cpu().numpy()
        chosen = np.sort(np.argsort(-scores, axis=1, kind="stable")[:, :config["keep_patch_tokens"]], axis=1)
        reduced = normalized(pruner.finish(tokens, torch.as_tensor(chosen, device=device)))[:count]
        values = tokens[:count].cpu().numpy()
        if prefixes is None:
            prefixes = np.lib.format.open_memmap(folder / "prefix_tokens.npy", mode="w+",
                                                 dtype=values.dtype,
                                                 shape=(len(panel),) + values.shape[1:])
        end = offset + count
        prefixes[offset:end] = values
        features[offset:end] = full
        attention_features[offset:end] = reduced
        attention[offset:end] = scores[:count]
        selected[offset:end] = chosen[:count]
        identity["images_checked"] += count
        identity["max_feature_error_vs_native"] = max(
            identity["max_feature_error_vs_native"], float(np.max(np.abs(full - native))))
        identity["prediction_disagreements_vs_native"] += int(np.sum(
            (full @ text.T).argmax(1) != (native @ text.T).argmax(1)))
        offset = end
        if offset % 512 == 0 or offset == len(panel):
            print(json.dumps({"stage": "features", "split": split, "images": offset,
                              "total": len(panel), "seconds": time.perf_counter() - began}), flush=True)
    assert offset == len(panel)
    assert identity["max_feature_error_vs_native"] <= config["identity_atol"], identity
    assert identity["prediction_disagreements_vs_native"] == 0, identity
    prefixes.flush()
    evidence = {"image_ids": ids, "labels": labels, "text_features": text,
                "full_features": features, "full_scores": features @ text.T,
                "attention_scores": attention, "attention_features": attention_features,
                "attention_full_scores": attention_features @ text.T,
                "attention_patch_indices": selected}
    np.savez_compressed(folder / "evidence.npz", **evidence)
    baseline = {"dataset": "cub", "split": split, "config": config,
                "classification": reference_summary(evidence, manifest, split),
                "full_class_accuracy": float(np.mean(evidence["full_scores"].argmax(1) == labels)),
                "attention_full_class_accuracy": float(np.mean(evidence["attention_full_scores"].argmax(1) == labels))}
    (folder / "baseline_summary.json").write_text(json.dumps(baseline, indent=2) + "\n")
    (folder / "tasks.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (folder / "class_metadata.json").write_text(json.dumps(classes, indent=2) + "\n")
    summary = {"stage": "features", "split": split, "code_commit": commit,
               "config": config, "images": len(panel), "identity": identity,
               "prefix_bytes": int(prefixes.nbytes), "wall_seconds": time.perf_counter() - began,
               "completed_utc": datetime.now(timezone.utc).isoformat()}
    (folder / "feature_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"split": split, "baseline": baseline["classification"],
                      "full_class_accuracy": baseline["full_class_accuracy"]}), flush=True)
    return summary


def prepare_probes(config, output, split, pruner, device, commit):
    import torch

    began = time.perf_counter()
    folder = output / split
    prefixes = np.load(folder / "prefix_tokens.npy", mmap_mode="r")
    with np.load(folder / "evidence.npz") as evidence:
        ids, text = evidence["image_ids"].copy(), evidence["text_features"].copy()
    masks, retained, seed = mask_bank(config, pruner.num_patches)
    responses = np.empty((len(ids), len(masks), len(text)), np.float32)
    batch_size = config["teacher_batch_size"]
    batches = 0
    for image_index in range(len(ids)):
        tokens = torch.from_numpy(np.array(prefixes[image_index:image_index + 1])).to(device)
        parts = []
        for start in range(0, len(retained), batch_size):
            selected = torch.as_tensor(retained[start:start + batch_size], device=device)
            valid = len(selected)
            if valid < batch_size:
                selected = torch.cat([selected, selected[-1:].expand(batch_size - valid, -1)])
            parts.append(normalized(pruner.finish(tokens.expand(len(selected), -1, -1), selected))[:valid])
            batches += 1
        responses[image_index] = np.concatenate(parts) @ text.T
        if (image_index + 1) % 32 == 0 or image_index + 1 == len(ids):
            print(json.dumps({"stage": "probes", "split": split, "images": image_index + 1,
                              "total": len(ids), "seconds": time.perf_counter() - began}), flush=True)
    inference_seconds = time.perf_counter() - began
    np.savez_compressed(folder / "teacher.npz", image_ids=ids, masks=masks,
                        responses=responses, response_available=np.ones(len(ids), dtype=bool))
    summary = {"stage": "probes", "split": split, "code_commit": commit, "config": config,
               "images": len(ids), "mask_design_seed": seed, "probe_count": len(masks),
               "response_shape": list(responses.shape), "response_bytes": int(responses.nbytes),
               "suffix_batches": batches, "suffix_examples": batches * batch_size,
               "inference_seconds": inference_seconds, "wall_seconds": time.perf_counter() - began,
               "completed_utc": datetime.now(timezone.utc).isoformat()}
    (folder / "probe_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


def run(config, output, stage="all", split="both"):
    import clip
    import torch
    from token_compression.pruning import SingleStagePruner

    device = torch.device(config["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(config.get("seed", 2026))
    model, preprocess = clip.load(config["model"], device=device, jit=False,
                                  download_root=config["model_cache"])
    model.eval().requires_grad_(False)
    pruner = SingleStagePruner(model.visual, config["prune_after"])
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    splits = ("train", "val") if split == "both" else (split,)
    with torch.inference_mode():
        if stage in ("all", "features"):
            for selected_split in splits:
                prepare_features(config, output, selected_split, model, preprocess, pruner, device, commit)
        if stage in ("all", "probes"):
            for selected_split in splits:
                prepare_probes(config, output, selected_split, pruner, device, commit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=("features", "probes", "all"), default="all")
    parser.add_argument("--split", choices=("train", "val", "both"), default="both")
    parser.add_argument("--device")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.device:
        config["device"] = args.device
    run(config, args.output or Path(config["cache_dir"]), args.stage, args.split)
