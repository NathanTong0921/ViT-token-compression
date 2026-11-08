"""Cache the reserved test images with the fixed frozen-CLIP inference path."""

import argparse
from itertools import combinations
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def test_partition(images, tasks):
    """Check the two held-out axes and return their original manifest rows."""
    image_sets = {split: {r["image_id"] for r in images if r["split"] == split}
                  for split in ("train", "val", "test")}
    pair_sets = {split: {tuple(sorted((r["class_a"], r["class_b"])))
                         for r in tasks if r["split"] == split}
                 for split in image_sets}
    for left, right in combinations(image_sets, 2):
        assert image_sets[left].isdisjoint(image_sets[right]), (left, right, "images")
        assert pair_sets[left].isdisjoint(pair_sets[right]), (left, right, "class pairs")
    panel = [r for r in images if r["split"] == "test"]
    assert panel and len(panel) == len(image_sets["test"])
    selected_tasks = {split: [r for r in tasks if r["split"] == split]
                      for split in ("train", "test")}
    assert all(selected_tasks.values())
    return panel, selected_tasks


def run(config, output):
    import clip
    import torch
    from torch.utils.data import DataLoader

    from token_compression.baseline import PetsImages as ManifestImages, write_csv
    from token_compression.pets import load_manifests
    from token_compression.pruning import SingleStagePruner

    before = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    images, classes, tasks = load_manifests(config["manifest_dir"])
    panel, selected_tasks = test_partition(images, tasks)
    output = Path(output)
    output.mkdir(parents=True)
    folder = output / "test"
    folder.mkdir()
    write_csv(output / "panel_images.csv", panel)
    write_csv(folder / "panel_images.csv", panel)
    for split, rows in selected_tasks.items():
        write_csv(output / f"{split}_tasks.csv", rows)

    device = torch.device(config["device"])
    torch.manual_seed(config["seed"])
    model, preprocess = clip.load(config["model"], device=device, jit=False,
                                 download_root=config["model_cache"])
    model.eval().requires_grad_(False)
    pruner = SingleStagePruner(model.visual, config["prune_after"])
    prompts = [config["prompt_template"].format(class_name=r["class_name"])
               for r in classes]

    def normalized(features):
        features = features.float()
        return (features / features.norm(dim=-1, keepdim=True)).cpu().numpy()

    with torch.inference_mode():
        text = normalized(model.encode_text(clip.tokenize(prompts).to(device)))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    setup_seconds = time.perf_counter() - before
    ids = np.asarray([r["image_id"] for r in panel])
    labels = np.asarray([r["class_id"] for r in panel])
    batch_size = config["image_batch_size"]
    loader = DataLoader(ManifestImages(config["data_root"], panel, preprocess),
                        batch_size=batch_size, shuffle=False,
                        num_workers=config["num_workers"], pin_memory=device.type == "cuda")
    features = np.empty((len(panel), text.shape[1]), np.float32)
    attention = np.empty((len(panel), pruner.num_patches), np.float32)
    prefixes, offset, physical = None, 0, 0
    identity = {"images_checked": 0, "max_feature_error_vs_native": 0.,
                "multiclass_prediction_disagreements_vs_native": 0}
    extract_started = time.perf_counter()
    with torch.inference_mode():
        for pixels, batch_labels, batch_ids in loader:
            valid = len(pixels)
            np.testing.assert_array_equal(ids[offset:offset + valid], batch_ids)
            np.testing.assert_array_equal(labels[offset:offset + valid], batch_labels.numpy())
            if valid < batch_size:
                pixels = torch.cat([pixels, pixels[-1:].expand(
                    (batch_size - valid,) + pixels.shape[1:])])
            pixels = pixels.to(device)
            tokens, batch_attention = pruner.prefix(pixels)
            keep_all = torch.arange(pruner.num_patches, device=device).expand(len(pixels), -1)
            recomposed = normalized(pruner.finish(tokens, keep_all))[:valid]
            native = normalized(model.encode_image(pixels))[:valid]
            prefix_values = tokens[:valid].cpu().numpy()
            if prefixes is None:
                prefixes = np.lib.format.open_memmap(
                    folder / "prefix_tokens.npy", mode="w+", dtype=prefix_values.dtype,
                    shape=(len(panel),) + prefix_values.shape[1:])
            prefixes[offset:offset + valid] = prefix_values
            features[offset:offset + valid] = native
            attention[offset:offset + valid] = batch_attention[:valid].cpu().numpy()
            identity["images_checked"] += valid
            identity["max_feature_error_vs_native"] = max(
                identity["max_feature_error_vs_native"], float(np.max(np.abs(recomposed - native))))
            identity["multiclass_prediction_disagreements_vs_native"] += int(np.sum(
                (recomposed @ text.T).argmax(1) != (native @ text.T).argmax(1)))
            offset += valid
            physical += len(pixels)
    assert offset == len(panel)
    assert identity["max_feature_error_vs_native"] <= config["identity_atol"], identity
    assert identity["multiclass_prediction_disagreements_vs_native"] == 0, identity
    prefixes.flush()
    scores = features @ text.T
    np.savez_compressed(folder / "evidence.npz", image_ids=ids, labels=labels,
                        text_features=text, full_features=features, full_scores=scores,
                        attention_scores=attention)
    counts = {"images": len(panel), "classes": len(np.unique(labels)),
              "tasks": len(selected_tasks["test"]), "prefix_shape": list(prefixes.shape),
              "prefix_bytes": prefixes.nbytes}
    summary = {
        "run_type": "reserved_test_frozen_clip_cache",
        "started_utc": started, "code_commit": commit, "config": config,
        "model_geometry": {"input_resolution": int(model.visual.input_resolution),
                           "patch_size": int(model.visual.conv1.weight.shape[-1]),
                           "patch_tokens": pruner.num_patches, "visual_blocks": len(pruner.blocks),
                           "token_dim": int(model.visual.conv1.weight.shape[0]),
                           "embedding_dim": int(text.shape[1]), "prune_after": config["prune_after"]},
        "counts": {"test": counts}, "identity": {"test": identity},
        "baseline": {"test": {"multiclass_accuracy": float(np.mean(scores.argmax(1) == labels)),
                               "class_count": len(classes)}},
        "preprocess": repr(preprocess), "prompts": prompts, "test_images_used": True,
        "task_splits_saved": ["train", "test"],
        "batching": {"batch_size": batch_size, "last_batch": "repeat final image; discard padded outputs",
                     "logical_images": len(panel), "physical_images_per_forward_path": physical},
        "cost": {"model_setup_seconds": setup_seconds,
                 "test_extraction_seconds": time.perf_counter() - extract_started,
                 "total_wall_seconds": time.perf_counter() - before,
                 "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(device)
                 if device.type == "cuda" else 0},
        "cost_scope": "Extraction includes image loading, keep-all validation and cache writing; not inference latency.",
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__},
    }
    (output / "prep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"split": "test", "counts": counts, "identity": identity,
                      "seconds": summary["cost"]["total_wall_seconds"]}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)
