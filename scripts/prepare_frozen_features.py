"""Cache frozen CLIP prefixes and native baselines for a manifested dataset."""

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def run(config, output):
    import clip
    import torch
    from torch.utils.data import DataLoader

    from token_compression.baseline import PetsImages as ManifestImages, write_csv
    from token_compression.metrics import binary_margin
    from token_compression.pets import load_manifests, balanced_task_rows
    from token_compression.pruning import SingleStagePruner

    before = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    images, classes, tasks = load_manifests(config["manifest_dir"])
    panels = {split: [row for row in images if row["split"] == split]
              for split in ("train", "val")}
    selected_tasks = {split: [row for row in tasks if row["split"] == split]
                      for split in panels}
    assert {r["image_id"] for r in panels["train"]}.isdisjoint(
        r["image_id"] for r in panels["val"])
    output.mkdir(parents=True)
    write_csv(output / "panel_images.csv", panels["train"] + panels["val"])
    for split in panels:
        write_csv(output / f"{split}_tasks.csv", selected_tasks[split])

    device = torch.device(config["device"])
    torch.manual_seed(config["seed"])
    model, preprocess = clip.load(config["model"], device=device, jit=False,
                                 download_root=config["model_cache"])
    model.eval().requires_grad_(False)
    pruner = SingleStagePruner(model.visual, config["prune_after"])
    prompts = [config["prompt_template"].format(class_name=row["class_name"])
               for row in classes]

    def normalized(features):
        features = features.float()
        return (features / features.norm(dim=-1, keepdim=True)).cpu().numpy()

    with torch.inference_mode():
        text = normalized(model.encode_text(clip.tokenize(prompts).to(device)))
    torch.cuda.reset_peak_memory_stats(device)
    setup_seconds = time.perf_counter() - before
    counts, identities, baselines, costs = {}, {}, {}, {}
    with torch.inference_mode():
        for split, panel in panels.items():
            start = time.perf_counter()
            folder = output / split
            folder.mkdir()
            write_csv(folder / "panel_images.csv", panel)
            ids = np.asarray([row["image_id"] for row in panel])
            labels = np.asarray([row["class_id"] for row in panel])
            loader = DataLoader(ManifestImages(config["data_root"], panel, preprocess),
                                batch_size=config["image_batch_size"], shuffle=False,
                                num_workers=config["num_workers"], pin_memory=True)
            features = np.empty((len(panel), text.shape[1]), np.float32)
            attention = np.empty((len(panel), pruner.num_patches), np.float32)
            prefixes, offset = None, 0
            identity = {"images_checked": 0, "max_feature_error_vs_native": 0.,
                        "multiclass_prediction_disagreements_vs_native": 0}
            for pixels, batch_labels, batch_ids in loader:
                pixels = pixels.to(device)
                tokens, batch_attention = pruner.prefix(pixels)
                keep_all = torch.arange(pruner.num_patches, device=device).expand(len(pixels), -1)
                recomposed = normalized(pruner.finish(tokens, keep_all))
                native = normalized(model.encode_image(pixels))
                np.testing.assert_array_equal(ids[offset:offset + len(pixels)], batch_ids)
                np.testing.assert_array_equal(labels[offset:offset + len(pixels)], batch_labels.numpy())
                prefix_values = tokens.cpu().numpy()
                if prefixes is None:
                    prefixes = np.lib.format.open_memmap(
                        folder / "prefix_tokens.npy", mode="w+", dtype=prefix_values.dtype,
                        shape=(len(panel),) + prefix_values.shape[1:])
                prefixes[offset:offset + len(pixels)] = prefix_values
                features[offset:offset + len(pixels)] = native
                attention[offset:offset + len(pixels)] = batch_attention.cpu().numpy()
                identity["images_checked"] += len(pixels)
                identity["max_feature_error_vs_native"] = max(
                    identity["max_feature_error_vs_native"], float(np.max(np.abs(recomposed - native))))
                identity["multiclass_prediction_disagreements_vs_native"] += int(np.sum(
                    (recomposed @ text.T).argmax(1) != (native @ text.T).argmax(1)))
                offset += len(pixels)
            assert offset == len(panel)
            assert identity["max_feature_error_vs_native"] <= config["identity_atol"], identity
            assert identity["multiclass_prediction_disagreements_vs_native"] == 0, identity
            prefixes.flush()
            scores = features @ text.T
            np.savez_compressed(folder / "evidence.npz", image_ids=ids, labels=labels,
                                text_features=text, full_features=features, full_scores=scores,
                                attention_scores=attention)
            predictions = scores.argmax(1)
            baselines[split] = {"multiclass_accuracy": float(np.mean(predictions == labels)),
                                "class_count": len(classes), "binary": {}}
            lookup = {str(image_id): i for i, image_id in enumerate(ids)}
            for task_split in ("train", "val"):
                binary = []
                for row in balanced_task_rows(images, tasks, split, task_split,
                                              seed=config["balance_seed"]):
                    ix = lookup[row["image_id"]]
                    a, b, label = row["class_a"], row["class_b"], row["class_id"]
                    competitor = b if label == a else a
                    prediction = (a, b)[int(np.argmax(scores[ix, [a, b]]))]
                    _, margin = binary_margin(scores[ix], text, label, competitor)
                    binary.append({**row, "task_split": task_split,
                                   "correct": int(prediction == label), "margin": margin})
                write_csv(folder / f"baseline_{task_split}_tasks.csv", binary)
                baselines[split]["binary"][task_split] = {
                    "image_task_count": len(binary),
                    "accuracy": float(np.mean([r["correct"] for r in binary])),
                    "geometric_margin": float(np.mean([r["margin"] for r in binary]))}
                if split == "train" and task_split == "train":
                    edges = np.quantile([r["margin"] for r in binary], [.25, .5, .75]).tolist()
                    (output / "exchange_strata.json").write_text(json.dumps({
                        "config": {"margin_quartile_edges": edges},
                        "source": "Uncompressed training images and training tasks only.",
                        "purpose": "Fixed descriptive exchange strata; H2 uses continuous controls.",
                    }, indent=2) + "\n")
            identities[split] = identity
            counts[split] = {"images": len(panel), "classes": len(np.unique(labels)),
                             "tasks": len(selected_tasks[split]),
                             "prefix_shape": list(prefixes.shape), "prefix_bytes": prefixes.nbytes}
            costs[f"{split}_seconds"] = time.perf_counter() - start
            print(json.dumps({"split": split, "identity": identity,
                              "baseline": baselines[split]}), flush=True)
            del prefixes
    summary = {"run_type": "frozen_clip_feature_cache_and_native_baseline",
               "started_utc": started, "code_commit": commit, "config": config,
               "model_geometry": {"input_resolution": int(model.visual.input_resolution),
                                  "patch_size": int(model.visual.conv1.weight.shape[-1]),
                                  "patch_tokens": pruner.num_patches,
                                  "visual_blocks": len(pruner.blocks),
                                  "token_dim": int(model.visual.conv1.weight.shape[0]),
                                  "embedding_dim": int(text.shape[1]),
                                  "prune_after": config["prune_after"]},
               "counts": counts, "identity": identities, "baseline": baselines,
               "preprocess": repr(preprocess), "prompts": prompts, "test_images_used": False,
               "cost": {**costs, "model_setup_seconds": setup_seconds,
                        "total_wall_seconds": time.perf_counter() - before,
                        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(device)},
               "cost_scope": "Feature extraction and baseline validation include loading and CPU/I/O; not deployment latency.",
               "gpu": torch.cuda.get_device_name(device),
               "environment": {"python": platform.python_version(), "torch": torch.__version__,
                               "numpy": np.__version__}}
    (output / "prep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    run(config, args.output or Path(config["feature_cache_dir"]))
