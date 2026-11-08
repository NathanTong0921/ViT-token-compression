"""Create balanced candidate sets and fit their teachers from cached mask scores."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from token_compression.budget_ranking import ridge_projector
from token_compression.nway import build_panel, margin_responses, normalized_targets, task_manifest


def prepare(config, output):
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(config["cache_dir"])
    device = torch.device(config["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    with np.load(cache / "train" / "evidence.npz") as f:
        text = f["text_features"].copy()
        train_ids = f["image_ids"].copy()
    manifest = task_manifest(len(text), config["ways"], config["task_seed"],
                             config["train_task_rounds"], config["val_task_rounds"])
    (output / "tasks.json").write_text(json.dumps(manifest, indent=2) + "\n")
    text_tensor = torch.as_tensor(text, device=device, dtype=torch.float64)
    records = []
    for split in ("train", "val"):
        with np.load(cache / split / "evidence.npz") as f:
            ids, labels = f["image_ids"].copy(), f["labels"].copy()
            np.testing.assert_array_equal(text, f["text_features"])
        if split == "val":
            assert not set(ids) & set(train_ids)
        with np.load(cache / split / "teacher.npz") as f:
            np.testing.assert_array_equal(ids, f["image_ids"])
            assert f["response_available"].all()
            responses = f["responses"].copy()
            masks = f["masks"].copy()
        assert np.all(masks.sum(axis=1) == config["keep_patch_tokens"])
        projection = torch.as_tensor(ridge_projector(masks, config["ridge"])["projection"],
                                     dtype=torch.float64, device=device)
        # Responses stay on the host; each batch transfers only the images needed.
        for way in config["ways"]:
            before = time.perf_counter()
            panel = build_panel(ids, labels, manifest["ways"][str(way)][split])
            targets = np.empty((len(panel["labels"]), masks.shape[1]), np.float32)
            for begin in range(0, len(targets), config.get("teacher_batch_size", 64)):
                end = min(begin + config.get("teacher_batch_size", 64), len(targets))
                ix = panel["image_indices"][begin:end]
                batch = torch.as_tensor(responses[ix], device=device, dtype=torch.float64)
                y = torch.as_tensor(panel["labels"][begin:end], device=device)
                candidates = torch.as_tensor(panel["candidates"][begin:end], device=device)
                margins = margin_responses(batch, y, candidates, text_tensor)
                utilities = margins @ projection.T
                targets[begin:end] = normalized_targets(utilities, config["temperature"]).cpu().numpy()
            panel["targets"] = targets
            folder = output / "panels" / f"{way}way"
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / f"{split}.npz", **panel)
            records.append({"split": split, "way": way, "images": len(ids),
                            "tasks": len(manifest["ways"][str(way)][split]),
                            "examples": len(targets), "seconds": time.perf_counter() - before})
            print(json.dumps(records[-1]), flush=True)
        del responses
    summary = {"config": config, "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "wall_seconds": time.perf_counter() - start, "panels": records,
               "teacher": "Ridge fit of the minimum normalized true-class versus competitor margin over cached fixed-budget masks.",
               "descriptor": "Concatenate candidate-text mean and twice its population standard deviation.",
               "evaluation_split": "development validation images and unseen candidate sets"}
    (output / "preparation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(json.loads(args.config.read_text()), args.output)
