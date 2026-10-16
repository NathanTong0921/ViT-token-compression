"""Original frozen CLIP baseline on explicit image and candidate-pair splits."""

import csv
import json
import platform
import random
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .metrics import binary_margin, direction_angle
from .pets import balanced_task_rows, load_manifests


class PetsImages(Dataset):
    def __init__(self, data_root, rows, preprocess):
        self.image_dir = Path(data_root) / "images"
        self.rows = rows
        self.preprocess = preprocess

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.image_dir / (row["image_id"] + ".jpg")) as image:
            pixels = self.preprocess(image.convert("RGB"))
        return pixels, row["class_id"], row["image_id"]


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_baseline(config, output_dir):
    started = datetime.now(timezone.utc).isoformat()
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images, classes, tasks = load_manifests(config["manifest_dir"])
    selected_images = [row for row in images if row["split"] == config["image_split"]]
    device = torch.device(config["device"])
    model, preprocess = clip.load(config["model"], device=device, jit=False,
                                  download_root=config["model_cache"])
    model.eval().requires_grad_(False)
    prompts = [config["prompt_template"].format(class_name=row["class_name"])
               for row in classes]
    loader = DataLoader(PetsImages(config["data_root"], selected_images, preprocess),
                        batch_size=config["batch_size"], shuffle=False,
                        num_workers=config["num_workers"], pin_memory=device.type == "cuda")

    with torch.inference_mode():
        text = model.encode_text(clip.tokenize(prompts).to(device)).float()
        text /= text.norm(dim=-1, keepdim=True)
        text_features = text.cpu().numpy()
        logit_scale = model.logit_scale.exp().item()
        image_features, labels, image_ids = [], [], []
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        before = time.perf_counter()
        for pixels, batch_labels, batch_ids in loader:
            features = model.encode_image(pixels.to(device, non_blocking=True)).float()
            features /= features.norm(dim=-1, keepdim=True)
            image_features.append(features.cpu().numpy())
            labels.extend(batch_labels.tolist())
            image_ids.extend(batch_ids)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        extraction_seconds = time.perf_counter() - before
    image_features = np.concatenate(image_features)
    labels = np.asarray(labels)
    scores = image_features @ text_features.T
    predictions = scores.argmax(axis=1)
    lookup = {image_id: index for index, image_id in enumerate(image_ids)}
    np.savez_compressed(output_dir / "embeddings.npz", image_ids=np.asarray(image_ids),
                        labels=labels, image_features=image_features,
                        text_features=text_features, cosine_scores=scores)

    full_rows = []
    for index, (image_id, label) in enumerate(zip(image_ids, labels)):
        others = scores[index].copy()
        others[label] = -np.inf
        competitor = int(others.argmax())
        full_rows.append({
            "image_id": image_id, "class_id": int(label),
            "prediction": int(predictions[index]),
            "correct": int(predictions[index] == label),
            "strongest_competitor": competitor,
            "true_class_cosine": float(scores[index, label]),
            "cosine_margin": float(scores[index, label] - scores[index, competitor]),
        })
    write_csv(output_dir / "predictions_37class.csv", full_rows)

    pair_rows, task_summaries = [], []
    for task_split in config["task_splits"]:
        by_task = defaultdict(list)
        for row in balanced_task_rows(images, tasks, config["image_split"], task_split,
                                       seed=config["balance_seed"]):
            index = lookup[row["image_id"]]
            a, b, label = row["class_a"], row["class_b"], row["class_id"]
            competitor = b if label == a else a
            prediction = (a, b)[int(np.argmax(scores[index, [a, b]]))]
            gap, margin = binary_margin(scores[index], text_features, label, competitor)
            result = {**row, "task_split": task_split, "prediction": prediction,
                      "correct": int(prediction == label), "cosine_margin": gap,
                      "geometric_margin": margin}
            pair_rows.append(result)
            by_task[row["task_id"]].append(result)
        for task_id, rows in by_task.items():
            task_summaries.append({
                "task_id": task_id, "task_split": task_split,
                "class_a": rows[0]["class_a"], "class_b": rows[0]["class_b"],
                "n": len(rows), "correct": sum(row["correct"] for row in rows),
                "accuracy": float(np.mean([row["correct"] for row in rows])),
                "mean_geometric_margin": float(np.mean([row["geometric_margin"] for row in rows])),
            })
    write_csv(output_dir / "predictions_binary.csv", pair_rows)
    write_csv(output_dir / "tasks.csv", task_summaries)

    # Geometric task descriptors do not require test-image predictions.
    angles = []
    for anchor in range(len(classes)):
        competitors = [c for c in range(len(classes)) if c != anchor]
        for a, b in combinations(competitors, 2):
            angles.append({"anchor_class": anchor, "competitor_a": a, "competitor_b": b,
                           "angle_degrees": direction_angle(text_features, anchor, a, b)})
    write_csv(output_dir / "task_angles.csv", angles)

    summary = {
        "run_type": "unpruned_frozen_clip_baseline",
        "started_utc": started,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config": config,
        "model_training": False,
        "backbone_dtype": str(model.dtype),
        "normalization_and_scoring_dtype": "float32",
        "logit_scale": logit_scale,
        "preprocess": repr(preprocess),
        "prompts": prompts,
        "image_count": len(labels),
        "accuracy_37class": float(np.mean(predictions == labels)),
        "correct_37class": int(np.sum(predictions == labels)),
        "binary": {},
        "image_extraction_wall_seconds": extraction_seconds,
        "timing_scope": "One feature extraction pass, including data loading and CPU transfers; not an inference latency benchmark.",
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "cuda": torch.version.cuda},
    }
    for split in config["task_splits"]:
        rows = [row for row in task_summaries if row["task_split"] == split]
        total = sum(row["n"] for row in rows)
        summary["binary"][split] = {
            "task_count": len(rows), "image_task_count": total,
            "accuracy_micro": sum(row["correct"] for row in rows) / total,
            "accuracy_macro": float(np.mean([row["accuracy"] for row in rows])),
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
