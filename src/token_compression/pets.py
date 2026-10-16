"""Oxford-IIIT Pets image splits and candidate-class task manifests."""

import csv
import json
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


IMAGE_FIELDS = (
    "image_id", "class_id", "class_name", "species", "split", "official_split"
)
TASK_FIELDS = ("task_id", "class_a", "class_b", "split")


def _read_annotations(data_root):
    images = []
    classes = {}
    for official_split, filename in (("trainval", "trainval.txt"), ("test", "test.txt")):
        path = Path(data_root) / "annotations" / filename
        for line in path.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            image_id, label, species_id, _ = line.split()
            class_id = int(label) - 1
            class_name = image_id.rsplit("_", 1)[0].replace("_", " ").lower()
            species = {1: "cat", 2: "dog"}[int(species_id)]
            class_row = {"class_id": class_id, "class_name": class_name, "species": species}
            if class_id in classes:
                assert classes[class_id] == class_row, f"Inconsistent class metadata: {image_id}"
            classes[class_id] = class_row
            images.append({
                "image_id": image_id,
                **class_row,
                "split": "test" if official_split == "test" else "train",
                "official_split": official_split,
            })
    assert len({row["image_id"] for row in images}) == len(images), "Repeated image IDs"
    assert sorted(classes) == list(range(len(classes))), "Class IDs must be contiguous"
    return images, [classes[class_id] for class_id in sorted(classes)]


def _write_csv(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare_manifests(data_root, output_dir, seed=2026, task_seed=2027, val_fraction=0.2):
    """Preserve official test images and independently split unordered class pairs.

    The train/validation image split uses only official trainval annotations.
    Task splits hold out class combinations, not individual classes.
    """
    images, classes = _read_annotations(data_root)
    trainval_by_class = defaultdict(list)
    for row in images:
        if row["official_split"] == "trainval":
            trainval_by_class[row["class_id"]].append(row)
    image_rng = random.Random(seed)
    for class_id in sorted(trainval_by_class):
        rows = sorted(trainval_by_class[class_id], key=lambda row: row["image_id"])
        image_rng.shuffle(rows)
        n_val = round(len(rows) * val_fraction)
        for row in rows[:n_val]:
            row["split"] = "val"
    images.sort(key=lambda row: row["image_id"])

    pairs = list(combinations(range(len(classes)), 2))
    random.Random(task_seed).shuffle(pairs)
    n_train, n_val = int(0.8 * len(pairs)), int(0.1 * len(pairs))
    tasks = []
    for index, (class_a, class_b) in enumerate(pairs):
        split = "train" if index < n_train else "val" if index < n_train + n_val else "test"
        tasks.append({
            "task_id": f"{class_a:02d}_{class_b:02d}",
            "class_a": class_a,
            "class_b": class_b,
            "split": split,
        })
    tasks.sort(key=lambda row: row["task_id"])

    image_counts = Counter(row["split"] for row in images)
    task_counts = Counter(row["split"] for row in tasks)
    task_degrees = {split: Counter() for split in ("train", "val", "test")}
    for task in tasks:
        task_degrees[task["split"]].update((task["class_a"], task["class_b"]))
    summary = {
        "dataset": "Oxford-IIIT Pet",
        "image_seed": seed,
        "task_seed": task_seed,
        "validation_fraction_of_official_trainval": val_fraction,
        "image_counts": {split: image_counts[split] for split in ("train", "val", "test")},
        "task_counts": {split: task_counts[split] for split in ("train", "val", "test")},
        "class_count": len(classes),
        "per_class_image_counts": {
            str(row["class_id"]): {
                split: sum(image["class_id"] == row["class_id"] and image["split"] == split
                           for image in images)
                for split in ("train", "val", "test")
            }
            for row in classes
        },
        "task_class_coverage": {
            split: sorted({task[key] for task in tasks if task["split"] == split
                           for key in ("class_a", "class_b")})
            for split in ("train", "val", "test")
        },
        "per_class_task_degrees": {
            str(row["class_id"]): {
                split: task_degrees[split][row["class_id"]]
                for split in ("train", "val", "test")
            }
            for row in classes
        },
        "two_task_coverage": {
            split: {
                "anchor_classes_with_degree_at_least_two": sum(
                    degree >= 2 for degree in task_degrees[split].values()
                ),
                "two_task_combinations": sum(
                    degree * (degree - 1) // 2 for degree in task_degrees[split].values()
                ),
            }
            for split in ("train", "val", "test")
        },
        "image_split_protocol": "Stratified train/val from official trainval; official test unchanged.",
        "task_split_protocol": "Unordered class pairs shuffled independently; 80% train, 10% val, remainder test.",
        "balanced_task_protocol": "A shared seeded image order per class; equal examples from both task members.",
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "images.csv", IMAGE_FIELDS, images)
    _write_csv(output_dir / "tasks.csv", TASK_FIELDS, tasks)
    (output_dir / "classes.json").write_text(json.dumps(classes, indent=2) + "\n")
    (output_dir / "protocol_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def load_manifests(output_dir):
    """Return image rows, class metadata, and unordered task rows."""
    output_dir = Path(output_dir)
    with (output_dir / "images.csv").open(newline="") as handle:
        images = list(csv.DictReader(handle))
    for row in images:
        row["class_id"] = int(row["class_id"])
    classes = json.loads((output_dir / "classes.json").read_text())
    with (output_dir / "tasks.csv").open(newline="") as handle:
        tasks = list(csv.DictReader(handle))
    for row in tasks:
        row["class_a"], row["class_b"] = int(row["class_a"]), int(row["class_b"])
    return images, classes, tasks


def balanced_task_rows(images, tasks, image_split, task_split, seed=2028):
    """Yield balanced image/task examples without copying image files.

    A class's image ordering is fixed across tasks, so the same image can be
    evaluated under different competing classes. Image and task split axes are
    separate: e.g. test images can evaluate seen or held-out class pairs.
    """
    by_class = defaultdict(list)
    for row in images:
        if row["split"] == image_split:
            by_class[row["class_id"]].append(row["image_id"])
    rng = random.Random(seed)
    for class_id in sorted(by_class):
        by_class[class_id].sort()
        rng.shuffle(by_class[class_id])

    for task in sorted(tasks, key=lambda row: row["task_id"]):
        if task["split"] != task_split:
            continue
        class_a, class_b = task["class_a"], task["class_b"]
        n = min(len(by_class[class_a]), len(by_class[class_b]))
        for class_id in (class_a, class_b):
            for image_id in by_class[class_id][:n]:
                yield {
                    "image_id": image_id,
                    "class_id": class_id,
                    "task_id": task["task_id"],
                    "class_a": class_a,
                    "class_b": class_b,
                }
