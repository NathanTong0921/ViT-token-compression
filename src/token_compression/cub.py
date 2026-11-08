"""CUB-200-2011 metadata and fixed development-image subsets."""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random

from PIL import Image
from torch.utils.data import Dataset

from .pets import _write_csv


IMAGE_FIELDS = (
    "image_id", "official_image_id", "relpath", "class_id", "class_name",
    "class_text", "species", "split", "official_split",
)
PROMPT_TEMPLATE = "a photo of a {class_name}, a type of bird."


def _indexed_text(path):
    return {int(index): value for index, value in
            (line.split(maxsplit=1) for line in path.read_text().splitlines() if line.strip())}


def prepare_manifests(data_root, output_dir, seed=2615,
                      train_per_class=20, val_per_class=5):
    """Sample only official training images; retain test membership unchanged.

    Each class's shuffled order first supplies validation images, then training
    images. The remaining official training images are marked ``unused``.
    Candidate sets are generated separately by the existing n-way pipeline.
    """
    data_root, output_dir = Path(data_root), Path(output_dir)
    paths = _indexed_text(data_root / "images.txt")
    labels = _indexed_text(data_root / "image_class_labels.txt")
    membership = _indexed_text(data_root / "train_test_split.txt")
    names = _indexed_text(data_root / "classes.txt")
    assert paths.keys() == labels.keys() == membership.keys()
    assert sorted(names) == list(range(1, len(names) + 1))
    assert train_per_class > 0 and val_per_class > 0
    classes = []
    for official_id, name in sorted(names.items()):
        class_name = name.split(".", 1)[1].replace("_", " ").lower()
        classes.append({"class_id": official_id - 1, "class_name": class_name,
                        "species": "bird", "official_class_name": name,
                        "class_text": PROMPT_TEMPLATE.format(class_name=class_name)})
    images, training_by_class = [], defaultdict(list)
    for official_id, relpath in sorted(paths.items()):
        class_id = int(labels[official_id]) - 1
        is_train = int(membership[official_id])
        assert is_train in (0, 1)
        metadata = classes[class_id]
        row = {"image_id": str(Path(relpath).with_suffix("")),
               "official_image_id": official_id, "relpath": relpath,
               **{key: metadata[key] for key in
                  ("class_id", "class_name", "class_text", "species")},
               "official_split": "train" if is_train else "test",
               "split": "unused" if is_train else "test"}
        images.append(row)
        if is_train:
            training_by_class[class_id].append(row)
    rng = random.Random(seed)
    for class_id in range(len(classes)):
        rows = training_by_class[class_id]
        assert len(rows) >= train_per_class + val_per_class, (
            f"Class {class_id} has only {len(rows)} official training images")
        rng.shuffle(rows)
        for row in rows[:val_per_class]:
            row["split"] = "val"
        for row in rows[val_per_class:val_per_class + train_per_class]:
            row["split"] = "train"
    images.sort(key=lambda row: row["official_image_id"])
    counts = Counter(row["split"] for row in images)
    per_class = Counter((row["class_id"], row["split"]) for row in images)
    annotation_candidates = {
        "bounding_boxes": "bounding_boxes.txt",
        "part_locations": "parts/part_locs.txt",
        "part_names": "parts/parts.txt",
        "image_attributes": "attributes/image_attribute_labels.txt",
        "class_attributes": "attributes/class_attribute_labels_continuous.txt",
        "attribute_names": "attributes/attributes.txt",
    }
    summary = {
        "dataset": "CUB-200-2011", "class_count": len(classes),
        "image_count": len(images), "image_seed": seed,
        "train_per_class": train_per_class, "val_per_class": val_per_class,
        "image_counts": {split: counts[split] for split in ("train", "val", "test", "unused")},
        "per_class_image_counts": {
            str(c): {split: per_class[c, split] for split in ("train", "val", "test", "unused")}
            for c in range(len(classes))},
        "image_split_protocol": (
            "Within each official training class, shuffle numeric image IDs using "
            "Python random.Random(image_seed), take validation first and training next; "
            "mark the rest unused. Official test images are not sampled or evaluated."),
        "prompt_template": PROMPT_TEMPLATE,
        "class_name_protocol": "Remove numeric prefix, replace underscores with spaces, lowercase.",
        "candidate_task_protocol": "Generated separately by token_compression.nway.task_manifest.",
        "annotation_files": {key: path for key, path in annotation_candidates.items()
                             if (data_root / path).is_file()},
        "annotation_path_base": "dataset root",
        "source": "https://www.vision.caltech.edu/datasets/cub_200_2011/",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "images.csv", IMAGE_FIELDS, images)
    (output_dir / "classes.json").write_text(json.dumps(classes, indent=2) + "\n")
    (output_dir / "protocol_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def load_manifests(output_dir):
    """Return image records and class metadata; candidate tasks live separately."""
    output_dir = Path(output_dir)
    with (output_dir / "images.csv").open(newline="") as stream:
        images = list(csv.DictReader(stream))
    for row in images:
        row["class_id"] = int(row["class_id"])
        row["official_image_id"] = int(row["official_image_id"])
    return images, json.loads((output_dir / "classes.json").read_text())


class CUBImages(Dataset):
    def __init__(self, data_root, rows, preprocess):
        self.image_dir = Path(data_root) / "images"
        self.rows = rows
        self.preprocess = preprocess

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.image_dir / row["relpath"]) as image:
            pixels = self.preprocess(image.convert("RGB"))
        return pixels, row["class_id"], row["image_id"]
