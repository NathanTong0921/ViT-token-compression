"""Official Flowers102 image splits and balanced candidate-class task graphs."""

from collections import Counter
import html
import json
from pathlib import Path
import random
import re

from .pets import IMAGE_FIELDS, TASK_FIELDS, _write_csv


OFFSETS = {"train": tuple(range(1, 7)), "val": (7, 8), "test": (9, 10)}
OFFICIAL_FILES = {"train": "trnid", "val": "valid", "test": "tstid"}
SOURCE_BASE = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"


def class_metadata(labels, category_html):
    """Link official named thumbnails to MAT labels, checking every class count.

    The category webpage is alphabetical, whereas MAT class IDs are not. Each
    thumbnail filename contains its original image ID, providing the join.
    """
    counts = Counter(int(label) for label in labels)
    entries = re.findall(
        r'<img src="thumbs/thumbim_(\d+)\.jpg"[^>]*alt="([^"]+)"[^>]*>'
        r'</td><td class=lab[LCR]>.*?</td><td class=num[LCR]>(\d+)</td>',
        category_html, flags=re.S,
    )
    assert len(entries) == len(counts), "Category webpage must cover every MAT class"
    mapped = {}
    for image_number, name, declared_count in entries:
        image_number, declared_count = int(image_number), int(declared_count)
        class_id = int(labels[image_number - 1]) - 1
        assert class_id not in mapped, "Repeated category mapping"
        assert counts[class_id + 1] == declared_count, "Official category count disagrees with MAT labels"
        mapped[class_id] = {
            "class_id": class_id, "class_name": html.unescape(name).strip(),
            "species": "flower", "thumbnail_image_id": f"image_{image_number:05d}",
            "official_image_count": declared_count,
        }
    assert sorted(mapped) == list(range(len(counts)))
    assert len({row["class_name"] for row in mapped.values()}) == len(counts)
    audit = [mapped[i] for i in sorted(mapped)]
    classes = [{key: row[key] for key in ("class_id", "class_name", "species")}
               for row in audit]
    return classes, audit


def balanced_task_graph(class_count=102, seed=2027):
    """Use disjoint circular offsets after one outcome-independent permutation."""
    assert max(max(offsets) for offsets in OFFSETS.values()) < class_count / 2
    permutation = list(range(class_count))
    random.Random(seed).shuffle(permutation)
    tasks = []
    for split, offsets in OFFSETS.items():
        for offset in offsets:
            for index, class_a in enumerate(permutation):
                class_a, class_b = sorted((class_a, permutation[(index + offset) % class_count]))
                tasks.append({"task_id": f"{class_a:03d}_{class_b:03d}",
                              "class_a": class_a, "class_b": class_b, "split": split})
    assert len({row["task_id"] for row in tasks}) == len(tasks)
    return sorted(tasks, key=lambda row: row["task_id"]), permutation


def prepare_manifests(data_root, output_dir, task_seed=2027):
    """Preserve official train/validation/test membership without image sampling."""
    from scipy.io import loadmat

    data_root, output_dir = Path(data_root), Path(output_dir)
    labels = loadmat(data_root / "imagelabels.mat")["labels"].ravel()
    official = loadmat(data_root / "setid.mat")
    assert len(labels) == 8189 and set(map(int, labels)) == set(range(1, 103))
    classes, mapping_audit = class_metadata(labels, (data_root / "categories.html").read_text())
    split_ids = {split: [int(i) for i in official[key].ravel()]
                 for split, key in OFFICIAL_FILES.items()}
    assert {split: len(ids) for split, ids in split_ids.items()} == {
        "train": 1020, "val": 1020, "test": 6149}
    combined = [i for ids in split_ids.values() for i in ids]
    assert len(set(combined)) == len(combined), "Official image splits overlap"
    assert sorted(combined) == list(range(1, len(labels) + 1))
    images = []
    for split, ids in split_ids.items():
        for image_number in ids:
            class_id = int(labels[image_number - 1]) - 1
            images.append({"image_id": f"image_{image_number:05d}", **classes[class_id],
                           "split": split, "official_split": OFFICIAL_FILES[split]})
    images.sort(key=lambda row: row["image_id"])
    per_class = {str(c): {split: sum(row["class_id"] == c and row["split"] == split
                                    for row in images)
                         for split in OFFICIAL_FILES} for c in range(len(classes))}
    assert all(row["train"] == 10 and row["val"] == 10 for row in per_class.values())
    tasks, permutation = balanced_task_graph(len(classes), task_seed)
    degrees = {split: Counter() for split in OFFICIAL_FILES}
    for row in tasks:
        degrees[row["split"]].update((row["class_a"], row["class_b"]))
    all_pair_count = len(classes) * (len(classes) - 1) // 2
    summary = {
        "dataset": "Oxford Flowers102", "class_count": len(classes), "image_count": len(images),
        "image_counts": {split: len(ids) for split, ids in split_ids.items()},
        "per_class_image_counts": per_class,
        "task_counts": dict(Counter(row["split"] for row in tasks)),
        "all_possible_pairs": all_pair_count, "unused_pairs": all_pair_count - len(tasks),
        "task_seed": task_seed, "task_permutation_rng": "Python random.Random(seed).shuffle",
        "permuted_class_ids": permutation,
        "task_offsets": {split: list(offsets) for split, offsets in OFFSETS.items()},
        "per_class_task_degrees": {str(c): {split: degrees[split][c] for split in OFFICIAL_FILES}
                                   for c in range(len(classes))},
        "task_class_coverage": {split: sorted(degrees[split]) for split in OFFICIAL_FILES},
        "validation_image_task_examples": sum(degrees["val"][c] * per_class[str(c)]["val"]
                                               for c in range(len(classes))),
        "validation_exchange_examples": sum(degrees["val"][c] * (degrees["val"][c] - 1) // 2
                                              * per_class[str(c)]["val"] for c in range(len(classes))),
        "image_split_protocol": "Official trnid/valid/tstid unchanged; 10 train and 10 validation images per class.",
        "task_split_protocol": "One seeded class permutation; disjoint circular offsets 1..6 train, 7..8 validation, 9..10 test. Remaining pairs unused.",
        "task_selection_uses_images_or_model_outputs": False,
        "class_names_protocol": "Official category thumbnail image IDs joined to imagelabels.mat; all published class counts checked. Whitespace stripped only.",
        "sources": {name: SOURCE_BASE + name for name in
                    ("102flowers.tgz", "imagelabels.mat", "setid.mat", "categories.html", "README.txt")},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "images.csv", IMAGE_FIELDS, images)
    _write_csv(output_dir / "tasks.csv", TASK_FIELDS, tasks)
    (output_dir / "classes.json").write_text(json.dumps(classes, indent=2) + "\n")
    (output_dir / "protocol_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "class_mapping_audit.json").write_text(json.dumps({
        "category_source": SOURCE_BASE + "categories.html",
        "label_source": SOURCE_BASE + "imagelabels.mat",
        "mapping_method": summary["class_names_protocol"], "classes": mapping_audit,
        "all_published_counts_match": True,
    }, indent=2) + "\n")
    return summary
