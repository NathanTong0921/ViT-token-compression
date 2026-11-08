"""Check CUB split membership, class labels and nested image loading."""

from collections import Counter
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from token_compression.cub import CUBImages, load_manifests, prepare_manifests


class CUBManifestTests(unittest.TestCase):
    def fixture(self, root):
        raw = root / "raw"
        raw.mkdir()
        names = ["001.Black_footed_Albatross", "002.Laysan_Albatross", "003.Sooty_Albatross"]
        files = {name: [] for name in
                 ("images.txt", "image_class_labels.txt", "train_test_split.txt", "classes.txt")}
        expected_test, labels = set(), {}
        for c, name in enumerate(names, 1):
            files["classes.txt"].append(f"{c} {name}")
            for index in range(8):
                official_id = (c - 1) * 8 + index + 1
                relpath = f"{name}/bird_{official_id:04d}.jpg"
                files["images.txt"].append(f"{official_id} {relpath}")
                files["image_class_labels.txt"].append(f"{official_id} {c}")
                files["train_test_split.txt"].append(f"{official_id} {int(index < 6)}")
                labels[official_id] = c - 1
                if index >= 6:
                    expected_test.add(official_id)
        for name, lines in files.items():
            (raw / name).write_text("\n".join(reversed(lines)) + "\n")
        (raw / "parts").mkdir()
        (raw / "parts/part_locs.txt").write_text("1 1 2 3 1\n")
        (raw / "bounding_boxes.txt").write_text("1 0 0 6 4\n")
        return raw, expected_test, labels

    def test_fixed_counts_preserve_test_and_class_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw, expected_test, labels = self.fixture(root)
            summary = prepare_manifests(raw, root / "out", train_per_class=3, val_per_class=2)
            rows, classes = load_manifests(root / "out")
            self.assertEqual(summary["image_counts"], {"train": 9, "val": 6, "test": 6, "unused": 3})
            self.assertEqual({r["official_image_id"] for r in rows if r["split"] == "test"}, expected_test)
            self.assertTrue(all(r["official_split"] == "train" for r in rows
                                if r["split"] in ("train", "val", "unused")))
            self.assertEqual({r["official_image_id"]: r["class_id"] for r in rows}, labels)
            self.assertEqual(classes[0]["class_name"], "black footed albatross")
            self.assertEqual(classes[0]["class_text"], "a photo of a black footed albatross, a type of bird.")
            self.assertEqual(Counter((r["class_id"], r["split"]) for r in rows),
                             {(c, s): n for c in range(3) for s, n in
                              (("train", 3), ("val", 2), ("test", 2), ("unused", 1))})
            self.assertEqual(summary["annotation_files"]["part_locations"], "parts/part_locs.txt")
            self.assertEqual(len({r["image_id"] for r in rows}), len(rows))

    def test_reproducible_selection_and_seed_changes_only_official_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw, _, _ = self.fixture(root)
            for name, seed in (("out", 2615), ("repeat", 2615), ("other", 2616)):
                prepare_manifests(raw, root / name, seed, train_per_class=3, val_per_class=2)
            for p in (root / "out").iterdir():
                self.assertEqual(p.read_bytes(), (root / "repeat" / p.name).read_bytes())
            first, _ = load_manifests(root / "out")
            other, _ = load_manifests(root / "other")
            self.assertEqual([r for r in first if r["split"] == "test"],
                             [r for r in other if r["split"] == "test"])
            self.assertNotEqual([r["image_id"] for r in first if r["split"] == "val"],
                                [r["image_id"] for r in other if r["split"] == "val"])

    def test_nested_image_path_and_rgb_conversion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw, _, _ = self.fixture(root)
            prepare_manifests(raw, root / "out", train_per_class=3, val_per_class=2)
            rows, _ = load_manifests(root / "out")
            row = rows[0]
            path = raw / "images" / row["relpath"]
            path.parent.mkdir(parents=True)
            Image.new("L", (6, 4), color=90).save(path)
            dataset = CUBImages(raw, [row], lambda image: (image.mode, image.size))
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0], (("RGB", (6, 4)), row["class_id"], row["image_id"]))


if __name__ == "__main__":
    unittest.main()
