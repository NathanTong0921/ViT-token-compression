import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from token_compression.pets import balanced_task_rows, load_manifests, prepare_manifests


class PetsManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        annotations = self.root / "raw" / "annotations"
        annotations.mkdir(parents=True)
        class_names = ["Abyssinian", "american_bulldog", "Bengal", "beagle", "Birman"]
        lines = {"trainval": ["# image class species breed"], "test": []}
        self.official_test_ids = set()
        for class_id, name in enumerate(class_names):
            species = 1 if name[0].isupper() else 2
            for index in range(1, 11):
                split = "trainval" if index <= 8 else "test"
                image_id = f"{name}_{index}"
                lines[split].append(f"{image_id} {class_id + 1} {species} {class_id + 1}")
                if split == "test":
                    self.official_test_ids.add(image_id)
        for split, rows in lines.items():
            (annotations / f"{split}.txt").write_text("\n".join(rows) + "\n")
        self.output = self.root / "manifests"
        self.summary = prepare_manifests(self.root / "raw", self.output, val_fraction=0.25)
        self.images, self.classes, self.tasks = load_manifests(self.output)

    def test_label_mapping_and_official_test_isolation(self):
        self.assertEqual(self.classes[1], {
            "class_id": 1, "class_name": "american bulldog", "species": "dog"
        })
        actual_test = {row["image_id"] for row in self.images if row["split"] == "test"}
        self.assertEqual(actual_test, self.official_test_ids)
        self.assertTrue(all(row["official_split"] == "trainval"
                            for row in self.images if row["split"] in ("train", "val")))
        self.assertEqual(self.summary["image_counts"], {"train": 30, "val": 10, "test": 10})
        self.assertEqual({row["class_id"] for row in self.images}, set(range(5)))

    def test_manifests_are_deterministic_and_pairs_do_not_overlap(self):
        other = self.root / "repeat"
        prepare_manifests(self.root / "raw", other, val_fraction=0.25)
        for path in self.output.iterdir():
            self.assertEqual(path.read_bytes(), (other / path.name).read_bytes())
        pairs = {(row["class_a"], row["class_b"]) for row in self.tasks}
        self.assertEqual(len(pairs), 10)
        self.assertEqual(len(pairs), len(self.tasks))
        self.assertTrue(all(a < b for a, b in pairs))
        self.assertEqual(Counter(row["split"] for row in self.tasks), {"train": 8, "val": 1, "test": 1})
        self.assertEqual(self.summary["task_class_coverage"]["train"], list(range(5)))

    def test_two_task_coverage_counts_shared_anchor_comparisons(self):
        degrees = self.summary["per_class_task_degrees"]
        self.assertEqual([degrees[str(i)]["train"] for i in range(5)], [3, 4, 2, 3, 4])
        self.assertTrue(all(sum(row.values()) == 4 for row in degrees.values()))
        self.assertEqual(self.summary["two_task_coverage"]["train"], {
            "anchor_classes_with_degree_at_least_two": 5,
            "two_task_combinations": 19,
        })
        # One held-out edge contains two classes, but cannot support a
        # same-image comparison between two tasks sharing a class.
        for split in ("val", "test"):
            self.assertEqual(self.summary["two_task_coverage"][split], {
                "anchor_classes_with_degree_at_least_two": 0,
                "two_task_combinations": 0,
            })

    def test_balanced_task_rows_share_images_across_contexts(self):
        # Unequal class counts exercise the balancing rule; class 0 must use
        # the same prefix of its seeded image order in both candidate contexts.
        images = [
            {"image_id": f"class{class_id}_{i}", "class_id": class_id, "split": "test"}
            for class_id, count in ((0, 5), (1, 3), (2, 3)) for i in range(count)
        ]
        tasks = [
            {"task_id": "00_01", "class_a": 0, "class_b": 1, "split": "test"},
            {"task_id": "00_02", "class_a": 0, "class_b": 2, "split": "test"},
        ]
        rows = list(balanced_task_rows(images, tasks, "test", "test"))
        counts = Counter((row["task_id"], row["class_id"]) for row in rows)
        self.assertEqual(set(counts.values()), {3})
        class_zero_images = defaultdict(list)
        for row in rows:
            if row["class_id"] == 0:
                class_zero_images[row["task_id"]].append(row["image_id"])
        self.assertEqual(class_zero_images["00_01"], class_zero_images["00_02"])
        self.assertEqual(rows, list(balanced_task_rows(images, list(reversed(tasks)), "test", "test")))
        self.assertEqual(list(balanced_task_rows(images, tasks, "train", "test")), [])

    def test_duplicate_images_cannot_cross_official_split(self):
        path = self.root / "raw" / "annotations" / "test.txt"
        with path.open("a") as handle:
            handle.write("Abyssinian_1 1 1 1\n")
        with self.assertRaisesRegex(AssertionError, "Repeated image IDs"):
            prepare_manifests(self.root / "raw", self.root / "invalid")


if __name__ == "__main__":
    unittest.main()
