"""Verify official label joins, fixed image splits and balanced held-out tasks."""

from collections import Counter
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.io import savemat

from token_compression.flowers import balanced_task_graph, class_metadata, prepare_manifests
from token_compression.pets import load_manifests


class FlowersManifestTests(unittest.TestCase):
    def fixture(self, root):
        raw = root / "raw"
        raw.mkdir()
        labels, train, val, test, categories = [], [], [], [], []
        for c in range(102):
            count = 80 if c < 101 else 109  # Total8189, with10 train/val per class.
            first = len(labels) + 1
            labels.extend([c + 1] * count)
            train.extend(range(first, first + 10))
            val.extend(range(first + 10, first + 20))
            test.extend(range(first + 20, first + count))
            name = f"flower{101-c:03d}"
            categories.append(f'<img src="thumbs/thumbim_{first:05d}.jpg" alt="{name}"></td>'
                              f'<td class=labL>{name}</td><td class=numL>{count}</td>')
        savemat(raw / "imagelabels.mat", {"labels": np.asarray(labels)})
        savemat(raw / "setid.mat", {"trnid": train, "valid": val, "tstid": test})
        (raw / "categories.html").write_text("\n".join(reversed(categories)))
        return raw, {"train": train, "val": val, "test": test}

    def test_official_ids_survive_and_names_follow_labels_not_page_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw, expected = self.fixture(root)
            summary = prepare_manifests(raw, root / "out")
            images, classes, tasks = load_manifests(root / "out")
            self.assertEqual(classes[0]["class_name"], "flower101")
            self.assertEqual(classes[101]["class_name"], "flower000")
            self.assertEqual(summary["image_counts"], {"train": 1020, "val": 1020, "test": 6149})
            for split, ids in expected.items():
                self.assertEqual({r["image_id"] for r in images if r["split"] == split},
                                 {f"image_{i:05d}" for i in ids})
            self.assertEqual(summary["validation_image_task_examples"], 4080)
            self.assertEqual(summary["validation_exchange_examples"], 6120)
            prepare_manifests(raw, root / "repeat")
            for path in (root / "out").iterdir():
                self.assertEqual(path.read_bytes(), (root / "repeat" / path.name).read_bytes())

    def test_balanced_graph_has_no_split_overlap_or_missing_class(self):
        tasks, permutation = balanced_task_graph()
        self.assertEqual(len(set(permutation)), 102)
        self.assertEqual(Counter(r["split"] for r in tasks), {"train": 612, "val": 204, "test": 204})
        self.assertEqual(len({r["task_id"] for r in tasks}), 1020)
        for split, degree in (("train", 12), ("val", 4), ("test", 4)):
            counts = Counter(c for row in tasks if row["split"] == split
                             for c in (row["class_a"], row["class_b"]))
            self.assertEqual(dict(counts), dict.fromkeys(range(102), degree))
        self.assertEqual((tasks, permutation), balanced_task_graph())
        self.assertNotEqual(tasks, balanced_task_graph(seed=2028)[0])

    def test_wrong_category_count_is_rejected(self):
        page = '<img src="thumbs/thumbim_00001.jpg" alt="one"></td><td class=labL>one</td><td class=numL>1</td>'
        with self.assertRaisesRegex(AssertionError, "count disagrees"):
            class_metadata([1, 1], page)

    def test_overlap_in_official_splits_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw, ids = self.fixture(root)
            ids["test"][0] = ids["train"][0]
            savemat(raw / "setid.mat", {"trnid": ids["train"], "valid": ids["val"], "tstid": ids["test"]})
            with self.assertRaisesRegex(AssertionError, "splits overlap"):
                prepare_manifests(raw, root / "out")


if __name__ == "__main__":
    unittest.main()
