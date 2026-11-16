import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_selector_capacity import METHODS, CONTRASTS, PRIMARY, capacity_contrasts, read_panel
from analyze_direct_classification import METRICS, reconstruct_outcomes


class CapacityAnalysisTests(unittest.TestCase):
    def values(self, small, medium, large):
        data = np.zeros((2, len(METHODS), 3, 4))
        for capacity, values in zip(("small", "medium", "large"), (small, medium, large)):
            for arm, value in zip(("C", "U"), values):
                data[:, METHODS.index(f"{capacity}_{arm}")] = value
        return data

    def test_reader_consumes_evaluation_job_width_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            ids, labels = np.asarray(["a", "b"]), np.asarray([0, 1])
            pairs, text = np.asarray([[0, 1], [0, 1]]), np.eye(2)
            scores = np.asarray([[.8, .2], [.2, .8]])
            outcome = reconstruct_outcomes(scores, labels, pairs, text, 1.)
            fields = ["image_id", "image_index", "class_id", "task_id", "class_a", "class_b", "task_class_weight"]
            with (source / "examples.csv").open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for i in range(2):
                    writer.writerow(dict(image_id=ids[i], image_index=i, class_id=labels[i],
                                         task_id="task", class_a=0, class_b=1, task_class_weight=.5))
            np.savez(source / "evidence.npz", image_ids=ids, labels=labels, text_features=text, logit_scale=1.)
            jobs = []
            for capacity in ("small", "medium", "large"):
                for arm in ("conditioned", "image_only_wide"):
                    target = source / "seed_2041" / capacity
                    target.mkdir(parents=True, exist_ok=True)
                    np.savez(target / (arm + ".npz"), cosine_scores=scores,
                             patch_indices=np.tile(np.arange(49), (2, 1)),
                             **{metric: outcome[:, q] for q, metric in enumerate(METRICS)})
                    jobs.append(dict(seed=2041, capacity=capacity, arm=arm, width=64, parameter_count=100))
            (source / "evaluation_summary.json").write_text(json.dumps(dict(jobs=jobs)))
            data = read_panel(source, [2041])
            self.assertEqual(data["values"].shape, (1, 6, 2, 4))
            self.assertEqual(data["metadata"]["small_C"]["width"], 64)
            np.testing.assert_allclose(data["weights"], [.5, .5])

    def test_equal_capacity_improvements_do_not_increase_conditioning_gap(self):
        result = capacity_contrasts(self.values((.5, .4), (.7, .6), (.9, .8)))
        np.testing.assert_allclose(result[:, CONTRASTS.index(PRIMARY)], 0., atol=1e-15)
        np.testing.assert_allclose(result[:, CONTRASTS.index("C_large_minus_small")], .4)
        np.testing.assert_allclose(result[:, CONTRASTS.index("U_large_minus_small")], .4)

    def test_more_accurate_both_can_still_reduce_conditioning_value(self):
        result = capacity_contrasts(self.values((.6, .5), (.7, .7), (.8, .9)))
        np.testing.assert_allclose(result[:, CONTRASTS.index(PRIMARY)], -.2)
        np.testing.assert_allclose(result[:, CONTRASTS.index("gap_medium_minus_small")], -.1)
        np.testing.assert_allclose(result[:, CONTRASTS.index("gap_large_minus_medium")], -.1)

    def test_interaction_preserves_pairing_and_seed_differences(self):
        values = self.values((.6, .5), (.7, .7), (.8, .9))
        values[1, METHODS.index("large_C"), 0, 0] += .3
        result = capacity_contrasts(values)
        delta = result[1, CONTRASTS.index(PRIMARY)] - result[0, CONTRASTS.index(PRIMARY)]
        expected = np.zeros((3, 4)); expected[0, 0] = .3
        np.testing.assert_allclose(delta, expected, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
