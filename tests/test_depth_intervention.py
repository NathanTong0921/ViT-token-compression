import json
from itertools import permutations
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_depth_intervention as analysis
from evaluate_selectors import write_rows


def manual_image_bootstrap(ids, weights, values, replicates, seed):
    """Small independent implementation for checking the paired ratio estimate."""
    images = sorted(set(ids))
    totals = np.array([sum(weights[ids == image]) for image in images])
    numerators = np.array([sum(weights[ids == image] * values[ids == image])
                           for image in images])
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(images), size=(replicates, len(images)))
    samples = numerators[draws].sum(axis=1) / totals[draws].sum(axis=1)
    return numerators.sum() / totals.sum(), np.quantile(samples, [.025, .975])


def write_fixture(root, dataset):
    """Keep endpoints consistent while varying rows and protocol run ordering."""
    image_ids = np.array([f"{dataset}_{i}" for i in range(6)])
    labels = np.array([0, 1, 1, 2, 2, 2])
    text = np.array([[1., 0., 0.], [0., 1., 0.], [2**-.5, 2**-.5, 0.]])
    classes = ((0, 1), (0, 2), (1, 2))
    rows = []
    for image_index, (image, label) in enumerate(zip(image_ids, labels)):
        for a, b in classes:
            if label in (a, b):
                rows.append(dict(image_index=image_index, image_id=image,
                    task_id=f"{a}_{b}", class_id=int(label), class_a=a, class_b=b,
                    task_class_weight=1 / (6 * np.count_nonzero(labels == label))))
    n = len(rows)
    seeds = [11, 12, 13]
    pairs = list(permutations(seeds, 2))
    full = np.zeros((len(image_ids), 3))
    full[np.arange(len(image_ids)), labels] = .7
    row_effect = np.linspace(.7, 1.7, n) * (1.4 if dataset == "flowers" else 1.)
    truth = {}
    for di, depth in enumerate(analysis.DEPTHS):
        folder = root / f"{dataset}_d{depth}"
        folder.mkdir()
        order = np.random.default_rng(depth).permutation(n)
        write_rows(folder / "examples.csv", [rows[i] for i in order])
        np.savez(folder / "evidence.npz", image_ids=image_ids, labels=labels,
                 text_features=text, full_scores=full)
        (folder / "protocol.json").write_text(json.dumps(dict(
            setting=f"{dataset}_d{depth}", training_seeds=seeds[::-1] if di % 2 else seeds,
            primary_seed_pairs=pairs[di:] + pairs[:di],
            config=dict(prune_after=depth, model="tiny", keep_patch_tokens=49))))
        u, c, direct_raw, mix_raw = {}, {}, [], []
        for seed in seeds:
            shared = np.zeros((len(image_ids), 3))
            shared[np.arange(len(image_ids)), labels] = (
                .02 * (np.arange(len(image_ids)) - 2) + .002 * (seed - 11))
            u[seed] = shared[[row["image_index"] for row in rows]]
            shift = (-.04, .02, .08)[di] * row_effect * (1 + .1 * (seed - 11))
            c[seed] = u[seed].copy()
            c[seed][np.arange(n), [row["class_id"] for row in rows]] += shift
            direct_raw.append(shift)
        for s, t in pairs:
            uu = .7 * u[s] + .3 * u[t]
            cu = uu.copy()
            shift = (0., .03, .07)[di] * row_effect + .001 * (s - t)
            cu[np.arange(n), [row["class_id"] for row in rows]] += shift
            mix_raw.append(shift)
            scores = np.stack((c[s], u[s], u[t], cu, uu))
            np.savez(folder / f"pair_{s}_{t}.npz", methods=analysis.METHODS,
                     cosine_scores=scores[:, order])
        truth[depth] = dict(direct=np.stack(direct_raw), mix=np.stack(mix_raw))
    return rows, truth


class DepthInterventionTests(unittest.TestCase):
    def test_raw_geometric_margins_and_tie_rule(self):
        scores = np.array([[[.1, .1], [.0, .4]], [[.4, .1], [.0, .0]]])
        result = analysis.binary_outcomes(scores, np.array([0, 1]),
                                          np.array([1, 0]), np.eye(2))
        np.testing.assert_array_equal(result[..., 0], [[1, 1], [1, 0]])
        np.testing.assert_allclose(result[..., 2], [[0, .4], [.3, 0]])
        np.testing.assert_allclose(result[..., 1], result[..., 2] / np.sqrt(2))

    def test_average_fixed_runs_then_bootstrap_images_with_task_weights(self):
        ids = np.array(["a", "a", "b", "b", "c", "c"])
        weights = np.array([.1, .2, .15, .25, .1, .2])
        values = np.array([[0., 1, 3, 2, -1, 4], [2., 3, -1, 4, 0, 5],
                           [1., 0, 2, 3, 4, -2]])
        outputs = np.stack((values, values * .1, values * .3), axis=-1)
        result = analysis.summarize_runs(ids, weights, outputs, [11, 12, 13], 250, 17, 4)
        mean, interval = manual_image_bootstrap(ids, weights, values.mean(0), 250, 17)
        self.assertAlmostEqual(result["accuracy"]["mean"], mean)
        np.testing.assert_allclose(result["accuracy"]["ci95"], interval)
        for i, run in enumerate(result["per_run"]):
            self.assertEqual(run["run_id"], 11 + i)
            self.assertAlmostEqual(run["accuracy"], weights @ values[i])
        self.assertEqual(result["geom_margin"]["individual_interval_level"], .9875)

    def test_cross_depth_alignment_and_known_changes_survive_shuffled_rows_and_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            truths = {dataset: write_fixture(source, dataset) for dataset in analysis.DATASETS}
            result = analysis.analyze(source, root / "output", replicates=250, bootstrap_seed=17)
            self.assertEqual(result["compared_depths"], [3, 9])
            self.assertEqual(result["primary_family_size"], 4)
            for dataset, (rows, truth) in truths.items():
                stats = result["datasets"][dataset]
                order = np.asarray(sorted(range(len(rows)),
                    key=lambda i: (rows[i]["image_id"], rows[i]["task_id"])))
                weights = np.array([row["task_class_weight"] for row in rows])[order]
                ids = np.array([row["image_id"] for row in rows])[order]
                with np.load(root / "output" / dataset / "paired_cases.npz") as saved:
                    self.assertEqual(saved["endpoint_outcomes"].shape, (3, 3, 2, len(rows), 3))
                    self.assertEqual(saved["mixture_outcomes"].shape, (3, 6, 2, len(rows), 3))
                    for i, depth in enumerate(analysis.DEPTHS):
                        np.testing.assert_allclose(saved["conditioned_minus_shared"][i, ..., 2],
                                                   truth[depth]["direct"][:, order], atol=1e-16)
                        np.testing.assert_allclose(saved["cu_minus_uu"][i, ..., 2],
                                                   truth[depth]["mix"][:, order], atol=1e-16)
                    for panel, endpoint in (("direct", "conditioned_minus_shared"),
                                            ("mix", "cu_minus_uu")):
                        for depth in (3, 9):
                            contrast = f"depth_{depth}_minus_6"
                            expected = (truth[depth][panel] - truth[6][panel])[:, order]
                            np.testing.assert_allclose(saved[f"{endpoint}_{contrast}"][..., 2],
                                                       expected, atol=1e-16)
                            estimate = stats["depth_changes"][endpoint][contrast]["raw_gap"]
                            mean, interval = manual_image_bootstrap(ids, weights, expected.mean(0), 250, 17)
                            self.assertAlmostEqual(estimate["mean"], mean)
                            np.testing.assert_allclose(estimate["ci95"], interval, atol=1e-16)
                            self.assertEqual(estimate["individual_interval_level"], .9875)
                self.assertEqual(set(stats["depth_changes"]["conditioned_minus_shared"]),
                                 {"depth_3_minus_6", "depth_9_minus_6"})
                contributions = analysis.read_rows(root / "output" / dataset / "image_contributions.csv")
                self.assertEqual(len(contributions), 6)
                weighted = sum(float(row["conditioned_minus_shared_depth_9_minus_6_raw_gap_weighted_sum"])
                               for row in contributions)
                self.assertAlmostEqual(weighted, stats["depth_changes"]["conditioned_minus_shared"]
                                       ["depth_9_minus_6"]["raw_gap"]["mean"])


if __name__ == "__main__":
    unittest.main()
