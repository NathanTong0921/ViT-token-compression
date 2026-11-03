from itertools import permutations
import unittest

import numpy as np

from token_compression.geometry_permutation import (
    image_permutations, prepare_image, prepare_statistics, run_permutations,
    upper_tail_p,
)
from token_compression.selector_geometry import adjusted_angle_association


def unit(values):
    values = np.asarray(values, dtype=float)
    return values / np.linalg.norm(values, axis=-1, keepdims=True)


def synthetic_images(count=18, seed=47):
    rng = np.random.default_rng(seed)
    result = []
    for i in range(count):
        n = 3 + i % 3
        text = unit(rng.normal(size=(n + 1, 9)))
        directions = unit(text[0] - text[1:])
        full = unit(rng.normal(size=9))
        features = unit(full + .3 * directions[None] + rng.normal(0, .1, (3, n, 9)))
        result.append(prepare_image(f"image_{i:03d}", [f"task_{j}" for j in range(n)],
                                    features.astype(np.float32), directions,
                                    directions @ full, namespace="toy"))
    return result


class CompleteGraphTests(unittest.TestCase):
    def test_every_assignment_matches_direct_vectors_and_preserves_all_distances(self):
        image = synthetic_images(3)[2]
        assignments = np.asarray(list(permutations(range(len(image.task_ids)))))
        a, b = image.edges.T
        delta_u = image.task_directions[a] - image.task_directions[b]
        delta_z = image.mean_features[assignments[:, a]] - image.mean_features[assignments[:, b]]
        expected = np.einsum("bei,ei->be", delta_z, delta_u)
        expected /= np.linalg.norm(delta_z, axis=-1) * np.linalg.norm(delta_u, axis=-1)
        np.testing.assert_allclose(image.alignment(assignments), expected, atol=1e-14)
        np.testing.assert_allclose(image.displacement_norms(assignments),
                                   np.linalg.norm(delta_z, axis=-1), atol=1e-14)
        for distances in image.displacement_norms(assignments):
            np.testing.assert_allclose(np.sort(distances), np.sort(image.feature_norm), atol=1e-14)
        # The lookup includes negative orientations when a permutation reverses an edge.
        reversed_tasks = np.arange(len(image.task_ids))[::-1]
        self.assertTrue(np.all(image.edge_sign[reversed_tasks[a], reversed_tasks[b]] == -1))

    def test_identity_obeys_exchange_geometry_identity(self):
        image = synthetic_images(1)[0]
        a, b = image.edges.T
        direct_loss = .5 * np.einsum("ij,ij->i", image.mean_features[a] - image.mean_features[b],
                                     image.task_directions[a] - image.task_directions[b])
        reconstructed = image.feature_norm * np.sin(np.radians(image.angle_degrees) / 2) * image.alignment()
        np.testing.assert_allclose(reconstructed, direct_loss, atol=1e-14)

    def test_seed_average_is_before_normalizing_displacement(self):
        rng = np.random.default_rng(19)
        seeds = unit(rng.normal(size=(3, 4, 7))).astype(np.float32)
        directions = unit(rng.normal(size=(4, 7)))
        image = prepare_image("one", ["a", "b", "c", "d"], seeds, directions, np.zeros(4))
        np.testing.assert_array_equal(image.mean_features, seeds.astype(np.float64).mean(axis=0))
        self.assertTrue(np.all(np.linalg.norm(image.mean_features, axis=1) < 1))
        a, b = image.edges.T
        delta_z = seeds.astype(np.float64)[:, a] - seeds.astype(np.float64)[:, b]
        delta_u = directions[a] - directions[b]
        seed_alignments = np.einsum("sei,ei->se", unit(delta_z), unit(delta_u))
        self.assertGreater(np.max(np.abs(image.alignment() - seed_alignments.mean(axis=0))), .01)
        np.testing.assert_allclose(image.mean_features[a] - image.mean_features[b], delta_z.mean(axis=0), atol=1e-15)

    def test_known_correspondence_and_exhaustive_random_reference(self):
        directions = unit(np.random.default_rng(28).normal(size=(4, 8)))
        image = prepare_image("aligned", ["a", "b", "c", "d"], directions[None],
                              directions, np.zeros(4))
        np.testing.assert_allclose(image.alignment(), 1., atol=1e-14)
        # Every source displacement appears in both orientations equally often.
        reference = image.alignment(np.asarray(list(permutations(range(4)))))
        np.testing.assert_allclose(reference.mean(axis=0), 0., atol=1e-14)
        self.assertGreater(image.alignment().mean(), reference.mean() + .9)

    def test_undefined_alignment_fails_instead_of_changing_pair_support(self):
        directions = np.eye(3)
        duplicate_features = np.array([[[1., 0, 0], [1., 0, 0], [0, 1., 0]]])
        with self.assertRaisesRegex(AssertionError, "Undefined feature alignment"):
            prepare_image("duplicate", ["a", "b", "c"], duplicate_features, directions, np.zeros(3))
        duplicate_directions = np.array([[1., 0, 0], [1., 0, 0], [0, 1., 0]])
        with self.assertRaisesRegex(AssertionError, "Undefined text alignment"):
            prepare_image("duplicate", ["a", "b", "c"], np.eye(3)[None], duplicate_directions, np.zeros(3))


class ReferenceAssignmentTests(unittest.TestCase):
    def test_key_ignores_features_and_task_input_order_but_includes_panel(self):
        image = synthetic_images(1)[0]
        order = np.array([2, 0, 1])
        reordered = prepare_image(image.image_id, image.task_ids[order],
                                  (image.mean_features[order] * .5)[None],
                                  image.task_directions[order], image.full_margins[order],
                                  namespace=image.namespace)
        expected = image_permutations(image, 100, 2094)
        np.testing.assert_array_equal(expected, image_permutations(reordered, 100, 2094))
        np.testing.assert_array_equal(np.sort(expected, axis=1),
                                      np.tile(np.arange(3), (100, 1)))
        # Uniform permutations include fixed points; there is no derangement filter.
        self.assertTrue(np.any(np.all(expected == np.arange(3), axis=1)))
        reordered.namespace = "other_dataset"
        self.assertFalse(np.array_equal(expected, image_permutations(reordered, 100, 2094)))
        self.assertFalse(np.array_equal(expected, image_permutations(image, 100, 2095)))

    def test_upper_tail_counts_ties_and_never_returns_zero(self):
        reference = np.array([[.1, 5], [.2, 6], [.3, 7]])
        np.testing.assert_allclose(upper_tail_p(np.array([.2, 8]), reference), [.75, .25])


class StatisticTests(unittest.TestCase):
    def test_identity_slopes_reproduce_existing_alignment_analysis(self):
        images = synthetic_images()
        statistics = prepare_statistics(images)
        alignments = [image.alignment() for image in images]
        cases = {**statistics.cases, "alignment": np.concatenate(alignments)}
        point = statistics.evaluate(alignments)
        for index, fixed in enumerate([False, True]):
            existing = adjusted_angle_association(cases, outcome="alignment", replicates=3,
                                                  image_fixed_effects=fixed)
            self.assertAlmostEqual(point[index], existing["angle_slope"], places=12)
        self.assertAlmostEqual(point[2], np.mean([a.mean() for a in alignments]), places=14)
        self.assertGreater(np.ptp([len(image.edges) for image in images]), 0)

    def test_fixed_effects_remove_per_image_offsets(self):
        images = synthetic_images()
        statistics = prepare_statistics(images)
        alignment = [image.alignment() for image in images]
        shifted = [a + index * 10 for index, a in enumerate(alignment)]
        first, second = statistics.evaluate(alignment), statistics.evaluate(shifted)
        self.assertAlmostEqual(first[1], second[1], places=12)
        self.assertGreater(abs(first[0] - second[0]), .01)

    def test_permutation_statistics_match_direct_refits_with_identical_assignments(self):
        images = synthetic_images(12)
        result = run_permutations(images, draws=4, seed=201)
        prepared = prepare_statistics(images)
        assignments = [image_permutations(image, 4, 201) for image in images]
        for draw in range(4):
            direct = []
            for image, permutations_for_image in zip(images, assignments):
                features = image.mean_features[permutations_for_image[draw]]
                a, b = image.edges.T
                direct.append(np.einsum("ij,ij->i", unit(features[a] - features[b]),
                                        unit(image.task_directions[a] - image.task_directions[b])))
            cases = {**prepared.cases, "alignment": np.concatenate(direct)}
            for j, fixed in enumerate([False, True]):
                fit = adjusted_angle_association(cases, outcome="alignment", replicates=2,
                                                 image_fixed_effects=fixed)
                self.assertAlmostEqual(result["null_statistics"][draw, j], fit["angle_slope"], places=12)
            self.assertAlmostEqual(result["null_statistics"][draw, 2],
                                   np.mean([a.mean() for a in direct]), places=14)

    def test_reference_is_reproducible_and_independent_of_image_iteration_order(self):
        images = synthetic_images(12)
        first = run_permutations(images, draws=21, seed=2094)
        second = run_permutations(images[::-1], draws=21, seed=2094)
        np.testing.assert_allclose(first["observed"], second["observed"], atol=1e-13)
        np.testing.assert_allclose(first["null_statistics"], second["null_statistics"], atol=1e-13)
        np.testing.assert_array_equal(first["upper_tail_p"], second["upper_tail_p"])
        np.testing.assert_array_equal(first["null_statistics"],
                                      run_permutations(images, draws=21, seed=2094)["null_statistics"])


if __name__ == "__main__":
    unittest.main()
