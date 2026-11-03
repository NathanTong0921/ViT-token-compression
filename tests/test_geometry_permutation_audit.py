import importlib.util
import itertools
import contextlib
import io
import tempfile
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('permutation_audit', Path(__file__).resolve().parents[1] / 'scripts/audit_geometry_permutation.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class IndependentPermutationAuditTests(unittest.TestCase):
    def test_residualized_slopes_match_direct_full_weighted_regression(self):
        rng = np.random.default_rng(12)
        ids = np.repeat(np.arange(7), [4, 5, 6, 8, 10, 12, 15])
        margins = rng.uniform(-.2, .3, (len(ids), 2))
        angle = rng.uniform(20, 90, len(ids))
        target = rng.normal(size=len(ids))
        linear = audit.residual_coefficients(ids, margins[:, 0], margins[:, 1], angle)
        unique, inverse, count = np.unique(ids, return_inverse=True, return_counts=True)
        weight = 1 / (len(unique) * count[inverse])
        lo, hi = margins.min(axis=1), margins.max(axis=1)
        base = np.column_stack([lo, hi, lo**2, lo*hi, hi**2, angle/10])
        expected = []
        for fixed in (False, True):
            x, y = base.copy(), target.copy()
            if fixed:
                for image in unique:
                    take = ids == image
                    x[take] -= x[take].mean(axis=0)
                    y[take] -= y[take].mean()
            else:
                x = np.column_stack([np.ones(len(ids)), x])
            expected.append(np.linalg.lstsq(x * np.sqrt(weight[:, None]), y*np.sqrt(weight), rcond=None)[0][-1])
        expected.append(weight @ target)
        np.testing.assert_allclose(linear @ target, expected, rtol=0, atol=1e-11)

    def test_every_small_assignment_matches_direct_signed_geometry(self):
        rng = np.random.default_rng(4)
        z, u = rng.normal(size=(4, 9)), rng.normal(size=(4, 9))
        z /= np.linalg.norm(z, axis=1)[:, None]
        u /= np.linalg.norm(u, axis=1)[:, None]
        edge, distances, du, _, score = audit.image_geometry(z, u)
        a, b = edge.T
        permutations = np.asarray(list(itertools.permutations(range(4))))
        actual = audit.assigned_alignment(score, distances, du, edge, permutations)
        for row, assignment in enumerate(permutations):
            dz = z[assignment[a]] - z[assignment[b]]
            expected = np.sum(dz*(u[a]-u[b]), axis=1) / (np.linalg.norm(dz, axis=1)*du)
            np.testing.assert_allclose(actual[row], expected, rtol=0, atol=1e-14)
            np.testing.assert_array_equal(np.sort(distances[assignment[a], assignment[b]]), np.sort(distances[a, b]))
        # Averaging all bijections cancels every pair's signed alignment.
        np.testing.assert_allclose(actual.mean(axis=0), 0, atol=1e-14)

    def test_keyed_assignments_keep_prefix_and_namespace(self):
        tasks = ['task_01', 'task_04', 'task_09']
        a = audit.permutations('pets', 'image_1', tasks, 80, 2094)
        b = audit.permutations('pets', 'image_1', tasks, 120, 2094)
        np.testing.assert_array_equal(a, b[:80])
        self.assertFalse(np.array_equal(a, audit.permutations('flowers', 'image_1', tasks, 80, 2094)))
        np.testing.assert_array_equal(np.sort(a, axis=1), np.tile(np.arange(3), (80, 1)))
        self.assertTrue(np.any(np.all(a == np.arange(3), axis=1)))

    def test_end_to_end_raw_panel_and_random_reference_audit(self):
        fixture_spec = importlib.util.spec_from_file_location('permutation_io_fixture', Path(__file__).with_name('test_geometry_permutation_io.py'))
        fixture_module = importlib.util.module_from_spec(fixture_spec)
        fixture_spec.loader.exec_module(fixture_module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            setting, seeds = fixture_module.fixture(root)
            fixture_module.save_existing_geometry(setting, seeds)
            config = {"settings": [setting], "training_seeds": seeds, "draws": 31,
                      "permutation_seed": 2094, "family_size": 1, "final_test_used": False}
            with contextlib.redirect_stdout(io.StringIO()):
                result = fixture_module.runner.run(config, root/'result', root/'panels')
            panel = audit.saved_arrays(root/'panels/fixture.npz')
            raw = audit.check_raw_panel(setting, panel, seeds)
            self.assertEqual(raw['image_count'], 36)
            self.assertEqual(raw['raw_feature_dtype'], 'float32')
            verified = audit.check_setting(setting, panel, root/'result/fixture',
                                           result['settings']['fixture'], 31, 2094, family_size=1)
            self.assertEqual(verified['status'], 'passed')
            self.assertLess(verified['all_reference_statistics_max_abs_error'], 1e-10)
            panel['seed_features'][0, 0, 0] += .01
            with self.assertRaises(AssertionError):
                audit.check_raw_panel(setting, panel, seeds)

    def test_auxiliary_zero_pair_keeps_primary_panel_and_reweights_valid_pairs(self):
        fixture_spec = importlib.util.spec_from_file_location('permutation_zero_fixture', Path(__file__).with_name('test_geometry_permutation_io.py'))
        fixture_module = importlib.util.module_from_spec(fixture_spec)
        fixture_spec.loader.exec_module(fixture_module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            setting, seeds = fixture_module.fixture(root)
            path = Path(setting['source']) / 'full/predictions.npz'
            raw = audit.saved_arrays(path)
            selected = np.flatnonzero((raw['image_ids'] == 'image_000') & (raw['seed'] == seeds[1])
                                     & (raw['method'] == 'conditioned') & (raw['task_split'] == 'unseen'))
            selected = selected[np.argsort(raw['task_ids'][selected])]
            first, second = selected[:2]
            for key in ('image_features', 'cosine_scores'):
                raw[key][second] = raw[key][first]
            np.savez_compressed(path, **raw)
            fixture_module.save_existing_geometry(setting, seeds)
            config = {"settings": [setting], "training_seeds": seeds, "draws": 19,
                      "permutation_seed": 2094, "family_size": 1, "final_test_used": False}
            with contextlib.redirect_stdout(io.StringIO()):
                result = fixture_module.runner.run(config, root/'result', root/'panels')
            panel = audit.saved_arrays(root/'panels/fixture.npz')
            observed = result['settings']['fixture']['observed_statistics_per_seed']
            self.assertEqual([r['excluded_pair_count'] for r in observed], [0, 1, 0])
            self.assertEqual(observed[1]['eligible_pair_count'], 107)
            self.assertEqual(observed[1]['excluded_pairs'], [{'image_id':'image_000', 'task_a':'0_1', 'task_b':'0_2'}])
            verified = audit.check_setting(setting, panel, root/'result/fixture',
                                           result['settings']['fixture'], 19, 2094, family_size=1)
            self.assertEqual(verified['pair_count'], 108)
            self.assertEqual(verified['status'], 'passed')
            identity = audit.saved_arrays(root/'result/fixture/observed_pair_cases.npz')
            alignment = identity['alignment'].copy()
            alignment[0] = np.nan
            stats = audit.auxiliary_seed_statistics(identity, alignment)
            image_means = [np.nanmean(alignment[identity['image_ids'] == image]) for image in np.unique(identity['image_ids'])]
            self.assertAlmostEqual(stats['mean_alignment'], np.mean(image_means), places=12)

    def test_seed_mean_must_not_be_renormalized(self):
        seeds = np.asarray([[[1., 0, 0], [0, 1., 0]], [[0., 0, 1], [0., 1., 0]]])
        mean = seeds.mean(axis=0)
        u = np.asarray([[1., 0, 0], [0, 1., 0]])
        edges, dist, du, _, scores = audit.image_geometry(mean, u)
        actual = audit.assigned_alignment(scores, dist, du, edges, np.arange(2))
        normalized = mean / np.linalg.norm(mean, axis=1)[:, None]
        _, dist2, _, _, score2 = audit.image_geometry(normalized, u)
        wrong = audit.assigned_alignment(score2, dist2, du, edges, np.arange(2))
        self.assertGreater(abs(actual[0] - wrong[0]), .01)


if __name__ == '__main__':
    unittest.main()
