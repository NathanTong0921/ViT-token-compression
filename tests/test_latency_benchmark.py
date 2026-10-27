"""Check scientific invariants of the latency protocol without a GPU."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

spec = importlib.util.spec_from_file_location(
    'benchmark_selector_latency', Path(__file__).parents[1] / 'scripts' / 'benchmark_selector_latency.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class LatencyProtocolTests(unittest.TestCase):
    def test_stable_gpu_ranking_preserves_evaluation_ties(self):
        scores = np.asarray([[2, 1, 2, 2, 0], [0, 0, 0, 0, 0], [-1, -2, -1, -3, -1]], dtype=np.float32)
        expected = np.argsort(-scores, axis=1, kind='stable')[:, :3]
        actual = benchmark.select_positions(torch.tensor(scores), 3).numpy()
        np.testing.assert_array_equal(actual, expected)

    def test_panel_is_outcome_free_class_covered_and_order_invariant(self):
        images = [{'image_id': f'image_{label}_{i}', 'class_id': label, 'split': 'val'}
                  for label in range(4) for i in range(5)]
        images += [{'image_id': 'unused_test', 'class_id': 0, 'split': 'test'}]
        tasks = [{'task_id': f'{a}_{b}', 'class_a': a, 'class_b': b, 'split': 'val'}
                 for a, b in ((0, 1), (1, 2), (2, 3), (0, 3))]
        panel = benchmark.choose_panel(images, tasks, size=8, seed=2081)
        self.assertEqual(panel, benchmark.choose_panel(images[::-1], tasks[::-1], size=8, seed=2081))
        self.assertEqual(len({r['image_id'] for r in panel}), 8)
        self.assertEqual({r['class_id'] for r in panel}, set(range(4)))
        self.assertTrue(all(r['class_id'] in (r['class_a'], r['class_b']) for r in panel))
        self.assertNotIn('unused_test', {r['image_id'] for r in panel})

    def test_all_seeds_and_slow_samples_remain_in_summary(self):
        rows = []
        for round_id in range(2):
            rows.append({'batch_size': 1, 'round': round_id, 'method': 'full_model', 'seed': -1,
                         'wall_ms': 10., 'cuda_event_ms': 8.})
            for seed, value in zip((2041, 2042, 2043), (5., 10., 20.)):
                rows.append({'batch_size': 1, 'round': round_id, 'method': 'conditioned', 'seed': seed,
                             'wall_ms': value, 'cuda_event_ms': value-1.})
        windows = [{'batch_size': 1, 'round': r, 'uncontended_at_both_checks': True} for r in range(2)]
        result = benchmark.summarize_timings(rows, windows)['1']['methods']['conditioned']
        self.assertEqual(result['wall_ms']['count'], 6)
        self.assertEqual(result['wall_ms']['maximum'], 20.)
        self.assertEqual([r['seed'] for r in result['per_seed']], [2041, 2042, 2043])
        self.assertEqual(result['paired_round_speedup']['median'], 1.)

        windows[1]['uncontended_at_both_checks'] = False
        invalid = benchmark.summarize_timings(rows, windows)['1']
        self.assertFalse(invalid['timing_window_valid'])
        self.assertIsNone(invalid['methods']['conditioned']['paired_round_speedup'])
        self.assertEqual(invalid['methods']['conditioned']['wall_ms']['count'], 6)


if __name__ == '__main__':
    unittest.main()
