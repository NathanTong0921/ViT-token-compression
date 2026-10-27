"""Numerical-audit checks that do not require CLIP weights or a GPU."""
import sys
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from audit_inference_numerics import evaluate_path, load_reference, pair_outcomes, set_comparison, summarize_effect


class InferenceNumericsTests(unittest.TestCase):
    def test_one_set_replacement_is_not_many_sorted_index_mismatches(self):
        old = np.array([[1, 2, 3, 4, 5], [0, 2, 4, 6, 7]])
        new = np.array([[0, 1, 2, 3, 4], [0, 2, 4, 6, 7]])
        intersection, replaced, jaccard = set_comparison(new, old, 8)
        np.testing.assert_array_equal(intersection, [4, 5])
        np.testing.assert_array_equal(replaced, [1, 0])
        np.testing.assert_allclose(jaccard, [4 / 6, 1])

    def test_pair_margin_orientation_and_tie_rule(self):
        text = np.array([[1., 0.], [0., 1.]])
        scores = np.array([[.8, .2], [.2, .8], [.5, .5], [.5, .5]])
        labels = np.array([0, 1, 0, 1])
        pairs = np.tile([0, 1], (4, 1))
        margin, correct = pair_outcomes(scores, labels, pairs, text)
        np.testing.assert_allclose(margin, [.6 / np.sqrt(2), .6 / np.sqrt(2), 0, 0])
        np.testing.assert_array_equal(correct, [1, 1, 1, 0])

    def test_unique_image_reuse_scores_every_query_and_records_real_tail(self):
        class ImagePath:
            method = 'image_only'
            count = 2
            pruner = SimpleNamespace(num_patches=4)

            def __init__(self):
                self.forward_batch_sizes = []

            def embeddings(self, pixels, candidate_text):
                self.forward_batch_sizes.append(len(pixels))
                return pixels, torch.tensor([[0, 2]]).expand(len(pixels), -1)

        path = ImagePath()
        pixels = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        text = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
        pairs = torch.tensor([[0, 1], [1, 2], [0, 2], [0, 1]])
        image_indices = torch.tensor([0, 1, 0, 2])
        scores, positions, execution = evaluate_path(path, pixels, text, pairs, image_indices,
                                                     torch.tensor([0, 1, 3]), batch_size=2)
        np.testing.assert_allclose(scores, [[1, 2], [4, 7], [1, 3], [5, 6]])
        self.assertEqual(path.forward_batch_sizes, [2, 1])
        self.assertEqual(execution['image_forwards'], 3)
        self.assertEqual(execution['forward_tail_query_indices'], [3])
        self.assertEqual(execution['score_tail_size'], 0)
        self.assertEqual(positions.shape, (4, 2))

    def test_unchanged_implementations_have_zero_paired_effect_change(self):
        examples = [{'image_id': str(i), 'task_class_weight': .25} for i in range(4)]
        reference, current = {}, {}
        for seed in (2041, 2042, 2043):
            for method, correct in [('image_only', [1, 0, 1, 1]), ('conditioned', [1, 1, 1, 1])]:
                reference[method, seed] = {'correct': np.array(correct), 'margin': np.array(correct) * .2}
                current[method, seed] = np.stack((correct, np.array(correct) * .2), axis=1)
        result = summarize_effect(examples, current, reference, [2041, 2042, 2043])
        self.assertEqual(result['online']['accuracy']['mean'], .25)
        for metric in ('accuracy', 'margin'):
            self.assertEqual(result['online_minus_original'][metric]['mean'], 0.)
            self.assertEqual(result['online_minus_original'][metric]['ci95'], [0., 0.])

    def test_reference_joins_csv_identity_to_npz_row_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for group, jobs in [('references', [('full_model', -1), ('attention', -1)]),
                                ('full', [(method, seed) for seed in (2041, 2042, 2043)
                                          for method in ('image_only', 'conditioned')])]:
                folder = root / group
                folder.mkdir()
                rows, scores, positions = [], [], []
                for method, seed in jobs:
                    for i in range(2):
                        value = .6 + i * .1 + (seed % 3) * .01
                        rows.append({'row_id': len(rows), 'method': method, 'seed': seed, 'task_split': 'unseen',
                                     'image_id': f'image{i}', 'task_id': 'task0', 'class_id': i,
                                     'class_a': 0, 'class_b': 1, 'task_class_weight': .5,
                                     'margin': (value-.1) / np.sqrt(2), 'correct': 1})
                        scores.append([value, .1] if i == 0 else [.1, value])
                        positions.append([0, 1, 2, 3] if method == 'full_model' else [i, i+2, -1, -1])
                with (folder / 'predictions.csv').open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows[::-1])
                np.savez(folder / 'predictions.npz', image_ids=[r['image_id'] for r in rows],
                         task_ids=[r['task_id'] for r in rows], cosine_scores=np.asarray(scores),
                         patch_indices=np.asarray(positions), text_features=np.eye(2))
            examples, reference, _ = load_reference(root, [2041, 2042, 2043])
            self.assertEqual([r['image_id'] for r in examples], ['image1', 'image0'])
            self.assertEqual(len(reference), 8)
            np.testing.assert_allclose(reference['conditioned', 2042]['pair_scores'], [[.1, .72], [.62, .1]])
            np.testing.assert_array_equal(reference['image_only', 2041]['positions'], [[1, 3, -1, -1], [0, 2, -1, -1]])


if __name__ == '__main__':
    unittest.main()
