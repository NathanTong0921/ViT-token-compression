import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from token_compression.selector import ImageOnlyWideSelector
from token_compression.shared_residual import SharedResidualSelector
from token_compression.set_objective import normalized_scores, set_losses
from run_hybrid_distillation import mask_distribution
from run_set_distillation import (FIT_METRICS, fit_branch, fit_rows, initial_gradient_diagnostics,
                                  load_targets)
from run_shared_residual import cache_shared


class DirectSetObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_matches_enumerated_structured_loss_and_bounds_missing_members(self):
        scores = torch.tensor([[1., -.8, .2, 1.4, -.1], [-2., 3., 1., .5, 0.]], dtype=torch.float64)
        target = torch.tensor([[0, 2], [1, 2]])
        z, _ = normalized_scores(scores)
        actual = set_losses(scores, target)
        for row in range(len(scores)):
            truth = set(target[row].tolist())
            target_score = sum(float(z[row, i]) for i in truth)
            enumerated = max(sum(float(z[row, i]) for i in chosen) + len(set(chosen)-truth)
                             for chosen in itertools.combinations(range(5), 2))
            self.assertAlmostEqual(float(actual[row]), (enumerated-target_score)/2, places=12)
            chosen = set(torch.topk(scores[row], 2).indices.tolist())
            self.assertGreaterEqual(float(actual[row])+1e-12, len(chosen-truth)/2)

    def test_positive_affine_invariance_and_well_separated_zero(self):
        scores = torch.tensor([[2., 1., -3., 0.], [1., -1., -2., .4]], dtype=torch.float64)
        target = torch.tensor([[0, 2], [0, 3]])
        expected = set_losses(scores, target)
        transformed = scores * torch.tensor([[.2], [20.]]) + torch.tensor([[9.], [-4.]])
        torch.testing.assert_close(set_losses(transformed, target), expected, rtol=1e-12, atol=1e-12)
        separated = torch.tensor([[1., 1., -1., -1.]], requires_grad=True)
        loss = set_losses(separated, torch.tensor([[0, 1]]))
        self.assertEqual(float(loss.detach()), 0.)
        loss.sum().backward()
        self.assertTrue(torch.isfinite(separated.grad).all())

    def test_gradient_improves_missed_target_and_is_finite_at_constant_rows(self):
        scores = torch.tensor([[1.2, .8, -.5, -.7]], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[0, 2]])
        loss = set_losses(scores, target).sum()
        grad, = torch.autograd.grad(loss, scores)
        self.assertLess(float(set_losses(scores-.01*grad, target).sum().detach()), float(loss.detach()))
        self.assertGreater(float(grad[0, 1]-grad[0, 2]), 0.)
        # The loss cannot be solved by a constant score vector; std-floor rows
        # have a defined finite gradient and maximal missing-set augmentation.
        constant = torch.ones((2, 4), requires_grad=True)
        constant_loss = set_losses(constant, target.expand(2, -1)).sum()
        constant_loss.backward()
        self.assertTrue(torch.isfinite(constant.grad).all())
        self.assertGreater(float(constant_loss.detach()), 0.)

    def fixture(self):
        torch.manual_seed(2371)
        shared = ImageOnlyWideSelector(hidden_dim=7, token_dim=3).eval().requires_grad_(False)
        tokens = torch.randn(3, 7, 3)
        masks = np.asarray([[0, 2, 4], [1, 3, 5], [0, 1, 3]])
        data = {'tokens': tokens, 'text': torch.randn(3, 4), 'weights': np.ones(3)/3,
                'image_ix': np.arange(3), 'pair_ix': np.asarray([0, 1, 0]),
                'image_ids': np.asarray(['a', 'b', 'c']), 'pairs_array': np.asarray([[0, 1], [1, 2]]),
                'examples': {'image_indices': torch.tensor([0, 1, 2]),
                             'class_pairs': torch.tensor([[0, 1], [1, 2], [0, 1]])},
                'targets': mask_distribution(masks, 6, 'cpu'),
                'target_masks': torch.tensor(masks)}
        return shared, data, cache_shared(shared, tokens, 3), masks

    def test_small_paired_fits_keep_base_fixed_and_diagnostics_do_not_change_rng(self):
        shared, data, cached, _ = self.fixture()
        base = {k: v.clone() for k, v in shared.state_dict().items()}
        config = {'learning_rate': .01, 'weight_decay': .01, 'batch_size': 3,
                  'training_steps': 6, 'checkpoint_interval': 3, 'temperature': 1.}
        order = torch.tensor([0, 1, 2]*6)
        for conditioned in (True, False):
            torch.manual_seed(2372)
            start = SharedResidualSelector(shared, conditioned, text_dim=4)
            initial = {k:v.clone() for k,v in start.branch.state_dict().items()}
            state = torch.random.get_rng_state().clone()
            gradient = initial_gradient_diagnostics(start, data, cached, order[:3], 1.)
            torch.testing.assert_close(state, torch.random.get_rng_state(), rtol=0, atol=0)
            for k,v in initial.items():
                torch.testing.assert_close(v, start.branch.state_dict()[k], rtol=0, atol=0)
            self.assertTrue(all(p.grad is None for p in start.branch.parameters()))
            self.assertGreater(gradient['set_hinge']['branch_gradient_norm'], 0.)
            model, curve, _, _ = fit_branch(shared, conditioned, data, cached, order, config, 2372)
            self.assertEqual([x['step'] for x in curve], [1, 3, 6])
            self.assertTrue(model.branch.output_projection.weight.abs().sum() > 0)
            for k,v in base.items():
                torch.testing.assert_close(v, shared.state_dict()[k], rtol=0, atol=0)
            self.assertTrue(all(p.grad is None for p in shared.parameters()))
            values = fit_rows(model, data, cached, 3, 1.)
            self.assertEqual(values.shape, (3, len(FIT_METRICS)))
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue(np.all(values[:, FIT_METRICS.index('set_loss')] + 1e-6 >= 1-values[:, 0]))

    def test_cached_target_identity_mismatch_is_rejected(self):
        _, data, _, masks = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp)/'targets.npz'
            np.savez(file, patch_indices=masks, image_indices=data['image_ix'], pair_indices=data['pair_ix'],
                     image_ids=data['image_ids'], class_pairs=data['pairs_array'])
            loaded, restored = load_targets(file, data, 3)
            np.testing.assert_array_equal(restored, masks)
            torch.testing.assert_close(loaded['targets'], data['targets'])
            changed = {**data, 'image_ids': data['image_ids'][::-1]}
            with self.assertRaises(AssertionError):
                load_targets(file, changed, 3)


if __name__ == '__main__':
    unittest.main()
