import sys
import unittest
from pathlib import Path
import json

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from run_selector_capacity import training_config, checkpoint_path, check_checkpoint
from token_compression.selector import make_selector


class CapacityRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study = json.loads(Path('configs/selector_capacity.json').read_text())

    def test_three_capacity_pairs_have_matched_parameter_budgets(self):
        previous = 0
        for capacity in ('small', 'medium', 'large'):
            counts = [sum(p.numel() for p in make_selector(torch.zeros(1024), arm, width).parameters())
                      for arm, width in self.study['capacities'][capacity].items()]
            self.assertLess(abs(counts[0] - counts[1]) / counts[0], .005)
            self.assertGreater(counts[0], previous)
            previous = counts[0]

    def test_capacity_changes_do_not_change_the_training_recipe(self):
        for dataset in ('pets', 'flowers'):
            configs = [training_config(self.study, dataset, capacity, 'cuda:0')
                       for capacity in ('small', 'medium', 'large')]
            for key in ('cache_dir', 'source_cache_dir', 'learning_rate', 'weight_decay', 'batch_size',
                        'training_steps', 'temperature', 'training_seeds', 'objective'):
                self.assertEqual(configs[0][key], configs[1][key])
                self.assertEqual(configs[1][key], configs[2][key])
            self.assertIn('runs/', str(checkpoint_path(self.study, dataset, 'medium', 2041, 'conditioned')))
            self.assertIn('/data4/', str(checkpoint_path(self.study, dataset, 'small', 2041, 'conditioned')))

    def test_checkpoint_must_match_objective_step_width_and_sampling_recipe(self):
        config = training_config(self.study, 'flowers', 'medium', 'cuda:0')
        checkpoint = dict(hidden_dim=128, step=3978, conditioned=True, config=config)
        check_checkpoint(checkpoint, config, 128, 'conditioned')
        for key, value in [('objective', 'direct'), ('step', 1200), ('hidden_dim', 64)]:
            with self.assertRaises(AssertionError):
                check_checkpoint(dict(checkpoint, **{key: value}), config, 128, 'conditioned')


if __name__ == '__main__':
    unittest.main()
