import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_direct_classification as runner
from test_direct_classification import TinyVisual
from token_compression.direct_classification import DifferentiablePruner
from token_compression.selector import ImageOnlyWideSelector, build_examples
from token_compression.selector_scale import balanced_example_weights


def toy_data():
    labels = np.array([0, 0, 1, 1, 2, 2])
    pairs = np.array([[0, 1], [0, 2], [1, 2]])
    examples = build_examples(labels, pairs)
    ix, pi = examples["image_indices"].numpy(), examples["pair_indices"].numpy()
    text = F.normalize(torch.randn(3, 7), dim=-1)
    return dict(tokens=torch.randn(6, 10, 12), text=text, text_array=text.numpy(),
                labels=labels, labels_tensor=torch.tensor(labels), image_ids=np.asarray([f"i{i}" for i in range(6)]),
                pairs=torch.tensor(pairs), pairs_array=pairs, examples=examples, image_ix=ix, pair_ix=pi,
                targets=F.softmax(torch.randn(len(ix), 9), dim=-1),
                weights=balanced_example_weights(labels, pairs, ix, pi),
                task_ids=["01", "02", "12"])


class DirectTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.data = toy_data()
        self.visual = TinyVisual().float().eval().requires_grad_(False)
        self.pruner = DifferentiablePruner(self.visual, 1)
        self.model = ImageOnlyWideSelector(7, 12)
        self.config = dict(keep_patch_tokens=4, gate_temperature=.5, teacher_temperature=1.,
                           learning_rate=.001, weight_decay=.01, gradient_clip_norm=1.,
                           training_steps=4, batch_size=3, eval_batch_size=4, log_interval=2,
                           checkpoint_interval=2)

    def test_true_candidate_loss_is_label_order_invariant(self):
        feature = F.normalize(torch.randn(2, 7), dim=-1)
        text = F.normalize(torch.randn(2, 2, 7), dim=-1)
        target = torch.tensor([0, 1])
        left = runner.classification_metrics(feature, text, target, 5.)
        right = runner.classification_metrics(feature, text.flip(1), 1-target, 5.)
        for key in left: torch.testing.assert_close(left[key], right[key])
        torch.testing.assert_close(left["ce"], F.softplus(-5.*left["raw_gap"]))

    def test_direct_gradient_updates_selector_and_preserves_backbone(self):
        initial = copy.deepcopy(self.visual.state_dict())
        old = copy.deepcopy(self.model.state_dict())
        optimizer = runner.optimizer_for(self.model, self.config)
        loss = runner.training_loss(self.model, self.data, torch.arange(3), self.pruner,
                                    "direct", self.config, 5.)
        loss.backward(); optimizer.step()
        self.assertTrue(any(not torch.equal(old[k], v) for k, v in self.model.state_dict().items()))
        for key, value in self.visual.state_dict().items(): torch.testing.assert_close(value, initial[key], atol=0, rtol=0)
        self.assertTrue(all(p.grad is None for p in self.visual.parameters()))

    def test_resume_matches_uninterrupted_training_without_repeating_steps(self):
        for objective in ("kl", "direct"):
            full, resumed = copy.deepcopy(self.model), copy.deepcopy(self.model)
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                args = (self.data, self.pruner, self.config, 5., 2041, "image_only_wide", objective)
                runner.continue_training(full, *args, root/"full", 4, Path("original.pt"))
                runner.continue_training(resumed, *args, root/"resumed", 2, Path("original.pt"))
                # Restore occurs inside continue_training, including Adam moments.
                resumed = copy.deepcopy(self.model)
                runner.continue_training(resumed, *args, root/"resumed", 4, Path("original.pt"))
                for key, value in full.state_dict().items():
                    torch.testing.assert_close(value, resumed.state_dict()[key], atol=0, rtol=0)
                history = root/"resumed/training/seed_2041/image_only_wide"/objective/"training_history.csv"
                self.assertEqual([row.split(",")[0] for row in history.read_text().splitlines()[1:]], ["2", "4"])

    def test_evaluation_scores_reconstruct_accuracy_and_ce(self):
        _, arrays = runner.evaluate(self.model, self.data, self.pruner, self.config, 5.)
        labels = self.data["labels"][self.data["image_ix"]]
        pairs = self.data["pairs_array"][self.data["pair_ix"]]
        other = np.where(pairs[:, 0] == labels, pairs[:, 1], pairs[:, 0])
        row = np.arange(len(labels))
        raw = arrays["cosine_scores"][row, labels] - arrays["cosine_scores"][row, other]
        np.testing.assert_allclose(raw, arrays["raw_gap"], atol=1e-6)
        np.testing.assert_allclose(np.logaddexp(0, -5.*raw), arrays["ce"], atol=1e-6)
        self.assertEqual(arrays["patch_indices"].shape, (len(labels), 4))


if __name__ == "__main__":
    unittest.main()
