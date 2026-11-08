import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_direct_classification import (
    METHODS, CONTRASTS, reconstruct_outcomes, contrast_values, image_contributions, summarize,
)


class DirectClassificationAnalysisTests(unittest.TestCase):
    def test_outcomes_handle_candidate_order_ties_and_large_logits(self):
        scores = np.asarray([[.2, .2], [.8, -.8], [-.8, .8]])
        labels = np.asarray([0, 0, 0])
        pairs = np.asarray([[1, 0], [1, 0], [0, 1]])
        text = np.eye(2)
        actual = reconstruct_outcomes(scores, labels, pairs, text, 100)
        np.testing.assert_array_equal(actual[:, 0], [1, 1, 0])
        np.testing.assert_allclose(actual[:, 1], [0, 1.6, -1.6])
        np.testing.assert_allclose(actual[:, 2], actual[:, 1]/np.sqrt(2))
        np.testing.assert_allclose(actual[:, 3], [np.log(2), 0, 160], atol=1e-10)
        np.testing.assert_array_equal(actual, reconstruct_outcomes(scores, labels, pairs[:, ::-1], text, 100))

    def test_primary_interaction_is_gap_change_not_endpoint_gain(self):
        x = np.zeros((2, 6, 3, 4))
        x[:, METHODS.index("kl_C")] = 5
        x[:, METHODS.index("kl_U")] = 4
        x[:, METHODS.index("direct_C")] = 8
        x[:, METHODS.index("direct_U")] = 6
        differences = contrast_values(x)
        names = list(CONTRASTS)
        np.testing.assert_array_equal(differences[:, names.index("direct_C_minus_direct_U")], 2)
        np.testing.assert_array_equal(differences[:, names.index("direct_gap_minus_kl_gap")], 1)

    def test_cluster_contributions_preserve_weighted_repeated_tasks(self):
        x = np.arange(2*2*3*4).reshape(2, 2, 3, 4).astype(float)
        w = np.asarray([.2, .3, .5])
        ids = np.asarray(["a", "a", "b"])
        unique, sums, totals = image_contributions(x, w, ids)
        np.testing.assert_array_equal(unique, ["a", "b"])
        np.testing.assert_allclose(totals, [.5, .5])
        np.testing.assert_allclose(sums.sum(axis=0), np.einsum("n,smnq->mq", w, x)/2)
        result, per_seed, _, _ = summarize(x, w, ids, ["first", "second"], replicates=50)
        np.testing.assert_allclose(result["first"]["raw_gap"]["mean"], per_seed[:, 0, 1].mean())


if __name__ == "__main__":
    unittest.main()
