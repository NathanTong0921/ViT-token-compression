import sys
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_direct_classification import (
    METHODS, CONTRASTS, reconstruct_outcomes, contrast_values, image_contributions, summarize,
    error_transitions,
    load_evaluation,
)
from compare_direct_classification_budgets import paired_budget_effects
from analyze_direct_budget_tradeoff import tradeoff_values


class DirectClassificationAnalysisTests(unittest.TestCase):
    def test_explicit_keep_count_accepts_98_and_preserves_default_49(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); ev=root/"evaluation"; ev.mkdir()
            rows=[dict(image_id="i0",image_index=0,class_id=0,class_a=0,class_b=1,
                       task_class_weight=.5,task_id="pair"),
                  dict(image_id="i1",image_index=1,class_id=1,class_a=0,class_b=1,
                       task_class_weight=.5,task_id="pair")]
            with (ev/"examples.csv").open("w") as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            np.savez(ev/"evidence.npz",image_ids=["i0","i1"],labels=[0,1],text_features=np.eye(2),logit_scale=1.)
            scores=np.asarray([[.8,.2],[.1,.9]])
            outcomes=reconstruct_outcomes(scores,np.asarray([0,1]),np.asarray([[0,1],[0,1]]),np.eye(2),1.)
            for keep in (49,98):
                for arm in ("conditioned","image_only_wide"):
                    p=ev/"seed_2041"/arm;p.mkdir(parents=True,exist_ok=True)
                    for stage in ("initial","kl_step_1200","direct_step_1200"):
                        np.savez(p/(stage+".npz"),cosine_scores=scores,
                            patch_indices=np.tile(np.arange(keep),(2,1)),
                            **{name:outcomes[:,i] for i,name in enumerate(("accuracy","raw_gap","geometric_margin","ce"))})
                if keep==49:
                    default=load_evaluation(root,1200,[2041])
                    explicit=load_evaluation(root,1200,[2041],49)
                    np.testing.assert_array_equal(default["outcomes"],explicit["outcomes"])
                else:
                    actual=load_evaluation(root,1200,[2041],98)
                    self.assertEqual(actual["masks"].shape,(1,6,2,98))
                    with self.assertRaises(AssertionError): load_evaluation(root,1200,[2041])

    def test_budget_interaction_uses_paired_rows_and_rejects_mismatched_tasks(self):
        low=dict(outcomes=np.zeros((2,6,3,4)), methods=np.asarray(METHODS),
            metrics=np.asarray(("accuracy","raw_gap","geometric_margin","ce")),
            image_ids=np.asarray(["a","a","b"]),task_ids=np.asarray(["t1","t2","t1"]),
            seeds=np.asarray([2041,2042]),weights=np.asarray([.2,.3,.5]))
        high={k:v.copy() for k,v in low.items()}
        for data,dc,du,kc,ku in ((low,8,6,5,4),(high,10,7,8,6)):
            for name,value in (("direct_C",dc),("direct_U",du),("kl_C",kc),("kl_U",ku)):
                data["outcomes"][:,METHODS.index(name)]=value
        delta,lower,upper=paired_budget_effects(low,high)
        np.testing.assert_array_equal(lower[:,0],2);np.testing.assert_array_equal(upper[:,0],3)
        np.testing.assert_array_equal(delta[:,0],1)
        np.testing.assert_array_equal(delta[:,1],0)
        tradeoff,names=tradeoff_values(low,high)
        np.testing.assert_array_equal(tradeoff[:,names.index("direct_C")],2)
        np.testing.assert_array_equal(tradeoff[:,names.index("direct_U")],1)
        np.testing.assert_array_equal(tradeoff[:,names.index("objective_tradeoff_C")],-1)
        np.testing.assert_array_equal(tradeoff[:,names.index("objective_tradeoff_U")],-1)
        high["task_ids"]=high["task_ids"][::-1]
        high["task_ids"][0]="wrong"
        with self.assertRaises(AssertionError): paired_budget_effects(low,high)

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

    def test_error_rescue_minus_new_errors_recovers_accuracy_change(self):
        x = np.zeros((2, 6, 4, 4))
        x[:, METHODS.index("direct_U"), :, 0] = [[1, 0, 1, 0], [0, 0, 1, 1]]
        x[:, METHODS.index("direct_C"), :, 0] = [[0, 1, 1, 1], [1, 0, 0, 1]]
        weights = np.asarray([.1, .2, .3, .4])
        result = error_transitions(x, weights, [2041, 2042])["direct_C_vs_direct_U"]
        np.testing.assert_allclose([r["corrected_error_rate"] for r in result["per_seed"]], [.6, .1])
        np.testing.assert_allclose([r["introduced_error_rate"] for r in result["per_seed"]], [.1, .3])
        actual = np.mean((x[:, METHODS.index("direct_C"), :, 0] -
                          x[:, METHODS.index("direct_U"), :, 0]) @ weights)
        self.assertAlmostEqual(result["seed_mean"]["net_accuracy_change"], actual)


if __name__ == "__main__":
    unittest.main()
