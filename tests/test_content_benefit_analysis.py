import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_content_benefit as analysis


def example_data():
    rng = np.random.default_rng(15)
    images = np.repeat([f"i{i}" for i in range(10)], 10)
    tasks = np.tile([f"t{i}" for i in range(10)], 10)
    content = rng.normal(size=(3, 100, 2))
    y = np.stack((content[..., 0], .5 * content[..., 0], (content[..., 0] > 0).astype(float)), axis=-1)
    return dict(primary_X=rng.normal(size=(3, 100, 7)),
                primary_feature_names=np.asarray([f"x{i}" for i in range(7)]),
                content_X=content, content_names=["risk", "full_surrogate_abs_gap"],
                full_raw_gap=rng.normal(size=100), image_ids=images, task_ids=tasks,
                weights=np.ones(100)/100, primary_run_ids=np.asarray([2041,2042,2043]), y=y)


class ContentAnalysisTests(unittest.TestCase):
    def test_context_control_adds_only_projected_full_gap(self):
        data = example_data()
        X, names = analysis.design_inputs(data, "context")
        self.assertEqual(names[-1], "full_surrogate_abs_gap")
        self.assertEqual(X.shape, (3, 100, 8))
        np.testing.assert_array_equal(X[..., -1], data["content_X"][..., 1])
        difficulty, names = analysis.design_inputs(data, "difficulty_content")
        self.assertEqual(difficulty.shape[-1], 11)
        np.testing.assert_array_equal(difficulty[0, :, -2], data["full_raw_gap"])

    def test_tertiles_depend_on_training_scores_and_weights_only(self):
        train = np.asarray([[0., 1., 2., 3., 4., 5.], [0., 1., 2., 3., 4., 5.]])
        held = np.asarray([[-100., 2.5, 100.]])
        groups, cuts = analysis.assign_tertiles(train, held, np.ones(6))
        altered, other_cuts = analysis.assign_tertiles(train, held*100, np.ones(6))
        np.testing.assert_array_equal(cuts, other_cuts)
        self.assertEqual(groups[0,0], -1)
        self.assertEqual(groups[0,-1], 1)
        _, weighted_cuts = analysis.assign_tertiles(train, held, np.asarray([1,1,1,1,1,100.]))
        self.assertTrue(np.all(weighted_cuts >= cuts))

    def test_double_holdout_predictions_do_not_use_held_cell_targets(self):
        data = example_data()
        ifold, tfold = analysis.two_way_folds(data["image_ids"], data["task_ids"])
        chosen = (ifold == 0) & (tfold == 0)
        # Cheap ridge stand-ins exercise the common folding and grouping pipeline.
        with patch.object(analysis, "fit_model", side_effect=lambda model,X,y,w: analysis.fit_panel(X,y,w)):
            original = analysis.out_of_fold(data, ["risk"])
            changed = dict(data, y=data["y"].copy())
            changed["y"][:, chosen] += 1000
            rerun = analysis.out_of_fold(changed, ["risk"])
        np.testing.assert_array_equal(original["predictions"][:, :, chosen], rerun["predictions"][:, :, chosen])
        np.testing.assert_array_equal(original["groups"][:, :, chosen], rerun["groups"][:, :, chosen])
        mse = np.mean((original["predictions"][analysis.MODELS.index("content"), ..., 0]-data["y"][...,0])**2)
        self.assertLess(mse, 1e-4)

    def test_tree_predictor_recovers_nonlinear_signal(self):
        rng = np.random.default_rng(23)
        X = rng.uniform(-2,2,size=(3,200,2))
        y = (X[...,0] ** 2)[...,None]
        fit = analysis.fit_model("tree_content", X, y, np.ones(200)/200)
        held = rng.uniform(-2,2,size=(3,50,2))
        prediction = analysis.predict_panel(fit,held)[...,0]
        self.assertLess(np.mean((prediction-held[...,0]**2)**2), .04)

    def test_group_contributions_reconstruct_weighted_effect(self):
        data = example_data()
        groups = np.sign(data["content_X"][...,0]).astype(np.int8)[None]
        result = dict(predictions=np.zeros((len(analysis.MODELS), *data["y"].shape)), groups=groups, group_names=["risk"])
        names, contributions = analysis.contributions(data,result)
        sums = np.einsum("n,snk->k",data["weights"],contributions)/3
        metrics = analysis.derive_metrics(sums,names,["risk"])
        high, low = groups[0]==1, groups[0]==-1
        weights = np.broadcast_to(data["weights"],high.shape)
        expected = (weights[high]*data["y"][...,0][high]).sum()/weights[high].sum() - (weights[low]*data["y"][...,0][low]).sum()/weights[low].sum()
        self.assertAlmostEqual(metrics["group_risk_high_minus_low_raw"],expected,places=12)

    def test_accuracy_gate_compares_both_fixed_endpoints(self):
        data = example_data()
        rng = np.random.default_rng(19)
        base = rng.integers(0, 2, size=(3,100,3)).astype(float)
        alt = 1 - base
        data.update(primary_base=base, primary_alternative=alt)
        predictions = np.zeros((len(analysis.MODELS), 3, 100, 3))
        predictions[analysis.MODELS.index("content"),...,2] = alt[...,0]-base[...,0]
        predictions[analysis.MODELS.index("tree_content"),...,2] = -1
        result, choices, _, _ = analysis.routing_summary(data,predictions,replicates=40)
        e=result["estimates"]
        self.assertAlmostEqual(e["ridge_route_minus_U_accuracy"]["mean"],1-base[...,0].mean())
        self.assertAlmostEqual(e["ridge_route_minus_C_accuracy"]["mean"],1-alt[...,0].mean())
        self.assertEqual(e["tree_route_minus_U_accuracy"]["mean"],0)
        self.assertFalse(choices[1].any())


if __name__ == "__main__":
    unittest.main()
