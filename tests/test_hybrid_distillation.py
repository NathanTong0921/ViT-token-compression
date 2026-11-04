import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from analyze_hybrid_distillation import target_interaction, summarize_effect
from evaluate_matched_replacement import build_interventions
from evaluate_selectors import write_rows
from run_hybrid_distillation import (hybrid_masks, mask_distribution, generate_targets,
                                    fit_branch, historical_teacher_reference)
from run_shared_residual import cache_shared
from token_compression.selector import ImageOnlyWideSelector


class HybridDistillationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_target_exactly_matches_historical_top50_rule_including_ties(self):
        rng=np.random.default_rng(51)
        c=rng.integers(0,5,size=(8,12)).astype(float)
        u=rng.integers(0,5,size=(4,12)).astype(float)
        ix=np.repeat(np.arange(4),2)
        old=build_interventions(c,u,ix,np.asarray(['a','b','c','d']),np.asarray([str(i) for i in range(8)]),5,'fixture')
        got=hybrid_masks(c,u[ix],5)
        np.testing.assert_array_equal(got,old['patch_indices'][list(old['variants']).index('top_50')])
        p=mask_distribution(got,12,'cpu').numpy()
        np.testing.assert_allclose(p.sum(1),1)
        self.assertTrue(np.all((p>0).sum(1)==5))
        self.assertTrue(np.all(np.diff(got,axis=-1)>0))

    def test_target_generation_ignores_true_labels_and_candidate_order(self):
        class FakeConditioned(torch.nn.Module):
            def forward(self,tokens,text):
                return tokens[:,1:,0]+text.mean((1,2))[:,None]*tokens[:,1:,1]
        torch.manual_seed(52)
        data={'tokens':torch.randn(2,7,3),'image_ix':np.asarray([0,0,1]),
              'pair_ix':np.asarray([0,1,0]),'image_ids':np.asarray(['a','b']),
              'pairs_array':np.asarray([[0,1],[0,2]]),'text':torch.randn(3,4),
              'examples':{'image_indices':torch.tensor([0,0,1]),'class_pairs':torch.tensor([[0,1],[0,2],[0,1]])}}
        shared=torch.randn(2,6)
        with tempfile.TemporaryDirectory() as tmp:
            first,m=generate_targets(FakeConditioned(),data,shared,3,2,Path(tmp)/'a.npz')
            other={**data,'labels':np.asarray([999,999]),'examples':{**data['examples'],'class_pairs':data['examples']['class_pairs'].flip(1)}}
            second,n=generate_targets(FakeConditioned(),other,shared,3,2,Path(tmp)/'b.npz')
            np.testing.assert_array_equal(m,n)
            torch.testing.assert_close(first['targets'],second['targets'])

    def test_small_paired_training_changes_only_frozen_base_correction(self):
        torch.manual_seed(53)
        shared=ImageOnlyWideSelector(hidden_dim=7,token_dim=3).eval().requires_grad_(False)
        before={k:v.clone() for k,v in shared.state_dict().items()}
        tokens=torch.randn(2,7,3)
        data={'tokens':tokens,'text':torch.randn(3,4),'examples':{'image_indices':torch.tensor([0,1]),'class_pairs':torch.tensor([[0,1],[1,2]])},
              'targets':mask_distribution(np.asarray([[0,2,4],[1,3,5]]),6,'cpu')}
        cache=cache_shared(shared,tokens,2)
        config={'learning_rate':.01,'weight_decay':.01,'batch_size':2,'training_steps':5,'checkpoint_interval':2,'temperature':1.}
        for conditioned in [True,False]:
            m,curve,seconds=fit_branch(shared,conditioned,data,cache,torch.tensor([0,1]*5),config,54)
            self.assertTrue(m.branch.output_projection.weight.abs().sum()>0)
            self.assertEqual([r['step'] for r in curve],[1,2,4,5])
            for k,v in shared.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
            self.assertTrue(all(p.grad is None for p in shared.parameters()))

    def test_old_reference_alignment_uses_image_task_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            data={'image_ids':np.asarray(['a','b']),'image_ix':np.asarray([0,1]),'pair_ix':np.asarray([0,0]),'task_ids':['t']}
            write_rows(root/'examples.csv',[{'image_id':'b','task_id':'t'},{'image_id':'a','task_id':'t'}])
            variants=['conditioned','wide','top_50']
            masks=np.asarray([[[1,2],[0,1]],[[0,2],[1,2]],[[0,1],[0,2]]])
            scores=np.arange(12).reshape(3,2,2)
            np.savez(root/'seed_1.npz',variants=variants,patch_indices=masks,cosine_scores=scores)
            np.savez(root/'fresh.npz',conditioned_patch_indices=masks[0,::-1],shared_patch_indices=masks[1,::-1],patch_indices=masks[2,::-1])
            result=historical_teacher_reference(data,root/'fresh.npz',root,1,root/'out.npz')
            self.assertTrue(result['all_three_masks_exact'])
            with np.load(root/'out.npz') as out:np.testing.assert_array_equal(out['cosine_scores'],scores[:,::-1])

    def test_interaction_is_paired_difference_in_differences(self):
        old_image=np.zeros((2,3,2));old_task=np.ones((2,3,2))*.2
        new_image=np.ones((2,3,2))*.4;new_task=np.ones((2,3,2))*.5
        d=target_interaction(new_task,new_image,old_task,old_image)
        np.testing.assert_allclose(d,-.1)
        result=summarize_effect(np.asarray(['a','a','b']),np.asarray([.25,.25,.5]),d,[1,2],40)
        self.assertAlmostEqual(result['margin']['mean'],-.1)
        self.assertTrue(all(abs(r['accuracy']+.1)<1e-12 for r in result['per_seed']))


if __name__=='__main__':unittest.main()
