import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import evaluate_hybrid_specificity as evaluation
import analyze_hybrid_specificity as analysis
from evaluate_core_protection import build_masks,unique_image_masks
from evaluate_selector_scale import evaluation_examples
from evaluate_selectors import write_rows

class TinyPruner:
    def __init__(self,*args):pass
    def finish(self,tokens,selected):
        return tokens[torch.arange(len(tokens))[:,None],selected+1,:4].sum(1)+tokens[:,0,:4]

class HybridSpecificityTests(unittest.TestCase):
    def test_half_and_fixed_mask_have_known_removals_and_additions(self):
        a=np.array([10,9,8,7,6,5,4,3,2,1],float)
        b=np.array([10,9,2,1,0,-1,8,7,6,5],float)
        mask,n,d,ok=evaluation.hybrid_mask(a,b,6)
        np.testing.assert_array_equal(mask,[0,1,2,3,6,7]);self.assertEqual((n,d,ok),(4,2,True))
        mask,n,d,ok=evaluation.hybrid_mask(a,b,6,4)
        np.testing.assert_array_equal(mask,[0,1,6,7,8,9]);self.assertEqual((n,d,ok),(4,4,True))
        mask,n,d,ok=evaluation.hybrid_mask(a,b,6,5)
        np.testing.assert_array_equal(mask,np.arange(6));self.assertEqual((n,d,ok),(4,0,False))

    def test_shared_masks_are_task_independent_even_if_conditional_eligibility_changes(self):
        a=np.array([[10,9,8,7,6,5,4,3,2,1],[1,2,3,4,5,6,7,8,9,10]],float)
        u=np.array([[10,9,8,7,6,5,4,3,2,1]],float)
        v=np.array([[10,9,2,1,0,-1,8,7,6,5]],float)
        data=evaluation.build_pair(a,u,v,[0,0],6)
        for i in [1,2,4,6,8]:np.testing.assert_array_equal(data['patch_indices'][i,0],data['patch_indices'][i,1])
        self.assertFalse(data['fixed_d_eligible'][1,1]);self.assertTrue(data['fixed_d_eligible'][1,0])

    def test_all_ordered_distinct_seed_pairs_are_retained(self):
        pairs=evaluation.seed_pairs([1,2,3]);self.assertEqual(len(pairs),6)
        self.assertEqual(set(pairs),{(1,2),(1,3),(2,1),(2,3),(3,1),(3,2)})
        self.assertEqual(evaluation.seed_pairs([1,2,3],True),[(1,1),(2,2),(3,3)])

    def test_restricted_statistics_use_pair_row_weights_not_unweighted_available_means(self):
        ids=np.array(['a','b']);weights=np.array([.5,.5]);pairs=[(1,2),(2,1)]
        delta=np.array([[[1,1],[0,0]],[[3,3],[4,4]]],float)
        valid=np.array([[True,False],[True,True]])
        result=analysis.restricted_summary(ids,weights,delta,valid,pairs,30)
        self.assertAlmostEqual(result['accuracy']['mean'],8/3)
        self.assertEqual(result['weighted_pair_row_coverage'],.75)

    def test_replay_and_analysis_match_direct_scores_and_preserve_zero_shared_exchange(self):
        torch.set_num_threads(1)
        rng=np.random.default_rng(19)
        ids=np.array(['a','b','c','d']);labels=np.array([0,1,2,2])
        text=rng.normal(size=(3,4)).astype(np.float32);text/=np.linalg.norm(text,axis=1,keepdims=True)
        prefix=rng.normal(size=(4,11,6)).astype(np.float32)
        features=prefix[:,0,:4].copy();features/=np.linalg.norm(features,axis=1,keepdims=True)
        tasks=[dict(task_id='0_2',class_a=0,class_b=2),dict(task_id='1_2',class_a=1,class_b=2)]
        rows=evaluation_examples(ids,labels,[],tasks);ix=np.array([r['image_index'] for r in rows])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';source.mkdir();cache=root/'cache';(cache/'val').mkdir(parents=True)
            output=root/'output';np.save(cache/'val/prefix_tokens.npy',prefix)
            np.savez(source/'evidence.npz',image_ids=ids,labels=labels,text_features=text,full_scores=features@text.T)
            write_rows(source/'examples.csv',rows)
            config=dict(cache_dir=str(cache),keep_patch_tokens=5,image_batch_size=2,model='tiny',model_cache=str(root),prune_after=1,device='cpu')
            (source/'summary.json').write_text(json.dumps(dict(config=config,setting='fixture',git_commit='fixture',training_seeds=[101,102,103])))
            for seed in [101,102,103]:
                gen=np.random.default_rng(seed);c=gen.normal(size=(len(rows),10));u=gen.normal(size=(4,10))
                methods,masks=build_masks(c,u,ix,ids,5,'fixture')
                keys,inverse=unique_image_masks(ix,masks)
                f=TinyPruner().finish(torch.tensor(prefix[keys[:,0]]),torch.tensor(keys[:,1:]));f=(f/f.norm(dim=-1,keepdim=True)).numpy()
                np.savez(source/f'seed_{seed}.npz',variants=methods,patch_indices=masks,conditioned_logits=c,wide_logits=u,
                    unique_keys=keys,unique_features=f,cosine_scores=(f@text.T)[inverse])
            backbone=torch.nn.Module();backbone.visual=types.SimpleNamespace()
            fake=types.SimpleNamespace(load=lambda *a,**k:(backbone,None))
            with patch('token_compression.pruning.SingleStagePruner',TinyPruner),patch.dict(sys.modules,{'clip':fake}):
                evaluation.run(source,output)
            for s,t in evaluation.seed_pairs([101,102,103]):
                with np.load(output/f'pair_{s}_{t}.npz') as data:
                    for i in range(len(evaluation.METHODS)):
                        f=TinyPruner().finish(torch.tensor(prefix[ix]),torch.tensor(data['patch_indices'][i].astype(int)))
                        scores=(f/f.norm(dim=-1,keepdim=True)).numpy()@text.T
                        np.testing.assert_allclose(scores,data['cosine_scores'][i],atol=1e-7,rtol=1e-6)
            result=analysis.analyze(output,root/'analysis',replicates=40)
            self.assertEqual(result['primary_family_size'],4)
            self.assertAlmostEqual(result['classification']['contrasts']['cu_minus_uu']['margin']['individual_interval_level'],.9875)
            shared=next(x for x in result['exchange']['table'] if x['method']=='uu_half')
            self.assertEqual(shared['margin']['mean'],0.)
            self.assertIn('hybrid_increment_interaction',result['classification']['contrasts'])

if __name__=='__main__':unittest.main()
