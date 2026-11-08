import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from analyze_content_components import design,fit_predict,derive,ENERGY,GAP,CONTEXT


def fixture():
    rng=np.random.default_rng(14)
    names=[*ENERGY,*GAP[:2],CONTEXT,GAP[2]]
    content=rng.normal(size=(3,100,len(names)))
    return dict(primary_X=rng.normal(size=(3,100,7)),primary_feature_names=np.array([f"s{i}" for i in range(7)]),
        content=content,content_names=names,y=np.stack((content[...,2],2*content[...,2]),axis=-1),
        image_ids=np.repeat([f"i{i}" for i in range(10)],10),task_ids=np.tile([f"t{i}" for i in range(10)],10),
        weights=np.ones(100)/100)


class ContentComponentTests(unittest.TestCase):
    def test_component_designs_share_context_and_partition_new_features(self):
        data=fixture();designs={m:design(data,m) for m in ['context','energy','gap','full']}
        base=set(designs['context'][1]);energy=set(designs['energy'][1]);gap=set(designs['gap'][1]);full=set(designs['full'][1])
        self.assertEqual(energy-base,set(ENERGY));self.assertEqual(gap-base,set(GAP))
        self.assertEqual(energy & gap,base);self.assertEqual(energy | gap,full)
        self.assertEqual([designs[m][0].shape[-1] for m in designs],[8,10,11,13])

    def test_held_targets_do_not_change_held_predictions(self):
        data=fixture();first=fit_predict(data)
        held=(first['image_fold']==0)&(first['task_fold']==0)
        changed=dict(data,y=data['y'].copy());changed['y'][:,held]+=1000
        second=fit_predict(changed)
        np.testing.assert_array_equal(first['predictions'][:,:,held],second['predictions'][:,:,held])
        np.testing.assert_array_equal(first['null_predictions'][:,held],second['null_predictions'][:,held])

    def test_known_gap_signal_is_available_only_in_gap_and_full_groups(self):
        data=fixture();out=fit_predict(data)
        errors=np.mean((out['predictions']-data['y'][None])**2,axis=(1,2,3))
        self.assertLess(errors[2],1e-4);self.assertLess(errors[3],1e-4)
        self.assertGreater(errors[0],.5);self.assertGreater(errors[1],.5)


if __name__=='__main__': unittest.main()
