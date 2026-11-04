import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analyze_set_distillation import (align_rows, analyze, binary_outcomes, error_transitions,
                                      membership_overlap, summarize)
from analyze_selector_scale import panel_weights


class SetDistillationAnalysisTests(unittest.TestCase):
    def test_alignment_follows_identity_and_rejects_changed_task(self):
        rows = [dict(image_id=i, task_id='t', class_id=c, class_a=0, class_b=1)
                for i, c in [('a', 0), ('b', 1)]]
        np.testing.assert_array_equal(align_rows(rows, rows[::-1]), [1, 0])
        wrong = [dict(rows[1], class_b=2), rows[0]]
        with self.assertRaises(AssertionError):
            align_rows(rows, wrong)
        with self.assertRaises(AssertionError):
            align_rows(rows, [rows[0], rows[0]])

    def test_scores_use_true_class_direction_and_class_id_ties(self):
        text = np.eye(3)
        labels, rivals = np.array([0, 2]), np.array([1, 0])
        scores = np.array([[[.4, .2, .1], [.3, .1, .3]]])
        result = binary_outcomes(scores, labels, rivals, text)
        np.testing.assert_array_equal(result[..., 0], [[1, 0]])
        np.testing.assert_allclose(result[..., 1], [[.2 / np.sqrt(2), 0]])
        np.testing.assert_allclose(result[..., 2], [[.2, 0]])

    def test_error_changes_distinguish_corrections_from_new_errors(self):
        candidate = np.array([[1, .4, .2], [0, -.3, -.1], [1, .2, .1], [0, -.2, -.1]])
        reference = np.array([[0, -.1, -.1], [1, .1, .1], [1, .1, .1], [0, -.1, -.1]])
        got = error_transitions(candidate, reference)
        np.testing.assert_array_equal(got[:, 0], [1, 0, 0, 0])
        np.testing.assert_array_equal(got[:, 1], [0, 1, 0, 0])
        np.testing.assert_array_equal(got[:, 2], candidate[:, 0] - reference[:, 0])
        np.testing.assert_array_equal(got[:, 3], [1, 1, 0, 0])

    def test_mask_overlap_is_unordered_and_broadcasts_teacher(self):
        chosen = np.array([[[[0, 2], [1, 3]], [[2, 0], [0, 1]]]])
        target = np.array([[[[2, 0], [0, 3]]]])
        got = membership_overlap(chosen, target, 4)
        np.testing.assert_allclose(got, [[[1, .5], [1, .5]]])
        with self.assertRaises(AssertionError):
            membership_overlap(np.array([[0, 0]]), np.array([[0, 1]]), 4)

    def test_fixed_seed_average_precedes_image_bootstrap(self):
        values = np.array([[[0.], [0.], [0.]], [[2.], [2.], [2.]]])
        result = summarize(np.array(['a', 'a', 'b']), np.array([.2, .3, .5]),
                           values, [1, 2], ('value',), 50, family=6)
        self.assertEqual(result['value']['mean'], 1)
        self.assertEqual(result['value']['ci95'], [1, 1])
        self.assertEqual(result['value']['bonferroni_family_ci95'], [1, 1])
        self.assertEqual([r['value'] for r in result['per_seed']], [0, 2])

    def test_end_to_end_reordered_reference_recovers_known_net_accuracy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, old, output = root/'set', root/'kl', root/'analysis'
            source.mkdir(); old.mkdir()
            rows = [dict(image_index=i, image_id=['a','b','c','d'][i], class_id=[0,1,0,2][i],
                         task_id=f't{task}', class_a=0, class_b=1+task, task_split='unseen')
                    for i, task in [(0,0),(0,1),(1,0),(2,0),(2,1),(3,1)]]
            weights = panel_weights(rows, 'all')
            rows = [dict(row, task_class_weight=float(w)) for row,w in zip(rows,weights)]
            reverse = np.arange(len(rows))[::-1]
            methods = ['shared','task_residual','image_residual']
            protocol = dict(training_seeds=[2041,2042,2043], methods=methods,
                            setting='pets_b16_49',git_commit='fixture',kl_source=str(old))
            ids=np.array(['a','b','c','d']); labels=np.array([0,1,0,2]); text=np.eye(3)
            full=np.eye(3)[labels]; shared=full.copy(); shared[1]=[.8,.2,0]
            ix=np.array([r['image_index'] for r in rows]); pairs=np.array([[0,1],[0,2]])
            pair_ix=np.array([int(r['task_id'][1:]) for r in rows])
            for folder, data_rows in [(source,rows),(old,[rows[i] for i in reverse])]:
                (folder/'protocol.json').write_text(json.dumps(protocol))
                with (folder/'examples.csv').open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(data_rows)
                np.savez(folder/'evidence.npz',image_ids=ids,labels=labels,text_features=text,full_scores=full)
            (source/'training_records.json').write_text('[]')
            (source/'evaluation_records.json').write_text('[]')
            for seed in protocol['training_seeds']:
                (old/f'seed_{seed}').mkdir();(source/f'seed_{seed}').mkdir()
                old_scores=np.stack([shared[ix]]*3)
                new_scores=np.stack([shared[ix],full[ix],shared[ix]])
                old_masks=np.tile([0,2],(3,len(rows),1))
                new_masks=old_masks.copy();new_masks[1:]=[0,1]
                np.savez(old/f'seed_{seed}.npz',methods=methods,cosine_scores=old_scores[:,reverse],patch_indices=old_masks[:,reverse])
                np.savez(source/f'seed_{seed}.npz',methods=methods,cosine_scores=new_scores,patch_indices=new_masks)
                np.savez(old/f'seed_{seed}/teacher_reference.npz',methods=['conditioned','wide','top_50'],
                         cosine_scores=np.stack([full[ix],shared[ix],full[ix]])[:,reverse])
                np.savez(old/f'seed_{seed}/val_targets.npz',patch_indices=np.tile([0,1],(len(rows),1)),
                         image_ids=ids,image_indices=ix[reverse],pair_indices=pair_ix[reverse],class_pairs=pairs)
                vals=np.zeros((5,len(rows),7));vals[:3,:,0]=.5;vals[3:,:,0]=1
                for split in ('train','val'):
                    np.savez(source/f'seed_{seed}/{split}_fit_rows.npz',
                             model_names=['shared','kl_task','kl_image','set_task','set_image'],
                             metric_names=['overlap','exact_set','set_loss','hybrid_kl','teacher_boundary_gap','score_std','std_floor'],
                             values=vals,image_id=np.array([r['image_id'] for r in rows]),
                             task_id=np.array([r['task_id'] for r in rows]),
                             class_id=labels[ix],class_a=pairs[pair_ix,0],class_b=pairs[pair_ix,1],weights=weights,
                             target_masks=np.tile([0,1],(len(rows),1)))
            result=analyze(source,output,replicates=50)
            self.assertAlmostEqual(result['classification']['contrasts']['set_task_minus_set_image']['accuracy']['mean'],.25)
            self.assertAlmostEqual(result['classification']['contrasts']['objective_by_conditioning_interaction']['accuracy']['mean'],.25)
            self.assertAlmostEqual(result['recovery']['contrasts']['set_task_minus_kl_task']['overlap']['mean'],.5)
            self.assertAlmostEqual(result['error_transitions']['set_task_minus_shared']['corrected']['mean'],.25)
            self.assertEqual(result['error_transitions']['set_task_minus_shared']['broken']['mean'],0)
            self.assertEqual(result['fit_diagnostics']['online_cached_recovery_check']['set_task']['max_row_absolute_difference'],0)
            self.assertTrue((output/'paired_cases.npz').is_file())
            with np.load(output/'paired_cases.npz') as saved:
                np.testing.assert_array_equal(saved['set_task_minus_set_image'][...,0],saved['errors_set_task_minus_set_image'][...,2])


if __name__=='__main__':unittest.main()
