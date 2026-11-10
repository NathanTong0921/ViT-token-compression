import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_direct_classification_panel import load_panel
from evaluate_selector_scale import evaluation_examples
from run_direct_classification import write_evaluation_manifest


class DirectPanelTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.cache = Path(self.directory.name)
        folder = self.cache / 'test'
        folder.mkdir()
        self.labels = np.asarray([0, 0, 0, 1, 1, 2])
        self.ids = np.asarray([f'image_{i}' for i in range(len(self.labels))])
        self.tasks = [dict(task_id='task_01', class_a=0, class_b=1),
                      dict(task_id='task_12', class_a=1, class_b=2)]
        with (self.cache/'test_tasks.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=['task_id','class_a','class_b'])
            writer.writeheader(); writer.writerows(self.tasks)
        np.save(folder/'prefix_tokens.npy', np.zeros((6, 10, 12), dtype=np.float16))
        np.savez(folder/'evidence.npz', image_ids=self.ids, labels=self.labels,
                 text_features=np.eye(3,dtype=np.float32), full_scores=np.zeros((6,3)))

    def tearDown(self):
        self.directory.cleanup()

    def test_unequal_class_sizes_match_existing_final_panel_weights(self):
        data = load_panel(self.cache, 'test', 'test', torch.device('cpu'))
        expected = evaluation_examples(self.ids, self.labels, [], self.tasks)
        lookup = {(r['image_id'], r['task_id']):r['task_class_weight'] for r in expected}
        actual = {(self.ids[i], data['task_ids'][p]):float(w)
                  for i,p,w in zip(data['image_ix'],data['pair_ix'],data['weights'])}
        self.assertEqual(actual, lookup)
        self.assertEqual(len(actual),8)
        self.assertAlmostEqual(data['weights'].sum(),1.)
        self.assertNotIn('targets',data)
        self.assertEqual(data['tokens'].dtype,torch.float16)

    def test_manifest_preserves_example_and_evidence_correspondence(self):
        data = load_panel(self.cache,'test','test',torch.device('cpu'))
        write_evaluation_manifest(self.cache/'out',data,100.)
        folder=self.cache/'out/evaluation'
        with (folder/'examples.csv').open() as handle:
            rows=list(csv.DictReader(handle))
        with np.load(folder/'evidence.npz') as evidence:
            for row in rows:
                i=int(row['image_index'])
                self.assertEqual(evidence['image_ids'][i],row['image_id'])
                self.assertEqual(evidence['labels'][i],int(row['class_id']))
                self.assertIn(int(row['class_id']),[int(row['class_a']),int(row['class_b'])])
            self.assertEqual(float(evidence['logit_scale']),100.)


if __name__ == '__main__':
    unittest.main()
