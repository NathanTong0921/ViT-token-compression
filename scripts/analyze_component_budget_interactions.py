"""Exploratory paired budget interactions, after the main ablation results.

Reads already-saved development case arrays. It does not run a model or select
checkpoints. Each image is one bootstrap cluster; fixed training seeds are
averaged before resampling images.
"""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path("scripts").resolve()))
from analyze_selector_budgets import weighted_estimates

ROOT = Path('results')
METRICS = ('accuracy', 'margin')
SEEDS = np.array([2041, 2042, 2043])
COMPONENTS = ('mean_only', 'difference_only')
REPLICATES, BOOTSTRAP_SEED = 2000, 2093


def load(path):
    with np.load(path) as saved:
        return {key: saved[key] for key in saved.files}


def align(first, second):
    np.testing.assert_array_equal(first['training_seeds'], SEEDS)
    np.testing.assert_array_equal(second['training_seeds'], SEEDS)
    keys = [list(zip(x['image_id'].tolist(), x['task_id'].tolist())) for x in (first, second)]
    assert all(len(k) == len(set(k)) for k in keys)
    assert set(keys[0]) == set(keys[1])
    ordered = sorted(keys[0])
    indices = [{key: i for i, key in enumerate(k)} for k in keys]
    orders = [np.array([index[key] for key in ordered]) for index in indices]
    aligned = []
    for data, order in zip((first, second), orders):
        result = {key: data[key][order] for key in
                  ('image_id', 'task_id', 'class_id', 'class_a', 'class_b', 'task_class_weight')}
        for method in ('conditioned', *COMPONENTS):
            values = data[method + '_per_seed'][:, order, :]
            assert values.shape == (3, len(ordered), 2) and np.isfinite(values).all()
            assert np.isin(values[:, :, 0], (0., 1.)).all()
            result[method] = values
            for metric_index, metric in enumerate(METRICS):
                np.testing.assert_allclose(values.mean(axis=0)[:, metric_index],
                                           data[method + '_' + metric][order], atol=1e-14, rtol=0)
        aligned.append(result)
    for key in ('image_id', 'task_id', 'class_id', 'class_a', 'class_b', 'task_class_weight'):
        np.testing.assert_array_equal(aligned[0][key], aligned[1][key])
    base = aligned[0]
    assert np.all((base['class_id'] == base['class_a']) | (base['class_id'] == base['class_b']))
    counts = Counter(zip(base['task_id'], base['class_id']))
    task_count = len(set(base['task_id']))
    expected = np.array([1 / (2 * task_count * counts[task, label])
                         for task, label in zip(base['task_id'], base['class_id'])])
    np.testing.assert_allclose(base['task_class_weight'], expected, atol=1e-14, rtol=0)
    np.testing.assert_allclose(expected.sum(), 1, atol=1e-14, rtol=0)
    return aligned


def analyze(dataset):
    sources = [ROOT / f'{dataset}-selector-ablations-v1' / f'budget_{k}' / 'paired_cases.npz'
               for k in (49, 98)]
    a, b = align(*(load(path) for path in sources))
    weights = a['task_class_weight']
    ids = np.unique(a['image_id'])
    values, per_budget = {}, {}
    for component in COMPONENTS:
        gain49 = a['conditioned'] - a[component]
        gain98 = b['conditioned'] - b[component]
        values[component] = gain49 - gain98
        per_budget[component] = {'49': (gain49.mean(axis=0) * weights[:, None]).sum(axis=0),
                                 '98': (gain98.mean(axis=0) * weights[:, None]).sum(axis=0)}
    contrasts = {}
    for component in COMPONENTS:
        per_seed = np.einsum('n,snm->sm', weights, values[component])
        record = {'definition': f'(conditioned - {component})_49 - (conditioned - {component})_98',
                  'component_removed_by_ablation': 'absolute_difference' if component == 'mean_only' else 'mean',
                  'per_seed': [{ 'seed': int(seed), **dict(zip(METRICS, point.tolist()))}
                               for seed, point in zip(SEEDS, per_seed)],
                  'per_budget_conditioned_minus_ablation': {
                      k: dict(zip(METRICS, value.tolist())) for k, value in per_budget[component].items()}}
        estimates = weighted_estimates(a['image_id'], weights, values[component].mean(axis=0),
                                       METRICS, REPLICATES, BOOTSTRAP_SEED, family_size=2)
        for mi, metric in enumerate(METRICS):
            estimate = estimates[metric]
            record[metric] = {'mean': estimate['mean'], 'ci95': estimate['ci95'],
                              'ci97_5': estimate['bonferroni_family_ci95']}
            np.testing.assert_allclose(estimate['mean'], per_seed[:, mi].mean(), atol=1e-14, rtol=0)
            np.testing.assert_allclose(estimate['mean'], per_budget[component]['49'][mi] -
                                       per_budget[component]['98'][mi], atol=1e-14, rtol=0)
        contrasts[component] = record
    return {'sources': [str(p) for p in sources], 'image_count': len(ids),
            'image_task_examples': len(weights), 'task_count': len(set(a['task_id'])),
            'alignment': 'Exact image/task identities, class labels, class pairs and task/class weights agree across budgets.',
            'contrasts': contrasts}


def main():
    result = {'analysis_status': 'complete', 'analysis_type': 'exploratory',
              'rationale': 'Requested after observing the main component ablations: test the budget difference directly instead of comparing which separate intervals cross zero.',
              'source_code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'analysis_script': str(Path(__file__).resolve().relative_to(Path.cwd())),
              'source_array_schema': {'identity_and_weights': ['image_id', 'task_id', 'class_id', 'class_a', 'class_b', 'task_class_weight'],
                                      'values': ['conditioned_per_seed', 'mean_only_per_seed', 'difference_only_per_seed'],
                                      'value_axes': ['training_seed', 'image_task', 'metric'],
                                      'metric_axis': list(METRICS), 'seed_order_field': 'training_seeds'},
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'training_seeds': SEEDS.tolist(), 'bootstrap_replicates': REPLICATES,
              'bootstrap_seed': BOOTSTRAP_SEED,
              'estimand': 'Task-equal, class-within-task-equal means on unseen validation combinations. Average the three fixed seeds per image/task, then resample whole images with their original row weights and a resampled denominator.',
              'uncertainty': 'Ordinary 95% and two-component-contrast-adjusted 97.5% intervals, separately per dataset and metric. Conditional on existing tasks and three fixed checkpoints; exploratory, not a prespecified confirmatory test.',
              'units': {'accuracy': 'fraction (multiply by 100 for percentage points)', 'margin': 'normalized geometric margin'},
              'interpretation': 'Positive interaction means adding that missing descriptor component has a larger effect at 49 than at 98 retained patches.',
              'results': {dataset: analyze(dataset) for dataset in ('pets', 'flowers')}}
    target = ROOT / 'selector-ablations-v1' / 'budget_component_interactions.json'
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for dataset, data in result['results'].items():
        for component, effect in data['contrasts'].items():
            print(dataset, component, json.dumps({m: effect[m] for m in METRICS}))


if __name__ == '__main__':
    main()
