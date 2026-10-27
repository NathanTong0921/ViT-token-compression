"""Independent CPU audit of the prespecified Flowers external validation.

Uses NumPy and saved evidence only; imports no project analysis functions.
Run from the repository root after main and geometry analyses are complete.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

SEEDS = [2041, 2042, 2043]


def read_json(path):
    return json.loads(Path(path).read_text())


def close(actual, expected, tolerance=2e-10):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)


def image_bootstrap(values):
    """Input is sorted-image means, with equal mass on every image."""
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    assert len(values) == 1020
    rng = np.random.default_rng(2052)
    boot = np.empty((2000, values.shape[1]))
    for i in range(2000):
        boot[i] = values[rng.integers(0, len(values), len(values))].mean(0)
    return values.mean(0), np.quantile(boot, [.025, .975], axis=0), np.quantile(boot, [.0125, .9875], axis=0)


def load_prediction_group(path):
    wanted = ['row_id', 'method', 'seed', 'image_ids', 'task_ids', 'task_split',
              'labels', 'class_pairs', 'cosine_scores', 'margin', 'correct']
    with np.load(path) as data:
        chosen = data['task_split'] == 'unseen'
        out = {key: data[key][chosen] for key in wanted}
        out['text_features'] = data['text_features'].copy()
    scores, labels, pairs, text = [out[k] for k in ('cosine_scores', 'labels', 'class_pairs', 'text_features')]
    assert scores.shape[1] == len(text) == 102 and text.shape[1] == 512
    i = np.arange(len(labels))
    other = np.where(labels == pairs[:, 0], pairs[:, 1], pairs[:, 0])
    margin = (scores[i, labels] - scores[i, other]) / np.linalg.norm(text[labels] - text[other], axis=1)
    predicted = pairs[i, np.argmax(scores[i[:, None], pairs], axis=1)]
    correct = predicted == labels
    close(margin, out['margin'], 1e-7)
    np.testing.assert_array_equal(correct, out['correct'])
    out['recomputed'] = np.column_stack([correct, margin]).astype(np.float64)
    return out


def align(group, method, seed):
    ix = np.flatnonzero((group['method'] == method) & (group['seed'] == seed))
    ix = ix[np.lexsort((group['task_ids'][ix], group['image_ids'][ix]))]
    keys = np.column_stack([group['image_ids'][ix], group['task_ids'][ix]])
    assert len(ix) == len(np.unique(keys, axis=0)) == 4080
    ids, counts = np.unique(keys[:, 0], return_counts=True)
    assert len(ids) == 1020 and np.all(counts == 4)
    assert len(np.unique(keys[:, 1])) == 204
    _, class_counts = np.unique(group['labels'][ix], return_counts=True)
    assert len(class_counts) == 102 and np.all(class_counts == 40)
    return keys, group['recomputed'][ix]


def check_estimate(point, ci95, ci975, saved):
    close(point, saved['mean'])
    close(ci95, saved['ci95'])
    if ci975 is not None:
        close(ci975, saved['bonferroni_family_ci95'])
        assert saved['individual_interval_level'] == .975


def independent_angle_fit(cases, image_fixed_effects=False):
    a, b = cases['base_margin_a'], cases['base_margin_b']
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    x = np.column_stack([lo, hi, lo**2, lo*hi, hi**2, cases['angle_degrees']/10])
    y = cases['exchange_loss'].copy()
    # All 1020 images have exactly six pairs, so equal-pair OLS is image-equal.
    if image_fixed_effects:
        for image_id in np.unique(cases['image_ids']):
            ix = cases['image_ids'] == image_id
            x[ix] -= x[ix].mean(0)
            y[ix] -= y[ix].mean()
    scale = x.std(0)
    scale[scale < 1e-12] = 1.
    design = (x - x.mean(0)) / scale
    if not image_fixed_effects:
        design = np.column_stack([np.ones(len(y)), design])
    coef, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    return {'angle_slope': float(coef[-1]/scale[-1]), 'rank': int(rank),
            'columns': design.shape[1], 'angle_estimable': bool(rank > np.linalg.matrix_rank(design[:, :-1]))}


def geometry_audit(folder, saved, common_keys=None):
    with np.load(folder/'seed_mean_cases.npz') as raw:
        cases = {k: raw[k] for k in raw.files}
    keys = np.column_stack([cases['image_ids'], cases['task_a'], cases['task_b']])
    assert len(keys) == len(np.unique(keys, axis=0)) == 6120
    ids, counts = np.unique(cases['image_ids'], return_counts=True)
    assert len(ids) == 1020 and np.all(counts == 6) and np.all(cases['seed_count'] == 3)
    if common_keys is not None:
        np.testing.assert_array_equal(keys, common_keys)
    with np.load(folder/'per_seed_cases.npz') as raw:
        per_seed = {k: raw[k] for k in raw.files}
    losses = []
    seed_slopes = []
    assert sorted(np.unique(per_seed['seed']).tolist()) == SEEDS
    for seed in SEEDS:
        ix = np.flatnonzero(per_seed['seed'] == seed)
        ix = ix[np.lexsort((per_seed['task_b'][ix], per_seed['task_a'][ix], per_seed['image_ids'][ix]))]
        np.testing.assert_array_equal(np.column_stack([per_seed[k][ix] for k in ('image_ids', 'task_a', 'task_b')]), keys)
        losses.append(per_seed['exchange_loss'][ix])
        subset = {k: v[ix] for k, v in per_seed.items()}
        slope = independent_angle_fit(subset)['angle_slope']
        target = next(x for x in saved['per_seed'] if x['seed'] == seed)
        close(slope, target['difficulty_adjusted']['angle_slope'])
        seed_slopes.append({'seed': seed, 'angle_slope': slope})
    close(np.mean(losses, axis=0), cases['exchange_loss'])
    findings = {'image_count': len(ids), 'pair_count': len(keys), 'per_seed_slopes': seed_slopes}
    for key, fixed in [('primary_difficulty_adjusted', False), ('image_fixed_effects_sensitivity', True)]:
        fitted = independent_angle_fit(cases, fixed)
        target = saved[key]
        close(fitted['angle_slope'], target['angle_slope'])
        assert fitted['rank'] == target['design_rank'] and fitted['columns'] == target['design_columns']
        assert fitted['angle_estimable'] == target['angle_estimable_beyond_controls']
        assert target['family_size'] == 2 and target['familywise_interval_level'] == .975
        assert target['positive_after_budget_adjustment'] == (fitted['angle_estimable'] and target['ci_familywise_budgets'][0] > 0)
        assert target['bootstrap_rank_deficient_replicates'] == 0
        findings[key] = {**fitted, 'reported_ci95': target['ci95'],
                         'reported_ci975': target['ci_familywise_budgets'],
                         'positive_after_budget_adjustment': target['positive_after_budget_adjustment']}
    with np.load(folder/'cross_fitted_predictions.npz') as raw:
        cv = {k: raw[k] for k in raw.files}
    for k in ('image_ids', 'task_a', 'task_b'):
        np.testing.assert_array_equal(cv[k], cases[k])
    close(cv['exchange_loss'], cases['exchange_loss'])
    base_mse = float(np.mean((cv['exchange_loss']-cv['difficulty_only'])**2))
    angle_mse = float(np.mean((cv['exchange_loss']-cv['difficulty_plus_angle'])**2))
    ref_mse = float(np.mean((cv['exchange_loss']-cv['training_mean'])**2))
    target = saved['held_image_prediction']
    close([base_mse, angle_mse, (base_mse-angle_mse)/ref_mse],
          [target['difficulty_only_mse'], target['angle_augmented_mse'], target['incremental_out_of_fold_r2']])
    folds = []
    for image_id in ids:
        fold = np.unique(cv['fold'][cv['image_ids'] == image_id])
        assert len(fold) == 1 and int(fold[0]) == target['fold_assignment'][str(image_id)]
        folds.append(int(fold[0]))
    assert np.all(np.bincount(folds) == 204)
    findings['held_image_prediction'] = {'difficulty_only_mse': base_mse, 'angle_augmented_mse': angle_mse,
                                       'incremental_out_of_fold_r2': (base_mse-angle_mse)/ref_mse,
                                       'reported_r2_ci95': target['incremental_out_of_fold_r2_ci95']}
    return findings, keys


def training_audit(folder):
    summary = read_json(folder/'training_summary.json')
    with (folder/'training_history.csv').open() as handle:
        history = list(csv.DictReader(handle))
    findings = []
    assert len(summary['runs']) == 6
    for run in summary['runs']:
        assert run['seed'] in SEEDS and run['coverage'] == 'full'
        assert run['training_steps'] == 3978 and run['training_draws'] == 509184
        assert run['parameter_count'] == 348033 and run['available_training_images'] == 1020
        assert run['clip_parameters_in_optimizer'] == 0
        assert run['initial_state_exactly_matches_shared_seed_state'] and run['image_draws_reused_between_arms']
        values = {k: run[k] for k in ('initial_train_kl', 'initial_val_kl', 'final_train_kl', 'final_val_kl')}
        assert np.isfinite(list(values.values())).all()
        rows = [r for r in history if int(r['seed']) == run['seed'] and r['arm'] == run['arm']]
        assert [int(r['step']) for r in rows] == list(range(0, 3979, 153))
        assert np.isfinite([[float(r['train_kl']), float(r['val_kl'])] for r in rows]).all()
        close(float(rows[-1]['val_kl']), run['final_val_kl'])
        findings.append({'seed': run['seed'], 'arm': run['arm'], **values,
                         'train_kl_reduction_fraction': 1-run['final_train_kl']/run['initial_train_kl'],
                         'val_kl_reduction_fraction': 1-run['final_val_kl']/run['initial_val_kl'],
                         'final_val_minus_train_kl': run['final_val_kl']-run['final_train_kl']})
    return findings


def run(root, result_dir, output):
    main = read_json(result_dir/'analysis.json')
    geometry = read_json(result_dir/'geometry'/'analysis.json')
    assert main['budgets'] == [49, 98] and main['training_seeds'] == SEEDS
    assert main['bootstrap_replicates'] == 2000 and main['bootstrap_seed'] == 2052
    assert geometry['family_size'] == 2 and geometry['final_test_used'] is False
    report = {'status': 'passed', 'audit_scope': 'Independent reconstruction of unseen classification, paired primary bootstrap, H2 point estimates, saved OOF predictions and training-fit summaries.',
              'not_recomputed': 'H2 slope bootstrap intervals; reported intervals checked for multiplicity, rank and decision consistency only.',
              'uncertainty_scope': 'Fixed class-pair graph and three training seeds; images are bootstrap clusters. No final-test evaluation or causal dataset comparison.',
              'budgets': {}}
    contrasts, common_keys, common_geometry = {}, None, None
    for k in (49, 98):
        source = root/main['sources'][str(k)]['evaluation']
        learned = load_prediction_group(source/'full'/'predictions.npz')
        reference = load_prediction_group(source/'references'/'predictions.npz')
        np.testing.assert_array_equal(learned['text_features'], reference['text_features'])
        values = {}
        for method in ('conditioned', 'image_only'):
            arrays = []
            for seed in SEEDS:
                keys, arr = align(learned, method, seed)
                if common_keys is None:
                    common_keys = keys
                np.testing.assert_array_equal(keys, common_keys)
                arrays.append(arr)
            values[method] = np.mean(arrays, axis=0)
        keys, full = align(reference, 'full_model', -1)
        np.testing.assert_array_equal(keys, common_keys)
        contrasts[k] = (values['conditioned']-values['image_only']).reshape(1020, 4, 2).mean(1)
        point, ci95, ci975 = image_bootstrap(contrasts[k])
        outcome = {}
        target = main['task_splits']['unseen']['budgets'][str(k)]
        for j, metric in enumerate(('accuracy', 'margin')):
            check_estimate(point[j], ci95[:, j], ci975[:, j], target['conditioned_minus_image_only'][metric])
            outcome[metric] = {'mean': float(point[j]), 'ci95': ci95[:, j].tolist(), 'ci975': ci975[:, j].tolist()}
        for method, arr in [('full_model', full), *values.items()]:
            saved = next(row for row in target['table'] if row['method'] == method)
            close(arr.mean(0), [saved['accuracy']['mean'], saved['margin']['mean']])
        h2, common_geometry = geometry_audit(result_dir/'geometry'/f'budget_{k}', geometry['budgets'][str(k)], common_geometry)
        models = root/read_json(source/'eval_summary.json')['models_dir']
        report['budgets'][str(k)] = {'image_count': 1020, 'unseen_rows_per_seed_and_arm': 4080,
                                    'conditioned_minus_image_only': outcome, 'h2': h2,
                                    'training': training_audit(models)}
    point, ci95, _ = image_bootstrap(contrasts[98]-contrasts[49])
    contrast = main['task_splits']['unseen']['secondary_change_in_conditioning_advantage']['98_minus_49']
    report['98_minus_49'] = {}
    for j, metric in enumerate(('accuracy', 'margin')):
        check_estimate(point[j], ci95[:, j], None, contrast[metric])
        report['98_minus_49'][metric] = {'mean': float(point[j]), 'ci95': ci95[:, j].tolist()}
    e49, e98 = [report['budgets'][str(k)]['conditioned_minus_image_only']['margin'] for k in (49, 98)]
    report['prespecified_decisions'] = {
        'positive_margin_budget_contrast': bool(ci95[0, 1] > 0),
        'pets_margin_sign_pattern_replicated': e49['ci975'][1] < 0 and e98['ci975'][0] > 0,
        'positive_98_margin_advantage': e98['ci975'][0] > 0,
        'h2_positive_at_both_budgets': all(report['budgets'][str(k)]['h2']['primary_difficulty_adjusted']['positive_after_budget_adjustment'] for k in (49, 98))}
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': report['status'], 'output': str(output), 'decisions': report['prespecified_decisions']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--results', type=Path, default=Path('results/flowers-selector-budgets-v1'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result_dir = args.root/args.results
    run(args.root, result_dir, args.output or result_dir/'scientific_audit.json')
