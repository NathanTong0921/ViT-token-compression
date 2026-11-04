"""Independent arithmetic audit of fixed-mask set distillation.

Uses NumPy and the standard library only. No project analysis imports, training,
model inference, additional hypothesis tests or newly chosen comparisons.
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np


METHODS = ('shared', 'task_residual', 'image_residual')
METRICS = ('accuracy', 'margin', 'raw_gap')
ERRORS = ('corrected', 'broken', 'net_accuracy', 'prediction_disagreement')
SEEDS = (2041, 2042, 2043)
PAIRS = {
    'set_task_minus_set_image': ('set_task', 'set_image'),
    'kl_task_minus_kl_image': ('kl_task', 'kl_image'),
    'set_task_minus_shared': ('set_task', 'shared'),
    'set_image_minus_shared': ('set_image', 'shared'),
    'set_task_minus_kl_task': ('set_task', 'kl_task'),
    'set_image_minus_kl_image': ('set_image', 'kl_image'),
    'set_task_minus_hybrid_teacher': ('set_task', 'hybrid_teacher'),
    'set_image_minus_hybrid_teacher': ('set_image', 'hybrid_teacher'),
    'hybrid_teacher_minus_shared': ('hybrid_teacher', 'shared'),
}


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


class Check:
    def __init__(self):
        self.values = 0
        self.largest_error = 0.

    def __call__(self, actual, expected, label, tolerance=2e-10):
        a, b = np.asarray(actual), np.asarray(expected)
        assert a.shape == b.shape, label
        np.testing.assert_allclose(a, b, atol=tolerance, rtol=0, err_msg=label)
        self.values += a.size
        if a.size:
            self.largest_error = max(self.largest_error, float(np.abs(a-b).max()))


def identity(rows):
    return [(r['image_id'], r['task_id'], int(r['class_id']),
             int(r['class_a']), int(r['class_b'])) for r in rows]


def align(reference, candidate):
    lookup = {key: i for i, key in enumerate(candidate)}
    assert len(lookup) == len(candidate) == len(reference) == len(set(reference))
    assert set(lookup) == set(reference)
    return np.asarray([lookup[key] for key in reference])


def balanced_weights(meta):
    task_pairs = {}
    counts = Counter((task, y) for _, task, y, _, _ in meta)
    for _, task, y, a, b in meta:
        assert a < b and y in (a, b)
        assert task not in task_pairs or task_pairs[task] == (a, b)
        task_pairs[task] = (a, b)
    for task, pair in task_pairs.items():
        assert all(counts[task, label] for label in pair)
    return np.asarray([1/(2*len(task_pairs)*counts[task, y]) for _, task, y, _, _ in meta])


def outcomes(scores, meta, text):
    y = np.asarray([r[2] for r in meta])
    rival = np.asarray([b if y == a else a for _, _, y, a, b in meta])
    index = np.arange(len(y))
    a, b = scores[..., index, y], scores[..., index, rival]
    gap = a-b
    direction = text[y] - text[rival]
    length = np.sqrt(np.einsum('ij,ij->i', direction, direction))
    assert (length > 0).all()
    correct = (a > b) | ((a == b) & (y < rival))
    return np.stack([correct.astype(float), gap/length, gap], axis=-1)


def ordered_target(path, meta):
    with np.load(path) as x:
        ids = x['image_ids'][x['image_indices']]
        pairs = x['class_pairs'][x['pair_indices']]
        have = [(str(i), int(a), int(b)) for i, (a, b) in zip(ids, pairs)]
        want = [(i, a, b) for i, _, _, a, b in meta]
        masks = x['patch_indices'][align(want, have)].copy()
        assert np.issubdtype(masks.dtype, np.integer)
        assert np.all(np.diff(masks, axis=-1) > 0)
        assert np.all((masks >= 0) & (masks < 196))
        return masks


def overlap(chosen, target):
    # A different construction from the analysis: compare each retained ID with
    # target IDs directly, rather than constructing two patch-membership grids.
    return (chosen[..., :, None] == target[..., None, :]).any(-1).mean(-1)


def bootstrap_matrix(ids, weights, matrix, count, seed):
    groups = [np.flatnonzero(ids == image) for image in sorted(set(ids))]
    totals = np.asarray([(matrix[g]*weights[g, None]).sum(0) for g in groups])
    masses = np.asarray([weights[g].sum() for g in groups])
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(groups), size=(count, len(groups)))
    multiplicities = np.asarray([np.bincount(row, minlength=len(groups)) for row in draws])
    boot = multiplicities @ totals / (multiplicities @ masses)[:, None]
    point = totals.sum(0)/masses.sum()
    return point, boot


def audit_fit_rows(source, old, summary, meta, recovery, check):
    observed = {}
    for split in ('train', 'val'):
        per_seed, common = [], None
        for seed in SEEDS:
            with np.load(source/f'seed_{seed}/{split}_fit_rows.npz') as x:
                names = list(x['model_names']); metrics = list(x['metric_names'])
                rows = list(zip(x['image_id'].tolist(), x['task_id'].tolist(), x['class_id'].tolist(),
                                x['class_a'].tolist(), x['class_b'].tolist()))
                if common is None: common = (rows, names, metrics)
                else: assert common == (rows, names, metrics)
                w = balanced_weights(rows)
                check(x['weights'], w, f'{split}: fitting weights')
                target = ordered_target(old/f'seed_{seed}/{split}_targets.npz', rows)
                np.testing.assert_array_equal(x['target_masks'], target)
                value = x['values'].astype(float)
                assert np.isfinite(value).all()
                oi, ei, li = [metrics.index(k) for k in ('overlap', 'exact_set', 'set_loss')]
                check(value[..., ei], (value[..., oi] == 1).astype(float), f'{split}: exact sets')
                assert np.all(value[..., li] + 2e-6 >= 1-value[..., oi]), 'Set-error upper bound'
                std = value[..., metrics.index('score_std')]
                check(value[..., metrics.index('std_floor')], (std <= 1e-6).astype(float), f'{split}: floor')
                per_seed.append(value)
        values = np.stack(per_seed)
        rows, names, metrics = common
        observed[split] = {'rows': len(rows), 'models': names, 'metrics': metrics}
        for mi, name in enumerate(names):
            for ji, metric in enumerate(metrics):
                points = values[:, mi, :, ji] @ w/w.sum()
                saved = summary['fit_diagnostics'][split][name][metric]
                check(saved['mean'], points.mean(), f'{split}/{name}/{metric}: mean')
                assert [r['seed'] for r in saved['per_seed']] == list(SEEDS)
                check([r['value'] for r in saved['per_seed']], points, f'{split}/{name}/{metric}: seeds')
        if split == 'val':
            order = align(meta, rows)
            for mi, name in enumerate(names):
                diff = values[:, mi, order, metrics.index('overlap')] - recovery[name][..., 0]
                item = summary['fit_diagnostics']['online_cached_recovery_check'][name]
                check(item['max_row_absolute_difference'], abs(diff).max(), f'{name}: online cached max')
                check(item['weighted_mean_difference'], np.mean(diff @ balanced_weights(meta)), f'{name}: online cached weighted')
    return observed


def audit_setting(source, analysis, old):
    check = Check()
    protocol, previous = read_json(source/'protocol.json'), read_json(old/'protocol.json')
    summary = read_json(analysis/'summary.json')
    assert protocol['setting'] == previous['setting'] == summary['setting']
    assert protocol['final_test_used'] is False
    assert tuple(protocol['training_seeds']) == tuple(previous['training_seeds']) == SEEDS
    assert tuple(protocol['methods']) == tuple(previous['methods']) == METHODS
    for key in ('training_steps', 'training_seeds', 'batch_size', 'temperature', 'learning_rate',
                'weight_decay', 'keep_patch_tokens', 'prune_after', 'model', 'cache_dir', 'image_batch_size'):
        assert protocol['config'][key] == previous['config'][key], key
    assert protocol['config']['prune_after'] == 6
    k = protocol['config']['keep_patch_tokens']
    assert k in (49, 98)
    base, old_rows = read_csv(source/'examples.csv'), read_csv(old/'examples.csv')
    meta = identity(base); order = align(meta, identity(old_rows))
    assert all(row['task_split'] == 'unseen' for row in base)
    ids = np.asarray([r[0] for r in meta]); weights = balanced_weights(meta)
    check(weights, [float(r['task_class_weight']) for r in base], 'task/class weights')
    check(weights, [float(old_rows[i]['task_class_weight']) for i in order], 'old task/class weights')
    ix = np.asarray([int(r['image_index']) for r in base])
    old_ix = np.asarray([int(old_rows[i]['image_index']) for i in order])
    with np.load(source/'evidence.npz') as x, np.load(old/'evidence.npz') as y:
        np.testing.assert_array_equal(x['image_ids'][ix], ids)
        np.testing.assert_array_equal(x['labels'][ix], [r[2] for r in meta])
        np.testing.assert_array_equal(x['text_features'], y['text_features'])
        np.testing.assert_array_equal(x['full_scores'][ix], y['full_scores'][old_ix])
        text = x['text_features'].astype(float); full = x['full_scores'][ix].astype(float)
    new_values, old_values, teachers, new_masks, old_masks, targets = [], [], [], [], [], []
    groups = [np.flatnonzero(ids == i) for i in sorted(set(ids))]
    for seed in SEEDS:
        outputs = []
        for folder, arrange in ((source, np.arange(len(base))), (old, order)):
            with np.load(folder/f'seed_{seed}.npz') as x:
                np.testing.assert_array_equal(x['methods'], METHODS)
                scores = x['cosine_scores'][:, arrange].astype(float)
                masks = x['patch_indices'][:, arrange]
                assert scores.shape == (3, len(meta), len(text))
                assert masks.shape == (3, len(meta), k)
                assert np.issubdtype(masks.dtype, np.integer)
                assert np.isfinite(scores).all() and ((masks >= 0) & (masks < 196)).all()
                assert (np.diff(masks, axis=-1) > 0).all()
                chosen = np.sort(np.argsort(-x['selector_logits'][:, arrange], axis=-1, kind='stable')[..., :k], axis=-1)
                np.testing.assert_array_equal(masks, chosen)
                for mi in (0, 2):
                    for group in groups:
                        np.testing.assert_array_equal(masks[mi, group], np.repeat(masks[mi, group[:1]], len(group), axis=0))
                        check(scores[mi, group], np.repeat(scores[mi, group[:1]], len(group), axis=0), 'image-only score invariance', 1e-6)
                outputs.append((outcomes(scores, meta, text), masks, scores))
        check(outputs[0][2][0], outputs[1][2][0], 'unchanged shared scores', 0.)
        np.testing.assert_array_equal(outputs[0][1][0], outputs[1][1][0])
        new_values.append(outputs[0][0]); old_values.append(outputs[1][0])
        new_masks.append(outputs[0][1]); old_masks.append(outputs[1][1])
        with np.load(old/f'seed_{seed}/teacher_reference.npz') as x:
            np.testing.assert_array_equal(x['methods'], ('conditioned', 'wide', 'top_50'))
            teachers.append(outcomes(x['cosine_scores'][:, order].astype(float), meta, text))
        target = ordered_target(old/f'seed_{seed}/val_targets.npz', meta)
        assert target.shape == (len(meta), k)
        targets.append(target)
    new_values, old_values, teachers, new_masks, old_masks, targets = map(np.stack,
        (new_values, old_values, teachers, new_masks, old_masks, targets))
    check(teachers[:, 1], new_values[:, 0], 'teacher shared endpoint', 0.)
    values = {'shared': new_values[:, 0], 'set_task': new_values[:, 1], 'set_image': new_values[:, 2],
              'kl_task': old_values[:, 1], 'kl_image': old_values[:, 2],
              'conditioned_teacher': teachers[:, 0], 'hybrid_teacher': teachers[:, 2]}
    delta = {name: values[a]-values[b] for name, (a,b) in PAIRS.items()}
    delta['objective_by_conditioning_interaction'] = delta['set_task_minus_set_image']-delta['kl_task_minus_kl_image']
    errors = {}
    for name, (a,b) in PAIRS.items():
        ac, bc = values[a][..., 0], values[b][..., 0]
        repaired, broken = ((ac == 1) & (bc == 0)).astype(float), ((ac == 0) & (bc == 1)).astype(float)
        errors[name] = np.stack((repaired, broken, repaired-broken, repaired+broken), axis=-1)
        check(errors[name][..., 2], delta[name][..., 0], 'net accuracy identity')
    onew, oold = overlap(new_masks, targets[:, None]), overlap(old_masks, targets[:, None])
    recovery = {'shared': onew[:, 0, :, None], 'set_task': onew[:, 1, :, None], 'set_image': onew[:, 2, :, None],
                'kl_task': oold[:, 1, :, None], 'kl_image': oold[:, 2, :, None]}
    rdelta = {f'set_{a}_minus_kl_{a}': recovery[f'set_{a}']-recovery[f'kl_{a}'] for a in ('task','image')}
    with np.load(analysis/'paired_cases.npz') as x:
        np.testing.assert_array_equal(x['image_id'], ids)
        np.testing.assert_array_equal(x['task_id'], [r[1] for r in meta])
        check(x['weights'], weights, 'saved weights')
        for name, value in {**values, **delta}.items(): check(x[name], value, 'cases/'+name)
        for name, value in {**recovery, **rdelta}.items(): check(x['recovery_'+name], value, 'recovery cases/'+name)
        for name, value in errors.items(): check(x['errors_'+name], value, 'error cases/'+name)
    entries = []
    for name, value in values.items(): entries.append((name, value, METRICS, 1, summary['classification']['absolute'][name], True))
    for name, value in delta.items(): entries.append((name, value, METRICS, 3 if name=='set_task_minus_set_image' else 1, summary['classification']['contrasts'][name], True))
    entries.append(('full', outcomes(full, meta, text)[None], METRICS, 1, summary['classification']['full_reference'], False))
    for name, value in errors.items(): entries.append(('errors/'+name, value, ERRORS, 1, summary['error_transitions'][name], True))
    for name, value in recovery.items(): entries.append(('recovery/'+name, value, ('overlap',), 1, summary['recovery']['absolute'][name], True))
    for name, value in rdelta.items(): entries.append(('recovery delta/'+name, value, ('overlap',), 6, summary['recovery']['contrasts'][name], True))
    matrix = np.concatenate([e[1].mean(0) for e in entries], axis=-1)
    assert summary['bootstrap_seed'] == 2372
    assert summary['primary_classification_family'] == 3 and summary['primary_recovery_family'] == 6
    points, boot = bootstrap_matrix(ids, weights, matrix, summary['bootstrap_replicates'], 2372)
    offset = 0
    for name, value, metrics, family, saved, has_seeds in entries:
        for j, metric in enumerate(metrics):
            item = saved[metric]; col = offset+j
            check(item['mean'], points[col], name+'/'+metric+' mean')
            check(item['ci95'], np.quantile(boot[:, col], [.025,.975]), name+'/'+metric+' interval')
            if family>1:
                check(item['bonferroni_family_ci95'], np.quantile(boot[:,col], [.025/family,1-.025/family]), name+'/'+metric+' adjusted')
                check(item['individual_interval_level'], 1-.05/family, 'interval level')
        if has_seeds:
            assert [r['seed'] for r in saved['per_seed']] == list(SEEDS)
            check([[r[m] for m in metrics] for r in saved['per_seed']], np.einsum('n,snm->sm', weights,value)/weights.sum(), name+' per seed')
        offset += len(metrics)
    fit = audit_fit_rows(source, old, summary, meta, recovery, check)
    contributions = read_csv(analysis/'image_contributions.csv')
    assert [r['image_id'] for r in contributions] == sorted(set(ids))
    for row, group in zip(contributions, groups):
        check(float(row['weight']), weights[group].sum(), 'image weight')
        for name,value in {**values,**delta}.items():
            for j,metric in enumerate(METRICS):
                check(float(row[f'{name}_{metric}_weighted_sum']), np.sum(weights[group]*value.mean(0)[group,j]), 'image contribution')
    return {'status':'passed','setting':protocol['setting'],'images':len(groups),'rows':len(meta),
            'tasks':len(set(r[1] for r in meta)),'seeds':list(SEEDS),'fit_rows':fit,
            'numeric_values_checked':check.values,'maximum_numeric_error':check.largest_error,
            'bootstrap_replicates':summary['bootstrap_replicates'],'bootstrap_seed':2372,
            'all_saved_point_estimates_intervals_seed_values_reproduced':True,
            'same_shared_outputs_and_teacher_targets_verified':True,
            'scope':'Raw score/mask/identity reconstruction and all published arithmetic; no checkpoint replay, optimizer or CLIP inference.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--kl-source',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    folders=[args.source] if (args.source/'protocol.json').is_file() else sorted(p for p in args.source.iterdir() if (p/'protocol.json').is_file())
    assert folders, 'No completed setting folders'
    checks=[]
    for folder in folders:
        one=(args.source/'protocol.json').is_file()
        analysis=args.analysis if one else args.analysis/folder.name
        old=args.kl_source if (args.kl_source/'protocol.json').is_file() else args.kl_source/folder.name
        checks.append(audit_setting(folder,analysis,old))
        print(json.dumps({'setting':checks[-1]['setting'],'status':'passed'}),flush=True)
    result={'status':'passed','settings':checks,'scope':'Independent NumPy reconstruction, no project analysis imports and no additional comparisons.'}
    destination=args.output or args.analysis/'independent_check.json'
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'passed','settings':len(checks)}))


if __name__=='__main__': main()
