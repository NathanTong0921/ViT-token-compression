"""Fit matched residual branches directly to the existing hybrid retained sets."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from evaluate_selector_scale import selector_from_checkpoint
from evaluate_selectors import write_rows
from run_hybrid_distillation import mask_distribution
from run_semantic_identity_pilot import load_data, synchronize, write_json
from run_shared_residual import (ARMS, METHODS, SETTINGS, audit_shared_endpoint,
                                cache_shared, evaluate, paired_order)
from token_compression.selector import shared_descriptor
from token_compression.shared_residual import SharedResidualSelector
from token_compression.set_objective import STD_FLOOR, normalized_scores, set_losses

FIT_MODELS = ('shared', 'kl_task', 'kl_image', 'set_task', 'set_image')
FIT_METRICS = ('overlap', 'exact_set', 'set_loss', 'hybrid_kl',
               'teacher_boundary_gap', 'score_std', 'std_floor')


def load_targets(source, data, k):
    """Reuse historical masks only after matching complete image/task identities."""
    with np.load(source) as saved:
        for field, expected in (('image_indices', data['image_ix']),
                                ('pair_indices', data['pair_ix']),
                                ('image_ids', data['image_ids']),
                                ('class_pairs', data['pairs_array'])):
            np.testing.assert_array_equal(saved[field], expected)
        masks = saved['patch_indices'].copy()
    assert masks.shape == (len(data['image_ix']), k)
    assert np.all(np.diff(masks, axis=1) > 0)
    assert masks.min() >= 0 and masks.max() < data['tokens'].shape[1] - 1
    device = data['tokens'].device
    return {**data, 'targets': mask_distribution(masks, data['tokens'].shape[1]-1, device),
            'target_masks': torch.as_tensor(masks, dtype=torch.long, device=device)}, masks


def scores_for_rows(model, data, cached, rows):
    ix, pairs = data['examples']['image_indices'][rows], data['examples']['class_pairs'][rows]
    return cached[1][ix] if model is None else model.score_cached(cached[0][ix], cached[1][ix], data['text'][pairs])


@torch.no_grad()
def fit_rows(model, data, cached, batch_size, temperature):
    """Common diagnostics on each image/task, preserving the same teacher set."""
    if model is not None:
        model.eval()
    out = np.empty((len(data['weights']), len(FIT_METRICS)), dtype=np.float64)
    k = data['target_masks'].shape[1]
    for start in range(0, len(out), batch_size):
        end = min(start+batch_size, len(out))
        rows = torch.arange(start, end, device=data['tokens'].device)
        scores = scores_for_rows(model, data, cached, rows)
        target = data['target_masks'][rows]
        z, std = normalized_scores(scores)
        membership = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, target, True)
        chosen = np.argsort(-scores.cpu().numpy(), axis=-1, kind='stable')[:, :k]
        masks = target.cpu().numpy()
        overlap = (chosen[:, :, None] == masks[:, None, :]).any(-1).sum(-1) / k
        gap = z.gather(1, target).min(-1).values - z.masked_fill(membership, -torch.inf).max(-1).values
        kl = F.kl_div(F.log_softmax(scores / temperature, dim=-1), data['targets'][rows],
                      reduction='none').sum(-1)
        out[start:end] = np.stack((overlap, (overlap == 1).astype(float),
                                  set_losses(scores, target).double().cpu().numpy(),
                                  kl.double().cpu().numpy(), gap.double().cpu().numpy(),
                                  std.double().cpu().numpy(), (std <= STD_FLOOR).double().cpu().numpy()), axis=-1)
    return out


def initial_gradient_diagnostics(model, data, cached, rows, temperature):
    """Same first training minibatch, no parameter update or RNG-consuming work."""
    scores = scores_for_rows(model, data, cached, rows)
    objectives = {
        'hybrid_kl': F.kl_div(F.log_softmax(scores / temperature, dim=-1), data['targets'][rows],
                             reduction='none').sum(-1).mean(),
        'set_hinge': set_losses(scores, data['target_masks'][rows]).mean(),
    }
    params = tuple(model.branch.parameters())
    result = {}
    for name, loss in objectives.items():
        gradients = torch.autograd.grad(loss, params, retain_graph=True)
        norm = torch.sqrt(sum(g.square().sum() for g in gradients))
        result[name] = {'loss': float(loss.detach()), 'branch_gradient_norm': float(norm.detach())}
    return result


def fit_branch(shared, conditioned, data, cached, order, config, seed):
    torch.manual_seed(seed)
    model = SharedResidualSelector(shared, conditioned, text_dim=data['text'].shape[1]).to(data['tokens'].device)
    batch = config['batch_size']
    initial = initial_gradient_diagnostics(model, data, cached, order[:batch], config['temperature'])
    optimizer = torch.optim.AdamW(model.branch.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    model.train()
    synchronize(data['tokens'].device)
    before = time.perf_counter()
    curves = []
    for step in range(config['training_steps']):
        rows = order[step*batch:(step+1)*batch]
        optimizer.zero_grad(set_to_none=True)
        scores = scores_for_rows(model, data, cached, rows)
        loss = set_losses(scores, data['target_masks'][rows]).mean()
        loss.backward()
        optimizer.step()
        if step == 0 or (step+1) % config['checkpoint_interval'] == 0 or step+1 == config['training_steps']:
            curves.append({'step': step+1, 'batch_set_hinge': float(loss.detach())})
    synchronize(data['tokens'].device)
    return model.eval().requires_grad_(False), curves, time.perf_counter()-before, initial


def summarize_fit(data, values):
    return {name: dict(zip(FIT_METRICS, (data['weights'] @ value).tolist()))
            for name, value in zip(FIT_MODELS, values)}


def save_fit_rows(path, data, masks, values):
    np.savez_compressed(path, model_names=np.asarray(FIT_MODELS), metric_names=np.asarray(FIT_METRICS),
                        values=np.stack(values), image_id=data['image_ids'][data['image_ix']],
                        task_id=np.asarray(data['task_ids'])[data['pair_ix']],
                        class_id=data['labels'][data['image_ix']],
                        class_a=data['pairs_array'][data['pair_ix'], 0],
                        class_b=data['pairs_array'][data['pair_ix'], 1],
                        weights=data['weights'], target_masks=masks)


def run(config, output, setting, kl_source):
    import clip
    from token_compression.pruning import SingleStagePruner

    output, kl_source = Path(output), Path(kl_source)
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    old_protocol = json.loads((kl_source/'protocol.json').read_text())
    paired_fields = ('training_seeds', 'training_steps', 'batch_size', 'temperature', 'learning_rate',
                     'weight_decay', 'keep_patch_tokens', 'prune_after', 'model', 'cache_dir', 'image_batch_size')
    for key in paired_fields:
        assert config[key] == old_protocol['config'][key], key
    old_records = {(r['seed'], r['arm']):r for r in json.loads((kl_source/'training_records.json').read_text())}
    old_fit = {(r['seed'], r['arm']):r for r in json.loads((kl_source/'target_fit.json').read_text())}
    old_sampling = {r['seed']:r for r in json.loads((kl_source/'sampling_audit.json').read_text())}
    device = torch.device(config['device'])
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    temp, batch, k = config['temperature'], config['batch_size'], config['keep_patch_tokens']
    synchronize(device)
    before = time.perf_counter()
    train, val = [load_data(Path(config['cache_dir']), split, device, batch, temp) for split in ('train', 'val')]
    synchronize(device)
    data_load_seconds = time.perf_counter()-before
    assert not set(train['image_ids']) & set(val['image_ids'])
    assert not set(map(tuple, train['pairs_array'])) & set(map(tuple, val['pairs_array']))
    torch.testing.assert_close(train['text'], val['text'], rtol=0, atol=0)
    backbone, _ = clip.load(config['model'], device=device, jit=False, download_root=config['model_cache'])
    backbone.eval().requires_grad_(False)
    pruner = SingleStagePruner(backbone.visual, config['prune_after'])
    prior = shared_descriptor(train['text'], train['pairs'])
    examples = [{'image_index': int(i), 'image_id': str(val['image_ids'][i]), 'class_id': int(val['labels'][i]),
                 'task_id': val['task_ids'][p], 'class_a': int(val['pairs_array'][p, 0]),
                 'class_b': int(val['pairs_array'][p, 1]), 'task_split': 'unseen',
                 'task_class_weight': float(w)} for i, p, w in zip(val['image_ix'], val['pair_ix'], val['weights'])]
    write_rows(output/'examples.csv', examples)
    np.savez_compressed(output/'evidence.npz', image_ids=val['image_ids'], labels=val['labels'],
                        text_features=val['text_array'], full_scores=val['full_scores'])
    protocol = {'setting': setting, 'config': config, 'git_commit': commit, 'training_seeds': config['training_seeds'],
                'methods': list(METHODS), 'kl_source': str(kl_source.resolve()),
                'objective': 'Row-standardized cost-augmented top-k set hinge; outside-set cost 1; population std floor 1e-6.',
                'target': 'Historical fixed half-exclusive C/U hybrid mask; no target regeneration.',
                'primary': 'Set-vs-KL validation overlap within each arm (six comparisons); task-vs-image classification within set objective (three settings per metric).',
                'secondary': 'Objective-by-conditioning, each student versus U and hybrid teacher, actual corrected and broken classifications.',
                'validation': 'Final fixed step; old data and all three seeds; unseen validation images and candidate combinations.',
                'teacher_labels': 'Teacher masks use image tokens and both candidate texts. Frozen C/U previously fitted supervised utilities.',
                'data_load_seconds': data_load_seconds, 'final_test_used': False}
    write_json(output/'protocol.json', protocol)
    records, curves, sampling, evaluations, fit_summaries = [], [], [], [], []
    for seed in config['training_seeds']:
        folder = output/f'seed_{seed}'
        folder.mkdir(exist_ok=True)
        shared_path = Path(old_records[(seed, ARMS[0])]['shared_checkpoint'])
        assert str(shared_path) == old_records[(seed, ARMS[1])]['shared_checkpoint']
        shared = selector_from_checkpoint(torch.load(shared_path, map_location=device, weights_only=True), prior,
                                          'image_only_wide', train['tokens'].shape[-1], device)
        synchronize(device)
        before = time.perf_counter()
        caches = [cache_shared(shared, d['tokens'], config['image_batch_size']) for d in (train, val)]
        synchronize(device)
        cache_seconds = time.perf_counter()-before
        data, masks = zip(*(load_targets(kl_source/f'seed_{seed}/{split}_targets.npz', d, k)
                            for split, d in zip(('train', 'val'), (train, val))))
        shutil.copy2(kl_source/f'seed_{seed}/teacher_reference.npz', folder/'teacher_reference.npz')
        order, audit = paired_order(train, config['training_steps']*batch, seed)
        assert audit == old_sampling[seed]
        sampling.append(audit)
        synchronize(device)
        before = time.perf_counter()
        fit_values = [[fit_rows(None, d, cache, batch, temp)] for d, cache in zip(data, caches)]
        initial_val_kl = float(val['weights'] @ fit_values[1][0][:, FIT_METRICS.index('hybrid_kl')])
        for arm in ARMS:
            assert abs(initial_val_kl-old_records[(seed, arm)]['initial_val_teacher_kl']) < 1e-9
            checkpoint = torch.load(kl_source/f'seed_{seed}/{arm}.pt', map_location=device, weights_only=True)
            old_model = SharedResidualSelector(shared, arm == 'task_residual', text_dim=train['text'].shape[1]).to(device)
            old_model.branch.load_state_dict(checkpoint['branch_state_dict'])
            for si, (d, cache, split) in enumerate(zip(data, caches, ('train', 'val'))):
                rows = fit_rows(old_model, d, cache, batch, temp)
                fit_values[si].append(rows)
                report = dict(zip(FIT_METRICS, d['weights'] @ rows))
                historical = old_fit[(seed, arm)]['hybrid_trained'][split]
                assert abs(report['hybrid_kl'] - historical['hybrid_target_kl']) < 1e-8
                assert abs(report['overlap'] - historical['hybrid_mask_overlap']) < 1e-10
            del old_model
        synchronize(device)
        diagnostic_seconds = time.perf_counter()-before
        models = {}
        for arm in ARMS:
            model, curve, seconds, gradient = fit_branch(shared, arm == 'task_residual', data[0], caches[0], order, config, seed)
            models[arm] = model
            curves.extend({'seed': seed, 'arm': arm, **row} for row in curve)
            synchronize(device)
            before = time.perf_counter()
            for si, (d, cache) in enumerate(zip(data, caches)):
                fit_values[si].append(fit_rows(model, d, cache, batch, temp))
            synchronize(device)
            diagnostic_seconds += time.perf_counter()-before
            torch.save({'branch_state_dict': {key: value.detach().cpu().clone() for key, value in model.branch.state_dict().items()},
                        'conditioned': model.conditioned, 'hidden_dim': model.branch.hidden_dim, 'shared_hidden_dim': shared.hidden_dim,
                        'text_dim': train['text'].shape[1], 'shared_checkpoint': str(shared_path.resolve()),
                        'seed': seed, 'arm': arm, 'config': config, 'target_source': str((kl_source/f'seed_{seed}').resolve()),
                        'objective': protocol['objective'], 'git_commit': commit}, folder/f'{arm}.pt')
            record = {'seed': seed, 'arm': arm, 'training_seconds': seconds, 'parameters': model.parameter_counts(),
                      'shared_checkpoint': str(shared_path.resolve()), 'initial_val_hybrid_kl': initial_val_kl,
                      'initial_first_minibatch': gradient}
            records.append(record)
            write_json(output/'training_records.json', records)
            write_rows(output/'training_curve.csv', curves)
            print(json.dumps({'setting': setting, **record}), flush=True)
        fits = {'seed': seed}
        for split, d, mask, values in zip(('train', 'val'), data, masks, fit_values):
            save_fit_rows(folder/f'{split}_fit_rows.npz', d, mask, values)
            fits[split] = summarize_fit(d, values)
        fit_summaries.append(fits)
        write_json(output/'target_fit.json', fit_summaries)
        measured = evaluate(models, shared, val, pruner, k, config['image_batch_size'], output/f'seed_{seed}.npz')
        measured['historical_shared_endpoint'] = audit_shared_endpoint(val, output/f'seed_{seed}.npz', Path('runs/core-protection-v1')/setting, seed, config['identity_atol'])
        evaluations.append({'seed': seed, 'shared_feature_cache_seconds': cache_seconds,
                            'diagnostic_seconds': diagnostic_seconds, 'targets_reused_without_generation': True,
                            'initial_and_final_historical_kl_reproduced': True, **measured})
        write_json(output/'evaluation_records.json', evaluations)
        write_json(output/'sampling_audit.json', sampling)
        del models, model, shared, data, caches, fit_values
    return protocol


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setting', choices=SETTINGS, required=True)
    parser.add_argument('--device')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kl-source', type=Path)
    args = parser.parse_args()
    config = json.loads(Path(SETTINGS[args.setting][0]).read_text())
    if args.device:
        config['device'] = args.device
    run(config, args.output, args.setting, args.kl_source or Path('runs/hybrid-distillation-v1')/args.setting)
