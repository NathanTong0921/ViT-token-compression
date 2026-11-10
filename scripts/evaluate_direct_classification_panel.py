"""Evaluate the frozen direct/KL endpoints on the existing larger Flowers panel."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from run_direct_classification import (ARMS, evaluate, restore_selector,
                                       write_evaluation_manifest, write_json)
from token_compression.pruning import SingleStagePruner
from token_compression.selector import build_examples, shared_descriptor
from token_compression.selector_scale import balanced_example_weights


def read_pairs(path):
    with Path(path).open() as handle:
        tasks = list(csv.DictReader(handle))
    pairs = np.asarray([[int(t['class_a']), int(t['class_b'])] for t in tasks])
    assert np.all(pairs[:, 0] < pairs[:, 1])
    return tasks, pairs


def load_panel(cache, image_split, task_split, device):
    cache = Path(cache)
    tasks, pairs = read_pairs(cache / f'{task_split}_tasks.csv')
    with np.load(cache / image_split / 'evidence.npz') as saved:
        ids, labels, text_array, full_scores = [saved[k].copy() for k in
            ('image_ids', 'labels', 'text_features', 'full_scores')]
    labels_tensor = torch.as_tensor(labels, dtype=torch.long, device=device)
    pairs_tensor = torch.as_tensor(pairs, dtype=torch.long, device=device)
    examples = build_examples(labels_tensor, pairs_tensor)
    image_ix = examples['image_indices'].cpu().numpy()
    pair_ix = examples['pair_indices'].cpu().numpy()
    tokens = torch.as_tensor(np.load(cache / image_split / 'prefix_tokens.npy'), device=device)
    return dict(tokens=tokens, image_ids=ids, labels=labels, labels_tensor=labels_tensor,
                text=torch.as_tensor(text_array, dtype=torch.float32, device=device),
                text_array=text_array, full_scores=full_scores, pairs=pairs_tensor,
                pairs_array=pairs, task_ids=[t['task_id'] for t in tasks], examples=examples,
                image_ix=image_ix, pair_ix=pair_ix,
                weights=balanced_example_weights(labels, pairs, image_ix, pair_ix))


def restore_endpoint(source, run, seed, arm, stage, step, descriptor, device):
    model, _, path = restore_selector(source, seed, arm, descriptor, device)
    if stage != 'initial':
        path = Path(run) / 'training' / f'seed_{seed}' / arm / stage / f'step_{step:04d}.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert checkpoint['step'] == step and checkpoint['objective'] == stage
        model.load_state_dict(checkpoint['state_dict'])
    return model.eval().requires_grad_(False), path


def development_check(config, source, output, pruner, descriptor, scale, device):
    """Compare the no-teacher loader against the original direct-training evaluator."""
    from run_semantic_identity_pilot import load_data

    old = load_data(Path(source['cache_dir']), 'val', device,
                    source['batch_size'], source['teacher_temperature'])
    old['labels_tensor'] = torch.as_tensor(old['labels'], dtype=torch.long, device=device)
    new = load_panel(source['cache_dir'], 'val', 'val', device)
    for key in ('image_ids', 'labels', 'pairs_array', 'image_ix', 'pair_ix', 'weights'):
        np.testing.assert_array_equal(new[key], old[key])
    rows = torch.arange(32, device=device)
    checks = []
    for arm in ARMS:
        for stage in ('initial', 'kl', 'direct'):
            model, path = restore_endpoint(source, config['source_run'], config['training_seeds'][0],
                arm, stage, config['step'], descriptor, device)
            before, old_arrays = evaluate(model, old, pruner, source, scale, rows=rows)
            after, new_arrays = evaluate(model, new, pruner, source, scale, rows=rows)
            errors = {}
            for key in old_arrays:
                np.testing.assert_array_equal(new_arrays[key], old_arrays[key])
                errors[key] = float(np.max(np.abs(new_arrays[key] - old_arrays[key])))
            checks.append(dict(arm=arm, stage=stage, checkpoint=str(path), rows=32, maximum_errors=errors))
    write_json(output / 'development_replay.json', dict(passed=True, checks=checks,
        comparison='Same FP32 native evaluator; no-teacher panel loader versus original training loader.'))
    print(json.dumps(dict(development_replay_passed=True, endpoints=len(checks), rows_per_endpoint=32)), flush=True)


def main(args):
    import clip

    config = json.loads(args.config.read_text())
    source = json.loads(Path(config['source_config']).read_text())
    source.update(device=config['device'], eval_batch_size=config['eval_batch_size'])
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config['device'])
    torch.set_num_threads(4)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    backbone, _ = clip.load(source['model'], device=device, jit=False, download_root=source['model_cache'])
    backbone = backbone.float().eval().requires_grad_(False)
    pruner = SingleStagePruner(backbone.visual, source['prune_after'])
    scale = float(backbone.logit_scale.exp())
    with np.load(Path(source['cache_dir']) / 'train' / 'evidence.npz') as saved:
        source_text = saved['text_features'].copy()
        train_ids = saved['image_ids'].copy()
    _, train_pairs = read_pairs(Path(source['cache_dir']) / 'train_tasks.csv')
    descriptor = shared_descriptor(torch.as_tensor(source_text, device=device),
                                   torch.as_tensor(train_pairs, device=device))
    if args.mode == 'smoke':
        development_check(config, source, output, pruner, descriptor, scale, device)
        return
    data = load_panel(config['cache_dir'], config['image_split'], config['task_split'], device)
    assert not set(train_ids) & set(data['image_ids'])
    assert not set(map(tuple, train_pairs)) & set(map(tuple, data['pairs_array']))
    np.testing.assert_allclose(data['text_array'], source_text, atol=1e-6, rtol=0.)
    write_evaluation_manifest(output, data, scale)
    panel = dict(images=len(data['image_ids']), evaluated_images=len(set(data['image_ix'])),
                 tasks=len(data['pairs_array']), examples=len(data['weights']),
                 task_class_weight_sum=float(data['weights'].sum()))
    print(json.dumps(dict(panel=panel)), flush=True)
    jobs=[]
    for seed in config['training_seeds']:
        for arm in ARMS:
            for stage in ('initial', 'kl', 'direct'):
                model, path = restore_endpoint(source, config['source_run'], seed, arm, stage,
                    config['step'], descriptor, device)
                label = stage if stage == 'initial' else f'{stage}_step_{config["step"]:04d}'
                destination = output / 'evaluation' / f'seed_{seed}' / arm / f'{label}.npz'
                summary, _ = evaluate(model, data, pruner, source, scale, destination)
                write_json(destination.with_suffix('.json'), summary)
                job=dict(seed=seed, arm=arm, stage=label, checkpoint=str(path), **summary)
                jobs.append(job)
                print(json.dumps(job), flush=True)
                del model
    summary=dict(config=config, source_config=source, panel=panel, jobs=jobs, logit_scale=scale,
        backbone_dtype=str(next(backbone.parameters()).dtype), prefix_dtype=str(data['tokens'].dtype),
        evaluation_seconds=sum(j['elapsed_seconds'] for j in jobs), total_wall_seconds=time.perf_counter()-started,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        panel_history='Original reserved panel already evaluated with previous models; these fixed direct/KL endpoints are new evaluations.',
        cost_scope='Cached-prefix evaluation and output; no new training or deployment latency measurement.')
    write_json(output/'evaluation_summary.json', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mode', choices=('smoke','evaluate'), default='evaluate')
    main(parser.parse_args())
