"""Audit live batch-1/32 inference against the original cached validation results.

This evaluates numerical sensitivity, not inference speed. It retains all unseen
validation queries and fixed selector seeds; shared GPU use is permitted.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from analyze_selector_budgets import weighted_estimates
from benchmark_selector_latency import InferencePath, write_json, write_rows
from token_compression.pruning import SingleStagePruner
from token_compression.selector import TokenSelector


METHODS = ('full_model', 'attention', 'image_only', 'conditioned')
METRICS = ('accuracy', 'margin')


def pair_outcomes(scores, labels, pairs, text):
    """Signed geometric margin and the original evaluator's class-index tie rule."""
    assert np.all(pairs[:, 0] < pairs[:, 1])
    orientation = np.where(labels == pairs[:, 0], 1., -1.)
    norms = np.linalg.norm(text[pairs[:, 0]] - text[pairs[:, 1]], axis=1)
    margin = orientation * (scores[:, 0] - scores[:, 1]) / norms
    predicted = np.where(scores[:, 0] >= scores[:, 1], pairs[:, 0], pairs[:, 1])
    return margin, (predicted == labels).astype(np.int8)


def set_comparison(current, original, patch_count):
    """Count actual set intersection; sorted-list shifts are not replacements."""
    assert current.shape == original.shape
    indices = np.arange(len(current))[:, None]
    current_mask = np.zeros((len(current), patch_count), dtype=bool)
    original_mask = np.zeros_like(current_mask)
    current_mask[indices, current] = True
    original_mask[indices, original] = True
    assert np.all(current_mask.sum(1) == current.shape[1])
    assert np.all(original_mask.sum(1) == original.shape[1])
    intersection = (current_mask & original_mask).sum(1)
    union = (current_mask | original_mask).sum(1)
    return intersection, current.shape[1] - intersection, intersection / union


def load_reference(evaluation_dir, training_seeds):
    """Align every method/seed to native-full CSV image/task identities."""
    group_rows = {}
    for group in ('references', 'full'):
        with (evaluation_dir / group / 'predictions.csv').open() as handle:
            group_rows[group] = [r for r in csv.DictReader(handle)
                                 if r['task_split'] == 'unseen' and r['method'] in METHODS]
    native = [r for r in group_rows['references'] if r['method'] == 'full_model']
    assert all(int(r['seed']) == -1 for r in native)
    examples = [{key: r[key] for key in ('image_id', 'task_id')} | {
        key: int(r[key]) for key in ('class_id', 'class_a', 'class_b')} | {
            'task_class_weight': float(r['task_class_weight'])} for r in native]
    keys = [(r['image_id'], r['task_id']) for r in examples]
    assert len(keys) == len(set(keys))
    assert np.isclose(sum(r['task_class_weight'] for r in examples), 1., atol=1e-8)
    reference, text = {}, None
    for group, rows in group_rows.items():
        with np.load(evaluation_dir / group / 'predictions.npz') as saved:
            ids, task_ids = saved['image_ids'], saved['task_ids']
            scores, positions = saved['cosine_scores'], saved['patch_indices']
            group_text = saved['text_features']
            if text is not None:
                np.testing.assert_array_equal(text, group_text)
            text = group_text.copy()
            for method in METHODS:
                method_rows = [r for r in rows if r['method'] == method]
                if not method_rows:
                    continue
                expected = training_seeds if method in ('image_only', 'conditioned') else [-1]
                assert sorted({int(r['seed']) for r in method_rows}) == expected
                for seed in expected:
                    chosen = [r for r in method_rows if int(r['seed']) == seed]
                    lookup = {(r['image_id'], r['task_id']): r for r in chosen}
                    assert len(lookup) == len(chosen) == len(keys) and set(lookup) == set(keys)
                    aligned = [lookup[key] for key in keys]
                    ix = np.asarray([int(r['row_id']) for r in aligned])
                    np.testing.assert_array_equal(ids[ix], [r['image_id'] for r in examples])
                    np.testing.assert_array_equal(task_ids[ix], [r['task_id'] for r in examples])
                    pairs = np.asarray([[r['class_a'], r['class_b']] for r in examples])
                    for row, example in zip(aligned, examples):
                        assert all(int(row[field]) == example[field] for field in ('class_id', 'class_a', 'class_b'))
                        assert np.isclose(float(row['task_class_weight']), example['task_class_weight'], atol=1e-12)
                    pair_scores = np.take_along_axis(scores[ix], pairs, axis=1)
                    reference[method, seed] = {
                        'pair_scores': pair_scores.copy(), 'positions': positions[ix].copy(),
                        'margin': np.asarray([float(r['margin']) for r in aligned]),
                        'correct': np.asarray([int(r['correct']) for r in aligned], dtype=np.int8),
                    }
    return examples, reference, text


def summarize_effect(examples, current, reference, training_seeds):
    ids = np.asarray([r['image_id'] for r in examples])
    weights = np.asarray([r['task_class_weight'] for r in examples])
    current_effects, original_effects, per_seed = [], [], []
    for seed in training_seeds:
        online = current['conditioned', seed] - current['image_only', seed]
        original = np.stack([reference['conditioned', seed][name] - reference['image_only', seed][name]
                             for name in ('correct', 'margin')], axis=1)
        current_effects.append(online)
        original_effects.append(original)
        per_seed.append({'seed': seed,
                         'online': dict(zip(METRICS, (weights[:, None] * online).sum(0).tolist())),
                         'original': dict(zip(METRICS, (weights[:, None] * original).sum(0).tolist())),
                         'online_minus_original': dict(zip(METRICS, (weights[:, None] * (online-original)).sum(0).tolist()))})
    online, original = np.mean(current_effects, axis=0), np.mean(original_effects, axis=0)
    contrasts = {'online': online, 'original': original, 'online_minus_original': online-original}
    estimates = {name: weighted_estimates(ids, weights, values, METRICS, 2000, 2052)
                 for name, values in contrasts.items()}
    return {**estimates, 'per_seed': per_seed}


def evaluate_path(path, pixels, text, pairs, image_indices, first_query_indices, batch_size):
    """Reuse unique image forwards only for genuinely task-independent arms."""
    conditioned = path.method == 'conditioned'
    forward_indices = image_indices if conditioned else torch.arange(len(pixels), device=pixels.device)
    query_indices = torch.arange(len(pairs), device=pixels.device) if conditioned else first_query_indices
    retained = path.pruner.num_patches if path.method == 'full_model' else path.count
    features = torch.empty((len(forward_indices), text.shape[1]), dtype=torch.float32, device=pixels.device)
    positions = torch.empty((len(forward_indices), retained), dtype=torch.long, device=pixels.device)
    for start in range(0, len(forward_indices), batch_size):
        end = min(start + batch_size, len(forward_indices))
        candidate_text = text[pairs[query_indices[start:end]]]
        feature, selected = path.embeddings(pixels[forward_indices[start:end]], candidate_text)
        features[start:end] = feature
        positions[start:end] = (torch.arange(retained, device=pixels.device) if selected is None
                                else selected.sort(dim=1).values)
    # Score all queries at the requested batch size, including unique-image arms.
    # This is evaluation reuse only; the latency script always reruns the prefix.
    feature_indices = torch.arange(len(pairs), device=pixels.device) if conditioned else image_indices
    scores = torch.empty((len(pairs), 2), dtype=torch.float32, device=pixels.device)
    for start in range(0, len(pairs), batch_size):
        end = min(start + batch_size, len(pairs))
        candidate_text = text[pairs[start:end]]
        scores[start:end] = (features[feature_indices[start:end], None] * candidate_text).sum(-1)
    count = len(forward_indices)
    tail = count % batch_size
    return scores.cpu().numpy(), positions[feature_indices].cpu().numpy(), {
        'image_forwards': count, 'forward_calls': (count + batch_size - 1) // batch_size,
        'task_independent_image_reuse': not conditioned,
        'forward_tail_size': tail, 'forward_tail_query_indices': query_indices[-tail:].cpu().tolist() if tail else [],
        'score_tail_size': len(pairs) % batch_size,
        'padding': 'none',
    }


def run(args):
    import clip
    from PIL import Image

    before = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    config = json.loads(args.config.read_text())
    evaluation = json.loads((args.evaluation / 'eval_summary.json').read_text())
    for key in ('model', 'prune_after', 'keep_patch_tokens', 'manifest_dir', 'training_seeds'):
        assert config[key] == evaluation['config'][key]
    assert config['training_seeds'] == [2041, 2042, 2043]
    args.output.mkdir(parents=True)
    examples, reference, text_cpu = load_reference(args.evaluation, config['training_seeds'])
    ids = list(dict.fromkeys(row['image_id'] for row in examples))
    image_lookup = {image_id: i for i, image_id in enumerate(ids)}
    first_query = {row['image_id']: i for i, row in reversed(list(enumerate(examples)))}
    labels = np.asarray([r['class_id'] for r in examples])
    pairs_cpu = np.asarray([[r['class_a'], r['class_b']] for r in examples])
    weights = np.asarray([r['task_class_weight'] for r in examples])
    with np.load(Path(config['cache_dir']) / 'val' / 'evidence.npz') as saved:
        np.testing.assert_array_equal(text_cpu, saved['text_features'])
        assert set(ids) <= set(saved['image_ids'])
    for item in reference.values():
        margin, correct = pair_outcomes(item['pair_scores'], labels, pairs_cpu, text_cpu)
        np.testing.assert_allclose(margin, item['margin'], atol=2e-7, rtol=0)
        np.testing.assert_array_equal(correct, item['correct'])
    device = torch.device(config['device'])
    assert device.type == 'cuda'
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    backbone, preprocess = clip.load(config['model'], device=device, jit=False, download_root=config['model_cache'])
    backbone.eval().requires_grad_(False)
    pruner = SingleStagePruner(backbone.visual, config['prune_after'])
    cpu_pixels = []
    for image_id in ids:
        with Image.open(Path(config['data_root']) / 'images' / (image_id + '.jpg')) as image:
            cpu_pixels.append(preprocess(image.convert('RGB')))
    pixels = torch.stack(cpu_pixels).to(device)
    del cpu_pixels
    text = torch.as_tensor(text_cpu, device=device)
    pairs = torch.as_tensor(pairs_cpu, device=device)
    image_indices = torch.tensor([image_lookup[r['image_id']] for r in examples], device=device)
    first_query_indices = torch.tensor([first_query[image_id] for image_id in ids], device=device)
    jobs = [(method, -1) for method in ('full_model', 'attention')]
    jobs += [(method, seed) for seed in config['training_seeds'] for method in ('image_only', 'conditioned')]
    write_rows(args.output / 'examples.csv', [{'example_index': i, **row} for i, row in enumerate(examples)])
    write_rows(args.output / 'unique_images.csv', [{'image_index': i, 'image_id': image_id, 'first_query_index': first_query[image_id]}
                                                  for i, image_id in enumerate(ids)])
    results = {}
    with torch.inference_mode():
        for batch_size in args.batch_sizes:
            folder = args.output / f'batch_{batch_size}'
            folder.mkdir()
            records, parts, current, method_summaries = [], [], {}, []
            for method, seed in jobs:
                selector = None
                checkpoint_path = None
                if method in ('image_only', 'conditioned'):
                    checkpoint_path = args.models / 'full' / f'seed_{seed}' / f'{method}.pt'
                    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
                    selector = TokenSelector(checkpoint['state_dict']['shared_descriptor'],
                                             conditioned=method == 'conditioned', hidden_dim=checkpoint['hidden_dim'],
                                             token_dim=checkpoint.get('token_dim', backbone.visual.conv1.out_channels)).to(device)
                    selector.load_state_dict(checkpoint['state_dict'], strict=True)
                    selector.eval().requires_grad_(False)
                path = InferencePath(backbone, pruner, text, method, config['keep_patch_tokens'], selector)
                torch.cuda.synchronize(device)
                method_start = time.perf_counter()
                scores, positions, execution = evaluate_path(path, pixels, text, pairs, image_indices,
                                                             first_query_indices, batch_size)
                torch.cuda.synchronize(device)
                execution['wall_seconds'] = time.perf_counter() - method_start
                old = reference[method, seed]
                original_positions = old['positions'][:, :positions.shape[1]]
                intersection, replaced, jaccard = set_comparison(positions, original_positions, pruner.num_patches)
                margin, correct = pair_outcomes(scores, labels, pairs_cpu, text_cpu)
                current[method, seed] = np.stack((correct, margin), axis=1)
                score_error = np.abs(scores - old['pair_scores']).max(axis=1)
                changed = correct != old['correct']
                method_summaries.append({
                    'method': method, 'seed': seed, 'checkpoint': str(checkpoint_path) if checkpoint_path else None,
                    'online': {'accuracy': float(weights @ correct), 'margin': float(weights @ margin)},
                    'original': {'accuracy': float(weights @ old['correct']), 'margin': float(weights @ old['margin'])},
                    'online_minus_original': {'accuracy': float(weights @ (correct-old['correct'])),
                                              'margin': float(weights @ (margin-old['margin']))},
                    'prediction_disagreements': int(changed.sum()), 'weighted_disagreement_rate': float(weights @ changed),
                    'repairs': int(((correct == 1) & (old['correct'] == 0)).sum()),
                    'new_errors': int(((correct == 0) & (old['correct'] == 1)).sum()),
                    'changed_selected_sets': int((replaced > 0).sum()),
                    'max_replaced_tokens': int(replaced.max()), 'mean_replaced_tokens': float(replaced.mean()),
                    'mean_jaccard': float(jaccard.mean()),
                    'max_pair_score_error': float(score_error.max()), 'p95_pair_score_error': float(np.quantile(score_error, .95)),
                    'mean_pair_score_error': float(score_error.mean()), 'execution': execution,
                })
                offset = len(records)
                for i, row in enumerate(examples):
                    records.append({'row_id': offset+i, 'example_index': i, 'batch_size': batch_size,
                                    'method': method, 'seed': seed, **row,
                                    'score_a': float(scores[i, 0]), 'score_b': float(scores[i, 1]),
                                    'margin': float(margin[i]), 'correct': int(correct[i]),
                                    'original_score_a': float(old['pair_scores'][i, 0]), 'original_score_b': float(old['pair_scores'][i, 1]),
                                    'original_margin': float(old['margin'][i]), 'original_correct': int(old['correct'][i]),
                                    'retained_patches': positions.shape[1], 'intersection': int(intersection[i]),
                                    'replaced_tokens': int(replaced[i]), 'jaccard': float(jaccard[i])})
                padded_positions = np.full((len(examples), pruner.num_patches), -1, dtype=np.int16)
                padded_positions[:, :positions.shape[1]] = positions
                parts.append({'pair_scores': scores, 'patch_indices': padded_positions,
                              'margin': margin, 'correct': correct, 'intersection': intersection,
                              'replaced_tokens': replaced, 'jaccard': jaccard})
                print(json.dumps({'batch_size': batch_size, 'method': method, 'seed': seed,
                                  'seconds': execution['wall_seconds'], 'prediction_disagreements': int(changed.sum()),
                                  'changed_selected_sets': int((replaced > 0).sum())}), flush=True)
                del path, selector
            write_rows(folder / 'predictions.csv', records)
            packed = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
            packed.update(row_id=np.arange(len(records)),
                          method=np.asarray([r['method'] for r in records]), seed=np.asarray([r['seed'] for r in records]),
                          image_ids=np.asarray([r['image_id'] for r in records]), task_ids=np.asarray([r['task_id'] for r in records]),
                          class_pairs=np.asarray([[r['class_a'], r['class_b']] for r in records]),
                          labels=np.asarray([r['class_id'] for r in records]), text_features=text_cpu)
            np.savez_compressed(folder / 'predictions.npz', **packed)
            results[str(batch_size)] = {
                'methods_per_seed': method_summaries,
                'methods_seed_mean': {method: {version: {metric: float(np.mean([
                    row[version][metric] for row in method_summaries if row['method'] == method]))
                    for metric in METRICS} for version in ('online', 'original', 'online_minus_original')}
                    for method in METHODS},
                'conditioned_minus_image_only': summarize_effect(examples, current, reference, config['training_seeds']),
            }
            write_json(folder / 'summary.json', results[str(batch_size)])
    summary = {
        'started_utc': started, 'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'config': config, 'models_dir': str(args.models), 'evaluation_source': str(args.evaluation),
        'evaluation_code_commit': evaluation['code_commit'], 'image_count': len(ids),
        'image_task_examples_per_method': len(examples), 'task_count': len({r['task_id'] for r in examples}),
        'training_seeds': config['training_seeds'], 'batch_sizes': args.batch_sizes,
        'scope': 'Complete unseen validation queries only. Live image inference reuses the exact latency InferencePath. Task-independent arms reuse one image forward per unique image for this accuracy audit only; conditioned arms run every query. No new training, task selection, checkpoints or final-test evaluation.',
        'batching': 'Deterministic original-query order and first-appearance unique-image order, saved in examples.csv and unique_images.csv. Forward and scoring tails are unpadded and separately recorded; the final short batch is not claimed to have the full requested batch size.',
        'statistics': 'Equal tasks, equal class sides, equal images within side, then average all fixed seeds. Conditioned-minus-image-only and its change from the original use paired image-cluster ratio bootstrap, seed 2052, 2000 replicates, fixed tasks and seeds. Diagnostic intervals are unadjusted; they are not a new hypothesis-search family.',
        'original_results_policy': 'The original cached-batch analysis is retained. This separate implementation sensitivity audit neither selects a preferred numerical setting nor replaces an unfavorable result.',
        'cost_scope': 'Recorded wall time is execution cost on the available GPU, possibly shared. It is not a model latency or speedup measurement.',
        'results': results, 'wall_seconds': time.perf_counter()-before,
        'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
                        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(device),
                        'backbone_dtype': str(backbone.visual.conv1.weight.dtype), 'selector_dtype': 'torch.float32'},
    }
    write_json(args.output / 'summary.json', summary)
    print(json.dumps({'output': str(args.output), 'wall_seconds': summary['wall_seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 32])
    run(parser.parse_args())
