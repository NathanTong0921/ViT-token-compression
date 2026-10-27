"""Measure complete frozen-CLIP inference, including online token selection.

The timed request starts with a preprocessed GPU image and cached class text
embeddings, and ends with the two candidate-class cosine scores. No image prefix
is cached. CPU preprocessing and pinned host-to-device transfer are recorded
separately and are not added to the model timing.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from token_compression.pruning import SingleStagePruner
from token_compression.selector import TokenSelector


def normalized(features):
    features = features.float()
    return features / features.norm(dim=-1, keepdim=True)


def select_positions(scores, count):
    """Keep the original evaluator's stable tie rule on the GPU."""
    return torch.argsort(scores, dim=1, descending=True, stable=True)[:, :count]


class InferencePath:
    def __init__(self, backbone, pruner, text, method, count, selector=None):
        self.backbone, self.pruner, self.text = backbone, pruner, text
        self.method, self.count, self.selector = method, count, selector

    def embeddings(self, pixels, candidate_text):
        if self.method == 'full_model':
            return normalized(self.backbone.encode_image(pixels)), None
        tokens, attention = self.pruner.prefix(
            pixels, return_attention=self.method == 'attention')
        importance = (attention if self.method == 'attention'
                      else self.selector(tokens, candidate_text))
        positions = select_positions(importance, self.count)
        return normalized(self.pruner.finish(tokens, positions)), positions

    def __call__(self, pixels, pairs):
        candidate_text = self.text[pairs]
        features, _ = self.embeddings(pixels, candidate_text)
        return (features[:, None, :] * candidate_text).sum(dim=-1)


def choose_panel(images, tasks, size=128, seed=2081):
    """One random validation image per class, then fill without replacement."""
    rows = sorted((r for r in images if r['split'] == 'val'), key=lambda r: r['image_id'])
    labels = np.asarray([r['class_id'] for r in rows])
    classes = np.unique(labels)
    assert len(classes) <= size <= len(rows)
    rng = np.random.default_rng(seed)
    selected = [int(rng.choice(np.flatnonzero(labels == label))) for label in classes]
    remaining = np.setdiff1d(np.arange(len(rows)), selected)
    selected += rng.choice(remaining, size - len(selected), replace=False).tolist()
    rng.shuffle(selected)
    unseen = sorted((t for t in tasks if t['split'] == 'val'), key=lambda t: t['task_id'])
    panel = []
    for index in selected:
        row = rows[index]
        valid = [t for t in unseen if row['class_id'] in (t['class_a'], t['class_b'])]
        task = valid[int(rng.integers(len(valid)))]
        panel.append({'panel_index': len(panel), 'image_id': row['image_id'],
                      'class_id': row['class_id'], 'task_id': task['task_id'],
                      'class_a': task['class_a'], 'class_b': task['class_b']})
    return panel


def write_rows(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def gpu_snapshot(physical_gpu):
    fields = 'uuid,name,pstate,temperature.gpu,clocks.current.sm,clocks.current.memory,utilization.gpu,memory.used,power.draw'
    gpu = subprocess.check_output(
        ['nvidia-smi', '-i', str(physical_gpu), '--query-gpu=' + fields,
         '--format=csv,noheader,nounits'], text=True).strip()
    record = dict(zip(fields.split(','), [part.strip() for part in gpu.split(',')]))
    processes = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
         '--format=csv,noheader,nounits'], text=True)
    apps = []
    for line in processes.splitlines():
        uuid, pid, memory = [part.strip() for part in line.split(',')]
        if uuid == record['uuid']:
            apps.append({'pid': int(pid), 'used_memory_mib': memory})
    record.update(utc=datetime.now(timezone.utc).isoformat(), processes=apps,
                  foreign_compute_pids=[p['pid'] for p in apps if p['pid'] != os.getpid()])
    return record


def distribution(values):
    values = np.asarray(values, dtype=float)
    q25, median, q75, p95 = np.quantile(values, [.25, .5, .75, .95])
    return {'count': len(values), 'mean': float(values.mean()),
            'median': float(median), 'q25': float(q25), 'q75': float(q75),
            'p95': float(p95), 'minimum': float(values.min()), 'maximum': float(values.max())}


def summarize_timings(rows, windows):
    """Retain every round; do not select apparently fast measurements."""
    summary = {}
    for batch in sorted({r['batch_size'] for r in rows}):
        chosen = [r for r in rows if r['batch_size'] == batch]
        valid = all(w['uncontended_at_both_checks'] for w in windows if w['batch_size'] == batch)
        methods = {}
        for method in sorted({r['method'] for r in chosen}):
            records = [r for r in chosen if r['method'] == method]
            wall = distribution([r['wall_ms'] for r in records])
            events = distribution([r['cuda_event_ms'] for r in records])
            seeds = []
            for seed in sorted({r['seed'] for r in records}):
                items = [r for r in records if r['seed'] == seed]
                seeds.append({'seed': seed, 'wall_ms': distribution([r['wall_ms'] for r in items]),
                              'cuda_event_ms': distribution([r['cuda_event_ms'] for r in items])})
            ratios = []
            for round_id in sorted({r['round'] for r in records}):
                native = np.median([r['wall_ms'] for r in chosen
                                    if r['round'] == round_id and r['method'] == 'full_model'])
                for seed in sorted({r['seed'] for r in records}):
                    estimate = np.median([r['wall_ms'] for r in records
                                          if r['round'] == round_id and r['seed'] == seed])
                    ratios.append({'round': round_id, 'seed': seed, 'ratio': float(native / estimate)})
            methods[method] = {'wall_ms': wall, 'cuda_event_ms': events, 'per_seed': seeds,
                               'images_per_second_at_median': batch * 1000 / wall['median'],
                               'paired_round_speedup_samples': ratios,
                               'paired_round_speedup': distribution([r['ratio'] for r in ratios]) if valid else None}
        summary[str(batch)] = {'timing_window_valid': valid,
                               'invalid_window_policy': 'All raw timings retained. Speedup summary withheld if another compute process was observed; reruns require a separately recorded window.',
                               'methods': methods}
    return summary


def parity_checks(paths, batches, text, count):
    """Check the online path against the original host-ranked implementation."""
    records = []
    pruner = paths[0]['path'].pruner
    for batch_size, batch_inputs in batches.items():
        pixels, pairs, _ = batch_inputs[0]
        tokens, attention = pruner.prefix(pixels, return_attention=True)
        all_positions = torch.arange(pruner.num_patches, device=pixels.device).expand(len(pixels), -1)
        native = normalized(paths[0]['path'].backbone.encode_image(pixels))
        keep_all = normalized(pruner.finish(tokens, all_positions))
        torch.testing.assert_close(keep_all, native, rtol=0, atol=1e-6)
        keep_all_error = float((keep_all - native).abs().max())
        candidate_text = text[pairs]
        for entry in paths:
            path = entry['path']
            actual, positions = path.embeddings(pixels, candidate_text)
            if entry['method'] == 'full_model':
                reference, position_match = native, True
            else:
                value = attention if entry['method'] == 'attention' else path.selector(tokens, candidate_text)
                host_rank = np.argsort(-value.cpu().numpy(), axis=1, kind='stable')[:, :count]
                reference_positions = np.sort(host_rank, axis=1).copy()
                np.testing.assert_array_equal(positions.sort(dim=1).values.cpu().numpy(), reference_positions)
                reference = normalized(pruner.finish(tokens, torch.as_tensor(reference_positions, device=pixels.device)))
                position_match = True
            torch.testing.assert_close(actual, reference, rtol=0, atol=1e-6)
            expected_scores = (reference[:, None] * candidate_text).sum(-1)
            actual_scores = path(pixels, pairs)
            torch.testing.assert_close(actual_scores, expected_scores, rtol=0, atol=1e-6)
            np.testing.assert_array_equal(actual_scores.argmax(1).cpu(), expected_scores.argmax(1).cpu())
            records.append({'batch_size': batch_size, 'method': entry['method'], 'seed': entry['seed'],
                            'images_checked': len(pixels), 'selected_positions_match_numpy_stable': position_match,
                            'max_feature_error': float((actual - reference).abs().max()),
                            'max_score_error': float((actual_scores - expected_scores).abs().max()),
                            'keep_all_vs_native_max_feature_error': keep_all_error})
    return records


def cached_reference_checks(paths, batches, text, panel, evaluation_dir):
    """Quantify old cached-batch versus new live-batch differences, without timing."""
    requested = {(r['image_id'], r['task_id']) for r in panel[:max(batches)]}
    reference = {}
    for group in ('references', 'full'):
        with (evaluation_dir / group / 'predictions.csv').open() as handle:
            rows = [r for r in csv.DictReader(handle)
                    if (r['image_id'], r['task_id']) in requested
                    and r['method'] in ('full_model', 'attention', 'image_only', 'conditioned')]
        indices = np.asarray([int(r['row_id']) for r in rows])
        with np.load(evaluation_dir / group / 'predictions.npz') as saved:
            features = saved['image_features'][indices]
            positions = saved['patch_indices'][indices]
            scores = saved['cosine_scores'][indices]
        for i, row in enumerate(rows):
            pair = [int(row['class_a']), int(row['class_b'])]
            key = row['method'], int(row['seed']), row['image_id'], row['task_id']
            reference[key] = (features[i], positions[i], scores[i, pair])

    records = []
    for batch_size, bank in batches.items():
        pixels, pairs, _ = bank[0]
        candidate_text = text[pairs]
        for entry in paths:
            features, positions = entry['path'].embeddings(pixels, candidate_text)
            scores = (features[:, None] * candidate_text).sum(-1).cpu().numpy()
            features = features.cpu().numpy()
            positions = None if positions is None else positions.sort(dim=1).values.cpu().numpy()
            old = [reference[entry['method'], entry['seed'], row['image_id'], row['task_id']]
                   for row in panel[:batch_size]]
            old_features = np.stack([r[0] for r in old])
            old_scores = np.stack([r[2] for r in old])
            mask_changes = [] if positions is None else [
                int(np.count_nonzero(current != previous[previous >= 0]))
                for current, (_, previous, _) in zip(positions, old)]
            records.append({'batch_size': batch_size, 'method': entry['method'], 'seed': entry['seed'],
                            'images_checked': len(pixels),
                            'max_feature_error': float(np.abs(features-old_features).max()),
                            'max_pair_score_error': float(np.abs(scores-old_scores).max()),
                            'prediction_disagreements': int(np.count_nonzero(scores.argmax(1) != old_scores.argmax(1))),
                            'rows_with_changed_selected_set': sum(n > 0 for n in mask_changes),
                            'sorted_index_mismatches_per_row': mask_changes})
    return {'scope': 'First complete batch at each timed batch size versus saved full-validation rows for the exact image, pair, method and seed. Cached evaluation uses its original prefix and padded suffix batch shapes; live inference uses the stated unpadded batch. This is a numerical sensitivity audit, separate from the exact same-batch implementation check.',
            'reference': str(evaluation_dir), 'cases': records}


def run(args):
    import clip
    from PIL import Image
    from token_compression.pets import load_manifests

    output = args.output
    output.mkdir(parents=True)
    config = json.loads(args.config.read_text())
    evaluation = json.loads((args.evaluation / 'eval_summary.json').read_text())
    for key in ('model', 'prune_after', 'keep_patch_tokens', 'manifest_dir', 'training_seeds'):
        assert evaluation['config'][key] == config[key]
    device = torch.device(config['device'])
    assert device.type == 'cuda', 'This benchmark measures CUDA model inference.'
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    started = datetime.now(timezone.utc).isoformat()
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    before_setup = time.perf_counter()
    backbone, preprocess = clip.load(config['model'], device=device, jit=False,
                                     download_root=config['model_cache'])
    backbone.eval().requires_grad_(False)
    pruner = SingleStagePruner(backbone.visual, config['prune_after'])
    images, _, tasks = load_manifests(config['manifest_dir'])
    panel = choose_panel(images, tasks, args.panel_size, args.panel_seed)
    write_rows(output / 'panel.csv', panel)
    cache = Path(config['cache_dir'])
    with np.load(cache / 'val' / 'evidence.npz') as evidence:
        text_cpu = evidence['text_features'].copy()
        assert set(r['image_id'] for r in panel) <= set(evidence['image_ids'])
    text = torch.as_tensor(text_cpu, device=device)
    paths = [{'method': method, 'seed': -1,
              'path': InferencePath(backbone, pruner, text, method, config['keep_patch_tokens'])}
             for method in ('full_model', 'attention')]
    for seed in config['training_seeds']:
        for method in ('image_only', 'conditioned'):
            checkpoint_path = args.models / 'full' / f'seed_{seed}' / f'{method}.pt'
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            selector = TokenSelector(checkpoint['state_dict']['shared_descriptor'],
                                     conditioned=method == 'conditioned', hidden_dim=checkpoint['hidden_dim'],
                                     token_dim=checkpoint.get('token_dim', backbone.visual.conv1.out_channels)).to(device)
            selector.load_state_dict(checkpoint['state_dict'], strict=True)
            selector.eval().requires_grad_(False)
            paths.append({'method': method, 'seed': seed, 'checkpoint': str(checkpoint_path),
                          'path': InferencePath(backbone, pruner, text, method,
                                                config['keep_patch_tokens'], selector)})

    cpu_pixels, preprocessing = [], []
    for row in panel:
        before = time.perf_counter()
        with Image.open(Path(config['data_root']) / 'images' / (row['image_id'] + '.jpg')) as image:
            rgb = image.convert('RGB')
        decoded = time.perf_counter()
        cpu_pixels.append(preprocess(rgb))
        finished = time.perf_counter()
        preprocessing.append({'image_id': row['image_id'], 'disk_and_decode_ms': (decoded-before)*1000,
                              'transform_ms': (finished-decoded)*1000})
    cpu_pixels = torch.stack(cpu_pixels).pin_memory()
    device_pixels = cpu_pixels.to(device, non_blocking=True)
    pairs = torch.tensor([[r['class_a'], r['class_b']] for r in panel], device=device)
    batches = {}
    for batch_size in args.batch_sizes:
        assert len(panel) % batch_size == 0, 'The timing pool must form complete batches without padding.'
        batches[batch_size] = [(device_pixels[start:start+batch_size], pairs[start:start+batch_size], start)
                               for start in range(0, len(panel), batch_size)]
    torch.cuda.synchronize(device)
    setup_seconds = time.perf_counter() - before_setup

    with torch.inference_mode():
        correctness = parity_checks(paths, batches, text, config['keep_patch_tokens'])
        write_json(output / 'correctness.json', {
            'status': 'passed', 'reference': 'Same input batch, native prefix, original NumPy stable descending selection, original suffix and normalized pair scores.',
            'scope': 'First full batch at each timed batch size, all methods and all training seeds. This checks the device-only implementation rather than comparing different numerical batch shapes.',
            'cases': correctness})
        write_json(output / 'cached_reference_audit.json',
                   cached_reference_checks(paths, batches, text, panel, args.evaluation))
        h2d = []
        for batch_size in args.batch_sizes:
            for index in range(20):
                start = (index % (len(panel)//batch_size)) * batch_size
                torch.cuda.synchronize(device)
                before = time.perf_counter()
                copied = cpu_pixels[start:start+batch_size].to(device, non_blocking=True)
                torch.cuda.synchronize(device)
                h2d.append({'batch_size': batch_size, 'repeat': index,
                            'wall_ms': (time.perf_counter()-before)*1000})
                del copied
        write_json(output / 'preparation.json', {
            'scope': 'Separate descriptive setup costs; excluded from model timing and not summed into an end-to-end pipeline estimate. CPU values are one pass with current filesystem cache; H2D starts from pinned host memory and includes device allocation.',
            'setup_seconds': setup_seconds, 'preprocess': repr(preprocess),
            'per_image_cpu': preprocessing,
            'cpu_disk_and_decode_ms': distribution([r['disk_and_decode_ms'] for r in preprocessing]),
            'cpu_transform_ms': distribution([r['transform_ms'] for r in preprocessing]),
            'pinned_host_to_device': h2d,
            'text_policy': 'All class text embeddings were computed once by the matched experiment and are cached on GPU. Candidate embedding lookup and conditioned descriptor construction are inside every timed call.'})

        timings, windows, schedules = [], [], []
        rng = np.random.default_rng(args.order_seed)
        for batch_size in args.batch_sizes:
            bank = batches[batch_size]
            for entry in paths:
                for i in range(args.warmup):
                    entry['path'](*bank[i % len(bank)][:2])
            torch.cuda.synchronize(device)
            for round_id in range(args.rounds):
                before_hardware = gpu_snapshot(args.physical_gpu)
                order = rng.permutation(len(paths)).tolist()
                schedules.append({'batch_size': batch_size, 'round': round_id,
                                  'order': [{'method': paths[i]['method'], 'seed': paths[i]['seed']} for i in order]})
                first_row = len(timings)
                for entry_index in order:
                    entry = paths[entry_index]
                    for call in range(args.calls_per_round):
                        input_index = (round_id * args.calls_per_round + call) % len(bank)
                        pixels, candidate_pairs, start_index = bank[input_index]
                        event_start = torch.cuda.Event(enable_timing=True)
                        event_end = torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize(device)
                        before = time.perf_counter()
                        event_start.record()
                        result = entry['path'](pixels, candidate_pairs)
                        event_end.record()
                        torch.cuda.synchronize(device)
                        wall_ms = (time.perf_counter() - before) * 1000
                        cuda_ms = event_start.elapsed_time(event_end)
                        timings.append({'batch_size': batch_size, 'round': round_id, 'call': call,
                                        'method_order': order.index(entry_index), 'method': entry['method'],
                                        'seed': entry['seed'], 'panel_start_index': start_index,
                                        'wall_ms': wall_ms, 'cuda_event_ms': cuda_ms,
                                        'window_uncontended': False})
                        del result
                after_hardware = gpu_snapshot(args.physical_gpu)
                valid = not (before_hardware['foreign_compute_pids'] or after_hardware['foreign_compute_pids'])
                windows.append({'batch_size': batch_size, 'round': round_id,
                                'uncontended_at_both_checks': valid,
                                'before': before_hardware, 'after': after_hardware})
                for row in timings[first_row:]:
                    row['window_uncontended'] = valid
                write_rows(output / 'raw_timings.csv', timings)
                write_json(output / 'hardware_windows.json', windows)
                print(json.dumps({'batch_size': batch_size, 'round': round_id,
                                  'calls_per_condition': args.calls_per_round, 'uncontended': valid}), flush=True)

    summary = {
        'started_utc': started, 'code_commit': source_commit, 'config': config,
        'models_dir': str(args.models), 'evaluation_source': str(args.evaluation),
        'evaluation_code_commit': evaluation['code_commit'],
        'quality_scope': 'Full validation evaluation from evaluation_source; timing-pool outcomes are not used as an accuracy benchmark.',
        'measurement_scope': 'GPU-resident preprocessed images and candidate-class indices to two cosine scores. Includes complete image encoder, online descriptor/selector, stable GPU ranking, spatial sort, gather and suffix. No image-prefix cache, request reuse, hidden padding, compilation, CUDA graphs or model fusion.',
        'excluded_from_primary': ['disk reading', 'CPU image preprocessing', 'host-to-device input transfer', 'one-time class text encoding', 'one-time model/checkpoint loading', 'output transfer to CPU'],
        'primary_clock': 'CUDA-synchronized Python wall clock per request; CUDA events are supplementary.',
        'precision': {'backbone': str(backbone.visual.conv1.weight.dtype), 'selector': 'torch.float32', 'cosine_scores': 'torch.float32'},
        'text_policy': 'Cached individual class embeddings, identical across methods; online pair lookup/descriptor included.',
        'panel_size': len(panel), 'panel_seed': args.panel_seed, 'order_seed': args.order_seed,
        'batch_sizes': args.batch_sizes, 'warmup_calls_per_condition': args.warmup,
        'rounds': args.rounds, 'timed_calls_per_round_per_condition': args.calls_per_round,
        'training_seeds': config['training_seeds'],
        'timed_unique_images_by_batch_size': {
            str(b): len({i for row in timings if row['batch_size'] == b
                         for i in range(row['panel_start_index'], row['panel_start_index']+b)}) for b in args.batch_sizes},
        'batch_construction': 'Contiguous complete batches from the seeded panel; each method/seed receives the same input sequence in each round. Unique timed-image counts are recorded separately from the pool size.',
        'round_order': schedules, 'results': summarize_timings(timings, windows),
        'hardware': {'physical_gpu': args.physical_gpu, 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                     'device_name': torch.cuda.get_device_name(device), 'pid': os.getpid()},
        'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
                        'cuda': torch.version.cuda, 'cpu_threads': torch.get_num_threads()},
        'contention_check_limit': 'Compute-process snapshots before and after each round; they cannot rule out a transient process entirely inside a round. All samples are retained; no fastest-run selection.',
    }
    write_json(output / 'summary.json', summary)
    print(json.dumps({'output': str(output), 'valid': all(v['timing_window_valid'] for v in summary['results'].values())}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--physical-gpu', type=int, required=True)
    parser.add_argument('--panel-size', type=int, default=128)
    parser.add_argument('--panel-seed', type=int, default=2081)
    parser.add_argument('--order-seed', type=int, default=2082)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 32])
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--calls-per-round', type=int, default=20)
    run(parser.parse_args())
