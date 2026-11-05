"""Time full CLIP, attention, C, wide U and Patch Ranking on identical inputs.

Inputs are preprocessed GPU images and cached class embeddings. Each measured
call runs the image prefix, online selector, token selection, suffix and binary
classification. Image loading, preprocessing and transfer happen before timing.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import clip
import numpy as np
from PIL import Image
import torch
from torch import nn

from benchmark_selector_latency import InferencePath, choose_panel, distribution, gpu_snapshot
from token_compression.patch_ranking import PatchRankingPredictor
from token_compression.pets import load_manifests
from token_compression.pruning import SingleStagePruner
from token_compression.selector import ImageOnlyWideSelector, TokenSelector


class PreservationSelector(nn.Module):
    """Match the common inference interface without a candidate-text branch."""

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, tokens, candidate_text):
        return self.predictor(tokens)


def write_rows(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_selector(path, method, device, token_dim):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if method == 'patch_ranking':
        predictor = PatchRankingPredictor()
        predictor.load_state_dict(checkpoint['model_state'])
        selector = PreservationSelector(predictor)
    elif method == 'image_only_wide':
        selector = ImageOnlyWideSelector(checkpoint['hidden_dim'], token_dim)
        selector.load_state_dict(checkpoint['state_dict'])
    else:
        selector = TokenSelector(checkpoint['state_dict']['shared_descriptor'],
                                 conditioned=True, hidden_dim=checkpoint['hidden_dim'],
                                 token_dim=token_dim)
        selector.load_state_dict(checkpoint['state_dict'])
    return selector.to(device).eval().requires_grad_(False)


def timing_summary(rows, entries, image_parameters):
    result = []
    for batch_size in sorted({r['batch_size'] for r in rows}):
        for entry in entries:
            selected = [r for r in rows if r['batch_size'] == batch_size
                        and r['method'] == entry['method'] and r['seed'] == entry['seed']]
            wall = distribution([r['wall_ms'] for r in selected])
            cuda = distribution([r['cuda_event_ms'] for r in selected])
            result.append({
                'batch_size': batch_size, 'method': entry['method'], 'seed': entry['seed'],
                'selector_parameters': entry['selector_parameters'],
                'image_encoder_parameters': image_parameters,
                'online_image_path_parameters': image_parameters + entry['selector_parameters'],
                'calls': len(selected), 'wall_median_ms': wall['median'],
                'wall_q25_ms': wall['q25'], 'wall_q75_ms': wall['q75'],
                'wall_iqr_ms': wall['q75'] - wall['q25'],
                'cuda_median_ms': cuda['median'], 'cuda_q25_ms': cuda['q25'],
                'cuda_q75_ms': cuda['q75'], 'cuda_iqr_ms': cuda['q75'] - cuda['q25'],
                'images_per_second': batch_size * 1000 / wall['median'],
                'foreign_process_observed': any(r['foreign_process_observed'] for r in selected),
            })
    return result


def run(args):
    config = json.loads(args.config.read_text())
    dataset = config['dataset']
    seeds = args.seeds or config['seeds']
    device = torch.device(args.device or config['device'])
    if device.type != 'cuda':
        raise ValueError('This benchmark measures CUDA inference.')
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.manual_seed(args.panel_seed)
    args.output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    backbone, preprocess = clip.load(config['model'], device=device, jit=False,
                                     download_root=config['model_cache'])
    backbone.eval().requires_grad_(False)
    pruner = SingleStagePruner(backbone.visual, config['prune_after'])
    images, _, tasks = load_manifests(config['manifest_dir'])
    with np.load(Path(config['feature_cache_dir']) / 'val' / 'evidence.npz') as saved:
        text = torch.as_tensor(saved['text_features'].copy(), device=device)
        image_ids = set(saved['image_ids'].tolist())
    panel = choose_panel([r for r in images if r['image_id'] in image_ids], tasks,
                         args.panel_size, args.panel_seed)
    write_rows(args.output / 'panel.csv', panel)
    pixels = []
    for row in panel:
        path = Path(config['data_root']) / 'images' / (row['image_id'] + '.jpg')
        with Image.open(path) as image:
            pixels.append(preprocess(image.convert('RGB')))
    pixels = torch.stack(pixels).to(device)
    pairs = torch.tensor([[r['class_a'], r['class_b']] for r in panel], device=device)
    batches = {}
    for batch_size in args.batch_sizes:
        if len(panel) % batch_size:
            raise ValueError('Panel size must be divisible by every batch size.')
        batches[batch_size] = [(pixels[i:i+batch_size], pairs[i:i+batch_size], i)
                               for i in range(0, len(panel), batch_size)]

    count = config['keep_patch_tokens']
    entries = [{'method': method, 'seed': -1, 'checkpoint': None, 'selector_parameters': 0,
                'path': InferencePath(backbone, pruner, text, method, count)}
               for method in ('full_model', 'attention')]
    roots = {
        'conditioned': args.conditioned_models or Path(f'runs/{dataset}-selector-budget-98-training-v1'),
        'image_only_wide': args.shared_models or Path(f'runs/{dataset}-selector-ablations-98-training-v1'),
        'patch_ranking': args.patch_models or Path(f'runs/patch-ranking-v1/{dataset}/training'),
    }
    for seed in seeds:
        for method, root in roots.items():
            checkpoint = (root / f'seed_{seed}' / 'model.pt' if method == 'patch_ranking'
                          else root / 'full' / f'seed_{seed}' / f'{method}.pt')
            selector = load_selector(checkpoint, method, device, backbone.visual.conv1.out_channels)
            entries.append({'method': method, 'seed': seed, 'checkpoint': str(checkpoint),
                            'selector_parameters': sum(p.numel() for p in selector.parameters()),
                            'path': InferencePath(backbone, pruner, text, method, count, selector)})

    timings, windows = [], []
    rng = np.random.default_rng(args.order_seed)
    with torch.inference_mode():
        for batch_size, bank in batches.items():
            for entry in entries:
                for index in range(args.warmup):
                    entry['path'](*bank[index % len(bank)][:2])
            torch.cuda.synchronize(device)
            for round_id in range(args.rounds):
                before_gpu = gpu_snapshot(args.physical_gpu)
                first_row = len(timings)
                order = rng.permutation(len(entries))
                for order_index, entry_index in enumerate(order):
                    entry = entries[entry_index]
                    for call in range(args.calls_per_round):
                        batch = bank[(round_id * args.calls_per_round + call) % len(bank)]
                        begin = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize(device)
                        before = time.perf_counter()
                        begin.record()
                        scores = entry['path'](*batch[:2])
                        end.record()
                        torch.cuda.synchronize(device)
                        wall_ms = 1000 * (time.perf_counter() - before)
                        timings.append({'batch_size': batch_size, 'round': round_id, 'call': call,
                                        'method': entry['method'], 'seed': entry['seed'],
                                        'method_order': order_index, 'panel_start_index': batch[2],
                                        'wall_ms': wall_ms, 'cuda_event_ms': begin.elapsed_time(end),
                                        'foreign_process_observed': False})
                        del scores
                after_gpu = gpu_snapshot(args.physical_gpu)
                foreign = bool(before_gpu['foreign_compute_pids'] or after_gpu['foreign_compute_pids'])
                windows.append({'batch_size': batch_size, 'round': round_id,
                                'before': before_gpu, 'after': after_gpu})
                for row in timings[first_row:]:
                    row['foreign_process_observed'] = foreign
                write_rows(args.output / 'raw_timings.csv', timings)
                write_json(args.output / 'hardware_windows.json', windows)
                print(json.dumps({'batch_size': batch_size, 'round': round_id,
                                  'foreign_process_observed': foreign}), flush=True)

    image_parameters = sum(p.numel() for p in backbone.visual.parameters())
    summaries = timing_summary(timings, entries, image_parameters)
    write_rows(args.output / 'latency.csv', summaries)
    write_json(args.output / 'summary.json', {
        'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
        'config': config, 'seeds': seeds,
        'methods': [{k: v for k, v in entry.items() if k != 'path'} for entry in entries],
        'scope': 'GPU-resident image and cached text to binary cosine scores; full image prefix, selector, pooling, ranking, gather, suffix and classification.',
        'input_preparation': 'Real validation images preprocessed once and held on GPU. Candidate pairs are fixed validation combinations. Every method receives the same input sequence.',
        'excluded': ['image loading', 'CPU preprocessing', 'host-to-device transfer',
                     'one-time text encoding', 'model loading', 'output transfer to CPU'],
        'clock': 'CUDA-synchronized wall clock; CUDA event time is also recorded.',
        'panel_size': len(panel), 'panel_seed': args.panel_seed, 'order_seed': args.order_seed,
        'batch_sizes': args.batch_sizes, 'warmup': args.warmup, 'rounds': args.rounds,
        'calls_per_round': args.calls_per_round,
        'physical_gpu': args.physical_gpu, 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'device_name': torch.cuda.get_device_name(device), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'image_dtype': str(backbone.visual.conv1.weight.dtype),
        'selector_dtype': 'torch.float32', 'results': summaries,
    })
    print(json.dumps({'output': str(args.output), 'conditions': len(entries)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--physical-gpu', type=int, required=True)
    parser.add_argument('--device')
    parser.add_argument('--patch-models', type=Path)
    parser.add_argument('--conditioned-models', type=Path)
    parser.add_argument('--shared-models', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 32])
    parser.add_argument('--panel-size', type=int, default=128)
    parser.add_argument('--panel-seed', type=int, default=2081)
    parser.add_argument('--order-seed', type=int, default=2082)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--calls-per-round', type=int, default=20)
    run(parser.parse_args())
