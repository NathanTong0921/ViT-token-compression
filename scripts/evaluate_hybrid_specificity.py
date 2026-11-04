"""Compare conditional/shared mixtures with mixtures of independent shared selectors."""

import argparse
from itertools import product
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from analyze_selector_scale import read_rows
from evaluate_core_protection import SETTINGS, padded_batches, unique_image_masks

METHODS = ('conditioned', 'shared_recipient', 'shared_donor', 'cu_half', 'uu_half',
           'cu_d2', 'uu_d2', 'cu_d4', 'uu_d4')
FIXED_D = (2, 4)


def seed_pairs(seeds, diagonal=False):
    return [(s, t) for s, t in product(seeds, repeat=2) if (s == t) == diagonal]


def hybrid_mask(recipient, donor, k, fixed_d=None):
    """Keep the intersection; replace lowest recipient-only with highest donor-only."""
    a = np.argsort(-recipient, kind='stable')[:k]
    b = np.argsort(-donor, kind='stable')[:k]
    drop_pool, add_pool = a[~np.isin(a, b)], b[~np.isin(b, a)]
    available = len(add_pool)
    assert len(drop_pool) == available
    eligible = fixed_d is None or available >= fixed_d
    d = available // 2 if fixed_d is None else fixed_d if eligible else 0
    drop = drop_pool[-d:] if d else drop_pool[:0]
    result = np.sort(np.concatenate((a[~np.isin(a, drop)], add_pool[:d]))).astype(np.int16)
    return result, available, d, eligible


def build_pair(conditioned, shared_recipient, shared_donor, image_indices, k):
    n = len(image_indices)
    masks = np.empty((len(METHODS), n, k), np.int16)
    disagreement = np.zeros((2, n), np.int16)
    changed = np.zeros((len(METHODS), n), np.int16)
    eligible = np.empty((len(FIXED_D), n), bool)
    for row, image in enumerate(image_indices):
        c, u, donor = conditioned[row], shared_recipient[image], shared_donor[image]
        for mi, scores in enumerate((c, u, donor)):
            masks[mi, row] = np.sort(np.argsort(-scores, kind='stable')[:k])
        for offset, scores in enumerate((c, u)):
            mask, available, d, _ = hybrid_mask(scores, donor, k)
            masks[3 + offset, row], disagreement[offset, row], changed[3 + offset, row] = mask, available, d
        for di, d in enumerate(FIXED_D):
            valid = []
            for offset, scores in enumerate((c, u)):
                mask, _, actual, ok = hybrid_mask(scores, donor, k, d)
                mi = 5 + 2 * di + offset
                masks[mi, row], changed[mi, row] = mask, actual
                valid.append(ok)
            eligible[di, row] = all(valid)
    return dict(methods=np.asarray(METHODS), patch_indices=masks, disagreement_sizes=disagreement,
                changed_tokens_vs_recipient=changed, fixed_d=np.asarray(FIXED_D), fixed_d_eligible=eligible)


def run(source, output, device=None, historical=None):
    import clip
    import torch
    from token_compression.pruning import SingleStagePruner

    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    prior = json.loads((source / 'summary.json').read_text())
    config = dict(prior['config'])
    if device is not None:
        config['device'] = device
    seeds, setting = prior['training_seeds'], prior['setting']
    k, batch_size = config['keep_patch_tokens'], config['image_batch_size']
    for name in ('examples.csv', 'evidence.npz'):
        shutil.copyfile(source / name, output / name)
    rows = read_rows(source / 'examples.csv')
    ix = np.asarray([int(r['image_index']) for r in rows])
    with np.load(source / 'evidence.npz') as evidence:
        text = evidence['text_features'].copy()
    prefixes = np.load(Path(config['cache_dir']) / 'val/prefix_tokens.npy', mmap_mode='r')
    saved = {}
    feature_cache = {}
    for seed in seeds:
        with np.load(source / f'seed_{seed}.npz') as archive:
            saved[seed] = {key: archive[key].copy() for key in
                ('conditioned_logits', 'wide_logits', 'variants', 'patch_indices', 'cosine_scores')}
            keys, features = archive['unique_keys'], archive['unique_features']
            for key, feature in zip(keys, features):
                feature_cache[key.astype(np.int64).tobytes()] = feature
    backbone, _ = clip.load(config['model'], device=config['device'], jit=False, download_root=config['model_cache'])
    backbone.eval().requires_grad_(False)
    pruner = SingleStagePruner(backbone.visual, config['prune_after'])
    cross = seed_pairs(seeds)
    diagonals = seed_pairs(seeds, diagonal=True)
    protocol = dict(setting=setting, config=config, source=str(source.resolve()),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_git_commit=prior['git_commit'], training_seeds=seeds, methods=list(METHODS),
        primary_seed_pairs=cross, reference_seed_pairs=diagonals, fixed_d=list(FIXED_D),
        primary='C_s+U_t minus U_s+U_t, all ordered s != t; 50% of each pair\'s exclusive positions.',
        fixed_d_scope='Secondary: exactly d replacements on joint-feasible pair/image/task rows; invalid stored masks are recipient placeholders excluded from that analysis.',
        final_test_used=False, selector_parameter_counts={'cu': 348033+347399, 'uu': 2*347399},
        compute_scope='Two existing selector evaluations and one frozen suffix at equal k; no measured full inference latency.',
        sources_use='No new fit; all fixed seeds retained. Validation images and unseen validation combinations only.')
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    records = []
    def sync():
        if torch.device(config['device']).type == 'cuda':
            torch.cuda.synchronize(config['device'])
    with torch.inference_mode():
        for s, t in cross + diagonals:
            values = build_pair(saved[s]['conditioned_logits'], saved[s]['wide_logits'], saved[t]['wide_logits'], ix, k)
            masks = values['patch_indices']
            keys, inverse = unique_image_masks(ix, masks)
            keys = keys.astype(np.int64)
            features = np.empty((len(keys), text.shape[1]), np.float32)
            pending = []
            for i, key in enumerate(keys):
                cached = feature_cache.get(key.tobytes())
                if cached is None:
                    pending.append(i)
                else:
                    features[i] = cached
            pending = np.asarray(pending, dtype=int)
            sync()
            before = time.perf_counter()
            for start, end, positions in padded_batches(len(pending), batch_size):
                current = keys[pending[positions]]
                tokens = torch.as_tensor(np.asarray(prefixes[current[:, 0]]), device=config['device'])
                retained = torch.as_tensor(current[:, 1:], dtype=torch.long, device=config['device'])
                result = pruner.finish(tokens, retained).float()
                result = (result / result.norm(dim=-1, keepdim=True)).cpu().numpy()
                features[pending[start:end]] = result[:end-start]
            sync()
            seconds = time.perf_counter()-before
            for i in pending:
                feature_cache[keys[i].tobytes()] = features[i].copy()
            scores = (features @ text.T)[inverse]
            max_error = 0.
            for mi, (seed, method) in enumerate(((s, 'conditioned'), (s, 'wide'), (t, 'wide'))):
                old = saved[seed]
                oi = list(old['variants']).index(method)
                np.testing.assert_array_equal(masks[mi], old['patch_indices'][oi])
                error = float(np.abs(scores[mi]-old['cosine_scores'][oi]).max())
                max_error = max(max_error, error)
                np.testing.assert_allclose(scores[mi], old['cosine_scores'][oi], atol=1e-6, rtol=0)
            historical_error = None
            if s == t and historical is not None:
                with np.load(Path(historical) / f'seed_{s}.npz') as old:
                    oi = list(old['variants']).index('top_50')
                    np.testing.assert_array_equal(masks[3], old['patch_indices'][oi])
                    historical_error = float(abs(scores[3]-old['cosine_scores'][oi]).max())
                    np.testing.assert_allclose(scores[3], old['cosine_scores'][oi], atol=1e-6, rtol=0)
            np.savez_compressed(output / f'pair_{s}_{t}.npz', recipient_seed=s, donor_seed=t,
                **values, cosine_scores=scores)
            record = dict(recipient_seed=s, donor_seed=t, primary=s!=t, unique_masks=len(keys),
                reused_masks=len(keys)-len(pending), new_masks=len(pending), new_suffix_seconds=seconds,
                endpoint_masks_exact=True, endpoint_score_max_error=max_error,
                historical_half_score_max_error=historical_error)
            records.append(record)
            print(json.dumps({'setting': setting, **record}), flush=True)
    (output / 'execution.json').write_text(json.dumps(records, indent=2)+'\n')
    return protocol


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setting', choices=SETTINGS, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--historical', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device')
    args = parser.parse_args()
    run(args.source or Path('runs/core-protection-v1')/args.setting, args.output, args.device,
        args.historical or Path('runs/matched-replacement-v1')/args.setting)
