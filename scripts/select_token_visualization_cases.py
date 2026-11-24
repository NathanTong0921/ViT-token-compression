"""Select illustrative token maps from existing final-test predictions.

Scan all same-image task pairs, summarize replacement distributions, and return
separate typical, helpful, and large-change/no-benefit candidate lists. No model
is loaded. With --remote-target, only the small candidate JSON and optional
224-pixel CLIP crops are transferred to the local output directory.
"""

import argparse
import base64
from collections import defaultdict
import csv
from itertools import combinations
import json
from pathlib import Path
import shlex
import subprocess
import sys


def quantiles(values):
    import numpy as np
    return {key: float(value) for key, value in zip(
        ['min', 'q25', 'median', 'q75', 'max'],
        np.quantile(values, [0, .25, .5, .75, 1]))}


def scan_dataset(repository, dataset):
    import numpy as np
    folder = repository / f'runs/final-test-v1/{dataset}_b16_98'
    rows = list(csv.DictReader((folder / 'examples.csv').open()))
    classes = {r['class_id']: r['class_name'] for r in json.loads(
        (repository / f'data/splits/{dataset}-v1/classes.json').read_text())}
    by_image = defaultdict(list)
    for i, row in enumerate(rows):
        by_image[row['image_id']].append(i)
    outcomes = {}
    for seed in [2041, 2042, 2043]:
        for method in ['conditioned', 'image_only_wide']:
            with np.load(folder / method / f'seed_{seed}/predictions.npz') as a:
                values = {'margin': a['margin'].copy(), 'correct': a['correct'].copy()}
                if seed == 2041 and method == 'conditioned':
                    positions = a['patch_indices'][:, :98].copy()
                    assert np.all(a['patch_indices'][:, 98:] == -1)
                    assert np.all((positions >= 0) & (positions < 196))
                    cosine = a['cosine_scores'].copy()
                    text = a['text_features'].copy()
            outcomes[(seed, method)] = values
    records = []
    per_image_counts = {}
    for image_id, indices in by_image.items():
        indices.sort(key=lambda i: rows[i]['task_id'])
        image_records = []
        for ia, ib in combinations(indices, 2):
            count = 98 - len(np.intersect1d(positions[ia], positions[ib]))
            entry = {'dataset': dataset, 'image_id': image_id,
                     'rows': [ia, ib], 'replaced_tokens': int(count)}
            changes, seed_values = [], []
            for seed in [2041, 2042, 2043]:
                c = outcomes[(seed, 'conditioned')]
                u = outcomes[(seed, 'image_only_wide')]
                delta = c['margin'][[ia, ib]] - u['margin'][[ia, ib]]
                accuracy = c['correct'][[ia, ib]].astype(int) - u['correct'][[ia, ib]].astype(int)
                seed_values.append({'seed': seed, 'C_correct': c['correct'][[ia, ib]].astype(int).tolist(),
                                    'U_correct': u['correct'][[ia, ib]].astype(int).tolist(),
                                    'C_margin': c['margin'][[ia, ib]].tolist(),
                                    'U_margin': u['margin'][[ia, ib]].tolist(),
                                    'margin_gain': delta.tolist(), 'accuracy_gain': accuracy.tolist()})
                changes.append(delta)
            entry['seeds'] = seed_values
            entry['mean_margin_gain'] = float(np.mean(changes))
            entry['repair_seed_counts'] = [sum(v['accuracy_gain'][j] == 1 for v in seed_values) for j in [0, 1]]
            entry['positive_margin_seed_counts'] = [sum(v['margin_gain'][j] > 0 for v in seed_values) for j in [0, 1]]
            entry['primary_accuracy_gain'] = sum(seed_values[0]['accuracy_gain'])
            image_records.append(entry)
            records.append(entry)
        if image_records:
            per_image_counts[image_id] = float(np.median([r['replaced_tokens'] for r in image_records]))
    all_counts = np.asarray([r['replaced_tokens'] for r in records])
    distribution = quantiles(all_counts)
    image_distribution = quantiles(list(per_image_counts.values()))
    stats = {'dataset': dataset, 'evaluated_images': len(by_image),
             'images_with_two_or_more_tasks': len(per_image_counts),
             'image_task_rows': len(rows), 'unordered_image_task_pairs': len(records),
             'pair_weighted_replacement_count': distribution,
             'pair_replacement_histogram': {str(k): int(v) for k, v in zip(*np.unique(all_counts, return_counts=True))},
             'image_median_replacement_count': image_distribution,
             'primary_repair_task_rows': int(np.sum(
                 (outcomes[(2041, 'conditioned')]['correct'] == 1)
                 & (outcomes[(2041, 'image_only_wide')]['correct'] == 0))),
             'primary_harm_task_rows': int(np.sum(
                 (outcomes[(2041, 'conditioned')]['correct'] == 0)
                 & (outcomes[(2041, 'image_only_wide')]['correct'] == 1)))}

    def expand(entry, category):
        value = dict(entry)
        true = int(rows[entry['rows'][0]]['class_id'])
        value['true_class_id'] = true
        value['true_class_name'] = classes[true]
        value['category'] = category
        value['image_median_replacements'] = per_image_counts[value['image_id']]
        value['replacement_pair_percentile_interval'] = [
            float(100 * np.mean(all_counts < value['replaced_tokens'])),
            float(100 * np.mean(all_counts <= value['replaced_tokens']))]
        value['tasks'] = []
        for label, i in zip(['A', 'B'], value['rows']):
            row = rows[i]
            pair = [int(row['class_a']), int(row['class_b'])]
            value['tasks'].append({'name': label, 'row_id': i, 'task_id': row['task_id'],
                                   'candidate_ids': pair, 'candidate_names': [classes[c] for c in pair]})
        # Explicit crossed margins can be reconstructed from existing all-class scores.
        crossed = []
        for j, i in enumerate(value['rows']):
            row = rows[i]
            competitor = int(row['class_b']) if int(row['class_a']) == true else int(row['class_a'])
            other = value['rows'][1-j]
            crossed.append(float((cosine[other, true] - cosine[other, competitor]) /
                                 np.linalg.norm(text[true] - text[competitor])))
        value['C_exchanged_margin_seed2041'] = crossed
        value['source_directory'] = str(folder)
        return value

    def unique_first(candidates, count=6):
        result, seen = [], set()
        for entry in candidates:
            if entry['image_id'] in seen:
                continue
            result.append(entry)
            seen.add(entry['image_id'])
            if len(result) == count:
                break
        return result

    # Typical means close to the dataset's median across image-level medians,
    # then close to that image's own median. Outcome values do not affect rank.
    typical = sorted(records, key=lambda r: (
        abs(per_image_counts[r['image_id']] - image_distribution['median']),
        abs(r['replaced_tokens'] - per_image_counts[r['image_id']]),
        r['image_id'], r['rows']))
    helpful = [r for r in records if any(
        r['seeds'][0]['accuracy_gain'][j] == 1 and r['seeds'][0]['margin_gain'][j] >= .005
        for j in [0, 1])]

    def helpful_rank(r):
        repairs = [j for j in [0, 1] if r['seeds'][0]['accuracy_gain'][j] == 1
                   and r['seeds'][0]['margin_gain'][j] >= .005]
        j = max(repairs, key=lambda j: (r['repair_seed_counts'][j],
                                       r['positive_margin_seed_counts'][j],
                                       np.mean([v['margin_gain'][j] for v in r['seeds']])))
        return (-r['repair_seed_counts'][j], -r['positive_margin_seed_counts'][j],
                -float(np.mean([v['margin_gain'][j] for v in r['seeds']])),
                abs(r['replaced_tokens'] - distribution['median']), r['image_id'])
    helpful.sort(key=helpful_rank)
    boundary = [r for r in records if r['replaced_tokens'] >= distribution['q75']
                and all(g <= 0 for g in r['seeds'][0]['accuracy_gain'])
                and r['mean_margin_gain'] <= 0]
    # Prefer repeatable harmful effects, then no-accuracy-gain cases. Within
    # that ordering, favor a change count near Q75 rather than an extreme mask.
    boundary.sort(key=lambda r: (
        -sum(any(g < 0 for g in v['accuracy_gain']) for v in r['seeds']),
        abs(r['replaced_tokens'] - distribution['q75']),
        abs(r['mean_margin_gain']), r['image_id']))
    stats['helpful_pair_candidates_margin_gain_at_least_0_005'] = len(helpful)
    stats['helpful_distinct_images'] = len({r['image_id'] for r in helpful})
    stats['upper_quartile_no_accuracy_gain_nonpositive_mean_margin_pairs'] = len(boundary)
    choices = {name: [expand(r, name) for r in unique_first(values)]
               for name, values in [('typical', typical), ('helpful', helpful), ('boundary', boundary)]}
    return stats, choices


def analyze(repository):
    summaries, candidates = [], {}
    for dataset in ['pets', 'flowers']:
        stats, choices = scan_dataset(repository, dataset)
        summaries.append(stats)
        candidates[dataset] = choices
    # Four candidates per category: two from each dataset, unique images.
    gallery, used = [], set()
    for category in ['typical', 'helpful', 'boundary']:
        for dataset in ['pets', 'flowers']:
            chosen = 0
            for row in candidates[dataset][category]:
                key = (dataset, row['image_id'])
                if key in used:
                    continue
                value = dict(row)
                value['gallery_id'] = f'{category}_{dataset}_{chosen+1}'
                value['crop_filename'] = value['gallery_id'] + '_clip224.png'
                gallery.append(value)
                used.add(key)
                chosen += 1
                if chosen == 2:
                    break
    return {'setting': {'backbone': 'ViT-B/16', 'pruning_block': 6,
                        'keep_patches': 98, 'total_patches': 196, 'map_seed': 2041,
                        'outcome_check_seeds': [2041, 2042, 2043]},
            'selection': {'typical': 'Near median of image-level median replacement counts; no outcome-based ranking.',
                          'helpful': 'C correct, U wrong on the same task at seed 2041; geometric-margin improvement >=0.005. Rank by repair and positive-margin consistency across three existing seeds.',
                          'boundary': 'Replacement count at least pair-level Q75; no primary accuracy improvement on either task and nonpositive C-U margin gain averaged across tasks/seeds. Prioritize repeatable harm, then change count near Q75.',
                          'role': 'Purposefully selected qualitative examples, not aggregate performance estimates.'},
            'pool_statistics': summaries, 'candidate_lists': candidates, 'gallery': gallery}


def remote_call(target, root, code, arguments):
    command = f'cd {shlex.quote(root)} && .venv/bin/python - {shlex.join(arguments)}'
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                           target, command], input=code, text=True, capture_output=True, check=True).stdout


def fetch_crops(target, root, payload, output):
    request = json.dumps(payload['gallery'])
    code = """import base64,io,json,sys
from pathlib import Path
from PIL import Image
from torchvision.transforms import Compose
from clip.clip import _transform
root=Path(sys.argv[1]); rows=json.loads(sys.argv[2]); transform=Compose(_transform(224).transforms[:3])
ret=[]
for row in rows:
 ds='oxford-iiit-pet' if row['dataset']=='pets' else 'flowers102'
 path=root/'data/raw'/ds/'images'/(row['image_id']+'.jpg')
 with Image.open(path) as image:
  original_size=list(image.size); crop=transform(image.convert('RGB'))
  data=io.BytesIO();crop.save(data,format='PNG')
 ret.append({'filename':row['crop_filename'],'original_size':original_size,'source':str(path),'png':base64.b64encode(data.getvalue()).decode()})
print(json.dumps(ret))
"""
    records = json.loads(remote_call(target, root, code, [root, request]))
    for record, row in zip(records, payload['gallery']):
        (output / record['filename']).write_bytes(base64.b64decode(record['png']))
        row['original_size'] = record['original_size']
        row['remote_image'] = record['source']
    payload['crop_preprocessing'] = 'Server existing CLIP _transform(224) first three operations: bicubic shorter-side Resize224, CenterCrop224, RGB. Applied after original.convert(RGB), before normalization.'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=Path.cwd())
    parser.add_argument('--output', default='-')
    parser.add_argument('--remote-target')
    parser.add_argument('--remote-root', default='/data2/zhao.tong/experiments')
    parser.add_argument('--fetch-crops', action='store_true')
    args = parser.parse_args()
    if args.remote_target:
        source = Path(__file__).read_text()
        value = json.loads(remote_call(args.remote_target, args.remote_root, source,
                                      ['--repository', args.remote_root]))
    else:
        value = analyze(args.repository)
    if args.output == '-':
        print(json.dumps(value))
    else:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if args.fetch_crops:
            fetch_crops(args.remote_target, args.remote_root, value, output.parent)
        output.write_text(json.dumps(value, indent=2) + '\n')
        print(json.dumps({'output': str(output), 'pool_statistics': value['pool_statistics'],
                          'gallery_count': len(value['gallery'])}, indent=2))


if __name__ == '__main__':
    main()
