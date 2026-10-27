"""Join fixed-protocol inference timings to complete unseen-task validation quality.

Aggregation requires no plotting packages. Use --plot-only after copying the
small latency_summary.json to render the figure on another machine.
"""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import mean


CONDITIONS = (
    ('pets-b16-49', 'Pets', 'ViT-B/16', 49, 196, 'pets-selector-budgets-v1'),
    ('pets-b16-98', 'Pets', 'ViT-B/16', 98, 196, 'pets-selector-budgets-v1'),
    ('pets-b16-147', 'Pets', 'ViT-B/16', 147, 196, 'pets-selector-budgets-v1'),
    ('flowers-b16-49', 'Flowers102', 'ViT-B/16', 49, 196, 'flowers-selector-budgets-v1'),
    ('flowers-b16-98', 'Flowers102', 'ViT-B/16', 98, 196, 'flowers-selector-budgets-v1'),
    ('pets-l14-128', 'Pets', 'ViT-L/14', 128, 256, 'pets-vitl128-v1'),
    ('flowers-l14-128', 'Flowers102', 'ViT-L/14', 128, 256, 'flowers-vitl128-v1'),
)
METHODS = ('full_model', 'attention', 'image_only', 'conditioned')
SEEDS = (2041, 2042, 2043)


def read_quality(evaluation_dir):
    """Equal tasks, equal class sides, equal images, then equal fixed seeds."""
    groups = defaultdict(list)
    for group in ('references', 'full'):
        with (evaluation_dir / group / 'predictions.csv').open() as handle:
            for row in csv.DictReader(handle):
                if row['task_split'] == 'unseen' and row['method'] in METHODS:
                    groups[row['method'], int(row['seed'])].append(row)
    reference_keys = {(r['image_id'], r['task_id']) for r in groups['full_model', -1]}
    results = {}
    for method in METHODS:
        expected_seeds = SEEDS if method in ('image_only', 'conditioned') else (-1,)
        assert sorted(seed for name, seed in groups if name == method) == list(expected_seeds)
        per_seed = []
        for seed in expected_seeds:
            rows = groups[method, seed]
            keys = {(r['image_id'], r['task_id']) for r in rows}
            assert keys == reference_keys and len(rows) == len(keys)
            sides = defaultdict(list)
            for row in rows:
                sides[row['task_id'], int(row['class_id'])].append(int(row['correct']))
            task_means = defaultdict(list)
            for (task_id, _), values in sides.items():
                task_means[task_id].append(mean(values))
            assert all(len(values) == 2 for values in task_means.values())
            accuracy = mean(mean(values) for values in task_means.values())
            recorded_weights = [float(r['task_class_weight']) for r in rows]
            assert math.isclose(sum(recorded_weights), 1.0, abs_tol=1e-8)
            recorded_accuracy = sum(float(r['task_class_weight']) * int(r['correct']) for r in rows)
            assert math.isclose(accuracy, recorded_accuracy, abs_tol=1e-8)
            per_seed.append({'seed': seed, 'accuracy': accuracy})
        results[method] = {
            'mean': mean(r['accuracy'] for r in per_seed), 'per_seed': per_seed,
            'image_count': len({r['image_id'] for r in rows}),
            'image_task_examples_per_seed': len(rows), 'task_count': len(task_means),
        }
    return results


def crosscheck_curated(quality, analysis_path, config):
    """The archived analysis also supplies its image-cluster uncertainty."""
    if not analysis_path.exists():
        return None
    analysis = json.loads(analysis_path.read_text())
    key = str(config['keep_patch_tokens'])
    source_config = analysis['sources'][key]['config']
    for field in ('model', 'prune_after', 'keep_patch_tokens', 'manifest_dir', 'training_seeds'):
        assert source_config[field] == config[field]
    table = analysis['task_splits']['unseen']['budgets'][key]['table']
    for row in table:
        if row['method'] in quality:
            target = quality[row['method']]
            assert math.isclose(target['mean'], row['accuracy']['mean'], abs_tol=1e-8)
            assert [r['seed'] for r in target['per_seed']] == [r['seed'] for r in row['per_seed']]
            for actual, stored in zip(target['per_seed'], row['per_seed']):
                assert math.isclose(actual['accuracy'], stored['accuracy'], abs_tol=1e-8)
            target['ci95'] = row['accuracy']['ci95']
    return str(analysis_path)


def write_rows(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def aggregate(runs_root, results_root, output):
    rows, seed_rows, runs = [], [], []
    for name, dataset, backbone, budget, patches, curated in CONDITIONS:
        folder = runs_root / name
        latency = json.loads((folder / 'summary.json').read_text())
        config = latency['config']
        assert config['model'] == backbone and config['keep_patch_tokens'] == budget
        assert latency['training_seeds'] == list(SEEDS)
        assert latency['batch_sizes'] == [1, 32]
        evaluation_dir = Path(latency['evaluation_source'])
        evaluation = json.loads((evaluation_dir / 'eval_summary.json').read_text())
        for field in ('model', 'prune_after', 'keep_patch_tokens', 'manifest_dir', 'training_seeds'):
            assert evaluation['config'][field] == config[field]
        quality = read_quality(evaluation_dir)
        curated_source = crosscheck_curated(quality, results_root / curated / 'analysis.json', config)
        windows = json.loads((folder / 'hardware_windows.json').read_text())
        gpu_uuids = sorted({window[when]['uuid'] for window in windows for when in ('before', 'after')})
        with (folder / 'panel.csv').open() as handle:
            panel = list(csv.DictReader(handle))
        runs.append({
            'condition': name, 'dataset': dataset, 'backbone': backbone, 'condition_budget': budget,
            'source': str(folder), 'evaluation_source': str(evaluation_dir), 'curated_crosscheck': curated_source,
            'quality': quality, 'latency': latency, 'hardware_windows': windows, 'panel': panel,
        })
        for batch in (1, 32):
            result = latency['results'][str(batch)]
            for method in METHODS:
                timing, accuracy = result['methods'][method], quality[method]
                wall, speedup = timing['wall_ms'], timing['paired_round_speedup']
                valid = result['timing_window_valid']
                assert valid == all(w['uncontended_at_both_checks'] for w in windows if w['batch_size'] == batch)
                assert valid or speedup is None
                expected_seeds = SEEDS if method in ('conditioned', 'image_only') else (-1,)
                assert [x['seed'] for x in timing['per_seed']] == list(expected_seeds)
                ci = accuracy.get('ci95', (None, None))
                row = {
                    'condition': name, 'dataset': dataset, 'backbone': backbone,
                    'condition_budget': budget, 'retained_patches': patches if method == 'full_model' else budget,
                    'original_patches': patches, 'prune_after_block': config['prune_after'],
                    'batch_size': batch, 'method': method,
                    'unseen_accuracy_percent': 100 * accuracy['mean'],
                    'accuracy_ci95_low_percent': None if ci[0] is None else 100 * ci[0],
                    'accuracy_ci95_high_percent': None if ci[1] is None else 100 * ci[1],
                    'training_seed_count': len(SEEDS) if method in ('conditioned', 'image_only') else 0,
                    'wall_median_ms': wall['median'], 'wall_q25_ms': wall['q25'], 'wall_q75_ms': wall['q75'],
                    'wall_iqr_ms': wall['q75'] - wall['q25'], 'wall_p95_ms': wall['p95'],
                    'cuda_event_median_ms': timing['cuda_event_ms']['median'],
                    'images_per_second_at_median': timing['images_per_second_at_median'],
                    'timed_calls': wall['count'], 'timing_window_valid': valid,
                    'paired_round_speedup_median': speedup['median'] if valid else None,
                    'paired_round_speedup_q25': speedup['q25'] if valid else None,
                    'paired_round_speedup_q75': speedup['q75'] if valid else None,
                    'paired_round_speedup_count': len(timing['paired_round_speedup_samples']),
                    'gpu_uuid': ';'.join(gpu_uuids), 'gpu_name': latency['hardware']['device_name'],
                    'quality_image_count': accuracy['image_count'], 'quality_task_count': accuracy['task_count'],
                    'quality_image_task_examples_per_seed': accuracy['image_task_examples_per_seed'],
                }
                rows.append(row)
                for per_seed in timing['per_seed']:
                    matched_quality = next(r for r in accuracy['per_seed'] if r['seed'] == per_seed['seed'])
                    seed_rows.append({
                        'condition': name, 'batch_size': batch, 'method': method, 'seed': per_seed['seed'],
                        'unseen_accuracy_percent': 100 * matched_quality['accuracy'],
                        'wall_median_ms': per_seed['wall_ms']['median'],
                        'wall_q25_ms': per_seed['wall_ms']['q25'], 'wall_q75_ms': per_seed['wall_ms']['q75'],
                        'wall_p95_ms': per_seed['wall_ms']['p95'],
                        'timed_calls': per_seed['wall_ms']['count'], 'timing_window_valid': valid,
                    })
    for dataset in ('Pets', 'Flowers102'):
        panels = [r['panel'] for r in runs if r['dataset'] == dataset]
        assert all(panel == panels[0] for panel in panels[1:])
    summary = {
        'quality_scope': 'All validation images and unseen class combinations from the matched experiment; equal tasks, equal class sides, equal images within each side, then mean of all three fixed training seeds. Timing-pool accuracy is not used.',
        'quality_uncertainty': 'Where present, archived 95% image-cluster intervals condition on the fixed tasks and seeds. They are not across-seed intervals.',
        'latency_scope': 'GPU-resident preprocessed image to candidate-class cosine scores; includes full prefix, online selection and suffix. Excludes CPU preprocessing, H2D input transfer and one-time text encoding.',
        'latency_aggregation': 'Wall-clock median, IQR and p95 pool equal numbers of calls across all fixed seeds. Speedup uses native-full versus method medians paired by round; all seeds and rounds retained. Quantiles describe timing variability and are not confidence intervals.',
        'contended_window_policy': 'All measurements remain in tables/figures. A foreign compute process observed before or after a round invalidates the batch-size timing window; speedup fields remain empty. No fastest-window selection.',
        'rows': rows, 'per_seed_rows': seed_rows, 'runs': runs,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'latency_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    write_rows(output / 'latency_table.csv', rows)
    write_rows(output / 'latency_seed_table.csv', seed_rows)
    return summary


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    styles = {
        'full_model': ('#6a6a6a', 'D', 'Full CLIP'),
        'attention': ('#bc8c32', '^', 'Attention'),
        'image_only': ('#3476ae', 's', 'Image-only'),
        'conditioned': ('#cf564b', 'o', 'Conditioned'),
    }
    panels = [('Pets', 'ViT-B/16'), ('Flowers102', 'ViT-B/16'),
              ('Pets', 'ViT-L/14'), ('Flowers102', 'ViT-L/14')]
    fig, axes = plt.subplots(2, 4, figsize=(13.4, 6.8))
    rows = summary['rows']
    for column, (dataset, backbone) in enumerate(panels):
        for row_index, batch in enumerate((1, 32)):
            ax = axes[row_index, column]
            selected = [r for r in rows if (r['dataset'], r['backbone'], r['batch_size']) == (dataset, backbone, batch)]
            for method in METHODS:
                points = sorted((r for r in selected if r['method'] == method), key=lambda r: r['condition_budget'])
                color, marker, _ = styles[method]
                if method != 'full_model' and len(points) > 1:
                    ax.plot([p['wall_median_ms'] for p in points],
                            [p['unseen_accuracy_percent'] for p in points],
                            color=color, linewidth=.8, alpha=.6, zorder=1)
                for point in points:
                    x, y = point['wall_median_ms'], point['unseen_accuracy_percent']
                    ax.errorbar(x, y, xerr=[[x-point['wall_q25_ms']], [point['wall_q75_ms']-x]],
                                fmt=marker, markersize=5, color=color, elinewidth=.8, capsize=2,
                                markerfacecolor=color if point['timing_window_valid'] else 'white', zorder=3)
                    if method == 'conditioned':
                        crowded = dataset == 'Pets' and backbone == 'ViT-B/16' and batch == 1
                        offset = ({49: (9, -18), 98: (8, 17), 147: (36, -24)}[point['condition_budget']]
                                  if crowded else (4, 4))
                        ax.annotate(str(point['condition_budget']), (x, y), xytext=offset,
                                    textcoords='offset points', fontsize=8, color=color,
                                    arrowprops={'arrowstyle': '-', 'color': color, 'lw': .5, 'alpha': .6}
                                    if crowded else None)
            ax.set_title(f'{dataset} · {backbone}\nBatch {batch}', fontsize=10)
            ax.grid(alpha=.18)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=8)
            ax.margins(x=.15, y=.18)
            if row_index == 1:
                ax.set_xlabel('Measured request latency (ms)', fontsize=9)
            if column == 0:
                ax.set_ylabel('Unseen-task accuracy (%)', fontsize=9)
    handles = [Line2D([], [], marker=marker, color=color, linestyle='none', markersize=6, label=label)
               for color, marker, label in styles.values()]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .96), ncol=4, frameon=False)
    fig.suptitle('Validation accuracy and measured model inference latency', fontsize=13, y=.995)
    fig.text(.5, .055, 'Horizontal bars: timing IQR. Labels: retained patch count. Three fixed selector seeds; all budgets and timings retained.',
             ha='center', fontsize=8.5)
    invalid = any(not r['timing_window_valid'] for r in rows)
    note = ('Hollow markers: contended GPU windows; descriptive measurements only, with no speedup claim.' if invalid
            else 'GPU-resident input and cached text; includes full image encoding and online selection. Preprocessing and H2D excluded.')
    fig.text(.5, .025, note, ha='center', fontsize=8.5)
    fig.tight_layout(rect=[0, .09, 1, .90])
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'latency_accuracy.{extension}', dpi=220)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, default=Path('runs/selector-latency-v1'))
    parser.add_argument('--results-root', type=Path, default=Path('results'))
    parser.add_argument('--output', type=Path, default=Path('results/vitl-and-latency-v1'))
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    result = (json.loads((args.output / 'latency_summary.json').read_text()) if args.plot_only
              else aggregate(args.runs_root, args.results_root, args.output))
    if args.plot or args.plot_only:
        plot(result, args.output)
    print(json.dumps({'rows': len(result['rows']), 'per_seed_rows': len(result['per_seed_rows']),
                      'all_timing_windows_valid': all(r['timing_window_valid'] for r in result['rows']),
                      'output': str(args.output)}))
