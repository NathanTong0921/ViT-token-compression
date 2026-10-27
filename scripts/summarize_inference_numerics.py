"""Summarize all fixed-budget numerical audits; plotting is optional and local."""

import argparse
import csv
import json
import math
from pathlib import Path

from summarize_selector_latency import CONDITIONS


SETTINGS = ('original_cached', 'batch_1', 'batch_32')
SEEDS = [2041, 2042, 2043]
METRICS = {'accuracy': (100., 'percentage points'), 'margin': (1000., '1e-3 margin units')}


def write_rows(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def preflight_context(folder):
    if folder is None:
        return {'source_supplied': False,
                'interpretation': 'The same-batch implementation parity check and cross-batch numerical sensitivity are distinct checks. Supply --preflight to attach the recorded preflight evidence.'}
    correctness = json.loads((folder / 'correctness.json').read_text())
    cached = json.loads((folder / 'cached_reference_audit.json').read_text())
    comparisons = []
    for batch in sorted({r['batch_size'] for r in cached['cases']}):
        rows = [r for r in cached['cases'] if r['batch_size'] == batch]
        compressed = [r for r in rows if r['method'] != 'full_model']
        comparisons.append({
            'batch_size': batch, 'images_per_method_seed': sorted({r['images_checked'] for r in rows}),
            'prediction_disagreements': sum(r['prediction_disagreements'] for r in rows),
            'compressed_method_seed_rows': sum(r['images_checked'] for r in compressed),
            'rows_with_changed_selected_sets': sum(r['rows_with_changed_selected_set'] for r in compressed),
            'max_feature_coordinate_error': max(r['max_feature_error'] for r in rows),
            'max_pair_score_error': max(r['max_pair_score_error'] for r in rows),
        })
    cases = correctness['cases']
    exact = (correctness['status'] == 'passed'
             and all(r['selected_positions_match_numpy_stable'] for r in cases)
             and all(r['max_feature_error'] == 0 and r['max_score_error'] == 0 for r in cases))
    return {
        'source_supplied': True, 'source': str(folder), 'status': correctness['status'],
        'same_batch_cases': len(cases), 'exact_same_batch_selection_features_scores': exact,
        'max_same_batch_feature_error': max(r['max_feature_error'] for r in cases),
        'max_same_batch_score_error': max(r['max_score_error'] for r in cases),
        'cross_batch_comparisons': comparisons,
        'interpretation': 'Exact parity on the same input batch checks the GPU implementation against the original stable-ranking algorithm. Changed selected sets between cached and live batch shapes motivated the full-validation numerical audit; this is consistent with ranking-boundary sensitivity, rather than evidence of a different selection algorithm. Preflight sorted-index mismatch counts are not numbers of replaced tokens and are not used here.',
    }


def aggregate(runs_root, output, preflight=None):
    rows, context, conditions = [], [], []
    expected_jobs = {('full_model', -1), ('attention', -1)} | {
        (method, seed) for method in ('image_only', 'conditioned') for seed in SEEDS}
    for name, dataset, backbone, budget, patches, _ in CONDITIONS:
        source = runs_root / name / 'summary.json'
        summary = json.loads(source.read_text())
        config = summary['config']
        assert config['model'] == backbone and config['keep_patch_tokens'] == budget
        assert summary['training_seeds'] == SEEDS and summary['batch_sizes'] == [1, 32]
        effects = {batch: summary['results'][str(batch)]['conditioned_minus_image_only'] for batch in (1, 32)}
        for metric in METRICS:
            first, second = effects[1]['original'][metric], effects[32]['original'][metric]
            assert math.isclose(first['mean'], second['mean'], abs_tol=1e-12)
            assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(first['ci95'], second['ci95']))
        conditions.append({
            'condition': name, 'dataset': dataset, 'backbone': backbone, 'retained_patches': budget,
            'original_patches': patches, 'source': str(source), 'code_commit': summary['code_commit'],
            'evaluation_source': summary['evaluation_source'], 'evaluation_code_commit': summary['evaluation_code_commit'],
            'image_count': summary['image_count'], 'task_count': summary['task_count'],
            'image_task_examples_per_method': summary['image_task_examples_per_method'],
            'wall_seconds': summary['wall_seconds'], 'environment': summary['environment'],
            'per_seed_effects': {str(batch): effects[batch]['per_seed'] for batch in (1, 32)},
        })
        for setting in SETTINGS:
            batch = None if setting == 'original_cached' else int(setting.split('_')[1])
            selected = effects[1]['original'] if batch is None else effects[batch]['online']
            change = None if batch is None else effects[batch]['online_minus_original']
            for metric, (scale, units) in METRICS.items():
                value = selected[metric]
                delta = None if change is None else change[metric]
                rows.append({
                    'condition': name, 'dataset': dataset, 'backbone': backbone,
                    'retained_patches': budget, 'original_patches': patches,
                    'setting': setting, 'metric': metric,
                    'estimate': value['mean'], 'ci95_low': value['ci95'][0], 'ci95_high': value['ci95'][1],
                    'plot_scale': scale, 'plot_units': units,
                    'plot_estimate': scale * value['mean'],
                    'plot_ci95_low': scale * value['ci95'][0], 'plot_ci95_high': scale * value['ci95'][1],
                    'change_from_original': None if delta is None else delta['mean'],
                    'change_ci95_low': None if delta is None else delta['ci95'][0],
                    'change_ci95_high': None if delta is None else delta['ci95'][1],
                    'plot_change_from_original': None if delta is None else scale * delta['mean'],
                    'plot_change_ci95_low': None if delta is None else scale * delta['ci95'][0],
                    'plot_change_ci95_high': None if delta is None else scale * delta['ci95'][1],
                })
        for batch in (1, 32):
            method_rows = summary['results'][str(batch)]['methods_per_seed']
            assert {(r['method'], r['seed']) for r in method_rows} == expected_jobs
            assert len(method_rows) == len(expected_jobs)
            for row in method_rows:
                execution = row['execution']
                context.append({
                    'condition': name, 'dataset': dataset, 'backbone': backbone, 'retained_patches': budget,
                    'batch_size': batch, 'method': row['method'], 'seed': row['seed'],
                    'image_task_examples': summary['image_task_examples_per_method'],
                    'online_accuracy_percent': 100 * row['online']['accuracy'],
                    'original_accuracy_percent': 100 * row['original']['accuracy'],
                    'accuracy_change_pp': 100 * row['online_minus_original']['accuracy'],
                    'online_margin': row['online']['margin'], 'original_margin': row['original']['margin'],
                    'margin_change': row['online_minus_original']['margin'],
                    'prediction_disagreements': row['prediction_disagreements'],
                    'weighted_disagreement_percent': 100 * row['weighted_disagreement_rate'],
                    'repairs': row['repairs'], 'new_errors': row['new_errors'],
                    'changed_selected_sets': row['changed_selected_sets'],
                    'changed_selected_sets_percent': 100 * row['changed_selected_sets'] / summary['image_task_examples_per_method'],
                    'max_replaced_tokens': row['max_replaced_tokens'], 'mean_replaced_tokens': row['mean_replaced_tokens'],
                    'mean_jaccard': row['mean_jaccard'], 'max_pair_score_error': row['max_pair_score_error'],
                    'p95_pair_score_error': row['p95_pair_score_error'], 'mean_pair_score_error': row['mean_pair_score_error'],
                    'image_forwards': execution['image_forwards'], 'forward_calls': execution['forward_calls'],
                    'forward_tail_size': execution['forward_tail_size'], 'score_tail_size': execution['score_tail_size'],
                    'task_independent_image_reuse': execution['task_independent_image_reuse'],
                })
    result = {
        'scope': 'All seven fixed conditions; original cached evaluation and live batch sizes 1 and 32. Same frozen backbone weights, selector checkpoints, class text, unseen validation queries and three training seeds. No retraining, new checkpoint choice, hyperparameter search or final-test evaluation.',
        'comparison': 'Conditioned minus matched image-only. The original cached result remains alongside both live batch settings; no numerical setting is selected for having the strongest effect.',
        'uncertainty': 'Ordinary unadjusted 95% paired image-cluster bootstrap intervals, 2000 replicates, seed 2052; tasks and training seeds held fixed. These are diagnostic numerical-sensitivity intervals, not replacements for the original budget-adjusted inferential results.',
        'aggregation': 'Equal tasks, equal class sides within task, equal images within side, then average all three fixed seeds before paired image resampling.',
        'context_count_units': 'Disagreement and selection counts are image/task cases within each method and seed. Repeated queries and seeds are not independent images. Token replacement is retained-set size minus actual set intersection, not the number of shifted sorted indices.',
        'batching': 'Final short forward and scoring batches are unpadded and recorded. Unique-image reuse is limited to task-independent arms in this accuracy audit; the latency benchmark reruns the complete image path per request.',
        'cost_scope': 'These audit execution times are not latency or speedup measurements and may come from shared GPU use.',
        'preflight': preflight_context(preflight), 'conditions': conditions, 'rows': rows, 'context_rows': context,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'numerical_sensitivity.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    write_rows(output / 'numerical_sensitivity.csv', rows)
    write_rows(output / 'numerical_sensitivity_context.csv', context)
    return result


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    styles = {'original_cached': ('#777777', 'o', -.19, 'Original cached'),
              'batch_1': ('#3476ae', 'D', 0., 'Live batch 1'),
              'batch_32': ('#cc7a30', 's', .19, 'Live batch 32')}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for row_index, dataset in enumerate(('Pets', 'Flowers102')):
        conditions = [r for r in summary['conditions'] if r['dataset'] == dataset]
        names = [r['condition'] for r in conditions]
        labels = [('B/16' if r['backbone'] == 'ViT-B/16' else 'L/14') +
                  f"\n{r['retained_patches']}/{r['original_patches']}" for r in conditions]
        for column, metric in enumerate(METRICS):
            ax = axes[row_index, column]
            selected = [r for r in summary['rows'] if r['dataset'] == dataset and r['metric'] == metric]
            for row in selected:
                color, marker, offset, _ = styles[row['setting']]
                x = names.index(row['condition']) + offset
                low, high = row['plot_ci95_low'], row['plot_ci95_high']
                ax.vlines(x, low, high, color=color, linewidth=1.2)
                ax.hlines([low, high], x-.035, x+.035, color=color, linewidth=1.2)
                ax.plot(x, row['plot_estimate'], marker, color=color, markersize=5)
            ax.axhline(0, color='#777777', linewidth=.7)
            ax.set(xticks=range(len(names)), xticklabels=labels, xlim=(-.5, len(names)-.5))
            ax.set_title(f"{dataset} · {'accuracy' if metric == 'accuracy' else 'classification margin'}", fontsize=11)
            ax.set_ylabel('Conditioned − image-only (pp)' if metric == 'accuracy'
                          else 'Conditioned − image-only (×10⁻³)', fontsize=9)
            ax.tick_params(labelsize=9)
            ax.grid(axis='y', alpha=.16)
            ax.spines[['top', 'right']].set_visible(False)
    handles = [Line2D([], [], marker=marker, color=color, linestyle='none', label=label)
               for color, marker, _, label in styles.values()]
    fig.suptitle('Numerical sensitivity of the conditioning effect', fontsize=13, y=.995)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .955), ncol=3, frameon=False)
    fig.text(.5, .060, 'Same checkpoints and unseen validation queries; all three fixed seeds. Labels show retained/original patch counts.',
             ha='center', fontsize=8.5)
    fig.text(.5, .033, 'Bars: ordinary 95% paired image-bootstrap diagnostic intervals; these do not replace the original budget-adjusted analysis.',
             ha='center', fontsize=8.5)
    fig.tight_layout(rect=[0, .09, 1, .89])
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'numerical_sensitivity.{extension}', dpi=220)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, default=Path('runs/selector-numerics-v1'))
    parser.add_argument('--output', type=Path, default=Path('results/vitl-and-latency-v1'))
    parser.add_argument('--preflight', type=Path, help='Directory with the recorded correctness.json and cached_reference_audit.json.')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    result = (json.loads((args.output / 'numerical_sensitivity.json').read_text()) if args.plot_only
              else aggregate(args.runs_root, args.output, args.preflight))
    if args.plot or args.plot_only:
        plot(result, args.output)
    print(json.dumps({'effect_rows': len(result['rows']), 'context_rows': len(result['context_rows']),
                      'output': str(args.output)}))
