"""Collect recorded experiment costs; these are not deployment latency timings."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path


TRAIN_TIMERS = ('training_seconds', 'validation_seconds', 'initial_evaluation_seconds',
                'final_train_evaluation_seconds')


def read(root, path):
    source = root / path
    return json.loads(source.read_text()) if source.exists() else {}


def pipeline_stage(pipeline, name):
    return next((row for row in pipeline.get('stages', []) if row['name'] == name), {})


def add_stage(rows, details, root, group, dataset, model, budget, name, summary_path,
              pipeline, pipeline_path, uses_gpu=True, coverage='full', note=''):
    summary = read(root, summary_path) if summary_path else {}
    process = pipeline_stage(pipeline, name)
    cost = summary.get('cost', {})
    internal = summary.get('wall_seconds', cost.get('total_wall_seconds'))
    process_seconds = process.get('wall_seconds')
    accounted = process_seconds if process_seconds is not None else internal
    status = process.get('status', 'summary_complete' if summary else 'missing_record')
    if status not in ('complete', 'summary_complete'):
        accounted = None
    basis = ('pipeline_process_wall' if process_seconds is not None else
             'internal_stage_wall' if internal is not None else 'missing')
    row = {
        'record_type': 'stage', 'group': group, 'dataset': dataset, 'model': model,
        'budget': budget, 'stage': name, 'coverage': coverage, 'seed': None, 'arm': None,
        'status': status, 'uses_gpu': uses_gpu, 'process_wall_seconds': process_seconds,
        'internal_stage_wall_seconds': internal, 'accounted_wall_seconds': accounted,
        'accounting_basis': basis,
        'gpu_stage_wall_hours_proxy': accounted / 3600 if uses_gpu and accounted is not None else None,
        **{key: None for key in TRAIN_TIMERS},
        'hardware': summary.get('gpu', summary.get('device_name', pipeline.get('hardware', 'not recorded here'))),
        'summary_source': summary_path, 'pipeline_source': pipeline_path,
        'note': note,
    }
    rows.append(row)
    details.append({
        'group': group, 'stage': name, 'summary_source': summary_path,
        'counts': summary.get('counts'), 'cost_breakdown': cost or None,
        'feature_reuse': summary.get('feature_reuse'), 'source_cache': summary.get('source_cache'),
        'training_setup_seconds': summary.get('preparation_seconds'),
        'training_runs': summary.get('runs'), 'cost_scope': summary.get('cost_scope'),
    })
    for run in summary.get('runs', []):
        rows.append({**row, 'record_type': 'training_run', 'coverage': run.get('coverage', coverage),
                     'seed': run['seed'], 'arm': run['arm'],
                     'process_wall_seconds': None, 'internal_stage_wall_seconds': None,
                     'accounted_wall_seconds': None, 'gpu_stage_wall_hours_proxy': None,
                     'accounting_basis': 'non_additive_breakdown',
                     **{key: run.get(key) for key in TRAIN_TIMERS},
                     'note': 'Contained in the training stage; do not add again. Timers include host work and synchronization.'})
    return summary


def collect(root, allow_incomplete):
    rows, details = [], []
    lpath = 'runs/vitl-and-latency-pipeline-v1/pipeline_status.json'
    lp = read(root, lpath)
    if not allow_incomplete and lp.get('status') != 'complete':
        raise ValueError('ViT-L/14 pipeline is not complete locally. Use --allow-incomplete for a partial record.')
    for dataset in ('pets', 'flowers'):
        paths = {
            'features': f'runs/{dataset}-vitl128-features-v1/prep_summary.json',
            'prepare': f'runs/{dataset}-vitl128-supervision-v1/prep_summary.json',
            'train': f'runs/{dataset}-vitl128-training-v1/training_summary.json',
            'evaluate': f'runs/{dataset}-vitl128-eval-v1/eval_summary.json',
            'analysis': None, 'geometry': None,
        }
        for stage, path in paths.items():
            summary = add_stage(rows, details, root, 'ViT-L14', dataset, 'ViT-L/14', 128,
                                f'{dataset}_{stage}', path, lp, lpath,
                                uses_gpu=stage not in ('analysis', 'geometry'),
                                note='Feature cache is built once; teacher/training/evaluation reuse it.')
            if not allow_incomplete:
                assert pipeline_stage(lp, f'{dataset}_{stage}')['status'] == 'complete'
                if path:
                    assert summary, path
                if stage == 'train':
                    assert {(r['coverage'], r['seed'], r['arm']) for r in summary['runs']} == {
                        ('full', seed, arm) for seed in (2041, 2042, 2043)
                        for arm in ('conditioned', 'image_only')}

    # The scale expansion really re-encoded the full prefix cache but reused old
    # teacher targets. Record each executed preparation once, not each consumer.
    source = 'results/pets-selector-pilot-v1/supervision/prep_summary.json'
    add_stage(rows, details, root, 'B16-Pets49-source', 'pets', 'ViT-B/16', 49,
              'pilot_supervision_source', source, {}, None, coverage='small_teacher_source',
              note='Earlier teacher source reused by scale preparation; counted once. Original pilot training is outside this table.')
    scale = 'results/pets-selector-scale-v1'
    for stage, suffix in [('prepare', 'prep_summary.json'),
                          ('train', 'training/training_summary.json'), ('evaluate', 'eval_summary.json')]:
        add_stage(rows, details, root, 'B16-Pets49-scale', 'pets', 'ViT-B/16', 49,
                  stage, f'{scale}/{suffix}', {}, None, coverage='small_and_full',
                  note='Outer process timer was not retained; internal wall is available. Small/full coverage runs are distinct.')

    ppath = 'results/pets-selector-budgets-v1/compute_cost.json'
    pp = read(root, ppath)
    for budget in (98, 147):
        base = f'results/pets-selector-budgets-v1/execution/budget_{budget}'
        for stage, suffix in [('prepare', 'supervision/prep_summary.json'),
                              ('train', 'training/training_summary.json'), ('evaluate', 'eval/eval_summary.json')]:
            add_stage(rows, details, root, 'B16-Pets-budgets', 'pets', 'ViT-B/16', budget,
                      f'{stage}_{budget}', f'{base}/{suffix}', pp, ppath,
                      note='Reuses the scale prefix cache; no second feature preparation charge.')
    for name in ('random_49', 'shared_49', 'shared_98', 'shared_147'):
        budget = int(name.split('_')[1])
        add_stage(rows, details, root, 'B16-Pets-supplements', 'pets', 'ViT-B/16', budget,
                  name, None, pp, ppath, uses_gpu=name in ('random_49', 'shared_49'),
                  note='Previously recorded supplementary random/shared diagnostics.')

    fbase = 'results/flowers-selector-budgets-v1'
    fpath = f'{fbase}/compute_cost.json'
    fp = read(root, fpath)
    add_stage(rows, details, root, 'B16-Flowers-budgets', 'flowers', 'ViT-B/16', None,
              'features', f'{fbase}/execution/features.json', fp, fpath,
              note='One shared feature cache for budgets 49 and 98; charged once.')
    for budget in (49, 98):
        for stage in ('prepare', 'train', 'evaluate'):
            add_stage(rows, details, root, 'B16-Flowers-budgets', 'flowers', 'ViT-B/16', budget,
                      f'{stage}_{budget}', f'{fbase}/execution/{stage}_{budget}.json', fp, fpath)
    for name in ('budget_analysis', 'geometry_analysis'):
        add_stage(rows, details, root, 'B16-Flowers-budgets', 'flowers', 'ViT-B/16', None,
                  name, None, fp, fpath, uses_gpu=False)
    return rows, details, lp


def totals(rows):
    stages = [row for row in rows if row['record_type'] == 'stage']
    known = [r for r in stages if r['accounted_wall_seconds'] is not None]
    gpu = [r for r in known if r['uses_gpu']]
    process = [r['process_wall_seconds'] for r in known if r['process_wall_seconds'] is not None]
    return {
        'stage_records': len(stages), 'known_stage_records': len(known),
        'all_stage_costs_available': len(known) == len(stages),
        'recorded_stage_wall_seconds': sum(r['accounted_wall_seconds'] for r in known) if known else None,
        'recorded_pipeline_process_seconds': sum(process) if process else None,
        'recorded_internal_only_seconds': sum(r['accounted_wall_seconds'] for r in known
                                             if r['process_wall_seconds'] is None) if known else None,
        'recorded_gpu_stage_wall_hours_proxy': sum(r['gpu_stage_wall_hours_proxy'] for r in gpu) if gpu else None,
        'missing_stage_records': [f"{r['group']}/{r['stage']}" for r in stages
                                  if r['accounted_wall_seconds'] is None],
    }


def main(root, output, allow_incomplete=False):
    rows, details, pipeline = collect(root, allow_incomplete)
    elapsed = None
    if pipeline.get('completed_utc'):
        elapsed = (datetime.fromisoformat(pipeline['completed_utc']) -
                   datetime.fromisoformat(pipeline['started_utc'])).total_seconds()
    result = {
        'scope': 'Recorded ViT-L/14 validation and selected B/16 scale/budget experiments; not the full project lifetime cost.',
        'definitions': {
            'accounted_wall_seconds': 'Per stage: prefer outer pipeline process wall; otherwise use explicitly labelled internal stage wall. Never sum both.',
            'gpu_stage_wall_hours_proxy': 'Single-GPU stage wall / 3600, including CPU, loading, synchronization and I/O. Not GPU kernel utilization hours, exclusive allocation, or billed hours.',
            'hardware_context': 'Shared server. Per-stage GPU names are retained; these timings are affected by shared CPU/I/O and are not hardware-normalized comparisons.',
            'missing': 'Null means not recorded or not yet complete, never zero. Totals sum known completed records only.',
            'training_runs': 'Non-additive breakdown, including separate small/full coverage runs at Pets49. Already included in parent stages.',
            'evaluation': 'Cached-prefix research evaluation can reuse image-only outputs across tasks. These costs must not be interpreted as deployment latency.',
            'excluded': 'Earlier pilots except reused teacher source, downloads, transfers, archival, plotting, audits and latency benchmark execution are outside this selected cost ledger.',
        },
        'vitl_pipeline': {'status': pipeline.get('status', 'missing_record'),
                          'elapsed_start_to_finish_seconds': elapsed,
                          'environment': pipeline.get('environment'),
                          'note': 'Pipeline elapsed and summed stages overlap; do not add them.'},
        'totals': totals(rows),
        'by_group': {group: totals([r for r in rows if r['group'] == group])
                     for group in dict.fromkeys(r['group'] for r in rows)},
        'by_model_dataset': {f'{model}/{dataset}': totals([
            r for r in rows if r['model'] == model and r['dataset'] == dataset])
            for model, dataset in dict.fromkeys((r['model'], r['dataset']) for r in rows)},
        'rows': rows, 'stage_details': details,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'experiment_costs.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (output / 'experiment_costs.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'vitl_pipeline_status': result['vitl_pipeline']['status'],
                      'rows': len(rows), 'totals': result['totals'], 'output': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, default=Path('results/vitl-and-latency-v1'))
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    main(args.root, args.output, args.allow_incomplete)
