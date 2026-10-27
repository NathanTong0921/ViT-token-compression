"""Verify retained evidence and recover models/summary values from a restored ZIP."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile


def verify_archive(path):
    path = Path(path)
    record = json.loads(path.with_suffix('.json').read_text())
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    assert path.stat().st_size == record['bytes']
    assert digest.hexdigest() == record['sha256']
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        expected = {item['path']: item['bytes'] for item in record['members']}
        assert {item.filename: item.file_size for item in archive.infolist()} == expected
    result = {'archive': path.name, 'bytes': record['bytes'], 'members': len(expected),
              'local_digest_matches_manifest': True, 'all_zip_crc_checks_passed': True}
    path.with_suffix('.verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return result


def recover_models(path):
    import numpy as np
    import torch
    from token_compression.selector import TokenSelector

    recovered = []
    summaries = []
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        checkpoints = sorted(name for name in archive.namelist() if name.endswith('.pt'))
        assert len(checkpoints) == 12
        for name in checkpoints:
            state = torch.load(io.BytesIO(archive.read(name)), map_location='cpu', weights_only=True)
            selector = TokenSelector(state['state_dict']['shared_descriptor'], state['conditioned'], state['hidden_dim'])
            selector.load_state_dict(state['state_dict'])
            assert sum(parameter.numel() for parameter in selector.parameters()) == state['parameter_count']
            with torch.inference_mode():
                scores = selector(torch.zeros(1, 197, 768), torch.zeros(1, 2, 512))
            assert scores.shape == (1, 196) and torch.isfinite(scores).all()
            assert state['step'] == 3978 and state['coverage'] == 'full'
            recovered.append({'path': name, 'seed': state['seed'], 'conditioned': state['conditioned'],
                              'step': state['step'], 'parameter_count': state['parameter_count'],
                              'strict_state_load_and_forward_passed': True})
        for budget in (49, 98):
            prefix = f'runs/flowers-selector-budget-{budget}-training-v1/'
            summary = json.loads(archive.read(prefix + 'training_summary.json'))
            history = list(csv.DictReader(io.StringIO(archive.read(prefix + 'training_history.csv').decode())))
            for run in summary['runs']:
                rows = [row for row in history if row['coverage'] == run['coverage']
                        and int(row['seed']) == run['seed'] and row['arm'] == run['arm']]
                final = max(rows, key=lambda row: int(row['step']))
                assert int(final['step']) == run['training_steps'] == 3978
                final_val = float(final['val_kl'])
                train_seconds = sum(float(row['train_seconds']) for row in rows)
                assert math.isclose(final_val, run['final_val_kl'], rel_tol=1e-12, abs_tol=1e-12)
                assert math.isclose(train_seconds, run['training_seconds'], rel_tol=1e-12, abs_tol=1e-8)
                summaries.append({'budget': budget, 'seed': run['seed'], 'arm': run['arm'],
                                  'final_val_kl_reconstructed': final_val,
                                  'training_seconds_reconstructed': train_seconds,
                                  'matches_original_summary': True})
        prefix = 'runs/flowers-features-v1/'
        baseline_summary = json.loads(archive.read(prefix + 'prep_summary.json'))
        tasks = list(csv.DictReader(io.StringIO(archive.read(prefix + 'val_tasks.csv').decode())))
        with np.load(io.BytesIO(archive.read(prefix + 'val/evidence.npz'))) as evidence:
            labels, scores, text = evidence['labels'], evidence['full_scores'], evidence['text_features']
        correct, margins = [], []
        for task in tasks:
            a, b = int(task['class_a']), int(task['class_b'])
            for ix in np.flatnonzero((labels == a) | (labels == b)):
                label = int(labels[ix])
                competitor = b if label == a else a
                predicted = (a, b)[int(np.argmax(scores[ix, [a, b]]))]
                correct.append(predicted == label)
                margins.append(float((scores[ix, label] - scores[ix, competitor])
                                     / np.linalg.norm(text[label] - text[competitor])))
        baseline = baseline_summary['baseline']['val']['binary']['val']
        assert len(correct) == baseline['image_task_count'] == 4080
        accuracy = float(np.mean(correct))
        assert accuracy == baseline['accuracy']
        assert math.isclose(float(np.mean(margins)), baseline['geometric_margin'], abs_tol=1e-7)
        baseline_recovery = {'image_task_count': len(correct), 'correct': int(np.sum(correct)),
                             'accuracy': accuracy, 'matches_original_summary': True,
                             'source': 'Restored full_scores, labels, text_features and unseen validation tasks.'}
    result = {'restored_archive': Path(path).name, 'all_zip_crc_checks_passed': True,
              'checkpoint_count': len(recovered), 'checkpoints': recovered,
              'source_summary_reconstruction': summaries,
              'native_unseen_binary_baseline_reconstruction': baseline_recovery}
    destination = Path('runs/flowers-selector-pipeline-v1/github_recovery_check.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'restored_checkpoints': len(recovered), 'reconstructed_training_runs': len(summaries),
                      'record': str(destination)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['verify', 'recover-models'])
    parser.add_argument('path')
    args = parser.parse_args()
    if args.action == 'verify':
        verify_archive(args.path)
    else:
        recover_models(args.path)
