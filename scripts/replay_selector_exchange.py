"""Replay fixed task-exchange cases from saved selector weights and prefix tokens."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import clip

from evaluate_selectors import binary_outcomes
from token_compression.pruning import SingleStagePruner
from token_compression.selector import TokenSelector


def run(config, source, models, output):
    started = time.perf_counter()
    with np.load(source / 'evaluation/predictions.npz') as z:
        saved = {k: z[k] for k in z.files}
    seed0 = config['training_seeds'][0]
    reference = np.flatnonzero((saved['seed'] == seed0)
                              & (saved['method'] == 'conditioned')
                              & (saved['task_split'] == 'unseen'))
    plans = []
    for image_id in sorted(set(saved['image_ids'][reference])):
        rows = sorted((i for i in reference if saved['image_ids'][i] == image_id),
                      key=lambda i: saved['task_ids'][i])
        if len(rows) >= 2:
            plans.append((image_id, saved['task_ids'][rows[0]], saved['task_ids'][rows[1]]))
        if len(plans) == 8:
            break
    assert len(plans) == 8
    device = torch.device(config['device'])
    backbone, _ = clip.load(config['model'], device=device, jit=False,
                            download_root=config['model_cache'])
    backbone.eval().requires_grad_(False)
    pruner = SingleStagePruner(backbone.visual, config['prune_after'])
    tokens = torch.as_tensor(np.load(Path(config['cache_dir']) / 'val/prefix_tokens.npy'), device=device)
    text = saved['text_features']
    text_tensor = torch.as_tensor(text, device=device)
    batch_size = config['image_batch_size']
    assert batch_size == 64
    k = config['keep_patch_tokens']
    output.mkdir(parents=True, exist_ok=True)
    all_rows, all_scores, all_logits, all_positions, checks = [], [], [], [], []
    with torch.inference_mode():
        for seed in config['training_seeds']:
            for method in ('conditioned', 'image_only'):
                lookup = {(saved['image_ids'][i], saved['task_ids'][i]): i
                          for i in np.flatnonzero((saved['seed'] == seed) & (saved['method'] == method)
                                                 & (saved['task_split'] == 'unseen'))}
                rows = np.array([lookup[(image_id, task)] for image_id, a, b in plans for task in (a, b)])
                padded = np.concatenate([rows, np.repeat(rows[-1], batch_size - len(rows))])
                checkpoint = torch.load(models / f'seed_{seed}/{method}.pt', map_location=device, weights_only=True)
                model = TokenSelector(checkpoint['state_dict']['shared_descriptor'],
                                      conditioned=method == 'conditioned', hidden_dim=checkpoint['hidden_dim'],
                                      token_dim=tokens.shape[-1]).to(device)
                model.load_state_dict(checkpoint['state_dict'])
                model.eval().requires_grad_(False)
                prefix = tokens[torch.as_tensor(saved['image_index'][padded], device=device)]
                candidates = text_tensor[torch.as_tensor(saved['class_pairs'][padded], device=device)]
                logits = model(prefix, candidates).cpu().numpy()
                positions = np.sort(np.argsort(-logits, axis=1, kind='stable')[:, :k], axis=1)
                features = pruner.finish(prefix, torch.as_tensor(positions, device=device)).float()
                features = (features / features.norm(dim=-1, keepdim=True)).cpu().numpy()
                scores = (features @ text.T)[:len(rows)]
                logits, positions = logits[:len(rows)], positions[:len(rows)]
                score_error = float(np.max(np.abs(scores - saved['cosine_scores'][rows])))
                logit_error = float(np.max(np.abs(logits - saved['selection_logits'][rows])))
                assert score_error <= 1e-6 and logit_error <= 1e-5
                assert np.array_equal(positions, saved['patch_indices'][rows, :k])
                # Consecutive rows use descriptors A/B for the same image; evaluate each on the other task.
                exchanged = np.arange(len(rows)).reshape(-1, 2)[:, ::-1].reshape(-1)
                target_pairs = saved['class_pairs'][rows[exchanged]]
                margins, correct = binary_outcomes(scores, saved['labels'][rows], target_pairs, text)
                old_margins, old_correct = binary_outcomes(saved['cosine_scores'][rows], saved['labels'][rows], target_pairs, text)
                margin_error = float(np.max(np.abs(margins - old_margins)))
                assert margin_error <= 1e-6 and np.array_equal(correct, old_correct)
                checks.append({'seed': seed, 'method': method, 'logical_examples': len(rows),
                               'physical_examples': batch_size, 'max_score_error': score_error,
                               'max_logit_error': logit_error, 'max_crossed_margin_error': margin_error,
                               'positions_exact': True, 'crossed_correctness_exact': True})
                all_rows.append(rows); all_scores.append(scores); all_logits.append(logits); all_positions.append(positions)
                print(json.dumps(checks[-1]), flush=True)
                del model
    np.savez_compressed(output / 'replayed_predictions.npz', source_row=np.concatenate(all_rows),
                        cosine_scores=np.concatenate(all_scores), selection_logits=np.concatenate(all_logits),
                        patch_indices=np.concatenate(all_positions))
    summary = {'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               'source': str(source), 'config': config,
               'selection': 'First eight alphabetically sorted eligible unseen-panel images, first two task IDs per image; no outcome-based sampling.',
               'image_task_plans': [list(p) for p in plans], 'checks': checks,
               'score_atol': 1e-6, 'logit_atol': 1e-5, 'crossed_margin_atol': 1e-6,
               'wall_seconds': time.perf_counter() - started,
               'gpu': torch.cuda.get_device_name(device), 'torch_version': str(torch.__version__),
               'cost_scope': 'Small cached-prefix replay, including loading/checks/writes; not training or deployment latency.'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/pets_selector_pilot.json'))
    parser.add_argument('--source', type=Path, default=Path('results/pets-selector-pilot-v1'))
    parser.add_argument('--models', type=Path, default=Path('runs/pets-selector-training-v1'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.source, args.models, args.output)
