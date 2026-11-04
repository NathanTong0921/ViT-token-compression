# Pruning depth intervention — 2026-11-04

## Question and fixed scope

I27 found a Flowers49 average task-related mixing benefit; I30's simple signals
did not explain benefits across settings. This study changes the stage at which
selection occurs, asking whether the *relative* benefit of conditioning changes.
It extends proposal H1/H3 on the original images. The earlier composite-image
diagnosis used different tasks and masks.

Protocol, configurations, implementation and tests were committed and pushed
as `ebb08fc` before generating new supervision or training. See the
[fixed protocol](../pruning-depth-intervention.md). The six settings are
Pets/Flowers × blocks 3/6/9, retaining 49 patches. Block6 reuses historical
checkpoints and predictions; blocks3/9 add 24 selectors. C and the strong wide U
use the same fixed KL recipe and three paired seeds. No additional depths,
budgets, replacement seeds or direct-set training were added.

## Execution

Server repository: `/data2/zhao.tong/experiments`; Python: `.venv/bin/python`.
Two workers used GPU1 (Pets3, Flowers9) and GPU2 (Flowers3, Pets9). Each setting
ran fresh depth-specific prefix extraction, teacher generation and selector
training before evaluation. The following reproduces one setting; repeat for
the four committed configuration files:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/run_depth_training.py \
  --config configs/pets_selector_depth3_49.json --stage all

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:scripts \
  .venv/bin/python scripts/evaluate_depth_intervention.py \
  --dataset pets --config configs/pets_selector_depth3_49.json \
  --models runs/depth-intervention-v1/pets_b16_d3_49/training \
  --output runs/depth-intervention-v1/pets_d3
```

The raw archive retains worker commands, logs and per-setting execution records.
Old block6 teacher-fit diagnostics were recomputed with the existing final
weights; this did not retrain or replace the historical predictions. Additional
audit, diagnostic, plotting and packaging tools were committed as `f6d3c87`.

CPU analysis and independent reconstruction:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:scripts \
  python3 scripts/analyze_depth_intervention.py \
  --sources runs/depth-intervention-v1 \
  --output results/depth-intervention-v1

python3 scripts/audit_depth_intervention.py \
  --sources runs/depth-intervention-v1 \
  --analysis results/depth-intervention-v1

python3 scripts/plot_depth_intervention.py \
  --source results/depth-intervention-v1 \
  --output results/depth-intervention-v1
```

## Findings and consequence

Pets C−U geometric margin changes from −0.002524 at block6 to +0.001331 at
block9. The paired change is +0.003855, with four-comparison-adjusted interval
[0.002913, 0.004776]; all three seeds agree. CU−UU changes by +0.001277
[0.000738, 0.001825], with all six ordered distinct-seed pairs agreeing.
Raw cosine-gap analysis supports both changes. Flowers does not establish the
same depth interaction, although its mixing margin benefit remains positive at
all three depths. All accuracy depth-interaction intervals include zero.

The original Pets49 negative result is therefore specific to the original
stage, rather than a result applying across these stages. The intervention
changes prefix representations, suffix computation and teacher responses
together. It establishes stage dependence of the fixed recipe; distinguishing
the underlying causes is a separate question. I31 records this change in the
research argument. The fixed matrix ends here. The next method question still
comes from I28/I29: whether direct set supervision improves selected sets and
classification, and whether a task branch adds value beyond an image branch.

## Validation and cost

The 11 focused unit tests pass. Keep-all output agrees with native CLIP and
the stored full reference. The independent raw-score/mask audit checks all six
panels and reproduces point estimates and paired bootstrap intervals, including
all seeds and seed pairs. Training, archive and checkpoint replay verification
is recorded separately in `training_backup_check.json`; its stated scope does
not include rerunning full CLIP suffix inference. No final reserved data was used.

The 24 optimizer runs total 0.18155 hours. Summed GPU-assigned process wall time
is 0.52840 hours, including preparation, loading, I/O, evaluation and the new
block6 diagnostics; this is not device-kernel utilization. Frozen prefix reuse
makes selector optimization inexpensive. Detailed stage costs are in
[`costs.json`](../../results/depth-intervention-v1/costs.json).

## Retained evidence

Git retains summaries, paired case arrays, image contributions, all contrasts,
plots, diagnostics, audit records and backup manifests in
[`results/depth-intervention-v1`](../../results/depth-intervention-v1/README.md).
Two ZIPs retain the six raw evaluation panels and the 24 final checkpoints,
training configurations/history, thin depth-specific teacher arrays and eight
prefix replay examples. Existing block6 model/teacher backups are referenced
without duplication. Full prefix caches and raw teacher responses remain on
the server. Local and GitHub copies are listed in
[`backup_manifest.json`](../../results/depth-intervention-v1/backup_manifest.json).
