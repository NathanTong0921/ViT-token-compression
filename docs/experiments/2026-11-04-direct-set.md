# Direct retained-set supervision — 2026-11-04

## Purpose and fixed design

This is the planned H3 follow-up to I28/I29. The earlier hybrid-target KL fits
improved probability loss without improving mask recovery, and positive score
rescaling explained much of that loss change. This experiment changes the
objective to a row-standardized, cost-augmented top-k set surrogate while keeping
the same saved hybrid masks, frozen shared U, residual architectures, initialization,
sampling stream and optimization budget. See the
[protocol](../direct-set-supervision.md).

The protocol, loss, paired training runner and tests were committed and pushed
as `0f51dc9` before execution. Analysis/plotting and independent arithmetic checks
were committed as `f46a0c2` and `e1bd120`. The full matrix is three original
block-6 settings × two residual arms × three seeds: 18 new fits. Historical
hybrid-KL fits are the matched objective reference. No depth or teacher change is
combined with this experiment.

## Execution

Server repository: `/data2/zhao.tong/experiments`. GPUs1/2/3 ran Pets49,
Flowers49 and Flowers98 respectively, with one worker per setting. All workers
completed with exit0. Example command (the worker restricts the visible GPU):

```bash
CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=src:scripts .venv/bin/python scripts/run_set_distillation.py \
  --setting pets_b16_49 --device cuda:0 \
  --output runs/set-distillation-v1/pets_b16_49

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:scripts \
  .venv/bin/python scripts/analyze_set_distillation.py \
  --source runs/set-distillation-v1/pets_b16_49 \
  --output runs/set-distillation-v1/pets_b16_49/analysis

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/audit_set_distillation.py \
  --source runs/set-distillation-v1/pets_b16_49 \
  --analysis runs/set-distillation-v1/pets_b16_49/analysis \
  --kl-source runs/hybrid-distillation-v1/pets_b16_49
```

Repeat for `flowers_b16_49` and `flowers_b16_98`. Worker commands, exit status,
stage timings, training curves and source revisions are retained in the archive.
Plots use the copied per-setting summaries:

```bash
PYTHONPATH=src:scripts python3 scripts/plot_set_distillation.py \
  --source results/set-distillation-v1 --output results/set-distillation-v1
```

## Results

All six new-vs-KL validation overlap comparisons are positive after the fixed
six-comparison adjustment: task/image gains are +2.300/+1.821pp on Pets49,
+3.566/+2.648pp on Flowers49, and +1.595/+1.105pp on Flowers98. Compared with
the frozen U starting point, overlap point estimates increase only 0.290–0.551pp.
Part of the larger contrast therefore restores loss from the previous KL training.

Task-minus-image accuracy and margin comparisons are inconclusive in all three
settings under the planned three-setting adjustment. Neither arm establishes an
accuracy gain over U. Flowers49 task accuracy improves +0.2451pp over its KL
counterpart (ordinary interval [0.0082, 0.4902]pp) and reaches U's 96.7320%.
Flowers98 task accuracy improves +0.1471pp over KL but remains below U in the
point estimate. Both Flowers students have positive mean margin differences
from U, while the hybrid teacher remains ahead in margin. Pets students remain
below U in margin. All error corrections and newly introduced errors are retained.

I32 records the consequence: the objective intervention improves retained-set
recovery on unseen combinations; task-specific downstream superiority remains
unestablished. This completes the fixed matrix. No additional tuning, seeds or
new teacher are launched as part of this round.

## Validation, costs and retained evidence

Sixteen focused local tests pass, including enumeration of the set objective,
its error bound and scale invariance, frozen-base training, reference alignment
and the corrected-minus-broken accuracy identity. The new objective's five tests
also pass in the server environment.

The independent NumPy audit reconstructs each setting's raw-score/mask outcomes,
all saved point estimates and intervals, seed summaries and paired case arrays.
Maximum numerical discrepancy is below 6e-14. Training checks cover all18 branch
weights, saved targets and sample streams, all train/development fit panels, and
the inactive standard-deviation floor. CPU replay of18 branches plus9 bases
recovers all saved retained positions exactly. Cached diagnostic and online
validation masks agree. Full records are in the
[results directory](../../results/set-distillation-v1/README.md).

New branch optimization totals434.67seconds. The three GPU-assigned process wall
times total535.53seconds (0.14876hours), including loading, fitting, diagnostics,
evaluation and other overhead. These totals must not be added together. Frozen
prefixes, teacher generation and historical models were reused; this is not a
new end-to-end inference-speed measurement.

The210,999,734-byte recovery ZIP contains110 files: all18 new branch weights,
9 frozen shared bases,3 prefix replay examples, complete train/development set
diagnostics, raw classification scores/masks, logs, protocols and analyses.
Historical KL models/targets are referenced through the existing hybrid-followup
backup manifest. Local archive/member integrity checks pass. Git stores figures,
tables, paired cases, diagnostics and audit records; the server retains raw runs.
Locations and upload verification are recorded in the
[backup manifest](../../results/set-distillation-v1/backup_manifest.json).

The archive completed local/member integrity checks and GitHub upload verification.
Server-direct upload took44.87seconds. Transfer-only scripts and bundles are removed
after repository synchronization; the local scientific archive is retained.
