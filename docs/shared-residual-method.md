# Learned shared-score correction

The frozen-mask interventions found useful mixtures of shared and conditioned
selections on Flowers, while matching actual replacement counts removed the
evidence for a consistent priority advantage within the shared-only candidate
pool. This method test asks whether task information can improve a *learned*
correction to a strong shared selector, beyond an equally budgeted image-only
correction.

## Architecture

Freeze the existing width-199 shared selector U, including its normalization and
two hidden layers. Its penultimate patch representation is `h` (199 dimensions),
and its scalar score is `s_U`. Predict `s = s_U + r` using one small residual branch.

- **Task correction:** normalize h and the unordered candidate-pair descriptor
  (text mean plus absolute difference, 1,024 dimensions), concatenate, and apply
  a linear layer to width 8, GELU, then a scalar linear head. The implementation
  splits the first projection into image and context terms to avoid physically
  repeating the full descriptor for every patch; the function is unchanged.
- **Image correction:** normalize the same h, apply width 49, GELU, and a scalar
  linear head. It has no candidate-class input.
- Both normalizations use fixed affine parameters (no trainable scale or bias).
  Both scalar output heads start at exactly zero. Thus both selectors initially
  reproduce U's scores, masks, and predictions. The residual coefficient is 1.

| Selector | Frozen parameters | Newly trained parameters | Total parameters |
| --- | ---: | ---: | ---: |
| Shared U | 347,399 | 0 | 347,399 |
| U + task correction | 347,399 | 9,801 | 357,200 |
| U + image correction | 347,399 | 9,850 | 357,249 |

The new methods add about 2.8% to U's parameter count; their total counts differ
by 49 parameters. They are shared-score corrections: the architecture does not
enforce a protected token subset or a fixed number of replacements.

## Fixed training and evaluation

The initial matrix is Flowers49, Flowers98, and Pets49 as a control, with training
seeds 2041–2043 and two residual arms: 18 new fits. Each seed uses its corresponding
existing shared checkpoint. Freeze that checkpoint for both arms and reuse the
same full-coverage training images, teacher KL targets, temperature, optimizer,
learning rate, weight decay, update count, and exact sampled image/task stream.
The settings retain the original 3,990 Pets or 3,978 Flowers steps, batch size 128,
AdamW learning rate 0.0003 and weight decay 0.01. Evaluate the final fixed step.

Frozen h and U scores are cached for branch fitting only. Evaluation executes the
shared selector and residual online from cached CLIP prefix tokens, then runs the
actual pruned CLIP suffix. Candidate text describes both classes; ground-truth
labels are used for training supervision and evaluation, never as selector input.
The original teacher targets and full CLIP classifier remain unchanged.

The primary comparison is task correction minus image correction, across the
three settings, using geometric margin and accuracy with separate three-comparison
adjustments. Compare both methods with U as secondary evidence. In the same-image
task-exchange experiment, keep the evaluated task fixed and substitute the other
task's selection. U and image correction must have exactly zero task-exchange
effect. Class/task weights and paired image-bootstrap follow the existing protocol.

Report every seed. Measure actual token changes relative to U, residual score
standard deviation/RMS, and U's score standard deviation: a small parameter count
does not imply that the learned correction changes only a few tokens. Record
training/validation teacher losses to distinguish fitting changes from downstream
classification changes.

## What would count as progress

- A task correction that beats the image correction and U on independent
  validation task combinations, while retaining useful task-exchange effects,
  provides a learnable use of task information beyond extra image-only capacity.
- If both corrections improve similarly, extra shared-feature processing explains
  the improvement without a clear contribution from candidate text.
- If task exchange is positive but task correction does not beat image correction,
  using task information and making its inclusion worthwhile remain distinct.
- If neither correction improves U, the bounded method does not convert the
  frozen-mask complementarity into a gain under the original KL objective.

## Commands and artifacts

```bash
PYTHONPATH=src .venv/bin/python scripts/run_shared_residual.py --setting flowers_b16_49 --device cuda:7 --output runs/shared-residual-v1/flowers_b16_49
PYTHONPATH=src .venv/bin/python scripts/run_shared_residual.py --setting flowers_b16_98 --device cuda:6 --output runs/shared-residual-v1/flowers_b16_98
PYTHONPATH=src .venv/bin/python scripts/run_shared_residual.py --setting pets_b16_49 --device cuda:2 --output runs/shared-residual-v1/pets_b16_49
PYTHONPATH=src:scripts:tests python3 -m unittest test_shared_residual -q
```

Each setting saves small correction checkpoints referencing the original shared
checkpoint, training curves, teacher losses, parameter counts, sampling audits,
online selector timing, CLIP outputs, masks, logits, correction diagnostics, and
paired analysis under `analysis/`. The historical U endpoint and full-model scores
are checked against the existing core-protection artifacts. Reused shared training,
CLIP weights, and teacher caches are explicitly distinguished from the new fitting
cost. Batched replay timing does not establish an end-to-end speed improvement.
