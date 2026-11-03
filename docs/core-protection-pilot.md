# Shared core and task-specific residual: frozen-model intervention

The question is whether task conditioning loses useful common visual evidence when
it replaces tokens that a strong image-only selector consistently favors. This is
an intervention on existing trained models, with no new training or test-set use.
It extends the proposal's comparison with a strong shared selector: instead of
only comparing their final accuracies, it tests which part of their selections
can be combined productively.

## Fixed experiment

- Pets, CLIP ViT-B/16, block 6, 49 tokens; Flowers102, same backbone and block,
  98 tokens. All existing validation images and unseen validation class pairs.
- Original conditioned selector and the parameter-matched wide image-only
  selector, with all three trained seeds 2041, 2042, and 2043.
- Protect `q = floor(k × fraction)` image-only tokens at fractions 0, 0.5,
  0.75, and 1; fill the remaining slots in conditioned rank order, skipping
  already protected positions. Total retained tokens remain exactly `k`.
- At interior fractions compare the highest-ranked image-only core with a
  uniformly sampled subset of the same image-only top-k set. Random-core seeds
  are 2211, 2212, and 2213. The random ordering is fixed per image, shared across
  its tasks, and nested across core sizes. Endpoints are the original conditioned
  selector and original wide image-only selector.

The protected counts are 0/24/36/49 for Pets and 0/49/73/98 for Flowers. No core
fraction is selected using the evaluation outcomes. All fractions and seeds are
reported. Ties in token score use original patch order, and retained positions
remain spatially ordered.

## Evidence and comparisons

The primary mechanism comparison is top-ranked core minus size-matched random
core, at each interior fraction. This distinguishes protecting the shared
selector's most preferred evidence from simply injecting some shared tokens.
The normalized classification margin is the primary mechanism metric; accuracy
is also reported. The four comparisons across two fractions and two datasets
receive simultaneous intervals separately for each metric. Ordinary paired
intervals and every training seed's point estimate are also retained.

Classification uses the existing equal-task, equal-class-side weighting. Within
each image, exchange the task descriptors used to select the residual while
holding the evaluated task fixed. All unordered pairs of that image's validation
tasks are evaluated; exchange averages equally across images and then across
task pairs within an image. The wide endpoint must have exactly zero exchange
effect. Random-core outcomes are averaged after scoring, not ensembled as logits.
Fixed training seeds are averaged before the paired image bootstrap.
Actual replacements relative to each endpoint, and changed tokens across tasks,
are also recorded: equal protected-core size can replace different numbers of
the original conditioned tokens when the two rankings already overlap.

Interpretations:

- A top-ranked core beats the random core and the conditioned endpoint, while
  the residual retains a positive exchange effect: useful shared evidence and
  task-adapted evidence can coexist at a fixed token budget.
- A hybrid also beats the wide endpoint: task information adds value after
  protecting shared evidence, beyond the earlier comparison of separate models.
- Top-ranked and random cores behave similarly: improvements do not specifically
  depend on preserving the wide selector's highest-ranked tokens.
- Every mixture approaches the wide endpoint monotonically, and exchange benefit
  vanishes: this intervention finds little useful residual task specialization.

These outcomes concern frozen learned rankings. A positive result motivates a
single trained core-plus-residual selector; the replay itself uses two selectors
and is not a deployment-speed result.

## Running and stored evidence

From the repository root on the server:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_core_protection.py --setting pets_b16_49 --device cuda:0 --output runs/core-protection-v1/pets_b16_49
PYTHONPATH=src .venv/bin/python scripts/evaluate_core_protection.py --setting flowers_b16_98 --device cuda:0 --output runs/core-protection-v1/flowers_b16_98
PYTHONPATH=src .venv/bin/python scripts/analyze_core_protection.py --source runs/core-protection-v1/pets_b16_49 --output results/core-protection-v1/pets_b16_49
PYTHONPATH=src .venv/bin/python scripts/analyze_core_protection.py --source runs/core-protection-v1/flowers_b16_98 --output results/core-protection-v1/flowers_b16_98
```

The evaluator stores image/task identities, all masks, original selector logits,
unique frozen-CLIP output features, cosine scores, model paths, settings, and
runtime. Identical image/mask pairs share one suffix result, eliminating numerical
differences caused by evaluating an identical mask in different variant batches.
The analysis stores per-row and per-image paired outcomes, original per-seed and
per-random-mask outcomes, classification and exchange summaries, and bootstrap
intervals. No original images or model weights need to be copied into GitHub.

CPU verification:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_core_protection.py'
```
