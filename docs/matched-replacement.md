# Shared-ranking value at equal actual displacement

The core-protection pilot compared equally sized protected sets, but the top-ranked
core often already appeared in the conditioned selection. It therefore replaced
fewer original tokens than a random core. This follow-up fixes both the actual
number of changed tokens and the removed positions, isolating the added tokens.

## Intervention

For each image and candidate task, let `C` be the conditioned top-k and `U` the
wide image-only top-k. Retain their intersection throughout. Let
`m = |U minus C| = |C minus U|` and `d = floor(fraction × m)`.

At fractions 0.5 and 0.75:

1. Remove the same `d` lowest-conditioned-ranked positions from `C minus U`.
2. Add either the `d` highest-shared-ranked positions in `U minus C`, or a
   uniformly selected `d` positions from that same candidate set.
3. Keep exactly `k` unique positions in spatial order.

Random-addition seeds are 2211, 2212, and 2213. Each uses a fixed image/task-local
ordering of the disagreement candidates, nested across fractions. The candidate
set depends on the task through `C`; its per-row identity and the resulting
insertions/removals are saved. No labels select tokens. At fraction 0 the mask
is exactly `C`; at fraction 1 it is exactly `U`. Where rounding gives `d=0`, all
arms retain `C` and contribute zero top-versus-random difference.

## Fixed matrix and analysis

All four existing CLIP ViT-B/16 settings—Pets49, Pets98, Flowers49, Flowers98—use
block 6, all validation images/unseen validation class pairs, and all trained
seeds 2041–2043. Existing saved logits determine the rankings. Previously computed
features are reused only for identical image/mask pairs; remaining unique masks
run through the frozen CLIP suffix. There is no training or reserved-test use.

The primary comparison is top-ranked minus random additions in geometric margin,
at the two interior fractions across all four settings: eight comparisons with
ordinary and family-adjusted paired image-bootstrap intervals. Accuracy is
secondary and receives its own eight-comparison adjustment. Endpoint comparisons
and descriptor exchange are secondary analyses, with ordinary intervals.

Classification retains equal task/class-side weights. Random outcomes are averaged
after scoring, then training seeds are averaged before image resampling. Exchange
holds the evaluated task fixed while using the other task's complete intervention
mask; images receive equal weight, followed by equal within-image task-pair weight.
The wide endpoint must have zero exchange effect. Report each training seed and
the actual per-row displacement counts, including rows with no displacement.

## What distinguishes the outcomes

- A top-ranked advantage over matched random additions shows value in the shared
  ranking after displacement and removed positions are held fixed.
- If the earlier advantage disappears, actual intervention size or which original
  tokens were removed explains part of the earlier contrast.
- If both addition policies improve both endpoints, the frozen selections contain
  useful complementary evidence; a top-ranked advantage is a separate question.
- Positive residual exchange effects show retained task sensitivity, while beating
  the strong shared endpoint tests whether that sensitivity has practical value.

This experiment is an intervention on two frozen rankings. A successful outcome
can motivate a single trained replacement selector; the replay does not establish
inference acceleration.

## Commands and artifacts

For each setting (`pets_b16_49`, `pets_b16_98`, `flowers_b16_49`, `flowers_b16_98`):

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_matched_replacement.py --setting pets_b16_49 --device cuda:7 --output runs/matched-replacement-v1/pets_b16_49
PYTHONPATH=src .venv/bin/python scripts/analyze_matched_replacement.py --source runs/matched-replacement-v1/pets_b16_49 --output results/matched-replacement-v1/pets_b16_49
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_matched_replacement.py'
```

The default source is `runs/core-protection-v1/<setting>`. Evaluation saves masks,
explicit removals/additions, actual counts, unique output features, cosine scores,
reused-feature identities and runtime. Analysis saves paired per-row/per-image
classification and exchange results, original per-seed/per-random-seed outcomes,
actual replacement counts, and the eight-comparison interval summaries.
