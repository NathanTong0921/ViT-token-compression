# Direct supervision of a fixed retained set

Fixed on 2026-11-04 before new fits. This follows the I28/I29 finding that
hybrid-target KL can improve substantially without improving the retained set.
The question is whether directly optimizing set membership transfers the existing
hybrid teacher's benefit, and whether candidate information helps beyond the same
additional image-only parameter budget. The block-3/6/9 experiment is complete
and remains a separate intervention.

## One objective change, the same teacher and students

Reuse the existing block-6 hybrid mask S for each image/task/seed from
`runs/hybrid-distillation-v1`. Its rule, frozen C/U teachers and train/development
splits are unchanged. Both student arms start from the same frozen wide shared U
and a zero-output residual. The new loss compares directly with the historical
hybrid-mask softmax KL fits; it is not compared with a different teacher target.

For patch scores s, define z = (s − mean(s)) / max(std(s), 1e-6), using population
standard deviation and differentiating through it. For N=196 patches and |S|=k,
use the fixed cost-augmented set loss

```
L_set(s,S) = [sum TopK(z + 1[outside S], k) − sum_{i in S} z_i] / k.
```

The cost is fixed at 1 in standardized units, with no validation tuning. The
maximization selects exactly k positions. This is a structured set surrogate
with row normalization, not a claim to invent a new loss family.

Let P be the predicted top-k set. The maximization is at least its value at P,
and sum_P z ≥ sum_S z, so L_set ≥ 1 − |P intersect S|/k. The loss therefore
upper-bounds the fraction of missed target members. Away from the numerical
floor, adding a constant or multiplying all scores by a positive scalar leaves
the loss unchanged. This prevents the scale-only loss reduction studied in I29.
It does not guarantee better classification: multiple masks can retain useful
evidence, and reproducing this particular teacher is an intermediate outcome.

This differs from the earlier original-utility ranking experiment: that study
used all strict teacher top-k/complement pairs in an unnormalized softplus loss
and trained full selectors. Here the target is the fixed mixed selection, the
students are shared residuals, and the normalized objective operates on whole
sets. The intervention changes the complete objective; normalization also
couples token gradients.

## Fixed matrix and optimization

| Setting | Patches kept | Steps per fit | Seeds | New arms |
|---|---:|---:|---|---|
| Pets B/16, block6 | 49 | 3,990 | 2041/2042/2043 | task / image residual |
| Flowers B/16, block6 | 49 | 3,978 | 2041/2042/2043 | task / image residual |
| Flowers B/16, block6 | 98 | 3,978 | 2041/2042/2043 | task / image residual |

This is 18 new fits and 18 reused hybrid-KL fits. Task/image branches have
9,801/9,850 trainable parameters around the same 347,399-parameter frozen U.
Keep historical same-arm initialization, sampled image/task order, batch128,
AdamW learning rate0.0003, weight decay0.01 and final fixed-step evaluation.
No new depth, schedule, architecture, checkpoint selection or hyperparameter
search is included. All fixed seeds and settings are retained.

Record initial KL/set losses and branch-parameter gradient norms on the first
training minibatch, without changing weights. Do not adjust learning rates or
loss multipliers from validation outcomes. These diagnostics describe the
fixed-recipe objective comparison, rather than isolating a single component of
the optimization dynamics. Record score standard deviations and the frequency
of numerical-floor activation throughout diagnostics.

## Outcomes and paired comparisons

Two planned questions organize the results:

1. **Does the set objective improve held-out set recovery?** New minus historical
   hybrid-KL overlap, for each of three settings and both arms. Use a six-comparison
   adjustment. Report training and development overlap, exact-set recovery and
   set loss separately. KL is a shared-target diagnostic only.
2. **Does the task student improve actual classification beyond the image
   student?** Task minus image accuracy and geometric margin, with a separate
   three-setting adjustment for each metric. Also retain raw cosine gaps.

Secondary results: objective-by-conditioning interaction; each student's change
from its KL counterpart and difference from frozen U and the hybrid teacher.
Use ordinary paired 95% intervals, label them accordingly and retain all settings.
For classification contrasts report corrected errors and newly introduced errors;
their difference equals the net accuracy change. Report absolute accuracy alongside
margin. A smaller training loss alone is not a method improvement.

Use original equal-task/equal-class weights. Average the three fixed runs within
each image/task row before paired image bootstrap (2,000 resamples, seed2372).
Resample all tasks of each image together. Keep per-seed results and raw case
arrays. Development evaluation uses the existing unseen images and candidate
combinations; reserved final-test images and combinations remain unused.

## Execution, validation and stopping point

Reconstruct the saved target identity from image/pair metadata. Verify initial
shared predictions and sample order against the historical run. Unit checks cover
the set-error bound, positive-scale/shift invariance, constant-score gradients and
small paired training with a frozen base. Classification evaluation executes the
online selector from CLIP prefixes and then the actual pruned suffix.

Save final branch weights, per-row train/development fit diagnostics, raw scores,
masks, curves, sampling records and incremental costs. Teacher arrays and frozen
base weights are reused with explicit paths and prior backup references. Cache
only frozen features during training. This round does not establish a new latency
claim; any promising method would need complete inference timing later.

Complete the 18-fit matrix and stop. If overlap improves without classification,
the target-transfer bottleneck has narrowed but the method has not improved its
task outcome. If both arms improve similarly, the benefit is not specific to
candidate conditioning. If the task arm improves beyond image capacity and U,
that is evidence for the intended task-conditioned method. Further tuning or a
new teacher is a subsequent decision, not an automatic extension of this run.
