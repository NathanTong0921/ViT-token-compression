# Hybrid supervision: does lower KL require better token ranking?

Fixed on 2026-11-04 before computing this diagnostic. This follows I28 and uses
only the existing validation outputs: Pets49, Flowers49, Flowers98; seeds
2041–2043; task and image residual arms under original and hybrid supervision.
The frozen shared U is an additional reference. No selector, CLIP model or new
training target is trained or generated. Reserved final-test data remains unused.

## Three probability evaluations, identical token selections

For logits z and the same saved hybrid mask S of size k, evaluate

`KL = logsumexp(alpha * z) - alpha * mean(z[S]) - log(k)`.

1. Raw logits, alpha=1.
2. One positive scale shared by all examples of a model.
3. Subtract each row mean and divide by its standard deviation, then fit one
   positive scale shared by all examples of that model. This secondary analysis
   removes per-example amplitude variation as well as a global scale mismatch.

Positive scaling and row-wise centering preserve the complete token ranking.
The selected sets must match their saved counterparts exactly after both
transformations. CLIP classification outputs consequently remain unchanged;
there is no new backbone inference. Uniform probabilities give the limiting
reference KL log(196/k), not a new selection rule.

## Calibration and aggregation

Use five image-disjoint folds: sort unique image IDs, shuffle with seed2341,
assign cyclic folds0–4, and keep all tasks of an image together. The same folds
are used for every seed, model and both Flowers budgets. Fit alpha on four folds
and evaluate on the fifth. Alpha is constrained to [1e-4, 1e4]; report any boundary
solution. Only hybrid mask targets and saved logits enter calibration. Original
task/class weights are retained and normalized within the calibration subset.

Combine held-fold predictions before aggregation, preserving the original
equal-task/equal-class weights. Average three fixed seeds first, then bootstrap
images2000 times (seed2342). Intervals condition on the fitted cross-fold scales;
they do not repeat calibration fitting in each bootstrap draw. Report ordinary
intervals and six-setting/arm adjusted intervals separately for each contrast.
These are development diagnostics, not final confirmation claims.

## Planned contrasts

For each setting and arm, let old/new refer to original-/hybrid-supervised models
and let cal refer to the out-of-fold global-scale losses:

- Raw reduction D = L_old_raw - L_new_raw.
- Remaining difference R = L_old_cal - L_new_cal.
- Scale-removable difference C = (L_old_raw - L_old_cal)
  - (L_new_raw - L_new_cal). The identity D=C+R is checked per row.
- Report C/D as a ratio of aggregated effects when D>0, without clipping to
  0–100%. Also report L_old_cal - L_new_raw: whether recalibrating the old model
  can match or beat the new model's uncalibrated KL.

Repeat the decomposition with row-standardized calibration as a secondary
contrast. Neither remaining difference is a pure ranking effect: relative score
spacing can change while ranks stay fixed. This is a controlled probability
diagnostic, not causal attribution of the training dynamics.

Save all fold scales, row-wise losses, logit standard deviations, old/new rank
agreement, hybrid-mask overlap and per-seed summaries. Reproduce prior raw KL
and overlap first. The result will determine whether a new ranking/set objective
has a concrete motivation; no new training is automatically included this round.
