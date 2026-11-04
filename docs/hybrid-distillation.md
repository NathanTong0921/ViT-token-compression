# Distilling a fixed shared/conditioned mixture

This predeclared follow-up keeps the existing zero-start residual architecture and
changes the target distribution. It asks whether the previously observed hybrid
selection benefit can be transferred into a small selector, and whether candidate
information helps beyond the same additional image-only budget.

## Fixed target and fits

For each seed2041–2043, reuse its frozen conditioned selector C and width199 shared
selector U. Retain their intersection, replace floor(|U minus C|/2) lowest-ranked
C-only positions with the same number of highest-ranked U-only positions, and
keep exactly k patches. This is the historical matched-replacement top50 rule,
not the earlier core-protection rule. Teacher construction receives image tokens
and both candidate class texts, with no true-class indicator. The frozen teachers
were themselves previously fitted to supervised utility targets.

The new target assigns probability1/k to each retained position and0 elsewhere.
Both residual arms minimize the **same softmax KL loss used previously**; the
supervision distribution changes. Absolute KL values across the original soft
utility and uniform mask distributions are not directly compared: their target
entropies differ. Report each fixed target's fitting separately and use mask
intersection/k and actual CLIP classification as common outcomes.

Pets49, Flowers49, Flowers98; task residual9,801 learned parameters versus image
residual9,850; same U347,399, zero output initialization, batch128, AdamW0.0003,
weight decay0.01, temperature1, and original3,990/3,978 steps. Three paired seeds
produce18 new branch fits. The historical18 original-target fits are reused.
Candidate sampling and original-target initial validation losses must reproduce
those prior fits. No validation checkpoint selection or new hyperparameter search.

Training masks use training images and training pairs only. Masks on the existing
held-image/held-combination validation split diagnose fitting and classification;
they do not optimize weights. Original test images and test pairs remain unused.
Frozen h and U scores are cached only for fitting. Evaluation runs the online
selector from CLIP prefixes and executes the chosen CLIP suffix.

## Evaluation

Primary: new-target task residual minus image residual, with three-setting adjusted
image-bootstrap intervals for margin and accuracy separately. Keep all seeds.
Secondary: target-by-conditioning interaction
`(task_hybrid − image_hybrid) − (task_original − image_original)`, with ordinary
paired95% intervals. Also retain each student's change from its original-target
counterpart, each student's difference from U and the hybrid teacher, and teacher
minus C/U. Images are resampled after averaging the three paired seeds.

Save initial/final training and validation hybrid-mask overlap and both target KLs
for the new and original fits. The teacher's validation masks are freshly generated
and compared exactly with historical matched-replacement masks, then its saved
CLIP predictions are reused. This verifies the actual teacher benefit instead of
assuming it survived the move into a new training pipeline. The ordinary residual
analysis also retains correct/swapped-task effects, changed-token counts, and
residual patch standard deviations.

If both image and task students capture the teacher benefit, task information may
be useful when constructing supervision without being needed in the deployed
selector. If only the task student benefits beyond equal image capacity, that
supports a task-conditioned method. Strong training overlap with weak validation
overlap identifies a transfer problem; good overlap without a downstream gain
indicates that reproducing this mask target has not yielded useful classification.
No outcome is selected before observing the full paired matrix.

## Run

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/run_hybrid_distillation.py \
  --setting flowers_b16_49 --device cuda:6 \
  --output runs/hybrid-distillation-v1/flowers_b16_49
PYTHONPATH=src:scripts:tests python3 -m unittest test_hybrid_distillation -q
```

Outputs include both target-fit diagnostics, training curves, final branch weights,
frozen-teacher paths, train/validation mask targets, raw predictions and the paired
interaction analysis. Costs separate new target generation, cached hidden features,
branch fitting and replay. Earlier teacher/U fitting is reused; this round is not
a fresh end-to-end training-cost or deployment-speed claim.
