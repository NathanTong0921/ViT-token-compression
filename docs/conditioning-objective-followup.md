# Conditioning potential and the training objective

Protocol fixed on 2026-10-28 before this follow-up's diagnostic predictions and new training outcomes. This extends proposal H1/H3 and insights I2–I3, I6 and I9–I10. Existing validation findings motivate the design; official test images and reserved test class combinations remain untouched.

## Research question

Why can a selector use task information beneficially, yet fail to outperform a strong image-only selector? We distinguish limited task-specific selection potential from imperfect realization of that potential, then test whether the supervision objective changes the learned advantage.

The first stage uses Pets B/16 at 49 and 98 patches, two previously observed settings with different margin effects. The intervention uses both Pets budgets and repeats the same fixed design on Flowers102 at 49 and 98 patches. L/14 is a possible targeted follow-up after these results, not an automatic repetition of the full matrix. No new token budget or pruning position is searched.

## 1. Potential and realization on the same task pairs

For one image and two tasks A/B, let q_A(S) and q_B(S) be true-class-oriented, normalized classification margins measured after keeping set S. Using the existing common candidate library, define:

- O_T = [max_S q_A(S) + max_S q_B(S)] / 2: separate choices in the library.
- O_S = max_S [q_A(S) + q_B(S)] / 2: one shared choice in the same library.
- C: the mean margin of the learned conditioned choices on their own tasks.
- U: the mean margin of the learned image-only choice on both tasks.

The identity C − U = (O_T − O_S) − [(O_T − C) − (O_S − U)] describes the library specialization gap and the difference in approximation gaps. Compute every quantity on identical image/task pairs, averaging the three seeds before image-equal aggregation. The existing augmented library includes the learned choices; verify nonnegative component gaps and the exact identity. This decomposition is accounting, not a new theorem or independent mechanism evidence.

Both library searches know the label and the two tasks. O_S is an offline joint-task reference, not an image-only deployable selector. A small gap in a finite library does not establish a small gap among all possible masks.

### A separate predictive check

Exclude all learned/teacher masks and use only the 512 existing random probe masks for the predictive library gap. Recover missing Pets49 probe responses from the already saved supplement. No new probes or teacher updates are introduced. Split these 512 masks into two fixed 256-mask subsets with seed 2083 to describe stability of the estimated gap; this is secondary to the full probe library.

Predict the actual pair-averaged C − U margin, not the exchange loss. Fit each budget separately. The baseline uses the existing symmetric quadratic controls of the two full-model margins plus the class-direction angle; the extended model adds the probe-only specialization gap. Use five class-stratified held-image folds (seed 2060), fit preprocessing only on the training folds and report incremental held-image R². The probe gap is label-informed and evaluated offline, so this is an explanatory generalization check, not a deployment gate.

Report image-cluster bootstrap intervals (2,000 draws, seed 2084) for the gap association and incremental R²: ordinary 95% and two-budget-adjusted 97.5%, separately by metric. R² intervals resample fixed out-of-fold predictions. Images, not repeated task pairs or seeds, are resampling units. The augmented library gap is not used as a supposedly independent predictor, and a direct correlation of exchange loss with C − U is not primary evidence because both share the conditioned outcome.

## 2. Task information × supervision objective

Keep the existing KL runs as the fixed reference. Train one new objective for both selector arms:

| Objective | Image-only | Candidate-conditioned |
| --- | --- | --- |
| Full teacher-distribution KL | Existing final checkpoints | Existing final checkpoints |
| Teacher Top-k boundary ranking | New matched training | New matched training |

For distinct teacher utilities, the new per-example loss is the mean of softplus(s_j − s_i) over teacher-retained i and teacher-discarded j. It supervises which side of the retention boundary a token belongs to rather than fitting the entire teacher probability distribution. Teacher ties at the boundary are treated symmetrically; equal-utility pairs are ignored and a uniform teacher contributes zero ranking loss. Record the tie incidence and common selection diagnostics.

The teacher, candidate inputs and scorer architecture are unchanged. Ranking is still teacher-based supervision, not a direct classification loss. No hard Top-k or CLIP operation needs a training gradient. Actual quality is always measured by running the pruned CLIP, not by the ranking loss alone.

### Fixed training recipe

- Frozen CLIP ViT-B/16, block 6/12, budgets 49/196 and 98/196; reuse the existing prefix caches and budget-matched 512-mask ridge teachers.
- Same 348,033-parameter selector, hidden width 128, float32 selector training, temperature 1, AdamW learning rate 0.0003 and weight decay 0.01.
- Seeds 2041/2042/2043, same initial state and balanced task/class/image stream as the existing runs, batch 128 and full training-image coverage.
- Pets: 2,944 training images, 3,990 updates. Flowers: 1,020 training images, 3,978 updates. Use the final step, without choosing checkpoints from validation outcomes.
- Record both objectives' initial gradient norms on the first eight training minibatches without parameter updates; do not change the learning rate, rescale the objective or tune on validation results. These measurements contextualize optimization-scale differences under the fixed recipe.
- Record the new objective loss with its actual name. Compare common teacher KL, Top-k agreement and teacher utility diagnostics where available; do not compare raw loss magnitudes across objectives as evidence of quality.

This produces 24 new selectors across four dataset/budget settings. Existing KL runs, including all unfavorable results and the Pets147/L14 studies, remain unchanged. Preserve sampling audits, all three seeds, training curves, final weights, preparation/training/evaluation costs and per-example predictions.

## 3. Primary intervention analysis

On all available validation images under unseen validation class combinations, compute conditioned minus image-only accuracy and geometric margin for each objective, then the paired interaction:

[(C − U)_rank − (C − U)_KL].

Use the same task-equal, class-within-task-equal weighting as the original main tables. Align image/task/seed/arm rows exactly, average the fixed seeds, and use 2,000 paired image-cluster bootstrap draws with seed 2085. Report ordinary 95% and two-budget-adjusted 97.5% intervals separately for each dataset and primary metric. Within-arm objective changes, per-seed effects and teacher-agreement measures are secondary. Keep the pair/image-equal diagnostic decomposition separate from these task/class-balanced classification comparisons.

Both selectors improving equally would support a general objective improvement, not an extra conditioning benefit. Improved teacher selection agreement without improved actual CLIP outcomes would support investigating the teacher/decision gap rather than claiming a successful method. A null or adverse interaction is retained and does not trigger another loss/seed search in this round. Any later design change needs a new recorded rationale.

## Deliverables and decision

Deliver a potential-versus-realization figure, the held-image explanatory check, the four-condition interaction table/plot, common teacher-fit diagnostics, and a fixed-rule Flowers replication. Record what changed in the explanation, not merely whether the new objective wins. The study can establish sensitivity to the training objective under this recipe; it does not alone identify a unique causal bottleneck.

After reviewing the complete development evidence, freeze the final-test plan and confirm the selected central claims on reserved data. Update the insight log, execution record and TODO, preserve evidence locally/on the server/on GitHub, and report the relation to the proposal and the project's scientific progress.
