# Three token budgets and H2

Protocol fixed on 2026-10-27 before generating or inspecting new 98/147-token results. The completed 49-token coverage experiment motivates this comparison; its outcome is already known. This is development analysis, not final-test confirmation.

## Questions and fixed training comparison

At a single pruning boundary after CLIP ViT-B/16 block 6, compare retaining 49, 98 and 147 of 196 patches. Test whether the conditioned selector improves over the matched image-only selector at each budget, and whether text-direction angles predict exchange losses beyond full-model task difficulty (H2).

Use all 2,944 training images, all 736 development images, the existing 532 training pairs and 66 development pairs. The primary unseen-combination benchmark contains 716 eligible images and 2,624 image/task rows; 616 images support 4,315 unordered task-pair exchanges. Keep the 3,669 official test images and 68 test pairs unused. Known classes with unseen combinations do not constitute unseen-class generalization.

Reuse the exact full-coverage 49-token checkpoints, reference predictions and learned-selector predictions from `pets-selector-scale-v1`. For each new budget generate fresh 512-mask, budget-matched ridge supervision (ridge 0.01, base mask seed 2031, budget-specific seed derivation as before) for all 3,680 training/development images. Reuse the frozen prefix and full-model evidence, which do not depend on retained token count. Validation teachers are used only for diagnostics. Each budget has its own trained selectors; applying the 49-token ranking at other budgets would answer a different transfer question.

Keep the same 348,033-parameter architecture, KL soft-utility objective, temperature 1, AdamW learning rate 0.0003 and weight decay 0.01. Use seeds 2041/2042/2043, 3,990 updates, batch size 128 and the same balanced task/class/image sampling algorithm. Use the final update for every model. No seed replacement, early stopping, alternative objective or hyperparameter search is included in this comparison. These design choices establish a common baseline rather than optimized performance at every budget.

## Main classification analysis

At each budget compare the full model, attention, random selection, image-only selector, conditioned selector, and the label-informed task teacher. The last is an offline diagnostic. Random selection uses three independent seeds 2055/2056/2057, with a fixed per-image random ordering shared across budgets (nested retained subsets). It is computed once per image, exactly reused across tasks, and averaged across those random seeds.

Primary estimands are conditioned-minus-image-only accuracy and signed geometric margin on unseen development combinations. Equal task weights and equal class-side weights define the benchmark; images within each class-side are uniform. Average the three fixed training seeds before image-cluster bootstrap (2,000 replicates, seed 2052). Report per-seed results, paired 95% intervals, and 98.333% Bonferroni intervals across the three budgets separately for each primary metric. Intervals are conditional on these tasks and checkpoints. Cross-budget changes, compression-induced errors/repairs, seen pairs and exchange effects are secondary descriptions.

Retain all per-example scores, selected positions and normalized 512-dimensional CLIP features. Report experiment compute separately from deployment latency; cached prefixes and per-image reuse do not establish actual inference speedup.

## Strong shared offline diagnostic

For each eligible development image and pair of tasks, form one common candidate library: the 512 probe masks at that budget, the two task-teacher masks, attention, the three image-only masks, and both tasks' conditioned masks for all three seeds. Every candidate has measured class scores, and both task-specific search and shared search use this same library. Search separately for each task, or select one mask maximizing the mean signed margin across the two tasks. Save chosen positions and outcomes, library provenance, both-correct feasibility and the cost in average margin of requiring a shared mask.

The 49-token cache lacks stored probe responses for 144 old development images; regenerate these exact mask responses in a separate supplement, without altering existing teachers or predictions. A common library and complete availability avoid changing search strength between budgets merely because evidence was missing.

These searches know the label and both tasks. They are diagnostic references, not deployable methods. Separate search weakly dominates shared search in the optimized mean margin by construction; a positive gap alone is not evidence of a learned contribution or a global optimum. The task teacher and actual trained models remain separately reported. This new library differs from the earlier 192-candidate pilot library, so their values are not directly interchangeable.

## H2 estimand and algebraic control

For two tasks sharing true class y, let u_A and u_B be unit-normalized class-separating text directions; theta is their angle. Let z_A and z_B be the normalized full-dimensional image features produced by the two task-dependent selections. Define symmetric exchange loss

`L = [(m_AA - m_BA) + (m_BB - m_AB)] / 2`

where the first subscript is the selection task and the second is the classification task. Then exactly

`L = 0.5 (z_A - z_B) dot (u_A - u_B)`

`  = ||z_A-z_B|| sin(theta/2) cos(phi)`.

This identity includes an angle-dependent factor even without a new learned mechanism. Consequently, raw angle/loss correlation is insufficient to claim a mechanism.

The primary H2 population is the common 616-image unseen task-pair panel for the full-coverage conditioned selector. Average the three seeds within each image/task pair before estimation; weight each image equally, then each of its eligible task pairs equally. Preserve per-seed fits. For seed-mean decomposition use the mean feature displacement, explicitly as a diagnostic of mean loss, not a deployed ensemble.

Fit a fixed low-order difficulty model: intercept, the smaller and larger full-model signed geometric margins, their squares, and their product. Add theta/10 degrees. Report the adjusted angle slope with image-cluster bootstrap intervals (2,000 draws, seed 2061), both 95% and 98.333% across the three budgets. A positive multiplicity-adjusted interval supports an association for that budget; inconsistent signs or intervals including zero do not establish it.

Two prespecified checks limit interpretation: (1) image fixed-effects sensitivity, demeaning outcome and predictors within each image; (2) five-fold held-image prediction (seed 2060, common folds across budgets), comparing difficulty-only versus difficulty-plus-angle weighted MSE and incremental R-squared. These folds assess prediction within the fixed development task collection, not final-test generalization. Disagreement between pooled, within-image and held-image results must be reported.

Mechanism diagnostics separately examine displacement magnitude, the deterministic sin(theta/2) factor and normalized alignment cos(phi). Report exact identity residuals and the zero-displacement image-only null. Alignment is defined only for nonzero displacement/direction difference (threshold 1e-12), with a prespecified displacement-threshold sensitivity at 1e-6 and explicit retained counts. Alignment regressions are exploratory; they cannot turn an algebraic identity into a causal claim.

## Completion and evidence

Complete the three-budget comparison, strong shared diagnostic and H2 regardless of result direction. Record current source versions before runs, supervision/training/evaluation costs, independent numerical checks and all seeds. Preserve small tables, figures, configurations and run records in Git; retain essential supervision, checkpoints and raw predictions locally and on the server with a GitHub Release copy and recovery verification. Keep recomputable prefix caches server-side. Update the insight log and TODO only after inspecting the evidence. Flowers102, targeted ViT-L/14, actual inference timing and final testing remain subsequent required stages.
