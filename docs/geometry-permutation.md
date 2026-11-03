# Within-image task–selection correspondence diagnostic

Protocol fixed on 2026-11-03 before generating new permutation outcomes. This is the final bounded development-stage extension of proposal H2, motivated by the distinction between its algebraic angle factor and empirical alignment. Existing positive development associations are already known. Reserved test images and test class combinations remain unused.

## Question and scope

Does the difficulty-adjusted alignment–angle association under the true task–selection correspondence exceed the association obtained by assigning the same selected-image features to other tasks on the same image?

Use all seven existing full-coverage KL-conditioned H2 settings: CLIP ViT-B/16 Pets with 49/98/147 retained patches and Flowers102 with 49/98, plus ViT-L/14 Pets/Flowers with 128. Reuse the three fixed seeds 2041/2042/2043, saved normalized image embeddings, class text embeddings, and full-model embeddings. No model training, new supervision, new image inference, or checkpoint selection is required.

For each image retain exactly the unseen development tasks used in its existing H2 analysis. Include images with at least two tasks and all unordered pairs of these tasks. Pets has 616 eligible images and 4,315 pairs; Flowers has 1,020 images and 6,120 pairs. Confirm exact identity alignment against the saved H2 cases, across seeds and matched dataset settings.

## Geometry and fixed statistic

For tasks a and b sharing the image's true class, let u_a and u_b be unit-normalized class-separating text directions pointing towards that class. Let z_a and z_b be the corresponding saved image features, averaged in float64 over the three fixed seeds before taking their difference. Do not renormalize the averaged features.

The existing symmetric exchange loss satisfies

`L_ab = 0.5 (z_a-z_b) dot (u_a-u_b)`

`     = ||z_a-z_b|| sin(theta_ab/2) alignment_ab`.

The present outcome is

`alignment_ab = cos(z_a-z_b, u_a-u_b)`.

It removes the explicit displacement-magnitude and angle factors. The primary statistic is the existing weighted regression coefficient on `theta_ab / 10 degrees`, controlling for an intercept and the symmetric quadratic basis of the two uncompressed signed geometric margins: min, max, min², min×max, max². Give each image equal total weight and each pair within that image equal weight. Reproduce the previous normalized-alignment point estimate under the identity assignment before computing the new reference distribution.

Secondary descriptive statistics are the same coefficient with image fixed effects, and the image-equal mean alignment. Retain observed per-seed versions of these statistics, but the primary reference analysis uses the seed-mean feature displacement, consistently with the existing H2 diagnostic. For these observed-only per-seed summaries, alignment is defined only on pairs with nonzero displacement and direction difference (threshold 1e-12); report excluded pair identities/counts and recompute image-equal weights on the valid pairs. No per-seed permutation claims use this reduced support.

## Reference assignments

Draw 1,999 independent uniform task permutations for each image, using master seed 2094. Replace each feature z_t assigned to task t by z_pi(t); leave the task directions, angles, full-model margins, image identity, and class labels fixed. Use the same mapping for all three seeds and for every budget/backbone with the same dataset, image, and sorted task IDs. A stable keyed random generator makes mappings independent of processing order. Fixed points, the identity, and repeated assignments are allowed; this is not a derangement test.

Each draw retains every saved selection-derived feature exactly once within its image. Because the task-pair graph is complete, it preserves the multiset of all within-image feature-displacement lengths, while changing their correspondence with the task directions. If a lookup stores only unordered feature pairs, reverse its sign whenever the permuted ordered pair reverses.

Existing geometry records show no zero or near-zero mean-displacement pairs at 1e-12 or 1e-6 in any setting. Verify that every eligible seed-mean feature and task-direction difference has norm greater than 1e-12. If this fails, stop and document the input discrepancy before interpreting results; do not introduce an outcome-dependent exclusion rule. Whole-graph permutations then preserve the valid support and permit fixed regression weights/design.

Input-stage clarification, 2026-11-03, before any permutation outcomes: the first extraction stopped while preparing an auxiliary seed summary. The existing Pets49 records contain one zero-displacement pair for seed 2042 (image `havanese_151`); all seed-mean pairs are valid. The observed-only per-seed rule above follows the existing normalized-alignment definition. The seven primary populations, statistics and reference permutations are unchanged; preserve the stopped input-stage log.

## Reporting

For each of seven primary statistics T, compute the upper reference-tail value

`p_ref = (1 + count(T_permuted >= T_observed)) / 2000`.

Report all observed values, reference means and central 95% ranges, exceedance counts, raw reference-tail values and `min(1, 7*p_ref)`. A primary result exceeds the reference after adjustment when the adjusted value is at most 0.05 and the observed slope is positive. The range of randomized statistics describes the reference distribution; it is not a confidence interval for the empirical slope. Report the two secondary statistics separately without using them to replace an unfavorable primary result.

This is a reference randomization of already observed development outputs, not randomized treatment assignment or an independent test of the original H2 hypothesis. Its correspondence null is stronger than the bare claim of zero angle slope. A positive result supports task-specific correspondence beyond the chosen scrambled reference; it does not establish that angle causes the effect, identify a unique semantic mechanism, or predict conditioned-minus-wide-image-only accuracy gains.

## Validation and deliverables

Meaningful numerical tests cover pair orientation, complete-graph preservation, seed averaging, mapping reproducibility, direct versus fast weighted regression, and exact small-task enumeration. An independent implementation will reconstruct compact panels from raw predictions, compare identity geometry with existing saved cases, check shared mapping behavior, and recompute reference statistics.

Save the reusable per-image/task feature panels, observed pair-level measurements, all randomized statistics, summaries and figures. Record source versions, input locations, commands, wall time, and data scope. Keep compact evidence in Git and larger essential archives in a GitHub Release and locally; the server retains the original source predictions. Update the insight log and TODO with results and their relation to H2.

After this analysis, finalize the reserved-data evaluation protocol and confirm the core findings. This round does not extend the selector architecture, optimize new losses, or use reserved data.
