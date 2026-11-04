# Does pruning depth change the value of task conditioning?

Fixed 2026-11-04 before new supervision or training. This is the next bounded
H1/H3 study after I27–I30: a positive average hybrid gain was found on Flowers49,
but simple selection statistics did not yield a reliable cross-setting benefit
predictor. The question is whether the stage at which selection happens changes
the incremental value of candidate-class information under the established
training recipe. This round does not change the selector objective.

## Scope and intervention

Use frozen CLIP ViT-B/16 on the existing Pets and Flowers train/development
splits. Retain 49 of 196 patch tokens, plus CLS, immediately after block 3, 6 or
9. All three positions and both datasets are fixed before results. Block 6
reuses the existing C and wide U checkpoints and raw evaluated predictions;
blocks 3 and 9 each receive newly computed prefixes and depth-matched teachers.

C is the conditioned width-128 network (348,033 parameters); U is the strong
image-only width-199 network (347,399 parameters). Train each architecture with
seeds 2041/2042/2043 at each new depth: 2 datasets × 2 new depths × 2 arms × 3
seeds = 24 new selectors. Keep KL, the 512-mask teacher design, ridge .01,
temperature 1, AdamW .0003/.01, batch 128, and final checkpoints. Pets uses
3,990 updates; Flowers uses 3,978. Same-architecture initialization and balanced
task/class/image streams are shared across depths. No validation selection.

Keep manifests, image order, preprocessing, prompts, text embeddings, 512
sampled masks and mask seeds unchanged. Rebuild suffix responses and ridge
coefficients at each new depth; earlier-depth prefixes or teacher coefficients
must not enter later-depth training. Prefix caches, configuration and checkpoint
depths are checked explicitly. Keep-all must reproduce the native full model
and match the existing image/text identities. Use the established FP16 CLIP,
FP32 selector, batch-64 suffix and stable spatial token ordering.

## Endpoints and comparisons

Primary endpoint is the signed geometric classification-margin difference
Delta_d = (C−U)_d. The two primary contrasts in each dataset are
Delta_3−Delta_6 and Delta_9−Delta_6, giving four primary comparisons. We test
these paired differences directly; differences in significance across depths do
not establish an interaction. There is no presumed monotonic direction.

Secondary endpoint is (CU−UU)_d using all six ordered distinct-seed pairs.
Use the existing half-exclusive-position replacement rule: retain intersection,
replace half the recipient-only positions from its lowest rank with the donor's
highest-ranked exclusive positions. Compare C_s + U_t against U_s + U_t under
the same rule. The same two depth contrasts give four secondary comparisons.
Only these five pair endpoints are evaluated: C_s, U_s, U_t, CU, UU. No new
core-protection fractions, fixed-d grid or diagonal pairs are added.

Save absolute C/U/CU/UU performance and the full-model reference at every
depth. Report accuracy and raw signed cosine-score gap alongside geometric
margin; the class-text denominator is unchanged across depths for a row.
Geometric margin is primary; accuracy establishes practical classification
changes and raw gap checks sensitivity to geometric normalization.

Images, candidate pairs and seeds/pairs are explicitly aligned across depths.
Use the historical equal-task / equal-class-within-task weights. Average the
fixed seeds or six pairs first, then bootstrap whole images and all their rows
with shared resampling across methods/depths (2,000 draws, seed2361). Report
ordinary 95% and four-comparison-adjusted intervals for each endpoint/metric
family. Absolute depth effects use ordinary intervals. Seeds sharing a model
are not independent statistical samples. Reserved test data remains unused.

## Diagnostics and interpretation

Record final train/validation KL and teacher top-k recovery, standardized-target
boundary ties, and pre-standardization teacher utility std / fraction below
the existing 1e-6 floor. These come from existing depth-specific teacher arrays
and final checkpoints, without changing targets or choosing new hyperparameters.
Check training streams, mask banks and depth identities against the fixed design.

At fixed depth, C/U have matched training and near-equal parameter budgets.
Different depths have different total backbone computation; this is not an
equal-latency comparison. Depth changes representations, suffix length and
teacher responses. An interaction establishes depth dependence of the complete
recipe, not by itself an isolated information-propagation mechanism. The earlier
synthetic-object depth study used different tasks and diagnostic masks; this
study evaluates learned selectors on the original images.

If only absolute performance changes while C−U and CU−UU do not, the proposed
conditioning-specific depth explanation is not supported in this range. If an
interaction is found, report its direction and cross-dataset consistency before
designing any further mechanism intervention. Finish the fixed matrix and stop
this round; do not add positions, budgets or replacement seeds to seek an effect.

## Retained evidence

Commit protocol/code before execution. Save per-example all-class scores and
masks, selector logits, complete run configurations, final checkpoints, paired
row metrics, all contrasts and plots. Keep generation/training/evaluation costs
separate. Prefix caches and source data stay on the server; necessary weights,
teacher targets and replay evidence receive local/GitHub artifact backups.
Summaries/figures and execution records belong in Git. No direct-set objective
experiment is launched in this round.
