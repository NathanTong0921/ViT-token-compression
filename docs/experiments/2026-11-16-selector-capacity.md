# Selector capacity and the value of candidate conditioning

## Research question

Does the advantage of candidate-conditioned selection change as the total selector parameter budget grows? The existing C versus wide-U comparison covers one capacity. This study adds a smaller and a larger capacity, keeping the model family, token budget and training objective fixed.

Use frozen CLIP ViT-B/16, selection after block 6, 49 retained patches plus CLS, and binary tasks on Pets and Flowers. C reads patch/CLS features and the unordered candidate-text descriptor; U allocates its parameters to patch/CLS processing. This evaluates the value of these alternatives across a limited capacity range, rather than optimizing arbitrary text/image branch allocation ratios.

| Capacity | C hidden width / parameters | U hidden width / parameters |
| --- | ---: | ---: |
| Small | 64 / 171,713 | 104 / 172,409 |
| Medium | 128 / 348,033 | 199 / 347,399 |
| Large | 256 / 725,249 | 378 / 726,163 |

Parameter differences within each pair are below 0.41%. Use seeds 2041, 2042 and 2043 throughout. Reuse the six medium-capacity checkpoints per dataset; train the 24 small/large endpoints. Re-evaluate every endpoint through the same FP32 native token-deletion path, batch size 32.

## Fixed training and evaluation

Reuse each dataset's existing prefix cache, budget-specific utility targets, image/task splits and full-coverage KL training procedure. AdamW learning rate 0.0003, weight decay 0.01, batch size 128 and temperature 1 remain unchanged. Pets uses 3,990 updates, Flowers 3,978. Sampling streams are paired across capacities and arms for each seed. Use the fixed final step. Direct-classification continuations are separate evidence and are not mixed into these capacity curves.

Record train/validation KL and endpoint fit alongside actual classification. Capacity may improve fitting without improving classification, so these measurements answer separate parts of the question. Keep the original step budget for every capacity; describe conclusions for that fixed training recipe.

Evaluate unseen development tasks for diagnostics and the existing test image/task panels for the primary capacity comparison. All model settings are fixed before evaluation. The test panels have been used for earlier models; this is a new fixed-model comparison on those panels. No data download, teacher regeneration or checkpoint selection is planned.

## Main comparisons

Let G(P) be C−U at capacity P. The primary contrast is G(large)−G(small), separately for Pets and Flowers. Analyze these two dataset contrasts as one family. Report the medium point and the small-to-medium and medium-to-large changes as secondary comparisons.

Accuracy is the primary metric. Raw cosine gap, geometric margin and cross-entropy accompany it, with full endpoint values, actual parameter counts and all three seeds. Average the fixed seeds, retain task/class-balanced weights, and use 2,000 paired image-cluster bootstrap resamples with seed 2721. Primary intervals adjust across the two datasets. The main research conclusion uses the test panel; development results describe fitting and transfer, not a separate opportunity to select a trend.

A rise in both C and U accuracy is not itself evidence of a conditioning-by-capacity interaction. Directly compare their changes. Three capacities can establish a trend or reversal within the studied range, not a universal scaling law. An absence of a clear interaction would support retaining the earlier single-capacity comparison without a new capacity-dependent design claim.

## Outputs and parallel evidence review

New run outputs go to `/data4/zhao.tong/experiments/selector-capacity-v1/`. Code and reused caches remain at `/data2/zhao.tong/experiments`. Curated analysis and figures go to `results/selector-capacity-v1/`; checkpoints and full native scores are backed up on data4 and GitHub Releases.

In parallel, review the already completed I36 content-based benefit prediction for scientific value, input availability and evaluation scope. No new prediction models or H2 experiments are part of this round. H2 retains its role as evidence about cross-task exchange sensitivity; I36 concerns a different target, C−U benefit. The previous choice not to include I36 in the report stays in place until its role is reconsidered using this review.

## Completed results

All 24 new fits and 72 development/test evaluations completed on 2026-11-16. Training streams match the reused middle-capacity runs. Re-evaluated Flowers middle-capacity scores and masks match the existing FP32 evaluation exactly. Total process time across two GPUs was 1,268.72 seconds (0.3524 GPU-process hours), including 751.48 seconds of optimizer updates; prefixes and supervision were reused.

On the test panels, the C−U accuracy differences at small/medium/large capacity are +0.0669/−0.0360/−0.0388 pp for Pets and −0.0331/+0.1521/+0.0865 pp for Flowers. The primary large-minus-small interactions are −0.1057 pp [−0.2092,−0.0100] and +0.1195 pp [−0.0836,+0.3427], with two-dataset-adjusted intervals. Their three-seed values are −0.0670/−0.2511/+0.0011 pp and +0.1902/+0.2835/−0.1150 pp.

Flowers' mean C−U geometric margin rises from −0.000520 to +0.001144; its large-minus-small change is positive in all three seeds. Both arms fit the training teacher better as capacity grows, while development KL and classification do not follow a common pattern. The result extends the resource comparison across capacities without establishing a shared accuracy scaling rule. Tables, paired outcomes, fitting histories and figures are in [selector-capacity-v1](../../results/selector-capacity-v1/README.md).

The [I36 review](../i36-evidence-review.md) found useful partial out-of-sample prediction but no established routing accuracy gain. It remains candidate supporting material, with the earlier report-exclusion decision retained. No new prediction or H2 run was performed.
