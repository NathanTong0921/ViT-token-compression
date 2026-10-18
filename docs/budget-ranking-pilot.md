# Does the ranking context need to match the compression budget?

This extends the proposal's offline ranking and supervision-generation step. In the previous development pilot, block-deletion task rankings worked better than attention at 147/98 patches but substantially worse at 49. One possible explanation is that importance estimated near the full image does not transfer to large joint deletions. This experiment tests that explanation; it does not assume it is true.

## Primary comparison fixed before outcomes

Use the existing 36-image smoke and 144-image development panels, original validation task split, frozen CLIP ViT-B/16 and block-6 boundary. Evaluate 147/98/49 retained patches. No test data is used.

Fit the **same additive response model** in two contexts:

1. Near-full context: 512 uniformly sampled masks, each retaining 192 of 196 patches.
2. Matched context: 512 uniformly sampled masks, each retaining exactly the evaluation budget (147, 98 or 49).

Each mask is evaluated by the actual frozen suffix. The response vector contains all 37 class cosine scores plus similarity to the full image embedding. Fit centered, column-standardized ridge regression with fixed regularization 0.01, then convert coefficients back to original mask coordinates. Standardizing is necessary because mask variance differs greatly across budgets. Fixed-cardinality masks have a constant-sum constraint; ridge identifies a representative coefficient vector, while common offsets do not affect rankings.

The mask design is content- and label-independent and shared across images for efficient paired comparison. Near-full probing is reused across the three target budgets. Equal probe counts mean equal numbers of suffix queries, not equal wall time: longer sequences cost more. All costs are recorded.

For each binary task, project the fitted class-score coefficients onto its true-class-oriented unit text direction. Keep the k highest coefficients, run the actual compressed model and evaluate predictions. A shared proxy averages both task-utility vectors. The image-only representation objective uses its fitted full-embedding similarity coefficients. All rankings are label-informed offline diagnostics where appropriate; no learned selector is claimed.

## Two independent checks

- **Predictive quality:** 128 independently seeded uniform masks at each target k, disjoint from that k's training masks. Predict their task margins using both fitted models. Compare correlation and centered MSE, averaging within images first. Raw MSE is secondary because an intercept shift under distribution change can dominate it. Held-out outcomes never select the top-k sets or regularization.
- **Selection quality:** actually execute each model's top-k set, compare the two contexts on the same images/tasks, and retain the earlier block-deletion method and attention for reference. Improvement of random-mask prediction does not guarantee improvement of the optimized top-k set.

The primary effect is matched-context minus near-full-context actual accuracy/margin. A gain supports a context mismatch in the previous ranking strategy. It does not by itself establish that conditioning is better than a strong shared selection.

## Shared references and exchange

Each fitted model is evaluated with the original 64-candidate bank construction. To compare against the same strong reference, the cross-arm analysis combines these two banks and the saved legacy bank into a common 192-candidate union, then selects the shared mean-margin and maxmin-margin winners. All arms see this same privileged reference. This union uses already computed scores, with no additional model calls. It is a larger bounded search than the previous 64-candidate reference, not an optimality guarantee or a deployable selector.

Preserve own-versus-exchanged selections, full-model difficulty controls and the H2 geometry qualification. Report the balanced 144-image single-task panel separately from the 124-image eligible exchange panel. Paired uncertainty resamples images and keeps the current tasks fixed.

## Decision and replication

First verify the 36-image pipeline, then run the preselected 144-image panel. If there is a signal, repeat the 36-image experiment with an independent probe-mask seed. If surrogate prediction improves but actual top-k selection does not, optimized selections may lie outside the sampled contexts or require token interactions; record that outcome. A failed proxy is not evidence that task information is useless.

Once useful supervision is demonstrated, start the matched conditional/unconditional selector trial promptly. If the shared reference already covers the apparent advantage, retain that finding and clarify which part of H1 remains unsupported.

## Numerical and evidence controls

Reuse the original validation image order and 64-image encoder batches; compare keep-all to native and saved outputs at the existing 1e-6 tolerance. Fixed-size suffix batches repeat their last row only to pad the batch, then discard padded predictions. Save probe masks/responses, fitted coefficients/intercepts, held-out prediction metrics, all selected positions and candidate outcomes. Record code/configuration and per-context computation before interpreting results. No persistent prefix cache is needed.
