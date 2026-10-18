# Budget-context ranking: independent artifact audit

Date: 2026-10-18. Result: **all checks passed; no blocking discrepancy found**.

This audit independently reconstructed quantities from the saved NumPy arrays and CSV rows. It did not rerun the GPU forward passes or change experiment results. The reconstruction used explicit NumPy algebra rather than the experiment's fitting or comparison helpers.

## Bundles examined

| Bundle under `results/` | Images | Mask seed | Reconstructed task top-k selections, both arms | Held-out prediction rows | Exchange margin/correctness cells |
|---|---:|---:|---:|---:|---:|
| `pets-budget-ranking-smoke-v1` | 36 | 2031 | 792 | 792 | 26,040 |
| `pets-budget-ranking-v1` | 144 | 2031 | 3,168 | 3,168 | 104,160 |
| `pets-budget-ranking-replication-v1` | 36 | 2032 | 792 | 792 | 26,040 |

The previous `pets-selection-smoke-v1` and `pets-selection-v1` candidate banks were also read to reconstruct the common shared references. The smoke and replication panels contain the same images. The primary run reuses the smoke mask designs; the replication uses different designs. Recorded image and task splits are validation splits throughout. All saved keep-all identity differences and prediction disagreement counts are zero.

## Reconstructed checks

- **Mask designs:** each of the four contexts contains 512 distinct masks with exactly 192, 147, 98, or 49 retained patches. Each target context has 128 distinct held-out masks, disjoint from its own training masks. Image ordering agrees with the saved evidence.
- **Fitted models:** independently centered and standardized the masks, solved the ridge system with the recorded regularization value 0.01, restored coefficients to original mask units, and reconstructed intercepts. This covers every saved image and all 38 responses, including the representation-preservation response. Maximum absolute coefficient/intercept discrepancy was below `1.1e-12` across the bundles.
- **Selections:** independently projected coefficients to each task's signed geometric-margin direction, ranked patches, and matched every saved task top-k set exactly. All saved single-task and candidate-bank indices are sorted, distinct, and within 0–195; padded single-task positions use the recorded sentinel.
- **Independent prediction:** reconstructed raw MSE, centered MSE, and Pearson correlation from held-out mask responses. In the primary run, maximum discrepancies were `1.70e-9`, `4.63e-10`, and `1.59e-7`, respectively. All bundles passed absolute tolerances of `1e-8` for MSE and `1e-6` for correlation.
- **Actual exchange outcomes:** reconstructed geometric margins and correctness directly from saved candidate cosine scores, including class-index tie handling. Correctness agreed exactly; maximum margin discrepancy was below `3.8e-8`. Stored task embeddings reproduced their candidate cosine scores within the `1e-6` tolerance; measured maxima in the primary and replication bundles were below `2.1e-7`.
- **Common shared reference:** concatenated the previous, near-full, and budget-matched candidate banks into the same 192-candidate reference. Independently reconstructed mean-margin and max-min shared winners, separately selected task winners, and all six image-averaged margin/accuracy summaries. Agreement was within `1e-7` for all budgets and bundles. This is an expanded privileged reference, not the previous 64-candidate search at unchanged search cost.
- **Tables and uncertainty:** the primary and replication balanced-accuracy tables matched the saved task outcomes exactly. For each of all three bundles, independently reconstructed 39 paired-effect or held-out-prediction means and 95% intervals by averaging within image and then resampling images 1,000 times with the recorded seed 2030. Counts, means, and interval endpoints agreed within `1e-12`.

## Interpretation retained

The audit establishes internal reproducibility of these development artifacts. Label-informed ranking and the privileged shared search remain diagnostics, not deployable predictions. Reusing a panel with a different mask seed tests sensitivity to the sampled mask design; it is not an independent-image generalization result. Improved prediction on random held-out masks and improved performance of the selected top-k sets are separate outcomes and should be reported separately.
