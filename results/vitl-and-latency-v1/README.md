# ViT-L/14 validation and measured inference efficiency

This milestone completes the proposal's targeted cross-model validation, inference-efficiency measurements and supervision/training cost accounting. It also adds a numerical-stability experiment motivated by the timing preflight. Official test images and reserved test class combinations remain unused.

## Cross-model findings

Frozen CLIP ViT-L/14 retains 128 of 256 patch tokens after block 12 of 24, matching the half-token, middle-block setting of B/16 at 98/196. Each dataset uses fresh supervision and six final-step selectors: conditioned/image-only pairs at seeds 2041, 2042 and 2043. Width, architecture, updates and supervision are matched within each pair. All 5,720 training/validation images reproduce native CLIP exactly when all tokens are retained.

| L/14 result | Oxford-IIIT Pets | Flowers102 |
| --- | ---: | ---: |
| Conditioned − image-only accuracy, percentage points | −0.0126 [−0.0763, +0.0392] | +0.1471 [+0.0245, +0.2859] |
| Conditioned − image-only geometric margin | −0.000638 [−0.001006, −0.000270] | +0.000222 [−0.000163, +0.000603] |
| Correct − exchanged task descriptor, margin | +0.000361 [+0.000272, +0.000455] | +0.001138 [+0.001003, +0.001272] |
| H2 difficulty-adjusted slope per 10° | +0.000228 [+0.000137, +0.000323] | +0.000579 [+0.000389, +0.000771] |
| Held-image incremental R² | +0.009865 [+0.001915, +0.017745] | +0.009725 [+0.003262, +0.016250] |

Intervals are 95% image-cluster bootstrap intervals, averaging the three training seeds before resampling. Classification weights tasks equally and classes equally within each task; exchange/H2 analyses weight images equally. Within-image fixed-effect H2 slopes are also positive on both datasets. Held-image R² intervals resample fixed out-of-fold predictions; they do not refit every bootstrap model.

**H2 generalizes more consistently than the classification advantage.** Flowers reproduces a small accuracy benefit, while Pets does not reproduce the positive B/16 half-budget margin effect. Both backbones change patch size, depth, width and text embeddings together, so this is a cross-backbone check rather than an isolated causal test of model size.

[Pets details](../pets-vitl128-v1/README.md) · [Flowers details](../flowers-vitl128-v1/README.md) · [Cross-model plot](cross_backbone_validation.png) · [Plot source table](cross_backbone_plot_table.csv)

## Complete inference measurements

Seven settings cover B/16 Pets at 49/98/147 tokens, B/16 Flowers at 49/98, and both L/14 datasets at 128. Measurements include native full CLIP, attention selection, image-only selection and task-conditioned selection. Learned methods retain all three seeds.

On one RTX A6000, each setting uses a fixed 128-image panel and eligible unseen class pairs, batches 1 and 32, ten warmups, and five rounds of twenty timed calls with randomized method order. The complete GPU path starts from preprocessed GPU-resident images and includes the model prefix, task descriptor, selector, ranking/gather, suffix, normalization and candidate scores. Class text is cached for every method; image prefixes are not cached. This measures one candidate-set query per independently supplied image. Repeated queries on the same cached image embedding are a different workload.

| Conditioned setting | Batch 32 median ms/batch | Median paired-round speedup |
| --- | ---: | ---: |
| Pets B/16, 49 | 16.992 | 1.473× |
| Pets B/16, 98 | 20.032 | 1.256× |
| Pets B/16, 147 | 23.242 | 1.080× |
| Flowers B/16, 49 | 17.066 | 1.473× |
| Flowers B/16, 98 | 20.037 | 1.255× |
| Pets L/14, 128 | 75.722 | 1.287× |
| Flowers L/14, 128 | 75.490 | 1.286× |

Half-token conditioning yields approximately 1.25–1.29× speedup at batch 32. **Batch 1 does not accelerate**, and some settings have substantial temporal variability. Speedup is computed from same-round paired method/full-model medians before aggregation; it need not equal the ratio of pooled latency medians. All rounds and seeds, including slow rounds, remain in the summaries.

The 70 measurement windows showed no other GPU compute process in their before/after snapshots. CPU/I/O resources remained shared, and snapshots are not continuous monitoring. CPU preprocessing and transfer measurements are recorded separately. Timing quantiles are not confidence intervals. The independent [timing audit](timing_audit.json) reproduces 3,360 summary values from raw records.

[Accuracy–latency plot](latency_accuracy.png) · [Summary](latency_summary.json) · [Table](latency_table.csv) · [Per-seed table](latency_seed_table.csv)

## Numerical follow-up

Same-batch implementation parity was exact in the preflight. Changing batch shape in the FP16 inference path nevertheless changed some near-boundary token choices. We therefore compared the original cached evaluation with live batch-1 and batch-32 inference on all validation images and unseen class combinations, for all seven settings and three paired seeds.

The largest change in the conditioned-minus-image-only effect was **0.01634 percentage points in accuracy** and **0.00002821 in geometric margin**. Some individual predictions changed; this was not an exactly invariant computation. Across all seven settings and both metrics, the ordinary 95% interval sign/zero pattern was unchanged. These sensitivity intervals are diagnostic and remain separate from the original budget-adjusted main analyses.

[Numerical plot](numerical_sensitivity.png) · [Effect table](numerical_sensitivity.csv) · [Context table](numerical_sensitivity_context.csv) · [Full summary](numerical_sensitivity.json)

## Costs, validation and evidence

L/14's twelve pipeline stages total 4,402.50 seconds of process wall time. The GPU-using stages correspond to 1.213 GPU-stage wall hours, including loading, CPU work and synchronization; this is not active GPU-kernel time or billed compute. The [cost ledger](experiment_costs.json) covers 35 selected historical/current stages and explicitly identifies reused supervision, per-run details and excluded timing/audit stages. It is not a project-lifetime GPU total. The seven numerical-audit execution summaries total another 928.74 seconds, recorded separately.

Both L/14 result sets passed independent scientific audits, and 117 server tests passed. Small evidence is tracked here and in the dataset result directories. Five scientific archives, totaling 3,204,072,261 bytes, preserve supervision, predictions, all twelve new selectors, timing records and numerical-audit predictions. The published [release](https://github.com/NathanTong0921/ViT-token-compression/releases/tag/vitl-and-latency-v1) accompanies local retained archives and original server run files. Regenerable frozen prefixes, public datasets and public CLIP weights are not duplicated in the release. See [archive verification](local_archive_verification.json) and [asset inventory](release_assets.json).

The models-and-records archive was independently downloaded from GitHub. All 209 ZIP members passed integrity checks; all twelve recovered selector checkpoints strictly loaded and produced finite, correctly shaped forward outputs, with the expected final training steps. This [recovery test](recovery_verification.json) checks backup integrity and model executability, not a new reproduction of experiment accuracy. Transfer-only archives and the temporary server timing checkout were [cleaned](transfer_cleanup.json); retained scientific archives and original run files remain available.

## Interpretation and next step

This is positive progress toward the proposal: H2 now has cross-dataset/backbone support, the classification comparisons are complete for the planned development settings, and the efficiency claim has a measured operating range. It does not establish universal superiority of conditioning. I10–I11 in the [insight log](../../docs/research-insights.md) record the new evidence and its limits.

Next, freeze the final evaluation protocol before using reserved test images and class combinations. Then confirm the core findings and complete the four-page report, slides and video. The [execution record](../../docs/experiments/2026-10-27-vitl-and-latency.md) and [fixed protocol](../../docs/vitl-and-latency-validation.md) retain the complete study scope.
