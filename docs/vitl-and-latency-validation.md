# ViT-L/14 targeted validation and measured inference latency

Protocol fixed on 2026-10-27 before ViT-L/14 model predictions and benchmark measurements. This stage tests whether the learned task dependence and H2 association persist with a second backbone, and whether token deletion produces actual inference savings after selection overhead.

## Questions and scope

1. At a 50% patch budget, does a conditioned selector improve classification over its matched image-only counterpart on unseen class combinations?
2. Does the class-direction angle predict task-descriptor exchange loss beyond uncompressed task difficulty on both datasets, including within-image and held-image checks?
3. How do accuracy and measured model latency trade off at the studied budgets, once selector, sorting and gathering costs are included?

Use standard OpenAI CLIP ViT-L/14 at its native 224-pixel resolution: 256 patches, 24 visual blocks, visual width 1,024 and output dimension 768. Prune after block 12 and retain 128 patches plus CLS. This preserves the 50% patch fraction and halfway pruning position used by the B/16 98-token condition. The focus is informed by the completed B/16 development results; it was not selected before those results. This is a targeted cross-backbone check, not a new full budget sweep or a final-test evaluation.

Patch size, depth, width, text representations and the number of fitted patch coefficients all change between backbones. The comparison assesses robustness across these configurations; it does not isolate a causal effect of model size or establish 128 as an optimal L/14 budget.

## Data and paired training

Reuse the existing image and task manifests without changing the splits. Pets uses 2,944 training images and 736 validation images, with 532 training and 66 validation class pairs. Flowers uses its official 1,020 training and 1,020 validation images, with the fixed 612 training and 204 validation pairs. Official test images and reserved test pairs remain unused for model training and predictions.

Create fresh L/14 frozen-feature caches and budget-specific supervision for both datasets. Check native CLIP against the prefix-plus-keep-all path on every training and validation image before pruning. Store all native output dimensions for later geometry. Each dataset's `source_cache_dir` is its own complete L/14 feature cache; the legacy old-panel flag therefore covers the whole panel and has no coverage-experiment interpretation.

Generate 512 distinct fixed-128 masks per image, with the existing seed derivation from base seed 2031 and ridge 0.01. Reuse each dataset's mask bank across its images. Training uses only training-image teachers; validation teachers provide diagnostic references and monitoring.

The actual 512-by-256 mask design has centered rank 255, the maximum permitted by its fixed row sum. Its positive covariance eigenvalues range from 0.09269 to 2.85138; ridge 0.01 gives a condition number of about 27.86 on the identifiable contrast subspace. Positive ridge fixes the constant-direction ambiguity. Adding a common constant to all raw patch utilities does not change rankings or the centered, standardized training targets. These checks support the numerical design, not the accuracy of an additive approximation to CLIP.

Train new L/14-specific selectors, rather than loading B/16 selector weights. Retain the same scorer design, hidden width 128, soft-importance KL objective, temperature 1, AdamW learning rate 0.0003 and weight decay 0.01. Adapt input normalization and projection dimensions to L/14; the paired arms must have identical architecture and parameter count within this backbone. The expected count is 480,641 per arm, compared with 348,033 for B/16.

Use seeds 2041, 2042 and 2043, paired initial states and identical example streams between conditioned and image-only arms. Preserve batch size 128 and the fixed final-checkpoint rule: Pets 3,990 updates with monitoring every 133; Flowers 3,978 updates with monitoring every 153. This produces six models per dataset, twelve new selectors in total. Record preparation, supervision, training and evaluation costs separately.

## Classification and H2

Evaluate native full CLIP, random selection (mask seeds 2055/2056/2057), attention selection, image-only, conditioned and the label-informed teacher. The teacher remains an offline diagnostic. The main comparison is conditioned minus image-only on validation images under unseen validation class combinations. Use the existing task/class-balanced metrics, average the three fixed seeds, and report accuracy, signed geometric margin, repairs/new errors and per-seed effects.

With one new budget per dataset, report ordinary 95% paired image-cluster intervals, with 2,000 draws and seed 2052. Analyze the datasets separately. These intervals do not claim simultaneous coverage of all metrics and auxiliary analyses. No cross-budget contrast is newly tested for L/14.

For H2, keep the actual classification task fixed while exchanging the conditioned selector's candidate-class descriptor on the same image. Average the same image/task-pair loss across seeds, then weight images equally and pairs equally within image. Use the existing symmetric quadratic controls of the two native full-model margins and an angle term in units of ten degrees. Use 2,000 image-cluster bootstrap draws with seed 2061 and one-budget families within each dataset.

A joint directional replication requires positive primary angle slopes with 95% intervals above zero in **both datasets**. Report image fixed effects, every seed's slope, and five-fold class-stratified held-image prediction (fold seed 2060), fitting preprocessing only in training folds. Report the incremental out-of-fold R-squared and its fixed-prediction image-bootstrap interval. Check the full-dimensional exchange identity and the exactly zero image-only exchange effect; retain normalized-alignment diagnostics as exploratory analysis. H2 concerns a predictive association, not a causal intervention on angle.

## Latency comparison

Measure the complete GPU model inference path from a preprocessed, GPU-resident image tensor to its two candidate-class scores. The timed path includes the visual prefix, selector and descriptor construction where applicable, dtype conversion, top-k, spatial sorting, gathering, visual suffix, feature normalization and pair scoring. Text embeddings are cached on the GPU for all methods. Record the model/selector dtypes and software environment.

Use native `encode_image` for the full CLIP baseline. Learned selectors need no attention-score recomputation, so their prefix may use `return_attention=False`; the attention method includes its actual scoring computation. Verify that any timing-path implementation retains the same selected positions and outputs as the evaluated model, subject to documented floating-point precision. No cached visual prefix, omitted selector or padded suffix batch is used to claim end-to-end speed.

The benchmark matrix is:

| Backbone / dataset | Retained patch budgets | Methods |
| --- | --- | --- |
| B/16 / Pets | 49, 98, 147 | Native full, attention, image-only, conditioned |
| B/16 / Flowers | 49, 98 | Native full, attention, image-only, conditioned |
| L/14 / Pets | 128 | Native full, attention, image-only, conditioned |
| L/14 / Flowers | 128 | Native full, attention, image-only, conditioned |

Time all three trained seeds for learned methods; deterministic baselines need one model configuration. Use batch sizes 1 and 32. Before timing, select a fixed pool of 128 validation images per dataset: one per class, then fill the pool without replacement using seed 2081. Assign each image one valid unseen pair containing its class using the same seeded RNG. Pool and pair choices do not use outcomes and are shared across methods and backbones.

Use ten warmup calls per configuration, followed by five rounds of twenty measured calls. Interleave configurations using seed 2082. Batch-1 measurements use 100 unique images from the fixed pool; batch-32 measurements cycle the four 32-image batches and cover all 128. Save the exact input IDs, candidate pairs, configuration order and every timing observation.

Primary latency is synchronized host wall time around the model call; CUDA event timing is supplementary. Report median, interquartile range and p95, with batch latency and throughput clearly separated. Round-level and seed-level results show timing stability. Compute speedups against native full CLIP at the same backbone, dataset and batch size. Plot accuracy from the complete validation evaluation against the corresponding representative-panel latency; identify the two measurement populations.

Measure CPU image preprocessing and host-to-device transfer separately as descriptive costs. Do not add medians from separate measurements and label their sum an observed service latency. GPU-resident tensor-to-score measurements exclude disk/JPEG work, text encoding and network service overhead. Keeping half the patches after halfway through the model does not imply halving total latency.

Use an available GPU and record process/utilization snapshots around timing rounds. Shared-server CPU conditions and any GPU contention are part of the timing record. If contention makes the benchmark unsuitable, retain that record and repeat the complete affected timing block under the same settings when an uncontended device is available; do not select individual fast samples.

## Execution and saved evidence

Commit code, configurations and this protocol before substantive runs. Preserve all model outcomes, independent statistics, raw timing samples, model checkpoints and enough scores/features/positions to reconstruct the scientific comparisons. Keep small results and figures in Git, larger scientific evidence locally and in a versioned GitHub Release, and original runs on the server. Remove transfer-only duplicates after verification. Record the observed source versions and costs; deployment timing stays separate from experiment wall time.

This stage completes the planned targeted backbone validation and actual-efficiency study. Its results inform the final frozen protocol; the reserved test images and class combinations remain for subsequent confirmation.
