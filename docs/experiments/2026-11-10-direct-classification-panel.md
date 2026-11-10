# Direct classification training: expanded Flowers49 evaluation

## Question

Does the improvement from direct classification supervision on Flowers49 extend to the existing larger test panel, and does it improve C relative to the strong shared U?

The preceding development experiment trained three paired seeds of C/U with either direct classification loss or continued teacher KL. Direct C improved over KL C by 0.278 percentage points. Its accuracy advantage over direct U remained uncertain. This evaluation keeps all those weights fixed.

## Fixed evaluation

- Flowers102, CLIP ViT-B/16, block 6, 49 patch tokens plus CLS.
- Existing test panel: 6,149 images, 204 candidate pairs, 24,596 image/task decisions. Original models have already been evaluated on this panel; this is an evaluation of the newly trained routes on the same panel.
- Seeds 2041, 2042 and 2043; initial, KL step 1,200 and direct step 1,200 for both C and strong U. No checkpoint or hyperparameter selection in this round.
- Reuse the saved prefixes, candidate texts and task manifests. Re-evaluate all initial and continued weights with the same FP32 native token-gathering path used for the preceding development comparison, batch size 32.
- Metrics: task/class-balanced accuracy, cross-entropy, raw cosine-score gap and geometric margin. Keep raw scores and selections.
- Main contrasts: direct C minus direct U, and (direct C minus direct U) minus (KL C minus KL U). Also report each arm's direct-minus-KL and direct-minus-initial change.
- Average the three fixed seeds, resample image clusters 2,000 times with seed 2717, and retain the same two-primary-contrast adjustment separately per metric. Save each seed and corrected/new-error decompositions.

## Execution and outputs

This round performs inference and analysis only. Flowers98 training will be considered after these results. The numerical pipeline will be compared to the prior evaluator on a small development batch before the larger panel is run.

Code and existing caches remain in `/data2/zhao.tong/experiments`. Because that filesystem has approximately 150 MB free at launch, the user specified data4 for new outputs. This run writes to `/data4/zhao.tong/experiments/direct-classification-panel-v1`; archives go to `/data4/zhao.tong/artifacts/direct-classification-panel-v1` and a GitHub Release. Curated analysis, figures and records stay in `results/direct-classification-panel-v1/` locally, on GitHub and on the server. Transfer intermediates will be removed after recovery verification.

## Completed result

All eighteen endpoints completed. Six endpoints on a 32-example development batch matched the previous evaluator exactly in scores, masks and metrics. The larger panel contained the planned 6,149 images, 204 pairs and 24,596 decisions. Cached-prefix inference took 164.58 seconds, total wall time 170.48 seconds, with 2.61 GB peak allocated GPU memory; no new training was performed.

Direct C reaches 97.5519% accuracy and direct U 97.2463%. Their difference is +0.3056 percentage points, with the two-primary-adjusted interval [+0.1139, +0.5069]. The C−U difference increases by +0.2467pp over KL continuation, interval [+0.0460, +0.4499]. C improves over its KL continuation by +0.4183pp and U by +0.1715pp. Both main comparisons also favor direct conditioning in raw gap, geometric margin and CE. The direct C−U accuracy difference is positive in all three seeds; the target interaction is positive in two of three.

The development C−U and objective-interaction estimates were +0.2941pp and +0.2533pp. Their similar magnitudes on this larger panel support continuing with a targeted Flowers98 replication. Flowers98 has not been run in this round. See [results and panel comparison](../../results/direct-classification-panel-v1/README.md).
