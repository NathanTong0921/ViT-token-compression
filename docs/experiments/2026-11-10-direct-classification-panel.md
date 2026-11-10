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

The project remains in `/data2/zhao.tong/experiments`. Because that filesystem has approximately 150 MB free at launch, temporary output goes to `/tmp/zhao-tong-direct-panel-v1`. Full artifacts will be archived under `/home/users/zhao.tong/experiment-artifacts/direct-classification-panel-v1` and in a GitHub Release. Curated analysis, figures and records stay in `results/direct-classification-panel-v1/` locally, on GitHub and on the server. Transfer intermediates will be removed after recovery verification.
