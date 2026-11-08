# Patch Ranking checkpoint selection

Completed 2026-11-08. Repeat the original frozen B/16, block-6, 98-token PR fits on Pets and Flowers using seeds 2041–2043. Each fit retains both epoch 50 and the epoch with minimum validation KL over epochs 1–50. Checkpoint selection uses the preservation objective, not classification accuracy. All evaluations here use development images and combinations.

| Method | Pets accuracy (%) | Flowers accuracy (%) |
|---|---:|---:|
| PR, epoch 50 | 98.7832 | 96.2500 |
| PR, minimum validation KL | 98.9693 | 96.2745 |
| Strong shared U | 99.1221 | 96.9118 |
| Conditioned C | 99.1481 | 97.1160 |

The selected epochs are **12/9/13** for Pets and **12/11/13** for Flowers. All six repeated epoch-50 predictors reproduce the original scores, retained sets and classification results exactly.

## Effect of checkpoint selection

| Selected minus epoch 50 | Accuracy difference (pp) | Two-dataset adjusted interval (pp) | Margin difference | Adjusted interval |
|---|---:|---:|---:|---|
| Pets | +0.1861 | [−0.0261, +0.4424] | −0.000264 | [−0.001300, +0.000732] |
| Flowers | +0.0245 | [−0.1961, +0.2206] | +0.000463 | [+0.000100, +0.000888] |

Accuracy uses equal task and class-side weights, averaging three seeds before image-paired bootstrap. Both accuracy-change intervals span zero. The Flowers margin improvement has a positive interval.

Against the selected PR checkpoint, **U−PR is +0.1528 pp on Pets**, ordinary 95% interval [−0.0426, +0.3556], and **+0.6373 pp on Flowers**, [+0.3266, +0.9559]. Thus Pets no longer establishes an accuracy advantage for U, while Flowers retains one. The C−PR differences are +0.1788 and +0.8415 pp; their ordinary intervals are positive. Reference intervals here are ordinary 95% intervals, separately labelled from the adjusted primary checkpoint comparison. U and C retain positive margin differences from PR on both datasets.

Auxiliary full-class accuracy changes from 82.4728% to 83.5145% on Pets and 67.8431% to 68.5621% on Flowers. Top-98 teacher overlap changes by −0.0287 and +0.5205 pp, respectively. Complete statistics and every seed are in each dataset's `analysis/` directory.

## Research consequence

This updates the external baseline comparison: the Pets accuracy gap is smaller than the original final-epoch table suggested; Flowers changes little. It does not alter the existing C/U task-information comparisons, task-exchange experiments or mixing results. The selected PR weights are available for the eventual reserved-test comparison. The original epoch-50 results remain in [the first external comparison](../patch-ranking-v1/README.md).

The model architecture and inference path are unchanged, so the previously measured latency remains the cost reference. No new timing or reserved-test evaluation was performed. This checkpoint check is complete; it adds no capacity search or further PR tuning.

## Records

- [Learning curves and classification comparison](checkpoint_comparison.png), with [SVG source](checkpoint_comparison.svg).
- [Experiment design and outcome](../../docs/experiments/2026-11-08-pr-checkpoints.md).
- Training plus diagnostics: 64.7 seconds for the three Pets fits and 22.9 seconds for the three Flowers fits, using previously cached features and ranking targets.
- The local and server `artifacts/patch-ranking-checkpoints-v1/` directories retain the complete archive: 12 checkpoints, per-image predictions, selected sets, embeddings and logs. The matching GitHub Release provides the remote copy. Reused features and teachers remain in the original milestone.
