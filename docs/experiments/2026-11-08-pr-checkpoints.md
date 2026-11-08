# Patch Ranking checkpoint comparison

The first PR adaptation used the final epoch-50 predictor. Its validation KL reaches a minimum around epochs 9–13 and later rises, especially on Flowers. This experiment asks whether choosing a checkpoint by the training objective improves its classification results and changes the external comparison.

Reuse the original Pets/Flowers configurations, frozen prefixes and preservation targets: B/16, block 6, 98 patches, original predictor size, Adam/cosine schedule and 50 epochs. Repeat seeds 2041, 2042 and 2043. For each seed, keep both the final weights and the checkpoint with minimum validation KL over epochs 1–50; use the earliest epoch for a tie. Selection uses no classification accuracy. The saved validation data is development data; reserved final-test data is unchanged.

Primary comparison: selected PR minus rerun-final PR, for task/class-balanced accuracy and geometric margin, averaging the three seeds before image-paired bootstrap. Adjust the two dataset comparisons per metric. Also compare the rerun finals with the original PR predictions and show the selected PR alongside the existing strong shared U and conditioned C. Record teacher-set recovery, full-class accuracy and all per-seed results.

This is a checkpoint-selection check within the existing external baseline. It does not change predictor capacity, ranking targets, hyperparameters or input information. Inference code and model size are unchanged, so the prior full-path latency measurements remain the cost reference. Results go to `results/patch-ranking-checkpoints-v1/`, with model and per-image evidence in the matching artifact archive.

## Outcome

Completed six repeated fits. Selected epochs are Pets 12/9/13 and Flowers 12/11/13, matching each history's minimum validation KL. Every repeated epoch-50 predictor matches its original scores, selected sets and outputs exactly.

Selected-checkpoint accuracy is 98.9693% on Pets (+0.1861 pp from epoch 50) and 96.2745% on Flowers (+0.0245 pp). Both accuracy-change adjusted intervals span zero. Flowers margin improves by 0.000463 with a positive adjusted interval. U−PR accuracy is now +0.153 pp on Pets (ordinary interval spans zero) and +0.637 pp on Flowers (positive interval). The previous broad statement that U establishes an accuracy lead on both datasets is updated accordingly.

This completes the checkpoint comparison. Twelve weights, all predictions and histories are retained; prior records remain available. No further PR optimization was started. [Results](../../results/patch-ranking-checkpoints-v1/README.md).
