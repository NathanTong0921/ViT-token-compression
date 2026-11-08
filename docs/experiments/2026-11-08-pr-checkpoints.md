# Patch Ranking checkpoint comparison

The first PR adaptation used the final epoch-50 predictor. Its validation KL reaches a minimum around epochs 9–13 and later rises, especially on Flowers. This experiment asks whether choosing a checkpoint by the training objective improves its classification results and changes the external comparison.

Reuse the original Pets/Flowers configurations, frozen prefixes and preservation targets: B/16, block 6, 98 patches, original predictor size, Adam/cosine schedule and 50 epochs. Repeat seeds 2041, 2042 and 2043. For each seed, keep both the final weights and the checkpoint with minimum validation KL over epochs 1–50; use the earliest epoch for a tie. Selection uses no classification accuracy. The saved validation data is development data; reserved final-test data is unchanged.

Primary comparison: selected PR minus rerun-final PR, for task/class-balanced accuracy and geometric margin, averaging the three seeds before image-paired bootstrap. Adjust the two dataset comparisons per metric. Also compare the rerun finals with the original PR predictions and show the selected PR alongside the existing strong shared U and conditioned C. Record teacher-set recovery, full-class accuracy and all per-seed results.

This is a checkpoint-selection check within the existing external baseline. It does not change predictor capacity, ranking targets, hyperparameters or input information. Inference code and model size are unchanged, so the prior full-path latency measurements remain the cost reference. Results go to `results/patch-ranking-checkpoints-v1/`, with model and per-image evidence in the matching artifact archive.
