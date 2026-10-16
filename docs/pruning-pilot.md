# Single-stage pruning pilot

This pilot checks the token-deletion implementation and measures the losses caused by two image-only selection rules. It does not train a selector or test the benefit of task conditioning.

## Fixed setup

- Frozen OpenAI CLIP ViT-B/16; the same 736 development images, text prompt, preprocessing and FP32 embedding normalization as `pets-baseline-v1`.
- Complete the first 6 of 12 transformer blocks with all 196 patch tokens and CLS. Delete patches once, then run the remaining 6 blocks.
- Keep 147, 98 or 49 patches (75%, 50% or 25%), plus CLS. A 196-patch keep-all run verifies the actual prefix/gather/suffix path.
- Preserve the selected tokens' feature values and original positional information. Gather them in their original spatial order. No merging, re-embedding or model fine-tuning.
- Attention: take block 6's inputs, apply its `ln_1` and Q/K projection, compute CLS attention separately per head, then average over heads. CLS participates in softmax but is excluded from patch ranking. Real hidden states still follow the unchanged native block forward. Projection uses backbone precision; attention logits, softmax and head averaging use FP32.
- Random: one NumPy patch permutation per image in manifest order, seeds 2026/2027/2028. Reuse each ranking across all budgets and candidate-class tasks. Sampling does not depend on DataLoader batching.

Both selectors receive image information only. Every candidate-class task on a given image uses the same retained set. Ranking ties in the attention baseline preserve the original patch order.

## Verification before pruning evaluation

Compare keep-all embeddings against both the native encoder in the current run and the saved original baseline. Require maximum absolute normalized embedding and cosine-score differences no greater than `1e-6`, and identical 37-class and evaluated binary predictions. The keep-all path must execute gather and the suffix; it cannot bypass them by calling the native encoder. The script stops before evaluating reduced budgets if this check fails.

The targeted unit tests also check attention against PyTorch's explicit attention weights, the preservation of CLS and original patch positions, and error/margin aggregation. Tests requiring Torch run in the server environment.

## Outcomes

Retain full 37-class accuracy, balanced binary macro/micro accuracy, newly introduced errors, recovered errors, and geometric margin loss relative to the uncompressed model. New errors and recoveries are separate because they can cancel in the net accuracy change.

Define difficulty groups before inspecting any pruning results. The uncompressed training-pair examples on validation images give margin boundaries `[0.09612737409770489, 0.12876027822494507, 0.15853078663349152]`. Use these same boundaries for every method, seed and task split. Negative margins remain in Q1. Full-task results remain the primary descriptive summaries; strata explain where losses occur.

The three random seeds measure selection variability. They do not create independent copies of the dataset; image/task rows also reuse images. This pilot does not estimate a new confidence interval by treating those rows as independent.

## Running and retention

From the repository root:

```bash
.venv/bin/python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/run_pruning_pilot.py \
  --config configs/pets_pruning_pilot.json --output runs/pets-pruning-v1
```

The small pilot keeps prefix states in host RAM and releases them at process exit. It reuses those states across variants; recorded wall times are experiment costs, not inference-speed measurements. End-to-end latency benchmarking requires a later separate run that includes the prefix and selection overhead.

Each variant saves image IDs, normalized embeddings, text features, all class scores and the actual retained patch positions. Together with the fixed manifests these reconstruct every image/task comparison. Summary, per-task and fixed-margin-group tables accompany the arrays. Curated copies belong in `results/pets-pruning-v1/` on the server, locally and in GitHub; no dataset or model transfer is needed for this run.
