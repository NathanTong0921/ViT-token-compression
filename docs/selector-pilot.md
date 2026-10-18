# Paired selector learnability pilot

Protocol fixed before training outcomes, 2026-10-18. This is the first H3 experiment: can a small selector use candidate-class information to learn the offline advantage and transfer to class combinations absent from its training pairs?

## Data and teacher

Freeze original CLIP ViT-B/16. Prune once after block 6, retaining 49 of 196 patches plus CLS. Use the existing image and unordered-pair split manifests.

- Training: 16 images per class, 592 images, all 532 training pairs and 17,024 balanced image/task examples. Sampling follows the existing seed-2028 class ordering.
- Validation: the same 144 development images as the offline pilot, four per represented class. Evaluate 500 eligible seen training pairs (4,000 rows) and all 66 unseen validation pairs (528 rows), using the same images. A seen pair is eligible only when both classes occur in this panel.
- Training and validation images are disjoint. Validation pairs never enter selector optimization. All classes are known; the transfer axis is new combinations of known classes. These validation data have informed prior project development, so the results remain developmental evidence. Official test images and 68 test pairs remain untouched.

Generate 512 fixed-49-patch probes for each training image, using the previous seed-2031 standardized ridge estimator with regularization 0.01. Fit all 37 class scores once, then form each training task's signed geometric-margin utility vector from its true class and competitor. Reuse the prior validation teacher only for evaluation diagnostics. Cache selected block-6 prefixes on the server, preserving original full-split encoder batch shapes for consistency checks.

## Matched models

Both arms use the same architecture, hidden width 128 and trainable parameter count. A patch scorer receives layer-normalized patch features, CLS context and a symmetric descriptor of the two normalized candidate text embeddings: their mean and absolute difference. Two GELU hidden layers produce one score per patch; the 49 highest scores are retained in original spatial order.

The conditioned arm receives the current candidate pair. The image-only arm receives a fixed descriptor averaged over training pairs. Its image path and scoring network are otherwise unchanged. The constant context acts as a learned prior: this matches architecture and nominal parameter count, not effective input-dependent degrees of freedom. A later strong method claim should also withstand a stronger image-only capacity control.

The selector forward interface takes image tokens and both candidate texts, without the true label or teacher outputs. Swapping the candidate order leaves its scores unchanged. The true label is used to form training supervision and compute evaluation metrics.

## Training fixed in advance

For each image/task, center and standardize the teacher utility vector across patches, then apply softmax with temperature 1. Train with KL divergence between this soft teacher distribution and the selector's score distribution. This avoids treating one sampled hard selection as the only correct answer; whether this objective transfers to downstream performance is itself tested.

Use AdamW, learning rate 0.0003, weight decay 0.01, batch size 128 and 30 epochs. Run three paired seeds: 2041, 2042 and 2043. Within each seed, initialize the two models with identical weights and use identical minibatch orders. Use the final checkpoint for both arms; do not select their best validation epochs. Record initial/final teacher-fit diagnostics, epoch losses, optimization time and supervision-generation cost.

## Actual model evaluation

Run each selector's selected tokens through the frozen CLIP suffix. Report balanced binary accuracy and signed geometric margin separately for seen and unseen pairs, together with full-token, attention and label-informed teacher references. Save all class scores and selected positions. Use fixed 64-row suffix batches, padding and discarding extra outputs as in the prior pilots. Evaluation wall time is not a deployment-latency benchmark.

For each seed, compare the two learned arms on exactly the same image/task rows. Also average paired effects over the three seeds and then within images, and bootstrap images while retaining their task rows. Report seed variation separately: an image interval conditional on these three seeds does not estimate all training-seed uncertainty. Preserve the balanced task-table estimand separately from image-equal paired summaries.

Interpret results in conjunction with teacher-fit losses/correlations. Gains on seen combinations alone do not establish combination transfer. Near-ceiling accuracy can coexist with different margins. If no benefit appears, distinguish a failed optimization pipeline, a poorly learned teacher, and a well-trained model with no downstream advantage before deciding what H3 supports.

## Retained evidence

Keep image/task manifests, teacher probe responses and coefficients, configurations, training curves, final weights, all per-example evaluation outputs and figure data. Prefix caches are reproducible server working files. Link resulting discoveries or revisions in [research-insights.md](research-insights.md), including negative findings.
