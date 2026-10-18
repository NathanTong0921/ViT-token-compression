# Fixed-update training-image coverage comparison

Protocol fixed on 2026-10-18 before running the new preparation, training or expanded evaluation. This follows I4 (teacher-imitation generalization gap) and I5 (useful task dependence without established superiority to the learned image-only baseline). Configuration: `configs/pets_selector_scale.json`.

## Question and scope

Does exposing the paired selectors to more distinct training images, at the same optimization budget, improve held-out performance and the value of candidate-class conditioning? Use frozen OpenAI CLIP ViT-B/16, one pruning boundary after block 6 and 49/196 patch tokens. No backbone update, architecture change, new training objective or extra epoch search is included in this comparison.

Compare two image pools: the exact prior 592-image training panel (16 per class) and all 2,944 training images in the existing manifest. The smaller pool is nested in the larger one. Both use the same 532 training pairs. All 736 validation images are evaluated, covering 532 seen pairs / 21,162 image-task rows and 66 unseen pairs / 2,624 rows. Class sizes differ slightly (training: 74–80; validation: 19–20). The old validation panel has 144 images; 592 additional validation images extend development evidence. Final test images and the 68 test pairs remain unused.

## Control the information and optimization budgets

Run small/full coverage × conditioned/image-only × seeds 2041, 2042, 2043: 12 models. Use the original 348,033-parameter architecture, hidden width 128, soft-utility KL target at temperature 1, AdamW with learning rate 0.0003 and weight decay 0.01, and batch size 128.

Each model receives exactly 3,990 optimizer updates, or 510,720 example presentations. This equals the original pilot's 133 batches × 30 epochs. Fix a balanced sequence of 532 task pairs × two target-class sides, each appearing 480 times per seed, and draw an image uniformly from the corresponding class pool. The same task/side sequence is used across both coverage conditions and both models; the same image draws are used within each paired condition. Start all four models in a seed from the same initial parameters.

Sampling is with replacement within class. This differs from the earlier pilot's epoch permutations, so retrain the small-pool control under the new sampler. Do not use the old checkpoints as the primary data-scale control. The training-task distribution and number of gradient updates stay fixed; repeat frequency per individual image necessarily decreases with greater coverage. This is a two-size comparison conditional on the chosen pools, not a universal data-scaling law.

Use the final step 3,990 checkpoint for every model. Record teacher-fit diagnostics at step 0 and every 133 updates; do not select a checkpoint using validation results. Record actual image exposure, training time and supervision cost. Same architecture and nominal parameter count do not imply identical effective capacity when one context is constant.

## Supervision and frozen features

Use the same 512-mask, 49-patch ridge supervision as the preceding experiment (mask seed 2031, regularization 0.01). Reuse prior teacher values and prefixes for overlapping images after alignment and consistency checks; generate missing supervision for the additional training and validation images. Labels form offline targets and evaluation outcomes; the selector receives only visual tokens and both candidate-class text embeddings.

Encode images in the existing full-split order and fixed batch size 64, retain keep-all/native CLIP consistency checks, and save the expanded prefix cache only on the server. Record reused and newly generated work separately. Validation teacher values are diagnostics, never training targets. Any missing old validation mask responses must be explicitly marked rather than represented as observed zeros.

## Outcomes fixed before evaluation

The primary question is the change in the conditioning advantage on unseen combinations:

`(conditioned − image_only)_full − (conditioned − image_only)_small`.

Report accuracy and signed geometric margin, with both component effects and each model's full-minus-small improvement. Equal task and equal class weighting prevents classes with 20 validation images from outweighing classes with 19; within a task, each class contributes half the weight. Average training seeds, then bootstrap images as clusters using the same image resample for all four models (2,000 replicates, seed 2052). Preserve the chosen metric's weights in its point estimate and interval; label any image-equal auxiliary summaries separately. Report seeds separately because image intervals are conditional on these checkpoints and task collections.

Secondary diagnostics: teacher-fit gap, compression-induced errors/repairs, performance on the old 144-image panel versus the additional 592 images, and correct-versus-exchanged task descriptors within the trained models on unseen pairs. Balance each eligible pair within each panel and report its coverage. The additional images are now development data, not an untouched final confirmation.

Retain normalized full image embeddings from actual evaluation for later H2 analysis, as well as all 37 class scores and selected patch positions. This round does not add a new H2 claim. Seen-pair outcomes are secondary and cannot alone establish combination generalization.

## Interpretation and next decision

If both models improve similarly, greater image coverage helps compression without establishing a larger conditioning advantage. If the conditional advantage increases reliably, the result supports training-image coverage as one limiting factor under this fixed-update setup. If the advantage disappears or reverses, preserve that result and inspect proxy fitting versus actual classification before choosing a targeted follow-up. The intervention tests coverage; teacher-fit correlations alone do not establish the cause of downstream changes.

Keep architecture and training choices fixed until this comparison finishes. Broader token-budget experiments, Flowers102 and the now-required targeted ViT-L/14 validation remain separate planned stages. Preserve all 12 final weights, supervision, per-example scores/features and figure source data locally, on the server and in GitHub Git/Release storage as appropriate.
