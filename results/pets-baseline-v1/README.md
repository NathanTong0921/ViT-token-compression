# Original CLIP baseline: Pets development images

Completed 2026-10-16 with frozen OpenAI CLIP ViT-B/16, the fixed single prompt, and the [Pets v1 protocol](../../docs/pets-protocol.md). All metrics below use the **736 validation images**, with balanced sampling within each binary task. The model has not been trained on the project data; “train pairs” names the candidate-combination split, not model fitting.

| Evaluation | Tasks | Image/task rows | Accuracy |
| --- | ---: | ---: | ---: |
| Full 37-class classification | 1 | 736 | 87.2283% (642 correct) |
| Binary training pairs, macro average | 532 | 21,054 | 99.1037% |
| Binary training pairs, micro average | 532 | 21,054 | 99.1071% |
| Binary validation pairs, macro average | 66 | 2,608 | 99.2544% |
| Binary validation pairs, micro average | 66 | 2,608 | 99.2715% |

Official test images and test pairs have not been evaluated. These are development results, not a standard official-test CLIP benchmark score or evidence for H1/H2/H3.

## Contents

- `summary.json`: exact configuration, code commit, prompt strings, preprocessing, precision, environment, aggregate metrics and extraction timing.
- `embeddings.npz`: `image_ids` (736), `labels` (736), unit-normalized `image_features` (736 × 512), `text_features` (37 × 512), and `cosine_scores` (736 × 37).
- `predictions_37class.csv`: individual predictions and strongest competing classes.
- `predictions_binary.csv`: all 23,662 balanced image/task predictions, signed cosine gaps and signed geometric margins.
- `tasks.csv`: per-task sample counts, accuracy and mean geometric margin.
- `task_angles.csv`: text-direction angles for all 23,310 shared-anchor task pairs. Text geometry does not require test-image inference.

## First observation

On this development sample, 435 of 532 training pairs and 58 of 66 validation pairs have 100% accuracy. The 19 validation-pair errors involve only 16 distinct images. This suggests an accuracy ceiling in aggregate binary evaluation: later pruning comparisons should retain the full task set and additionally report results by predeclared uncompressed-margin strata. Do not pick a handful of low-accuracy pairs as the sole evaluation set. Repeated image/task rows are correlated; later uncertainty estimates should resample images.

The full single image-feature extraction pass took 5.32 seconds on one RTX A6000, including data loading and CPU transfers. This is not a deployment latency benchmark. No selector training or label-informed selection supervision was performed.

All result files have been checked after transfer: predictions, margins, task balancing and aggregate metrics agree with the saved embeddings and manifests. The artifact bundle is approximately 4.1 MB and is retained in ordinary Git. The server also retains the original run under `runs/pets-baseline-v1/`.
