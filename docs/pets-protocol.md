# Oxford Pets baseline protocol

## Data and image splits

Use the [Oxford-IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), with its original `annotations/trainval.txt` and `annotations/test.txt`. Class IDs are converted from one-based annotation IDs to zero-based IDs. Multiword breed names retain their words, with underscores replaced by spaces for text prompts. Images are converted to RGB before preprocessing.

The official test set stays fixed. Shuffle official trainval images within each class with Python's `random.Random(2026)` and reserve 20% per class for validation. The manifests under `data/splits/pets-v1/` define the exact split. Only annotated images are used, rather than every file in the image archive.

## Candidate-class tasks

Each task is an unordered pair of distinct classes. Shuffle all 666 pairs with independent seed 2027, allocating 532 to train, 66 to validation, and 68 to test. All 37 classes remain known; test generalization means new class combinations, not entirely new classes.

Image splits and task splits are independent. Training will use training images with training pairs. Model development uses validation images and validation pairs, alongside training-pair diagnostics. Final evaluation can compare test images with seen versus unseen pairs. This first baseline evaluates validation images with train and validation pairs; it does not evaluate official test images or test pairs.

Within each image/task split, sample the same number of images from both candidate classes, using the smaller class count. A single per-class image order (seed 2028) is reused across tasks. For future same-image cross-task interventions, join on image ID and use the intersection of eligible examples. The protocol summary records per-class pair degrees and how many anchors support two-task comparisons; random pair splitting need not give every class this coverage in every split.

## Original CLIP baseline

Load OpenAI CLIP ViT-B/16 from the pinned source revision in `requirements.txt`; freeze both encoders and use evaluation mode. Use the preprocessing returned by `clip.load`: resize, center crop to 224, RGB conversion, and CLIP normalization. Fix one prompt per breed in advance:

`a photo of a {class_name}, a type of pet.`

No prompt fitting or model training is performed. On CUDA the original backbone runs in its default FP16 precision. Convert the output embeddings to FP32, unit-normalize them, and compute cosine scores. Later pruned variants must use the same normalization/scoring path. The positive CLIP logit scale does not change classification decisions; record it to reconstruct scaled logits when needed.

Report 37-class top-1 accuracy and, for balanced binary tasks, per-pair accuracy plus macro and micro averages. Preserve normalized image/text embeddings, all 37 cosine scores, per-example predictions and task margins. A repeated image across tasks is not a new independent observation; later uncertainty estimates will resample images rather than image/task rows.

For image embedding `z`, true class `y` and competitor `a`, use the signed geometric margin

`m = dot(z, t_y - t_a) / norm(t_y - t_a)`.

The raw cosine gap and this geometric margin are stored separately. For H2, both task directions share the same true-class anchor: `t_y - t_a` and `t_y - t_b`. Keep negative margins. A later analysis restricted to initially correct predictions must report that restriction separately. Text-angle descriptors alone do not establish H2.

## Commands

From the repository root after installing dependencies and the editable package:

```bash
.venv/bin/python scripts/prepare_pets.py \
  --data-root data/raw/oxford-iiit-pet --output-dir data/splits/pets-v1
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_baseline.py \
  --config configs/pets_baseline.json --output runs/pets-baseline-v1
```

The extraction timer includes image loading and CPU transfers. It is a pipeline timing record, not the deployment-latency benchmark for a pruning method. Curate outputs into `results/pets-baseline-v1/` after review and back them up locally and on GitHub.
