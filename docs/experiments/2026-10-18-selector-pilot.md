# 2026-10-18: first paired selector training pilot

## Question and fixed design

This is the first trained-model test of proposal H3: can a lightweight selector learn the task-specific offline advantage and transfer it to unseen combinations of known classes? The [protocol](../selector-pilot.md) was fixed before training outcomes; configuration: `configs/pets_selector_pilot.json`.

Freeze original OpenAI CLIP ViT-B/16, prune after block 6, retain 49/196 patch tokens plus CLS. Training uses 592 images (16 per class), 532 pairs, and 17,024 image/task examples. Evaluation uses the prior 144 development images, with 500 seen pairs / 4,000 rows and 66 unseen pairs / 528 rows. Images are disjoint from training; the official test images and 68 test pairs remain unused. This validation panel informed earlier research decisions, so the evidence is developmental.

Both arms have the same 348,033-parameter network, hidden width 128, initialized identically within each of seeds 2041/2042/2043. One receives the current unordered candidate pair; the other receives the fixed mean training-pair descriptor. This matches architecture and nominal parameter count, while the information available differs. The constant descriptor can be absorbed into a bias; equal effective function capacity is not claimed. Train both on the same soft budget-matched teacher targets, with identical epoch permutations, AdamW (0.0003 learning rate, 0.01 weight decay), batch 128 and 30 fixed epochs. Save the final epoch, without validation-based checkpoint selection.

## Preparation and execution

Preparation source: `94f83fd8adf05a63a32f33f2d6cacd1128e591dd`. The existing 512-probe, standardized-ridge teacher is generated for training images at the 49-patch target budget; validation coefficients are reused only for diagnostics. Selectors never receive true labels as inputs.

On the server, run from `/data2/zhao.tong/experiments` with `CUDA_VISIBLE_DEVICES=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4`, using `.venv/bin/python -u`:

```bash
scripts/prepare_selector_pilot.py --config configs/pets_selector_pilot.json
scripts/train_selectors.py --config configs/pets_selector_pilot.json --output runs/pets-selector-training-v1
scripts/evaluate_selectors.py --config configs/pets_selector_pilot.json --models runs/pets-selector-training-v1 --output runs/pets-selector-eval-v1
```

Hardware: one RTX A6000, physical GPU 4. Python 3.10.12, PyTorch 2.6.0+cu124, NumPy 1.26.4. The source commits and measured costs are recorded in the saved preparation, training and evaluation JSON files. No CLIP parameter is updated. Cached-prefix evaluation time is not end-to-end deployment latency.

A startup attempt at `328519e` stopped before any training because the memory-statistics call preceded CUDA initialization. Adding explicit device initialization fixed this at `ae6d57f`; all six model runs use that version and the original protocol. No training hyperparameters changed in response to results.

## Checks

All 48 server unit tests passed before training. Independent preparation audit confirmed all image/task split memberships, no overlap with official test data, exact reconstruction of three sampled training ridge fits, finite teacher arrays, 512 masks each retaining exactly 49 patches, and exact reuse of the prior validation teacher. Every optimizer step uses training data; validation is evaluated without gradients and never selects checkpoints.

Keep-all features match native CLIP exactly across 2,944 training-split images and 736 validation-split images. Saved validation baseline features match exactly; cosine scores differ by at most 1.49e-7, within the pre-fixed 1e-6 tolerance, with no prediction differences.

The final accuracy tables balance task examples. Paired confidence intervals resample images after averaging repeated tasks and training seeds within each image. They are conditional on these task collections and three seeds, rather than uncertainty over an arbitrary new draw of tasks or training seeds. The unseen-minus-seen contrast is descriptive because task composition and difficulty differ.

## Results and interpretation

The [result README](../../results/pets-selector-pilot-v1/README.md) reports full tables and intervals. Unseen-pair accuracy averages 99.116% conditioned and 98.737% image-only. Each seed gains two correct examples out of 528. Both learned selectors exceed attention (97.917%). The image-bootstrap accuracy interval touches zero and mean-margin intervals span zero, so this is an initial positive accuracy signal, not established superiority. The fixed image-only baseline already performs strongly.

Conditioned training teacher KL improves to about 0.133 versus 0.329 image-only; held-out KL is worse (0.831 versus 0.617), increasing after the first few epochs. This provides evidence of a teacher-imitation generalization gap alongside slightly better classification accuracy. It extends I2/I3 about distinguishing proxy fitting from downstream selection quality; see new I4 in the research insight log. Do not infer that the training failure explains the accuracy changes, or that additional epochs are the solution.

Preparation took 101.494 s, all six paired runs 166.107 s, and model evaluation 12.861 s: 280.462 s = 0.07791 GPU process hours. Training loops account for 158.233 s. These are process wall times including associated work, not GPU-active time or deployment latency. Teacher preparation includes 303,104 actual suffix examples; cached-prefix evaluation includes both learned arms and all references. Peak tensor allocation for training was about 0.899 GB. This small model/cached-prefix trial does not imply the whole project costs only minutes.

A read-only checkpoint audit on seed 2041 confirmed 348,033 parameters in each model, exact candidate-order invariance, exact image-only invariance across candidate pairs, and nontrivial conditioned score changes beyond a common bias. Checkpoints contain only selector tensors; CLIP is absent from the optimizer.

## Evidence and backup

Curated small evidence is under `results/pets-selector-pilot-v1/` (about 45 MB). Selected weights and the 70.93 MB training-teacher archive are retained outside Git in the local `artifacts/pets-selector-pilot-v1/` directory and uploaded to the versioned GitHub Release. Original server outputs remain in the three `runs/pets-selector-*-v1/` directories. The 222.71 MB prefix cache remains on the server. No public dataset or CLIP checkpoint needed downloading for this trial.

The completed raw-data audit reconstructed all 40,752 evaluation rows from saved class scores: correctness and margins match exactly. Every retained top-49 mask agrees exactly with its saved ranking; image-only rankings are identical across candidate pairs for the same image and seed. All 1,056 unseen attention/teacher reference rows reproduce prior positions and predictions exactly; margin differences are at most 2.38e-7. All numeric evidence is finite. See `results/pets-selector-pilot-v1/raw_audit.json`.

## GitHub recovery verification

After publishing the milestone, the server fetched the tracked result files from GitHub independently of its original `runs/` outputs. All 14 retained original-run files match those outputs byte for byte. Four NumPy archives, six JSON records and two PNGs opened successfully; all floating-point evidence is finite. Rebuilding `analysis.json` from the fetched predictions gives an exact match.

The server then downloaded the 78,768,182-byte release ZIP directly from GitHub. All seven scientific files match the original server teacher/weights exactly. All six checkpoints load with PyTorch, have 348,033 model parameters and final epoch 30, and contain finite tensors. The teacher arrays load with their expected dimensions. The temporary downloaded ZIP was removed automatically. The corresponding local ZIP was also compared with every original local teacher/checkpoint file; redundant extracted local copies were then removed, retaining the archive and Git evidence. See `backup_verification.json` and `release_asset.json` in the result bundle.

The milestone is tagged `pets-selector-pilot-v1`. Code and small evidence are on GitHub main; the release retains the larger teacher and all final weights. The quota interruption happened during an auxiliary audit after training/evaluation had completed; the main agent completed the raw-result and recovery audits afterward. No training run or saved scientific result was lost.
