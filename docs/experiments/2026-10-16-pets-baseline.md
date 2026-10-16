# Pets original CLIP baseline

- Run ID: `pets-baseline-v1`; 2026-10-16.
- Status: complete development baseline.
- Purpose: establish data/task splits and the uncompressed reference predictions for subsequent pruning and cross-task interventions.
- Code commit: `2623193452f219f7708ecc15dff5c2839ff5661b`.
- Config: `configs/pets_baseline.json`, no overrides.
- Command: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/run_baseline.py --config configs/pets_baseline.json --output runs/pets-baseline-v1`.
- Environment: Python 3.10.12, PyTorch 2.6.0+cu124, NumPy 1.26.4; pinned official CLIP revision in `requirements.txt`; one RTX A6000.
- Training: none. Selection-supervision generation: none. Image extraction: 5.323 seconds, including loading/transfers; peak allocated GPU memory approximately 737 MB. This does not measure end-to-end pruning latency.

## Protocol and checks

Use `data/splits/pets-v1`: image seed 2026, task seed 2027, balancing seed 2028. The official Oxford split files used to generate manifests match the fastai distribution's files byte for byte. The server contains all 7,349 referenced image files and reproduced all four manifests exactly from its raw annotations.

Only validation images and train/validation candidate pairs were evaluated. Preserve the official test images and test pairs for final evaluation. CLIP B/16 is frozen; use the loader's official preprocessing and the predetermined single text prompt. Backbone outputs are converted to FP32 before normalization and cosine scoring. No pruning is applied.

Seven targeted tests passed locally and on the server. After copying results locally, independently recomputed all 23,662 binary predictions and signed margins, task balancing, image membership and aggregate accuracies from the saved embeddings. The independent review confirmed these checks.

## Result and interpretation

Full 37-class accuracy is 642/736 = 87.2283%. Binary macro accuracy is 99.1037% on training pairs and 99.2544% on validation pairs. Detailed data and the initial observation are in [the result bundle](../../results/pets-baseline-v1/README.md).

Binary task accuracy is near ceiling on many development pairs. Next, implement one pruning boundary and verify that retaining all tokens recovers the same baseline scoring path. Then measure pruning-induced errors and exchanged-selection losses, reporting both the full task set and difficulty strata fixed before examining pruning outcomes. This run does not test any of H1/H2/H3 yet.

## Storage

The server checkout is now named `experiments`; its renamed virtual environment passed pip and CUDA checks. The environment, cached B/16 weight, raw Pets data and current outputs together occupy approximately 6.5 GiB. Shared filesystem free space after setup is approximately 72 GiB, not a project allocation.

Raw images remain only on the server. Local installation-transfer files and their matching new pip cache copies were removed (approximately 6.5 GB), followed by this run's approximately 812 MB dataset transfer bundle after successful server extraction and verification. Temporary server archives were also removed. Split manifests, source code and scientific result artifacts are retained.

The server run is at `runs/pets-baseline-v1/`; the reviewed local and GitHub evidence bundle is `results/pets-baseline-v1/`. No learned checkpoint exists for this untrained baseline.
