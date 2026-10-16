# Single-stage pruning: random and attention selection

- Run ID: `pets-pruning-v1`; date: 2026-10-16; status: complete development pilot.
- Purpose: verify a single pruning boundary and measure compression losses before task-specific selection experiments.
- Code commit: `cffe5165a630ae500cf4f99032210dc6620191ed`.
- Config: `configs/pets_pruning_pilot.json`, no overrides.
- Command: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/run_pruning_pilot.py --config configs/pets_pruning_pilot.json --output runs/pets-pruning-v1`.
- Hardware/environment: one RTX A6000, Python 3.10.12, PyTorch 2.6.0+cu124, NumPy 1.26.4; original pinned CLIP weights/source.
- Training and supervision-generation cost: none. Total pilot wall time: 16.49 seconds. Prefix states reused across variants; this timing is not deployment latency. Peak allocated GPU memory: approximately 756 MB; temporary host prefix states: approximately 223 MB.

## Fixed design and verification

Use Pets v1's 736 validation images, 532 training pairs and 66 validation pairs. Keep official test images and test pairs untouched. All preprocessing, prompts and normalized FP32 scoring match the saved original baseline.

After the first 6 of 12 blocks, retain 196/147/98/49 patches plus CLS. Attention ranking uses block 6's input and its CLS-to-patch attention, averaged over heads; a separate scoring calculation leaves the real hidden-state path unchanged. Random ranking uses a separate per-image permutation for each of seeds 2026/2027/2028, shared across budgets and candidate tasks. Selected tokens retain their original features and positions.

Fifteen targeted tests passed on the server. Before measuring any pruning outcomes, keep-all passed the full 736-image identity check: normalized embeddings and cosine scores matched native CLIP and the saved baseline exactly; all evaluated predictions matched. The tolerance was fixed at 1e-6 in advance, with measured maximum error 0. After transfer, all 13 variants' selection sets and aggregate results were independently reconstructed and checked. A later metadata wording clarification states that scoring uses the original CLIP LayerNorm (internally FP32, returning backbone dtype); no numerical implementation changed.

Margin group boundaries were computed from unpruned training-pair rows on validation images and committed before this run. Each method/seed uses the same groups, including negative baseline margins in Q1. New errors and recovered errors are counted separately.

## Findings and limits

See the [results and figure](../../results/pets-pruning-v1/README.md). At 25% patch retention, validation binary macro accuracy is 97.66% for attention and 92.33% for random selection averaged over three seeds, compared with 99.25% uncompressed. Attention introduces 43 errors and recovers 2; 33 new errors are in the lowest baseline-margin group.

The smaller-budget results establish measurable compression losses and sensitivity to the selection rule. They do not establish an advantage of task conditioning: neither selector reads candidate classes, and the stronger shared ranking from the proposal has not yet been tested. At 75% and 50% retention, binary accuracy remains close to ceiling. The full 37-class metric also remains useful: attention does not dominate random selection at every budget on that metric.

Next, investigate the planned task-specific and strong shared rankings using the verified `prefix`/`finish` interface, with fixed features and task-difficulty controls. No layer/budget search was conducted in this run.

## Retention

The server keeps the original run at `runs/pets-pruning-v1/`; curated arrays, tables and figure exports are stored at `results/pets-pruning-v1/` locally and in GitHub. Every variant retains all class scores and selected positions, sufficient to reconstruct the image/task comparisons with the fixed manifests. The figure is generated from `metrics.csv` by `scripts/plot_pruning_pilot.py`.

No new dataset or model download was needed. The prefix cache was kept only in RAM and released after execution. Local temporary copies used only to verify GitHub recovery are deleted after verification; scientific result copies are retained.
