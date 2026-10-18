# 2026-10-18: fixed learned-selector task exchange

## Purpose and protocol

Extend the proposal's offline cross-task selection intervention to the already trained models. Keep the input image, trained weights, block-6 pruning boundary and 49-patch budget fixed; vary the candidate pair supplied to the selector, then evaluate on the original task. This connects H1 and H3 without changing the training setup.

The [protocol](../selector-exchange-pilot.md) and `configs/pets_selector_exchange.json` were committed and pushed at `b4d2390` before crossed outcomes were inspected. The primary panel uses unseen combinations of known classes: 124 eligible development images and 868 unordered task pairs per seed. The secondary panel has 144 images and 53,904 pairs per seed. Enumerate all valid pairs, average repeated pairs and seeds within images, and bootstrap images. Difficulty bands come unchanged from the original pruning pilot.

## Execution and checks

The parent evaluation stores all 37 class cosine scores for every selected set. Because the frozen suffix does not consume candidate text, evaluating a saved descriptor-B vector on task A gives the exact crossed result. This avoids repeating inference solely to change the scored candidate pair.

Replay source: `b38e13fbc4e99cfaf22257ae20455ef1e276e1aa`. Server directory: `/data2/zhao.tong/experiments`. One RTX A6000, physical GPU 4; Python 3.10.12, PyTorch 2.6.0+cu124, NumPy 1.26.4:

```bash
CUDA_VISIBLE_DEVICES=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python -u scripts/replay_selector_exchange.py --output runs/pets-selector-exchange-replay-v1
```

The predetermined replay uses the first eight alphabetically sorted eligible images, their first two tasks, all three training seeds and both selector arms. All selected positions and crossed predictions match exactly. Maximum raw-logit, all-class score and crossed-margin errors are zero. It took 4.6929 s for 96 logical / 384 physical examples with the original fixed batch size. This checks saved-inference consistency; it adds no independent evaluation images.

Analysis source: `171db3c8fa0439cab8668708eb688d105663985f`, committed and pushed before the official archived regeneration. Local Python 3.9.6 runs the two commands recorded in [run_record.json](../../results/pets-selector-exchange-v1/run_record.json), taking 2.6238 s and 0.3795 s on CPU. An earlier local check of the error-analysis script occurred before its commit; the archived results were regenerated afterward. No training, checkpoint selection or hyperparameter changes occurred.

All 55 server unit tests passed, including exchange-direction, image-weighting, original-benchmark transitions and fixed-band checks. An independent raw-score audit reconstructed every conditioned/image-only margin and correctness cell for all 164,316 cases, checked image/task/seed keys and pair coverage, confirmed exact image-only invariance, and reproduced image-bootstrap results to 1e-10. The plot was visually checked. See `raw_audit.json` in the result bundle.

## Result and interpretation

On the primary unseen panel, correct descriptors improve margin by 0.004155 [0.003219, 0.005160] and accuracy by 0.533 percentage points [0.125, 1.030] over exchanged descriptors. Same-band restriction preserves positive effects. Image-only exchange has exactly zero effect. These results support useful learned task dependence under the current setup.

Correct-descriptor versus separately trained image-only accuracy is +0.484 percentage points [-0.027, 1.075] on the same panel; mean margin is slightly lower, with an interval spanning zero. Thus the evidence does not establish reliable superiority over that baseline. The original balanced benchmark and this image-equal exchange panel have different weights and eligible images; their accuracy levels must not be conflated.

The original benchmark's identical net gain of two examples per seed arose from different cases: repairs/new errors are 3/1, 4/2 and 2/0. All repair events fall in the original lowest-margin band, but none of the individual cases improves in all three seeds. This narrows the interpretation of the earlier pilot rather than overturning it.

Full tables, difficulty strata, case records and the figure are in the [result README](../../results/pets-selector-exchange-v1/README.md). New insight I5 records the distinction between useful task information within a model and a dependable advantage over a strong shared model. H2 is unchanged: this run does not test normalized feature-displacement geometry.

## Next decision

The controlled exchange is a positive step for the proposal. Keep the main question focused on when conditioning helps relative to strong shared selection. The earlier teacher-imitation curves suggest testing additional distinct training images at a controlled optimizer-step budget, rather than merely increasing epochs. Broader validation and the other token budgets remain necessary; the official test set stays reserved.

## Evidence and backup

New evidence is under `results/pets-selector-exchange-v1/`, approximately 3.9 MB before documentation. This includes all case arrays, summaries, figure exports, error records and the GPU replay. No new weights, public datasets or pretrained-model downloads are created. Existing final selector weights and training supervision remain in the `pets-selector-pilot-v1` release. Original replay files remain under `runs/pets-selector-exchange-replay-v1/` on the server. The new small bundle is retained locally and published through Git, with destination recovery checked after publication.

Recovery from GitHub commit `d15d273` passed on the server: five JSON records, two NumPy archives and the PNG opened; every numeric array is finite. Both retained replay files match the original server outputs byte for byte. Regeneration from the fetched parent predictions reproduces all 34 outcome/metadata arrays exactly and all 4,310 compared summary numbers exactly. The separately retained text-angle field differs by at most 1.42e-14 degrees across the local/server numerical environments; this was the sole difference caught by an initial exact-array comparison. Both CSVs reproduce their contents, allowing that same angle roundoff in the exchange CSV. Temporary regeneration outputs were automatically removed. See [backup_verification.json](../../results/pets-selector-exchange-v1/backup_verification.json). The completed milestone is tagged `pets-selector-exchange-v1`.
