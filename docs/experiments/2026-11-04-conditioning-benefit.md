# Conditioning-benefit prediction — 2026-11-04

## Purpose and prior design

Follow I27's task-specific mixing evidence and I29's scale diagnosis by testing whether observable selection signals predict gains over the strong shared baseline. The user authorized this priority over immediately training another objective. Protocol and implementation were committed and pushed as `0d97885` before fitting real outcomes, on `exp/conditioning-benefit-prediction`.

The fixed protocol is [conditioning-benefit-prediction.md](../conditioning-benefit-prediction.md). All four settings, both outcome scales and all three panels were retained. Feature construction was smoke-tested without fitting outcome predictors; separate mathematical and synthetic checks preceded the real run.

## Execution

Server: `/data2/zhao.tong/experiments`; Python: `.venv/bin/python`. The existing core-protection raw directory was packaged on the server because the archive previously existed only in local backup. Source masks/endpoints and metadata reproduce exactly when loaded.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:scripts \
.venv/bin/python -u scripts/analyze_conditioning_benefit.py \
  --core-archive /data2/zhao.tong/artifacts/parallel-research-pilots-v1/core-protection-v1.zip \
  --hybrid-archive /data2/zhao.tong/artifacts/hybrid-followup-v1/hybrid-specificity-v1.zip \
  --output runs/conditioning-benefit-v1
```

Started at 2026-11-04 09:04:43 UTC. All 12 double-holdout and 12 transfer panels completed in 38.11 seconds; no new neural training or CLIP inference. Original logs and launch metadata remain in the server run directory. Results were copied directly to the local repository; no download/extraction of the existing large source archives was needed.

## Findings

Pets49 direct selection inputs add within-image predictive information (ΔR² 0.0343, adjusted interval [0.0174, 0.0524]); the full direct probe has modest R² 0.0271 and an adjusted interval reaching zero. Raw-gap sensitivity and difficulty controls agree. Other direct-benefit settings do not replicate a comparable signal. Flowers49's average mixed-selection advantage remains, but its example-level final hybrid benefit is not successfully predicted. Cross-dataset predictions and offline routing do not establish a general adaptive advantage.

The extra mixing-increment outcome has positive R² at both 49-token settings. Because this target subtracts the starting C−U advantage, it cannot alone identify a mechanism for final mixture quality. See [all results](../../results/conditioning-benefit-v1/README.md).

## Checks and retained evidence

Independent review verifies all 24 panels' MSE/R², within-image centering and routing, all 25-cell split assignments in each setting, and separate-solver reproduction of one actual fold and one cross-dataset fit. Maximum reproduced prediction discrepancy is below 3e-16. Scope is recorded in `independent_check.json`; bootstrap and CLIP were not rerun.

All 28 input/prediction arrays are backed up in a 115,185,448-byte release ZIP, retained locally and on the server. Git keeps the small summaries, fitted probes, folds, figures and manifest. This adds no new model weights. The final reserved test remains untouched. I30 records the scientific consequence; next training remains a separately specified follow-up.

Serialization follow-up: generated CSV line endings were normalized to LF and the writer updated accordingly. Numerical values, folds, saved arrays and fits were unchanged; the analysis protocol records the original execution commit.
