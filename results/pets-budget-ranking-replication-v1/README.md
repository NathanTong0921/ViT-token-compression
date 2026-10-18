# Budget-context ranking: 36-image independent probe-seed check

Completed 2026-10-18. Same frozen CLIP ViT-B/16/block-6 design as the [144-image main experiment](../pets-budget-ranking-v1/README.md). This subset has 36 images, 66 tasks and 132 balanced image/task examples per budget; the exchange panel has 31 images. Probe mask seed: 2032. The second-seed check changes only probe masks, not the image panel or candidate-bank randomization. It tests mask-seed sensitivity, not new-image generalization.

All task rankings use true labels offline. No selector was trained, and final test data remains unused.

| Patches | Near-full ranking | Budget-matched ranking | Near-full correlation | Budget-matched correlation |
| --- | ---: | ---: | ---: | ---: |
| 147 | 100.00% | 100.00% | 0.917 | 0.896 |
| 98 | 98.48% | 100.00% | 0.597 | 0.578 |
| 49 | 93.94% | 100.00% | 0.243 | 0.651 |

Same evidence layout as the main bundle. Keep-all matches native and saved CLIP exactly. Run source: `dee4d0c`; comparison analysis: `d413010`. Measured total wall time: 53.70 seconds on one RTX A6000; not a deployment latency measurement.

![Comparison](budget_comparison.png)

See `seed_stability.json` for paired mask-overlap, coefficient-correlation and outcome-stability measurements against the first probe seed, including the exact formulas and image-bootstrap settings.
