# Reserved-test confirmation

Completed 2026-11-08 using the fixed [configuration](../../configs/final_test_v1.json) and [test plan](../../docs/experiments/2026-11-08-final-test.md). Seven existing model/budget settings were evaluated on the original reserved images and class combinations, averaging seeds 2041–2043.

The test confirms that useful task dependence and gains over strong image-only selection are distinct. Correct task descriptions improve classification margin in all seven settings. Against strong shared U, Flowers98 has both an accuracy and margin advantage; Flowers49 and Pets98 remain close, while Pets49 favors U in margin.

## Data and comparisons

- Pets: 3,669 official test images and 68 reserved pairs; binary evaluation covers 3,569 images and 13,529 image/task examples. The remaining 100 images have no pair in the original reserved task split.
- Flowers: 6,149 official test images, 204 reserved pairs, and 24,596 image/task examples.
- C is task-conditioned; I is the original image-only control; U reallocates a similar parameter budget to a wider image network. U is available at B/16 49/98 tokens. Pets147 and L/14 compare C with I.
- PR uses the previously selected minimum-development-KL weights at B/16, block 6, 98 tokens. All weights and evaluation settings were fixed before test inference.

Classification gives equal weight to candidate tasks and class sides. Three seeds are averaged before paired-image bootstrap. Tables below use family-adjusted intervals, separately for each named contrast and metric: four settings for C−U, two datasets for PR comparisons, and seven settings for exchange/H2. Ordinary 95% intervals and every seed remain in the linked tables.

## Conditioning versus strong shared selection

Accuracy differences are percentage points (pp); margin differences and intervals are multiplied by 1,000.

| Setting | C−U accuracy (pp) [adjusted interval] | C−U margin × 1,000 [adjusted interval] |
|---|---:|---:|
| Pets49 | −0.036 [−0.124, +0.040] | −1.770 [−2.164, −1.420] |
| Pets98 | −0.013 [−0.088, +0.058] | +0.059 [−0.225, +0.358] |
| Flowers49 | +0.148 [−0.077, +0.391] | +0.105 [−0.353, +0.541] |
| Flowers98 | +0.152 [+0.026, +0.287] | +1.081 [+0.826, +1.354] |

The direct 98−49 change in the C−U margin advantage is positive on both datasets: Pets +1.829 [1.444, 2.259] × 10⁻³; Flowers +0.976 [0.557, 1.410] × 10⁻³. The corresponding accuracy interactions remain unresolved. See [budget interactions](analysis/budget_changes.csv).

The [complete classification table](analysis/classification_table.csv) and [contrasts](analysis/contrasts.csv) include all seven C−I comparisons, full CLIP, attention and random selection. Flowers L/14 has C−I accuracy +0.142 pp: its ordinary 95% interval is positive, while the seven-setting interval [−0.0006, +0.2705] pp spans zero. Pets147 remains in the full results.

## External PR reference at 98 tokens

| Dataset | PR accuracy (%) | U accuracy (%) | C accuracy (%) | U−PR (pp) [adjusted interval] | C−PR (pp) [adjusted interval] |
|---|---:|---:|---:|---:|---:|
| Pets | 99.2698 | 99.6455 | 99.6325 | +0.3757 [+0.2460, +0.5085] | +0.3628 [+0.2317, +0.5049] |
| Flowers | 96.7005 | 97.3264 | 97.4785 | +0.6259 [+0.4431, +0.8140] | +0.7780 [+0.5779, +0.9826] |

Both U and C exceed this PR adaptation in test accuracy and margin. PR and U have different predictor capacities and training objectives; this table supplies the external accuracy reference alongside the [measured runtime comparison](../patch-ranking-v1/README.md). The C/U comparison evaluates task information at a comparable parameter budget.

## Same-image task exchange and H2

All seven own-task minus exchanged-task margin effects are positive, as are both the difficulty-adjusted and within-image H2 slopes. The table reports margin quantities × 1,000; H2 slopes are per 10 degrees. Intervals adjust over seven settings.

| Setting | Own minus exchanged margin [interval] | Difficulty-adjusted H2 slope [interval] |
|---|---:|---:|
| Pets49 | 1.788 [1.585, 1.996] | 1.334 [1.169, 1.500] |
| Pets98 | 1.246 [1.115, 1.403] | 0.525 [0.420, 0.624] |
| Pets147 | 0.262 [0.212, 0.308] | 0.076 [0.034, 0.121] |
| Pets L/14 128 | 0.491 [0.428, 0.554] | 0.104 [0.039, 0.175] |
| Flowers49 | 3.240 [3.097, 3.387] | 1.795 [1.600, 1.980] |
| Flowers98 | 3.069 [2.972, 3.159] | 1.136 [1.030, 1.247] |
| Flowers L/14 128 | 0.928 [0.868, 0.993] | 0.221 [0.125, 0.311] |

Exchange accuracy improves in Pets49 and all three Flowers settings; the other Pets settings do not establish an accuracy effect. Exchange uses 3,171 Pets images / 24,440 task pairs and 6,149 Flowers images / 36,894 task pairs, averaging task pairs within each image. See [exchange results](analysis/exchange.csv) and [H2 results](analysis/h2.csv).

## Evidence and artifacts

The [figures](figures/), [analysis summary](analysis/analysis.json), [per-seed results](analysis/per_seed.csv), [development replay check](development_replay_check.json), and [independent reconstruction](independent_audit.json) accompany the results. Independent reconstruction matched all method means, contrasts and budget interactions to numerical precision. Full-token inference matched native CLIP during test feature extraction.

Original outputs are under server `runs/final-test-v1/`. Six evidence archives total 5,536,287,748 bytes; backup transfers are being verified. Prefix caches remain server-only and can be regenerated. Existing selector weights are retained in their earlier milestone archives.

This completes the current core test: classification, task exchange, budget comparisons, H2 and the PR reference. Hybrid, depth, identity and training-objective extensions remain development studies; their results are separate from this test.
