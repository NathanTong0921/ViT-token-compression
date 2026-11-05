# Patch Ranking preservation baseline

2026-11-05. Frozen CLIP ViT-B/16, pruning after block 6, retaining 98 of 196 patches. Three predictor seeds: 2041, 2042, 2043. This is **Patch Ranking–Preservation (adapted)**, using the paper's sliding-window preservation ranking and the public predictor/KL implementation in our existing setup. See the [implementation correspondence](../../docs/patch-ranking-implementation.md).

## Classification on unseen development combinations

Accuracy uses equal task and class-side weighting. Pets contains 716 eligible validation images, 66 tasks and 2,624 image/task cases; Flowers contains 1,020 images, 204 tasks and 4,080 cases. Learned-method results average all three seeds.

| Method | Selector parameters | Pets accuracy (%) | Flowers accuracy (%) |
|---|---:|---:|---:|
| Full CLIP | 0 | 99.2644 | 96.7402 |
| Random | 0 | 98.5825 | 95.9314 |
| Attention | 0 | 98.9932 | 96.0784 |
| Original shared U₀ | 348,033 | 99.1089 | 96.8382 |
| Strong shared U | 347,399 | 99.1221 | 96.9118 |
| Conditioned C | 348,033 | 99.1481 | 97.1160 |
| Offline preservation ranking | — | 97.9147 | 96.0539 |
| Patch Ranking (adapted) | 46,246 | 98.7832 | 96.2500 |

| PR minus comparator | Accuracy difference (pp) | Adjusted interval (pp) |
|---|---:|---:|
| Pets: strong shared U | −0.3389 | [−0.7360, −0.0389] |
| Pets: conditioned C | −0.3648 | [−0.7447, −0.1072] |
| Flowers: strong shared U | −0.6618 | [−1.0499, −0.2491] |
| Flowers: conditioned C | −0.8660 | [−1.3235, −0.4330] |

Intervals use image-paired bootstrap, with adjustment for the four primary comparisons. Both margin comparisons also favor U and C in each dataset. Full estimates, ordinary intervals and every seed are in each dataset's `analysis/` directory.

## What this comparison adds

Our strong shared selector exceeds this independent method adaptation on both datasets. It also has more selector parameters: U is 7.5 times the size of the PR predictor. The existing C-versus-U comparison isolates the value of task information at similar parameter counts; the PR comparison places that controlled system alongside a published method's architecture and objective.

The offline preservation ranking is already weaker than U and C, even with no predictor approximation. Predictor capacity alone therefore does not explain the entire problem: the ranking target's classification utility also matters. The predictor improves over its teacher by 0.869 pp on Pets; Flowers has a +0.196 pp mean difference with an interval spanning zero. Top-98 teacher overlap averages 75.24% and 67.27%, respectively.

The public KL target gives equal target probability to the first 146 ranked patches, while our evaluated budget retains 98. This is a concrete distinction between the training target and the final selection. The present run has not isolated its contribution from local-window ranking, pruning depth or representation preservation. A candidate-class preservation objective is a possible next intervention using the saved deletion responses; it has not been run.

## Actual latency

RTX A6000, FP16 CLIP and FP32 selectors. Each call includes the full image prefix, online selector/pooling, ranking, gather, suffix and binary classification. Images and class embeddings start on the GPU. Batch 32 entries below are milliseconds per batch, taking the median of the three checkpoint medians. Each checkpoint has 60 timed calls after warmup; no foreign compute process was observed in either timing window.

| Method | Pets batch 1 | Pets batch 32 | Flowers batch 1 | Flowers batch 32 |
|---|---:|---:|---:|---:|
| Full CLIP | 4.415 | 24.534 | 4.499 | 24.536 |
| Attention | 5.271 | 19.535 | 5.099 | 19.546 |
| Strong shared U | 4.999 | 19.581 | 5.072 | 19.629 |
| Conditioned C | 4.961 | 19.551 | 5.158 | 19.567 |
| Patch Ranking (adapted) | 5.029 | 19.257 | 5.234 | 19.328 |

PR is about 1.5–1.7% faster than U in batch 32; batch 1 has no consistent advantage. The shared image encoder has 86.19 million parameters, so the selector size difference changes the total image-path parameter count by about 0.35%. Full timing samples and IQRs are retained in `latency/`.

## Cost and artifacts

Preservation rankings cover all training and validation images: Pets 2,944/736, Flowers 1,020/1,020. Their generation takes 680.4/417.7 seconds. Three 50-epoch predictor fits plus diagnostics take 114.1/84.2 seconds, of which optimizer training takes 43.9/14.6 seconds. These are measured elapsed times using cached frozen prefixes on the shared server, separate from the isolated inference measurements.

Git retains statistics, per-example classification CSVs, training curves and figures. The [release archive](https://github.com/NathanTong0921/ViT-token-compression/releases/tag/patch-ranking-v1) preserves all six checkpoints, train/validation rankings, deletion class scores, selected positions, embeddings and timing records at their original `runs/` paths. Local and server artifact directories retain the same archive. Frozen prefix caches are reused from earlier milestones.

This completes the external-reference extension to proposal H3. Reserved final-test images and class combinations remain unused. The next research choice is whether to test a candidate-class preservation target before fixing the final comparison.
