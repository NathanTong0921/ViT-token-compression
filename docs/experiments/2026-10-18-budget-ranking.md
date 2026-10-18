# 2026-10-18: budget-context ranking comparison

## Research decision

The prior offline pilot's 49-patch failure motivated a controlled test of importance-estimation context. The fixed [protocol](../budget-ranking-pilot.md) was committed before GPU execution. This is an extension of the proposal's offline ranking and supervision-generation stage, not a change of the project's central question.

Primary comparison: same 512-probe standardized ridge estimator, near-full 192-patch context versus target 147/98/49-patch context, followed by actual top-k execution. The 128 held-out masks per budget assess prediction separately. The prior block-deletion ranking is a historical reference, not the causal algorithm control. Shared-search comparison uses a common union of all three 64-slot banks.

## Runs and provenance

All runners used commit `dee4d0c`. Comparison/paired-prediction analysis uses `d413010`; the existing exchange analysis is unchanged. Frozen OpenAI CLIP ViT-B/16, original preprocessing/prompts, block-6 pruning, validation images/task pairs only. The full baseline was recomputed using original 64-image batches before panel sampling; every keep-all/native/saved feature, score and prediction identity check passes exactly. Suffix queries use fixed batches of 64, padding then dropping extra rows.

| Bundle | Images | Mask seed | Total wall seconds | Physical suffix examples | Requested suffix examples |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pets-budget-ranking-smoke-v1` | 36 | 2031 | 52.91 | 184704 | 172320 |
| `pets-budget-ranking-v1` | 144 | 2031 | 196.20 | 738816 | 689280 |
| `pets-budget-ranking-replication-v1` | 36 | 2032 | 53.70 | 184704 | 172320 |

Hardware: one RTX A6000, physical GPU 2, exposed as cuda:0. Python 3.10.12, PyTorch 2.6.0+cu124, NumPy 1.26.4; OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=4. Runs include loading, consistency checks, probing, fitting, selection evaluation and file writes. Combined time: 302.81 seconds = 0.08411 single-GPU process hours. This is neither GPU-active time nor end-to-end deployment latency. Peak allocated tensor memory was about 0.756 GB in each run. No cross-image selector was trained; per-image ridge fitting costs are reported separately in each run summary as part of supervision generation.

Server command pattern, from `/data2/zhao.tong/experiments`:

```bash
CUDA_VISIBLE_DEVICES=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python -u scripts/run_budget_ranking_pilot.py --config configs/pets_budget_ranking_pilot.json --output runs/pets-budget-ranking-v1
```

The smoke/replication configurations are `pets_budget_ranking_smoke.json` and `pets_budget_ranking_replication.json`; their output directories match the bundles above. Analyze both arm subdirectories with `analyze_selection_pilot.py --search-depth`, then run `compare_budget_ranking.py --results <bundle> --previous <matching old panel> --plot`. The two 36-image panels match `pets-selection-smoke-v1`; the 144-image panel matches `pets-selection-v1`.

## Main findings

The [main result README](../../results/pets-budget-ranking-v1/README.md) contains tables and interpretations. At 49 patches, the same-estimator control changes from 94.89% to 100.00% balanced accuracy and from 0.253 to 0.657 held-out margin correlation. The old deletion ranking reaches 93.37%; attention 97.92%. At 147 patches there is no clear margin gain over the near-full model. At 98, actual selections improve even though held-out correlation slightly declines. Thus context affects supervision quality and prediction quality is not interchangeable with optimized-selection quality.

The improved task ranking still does not beat the common privileged shared search, which knows the label and both tasks. H1's task-specialization evidence strengthens; H1's advantage over a strong shared choice is unresolved, H2 remains exploratory, and H3 has not begun. The 100% result is label-informed, including recovery of five full-model mistakes, not deployable performance.

The independent probe seed preserves the strong-compression result on the same 36 images. Those images are included in the main 144 panel, so this is a sampling-seed check rather than an independent-image confirmation.

The paired seed check also shows non-unique selections: budget-matched 49-patch sets overlap by 46.3% while both seeds achieve 100% on this subset and the paired mean-margin interval spans zero. The saved `seed_stability.json` reports all budgets, aggregation and image-bootstrap settings. Treat this as motivation to judge learned selectors by actual downstream behavior rather than exact agreement with one hard mask; it does not establish statistical equivalence.

## Next experiment

Proceed to the small matched conditional/image-only selector trial at 49 patches, using training images and training task pairs. Both models should use the same budget-matched task-target distribution and training budget; the image-only model omits candidate information. Evaluate downstream accuracy and margins on disjoint validation images and unseen task combinations. Keep the strong shared-search reference separate from deployed model comparisons. The next trial should test learnability promptly instead of continuing to optimize label-informed choices indefinitely.

## Checks and retained evidence

The runner version passed 36 tests on the server. The subsequent paired-prediction analysis change passed 11 budget tests locally, including its joint-valid Pearson pairing test. Raw reconstruction is described in the [audit note](2026-10-18-budget-ranking-audit.md). Primary plots were visually checked. The final server suite passed all 37 tests after the analysis change. GitHub recovery verification is recorded below.

All raw probes/responses, coefficients, selected positions, candidate scores, per-example tables, summary analyses and figures are retained locally and in Git; total bundle size before final notes is approximately 209.5 MB, with largest file 20.2 MB. Original outputs remain in server `runs/`; tracked curated copies live in `results/`. Public datasets/checkpoints were reused in place. No new local transfer archives or persistent prefix caches were created.

## GitHub recovery verification

After pushing the results, the server fetched and checked out GitHub `main`. These new tracked `results/` paths were obtained from GitHub, separate from the original GPU outputs under `runs/`. All 87original run files (excluding text logs) matched the fetched copies byte for byte. All 45 NumPy archives, 19 JSON files and 3 PNG figures opened successfully; floating-point arrays were finite. All three comparison summaries were rebuilt from the fetched artifacts and agreed with the saved values within 1e-10. No checkpoint is produced at this stage.

A redundant fresh full clone on the local machine was stopped because its download was slow; its partial directory was removed. Verification used the server's completed GitHub fetch instead. Local scientific evidence remains intact. Code and evidence are retained locally, on the server, and on GitHub; this milestone is tagged `pets-budget-ranking-v1`.
