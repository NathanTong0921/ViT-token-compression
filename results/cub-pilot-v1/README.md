# CUB candidate-set pilot

Completed 2026-11-08. **Task conditioning did not improve on strong shared selection in this CUB pilot.** With five candidates, C has lower accuracy than U; with two candidates, the accuracy difference remains unresolved. C has lower raw and geometric margins at both task sizes, even though using the correct task description improves margins over exchanging descriptions.

## Setting

Frozen CLIP ViT-B/16, pruning after block 6, 98 retained patch tokens, and seeds 2041–2043. All 200 species participate. The fixed image split selects 20 training and 5 development images per species from the official training set: 4,000 training and 1,000 development images. The official test set is unused.

At each task size there are 800 training and 400 development candidate sets, with no shared combinations. Training covers 32,000 two-way or 80,000 five-way image/task examples; development evaluation covers 4,000 or 10,000. Twelve selectors were trained for 3,990 updates each, using the existing candidate descriptor and fixed-budget supervision recipe. C has 348,033 parameters and wide U has 347,399. Part and attribute annotations are retained for possible analysis but were not used by these selectors. No PR or L/14 expansion was run.

See the [fixed protocol and execution record](../../docs/experiments/2026-11-08-cub-pilot.md) and [configuration](../../configs/nway_cub.json).

## Classification

Accuracy is averaged equally across tasks and candidate classes, then across the three seeds. C−U differences are percentage points. The intervals adjust over the two task-size comparisons for each metric.

| Candidate classes | Full CLIP (%) | Attention (%) | C (%) | U (%) | C−U (pp) [adjusted interval] |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 98.675 | 98.350 | 98.375 | 98.517 | −0.142 [−0.308, +0.017] |
| 5 | 95.800 | 95.170 | 95.330 | 95.590 | −0.260 [−0.507, −0.040] |

The five-minus-two change in the C−U accuracy difference is −0.118 pp, with interval [−0.327, +0.092]; the experiment does not resolve a task-size interaction in accuracy. For context, full CLIP's ordinary 200-species development accuracy is 54.5%. Randomly constructed candidate subsets therefore remain substantially easier than the complete species task. This observation comes from the [initial references](execution/cache/val/baseline_summary.json), not a separate trained model.

## Margins and task exchange

Values below are geometric margins multiplied by 1,000; intervals use the same two-setting adjustment. Raw-margin comparisons have the same signs and also exclude zero.

| Candidate classes | C−U geometric margin × 1,000 | Own-task minus exchanged-task geometric margin × 1,000 |
| --- | ---: | ---: |
| 2 | −1.204 [−1.643, −0.798] | +0.602 [+0.478, +0.729] |
| 5 | −0.857 [−1.232, −0.485] | +0.108 [+0.060, +0.160] |

The corresponding exchange accuracy differences remain unresolved at both task sizes. CUB thus extends the existing distinction between using task information and outperforming a similarly sized image-only selector. It does not provide a performance-based reason to replace Pets with CUB as the main reported dataset, or establish a new explanation based on bird parts. Part-level analysis has not been performed.

## Execution, cost and artifacts

Full-token inference matched native CLIP exactly for all 5,000 cached images. The new supervision required 512 masks per image, totaling 2,560,000 suffix examples. Feature extraction and initial references took 46.8 seconds of recorded stage wall time. Probe generation took 748.0 seconds across the parallel train/development jobs, including 626.9 seconds in the inference loops. The study then recorded 111.1 seconds for target preparation, 455.9 for training and 44.7 for evaluation; optimizer updates across the twelve selectors took 347.3 seconds.

Summed recorded stage wall time was 1,406.6 seconds (0.391 hours), including CPU work and I/O. These shared-server timings are execution-cost records, not an exclusive-GPU latency benchmark or a measurement of device-kernel time. Cache-stage model setup and dataset download are not included in that sum. Source timings are in [cache records](execution/cache/) and [pipeline status](execution/study/status.json).

The [analysis summary](analysis/summary.json), [classification table](analysis/classification.csv), [paired contrasts](analysis/contrasts.csv), [task exchange](analysis/exchange.csv), [per-seed results](analysis/per_seed.csv), and [figure](figures/candidate_count_effect.png) retain the small outputs. Complete raw outputs remain under server `runs/cub-nway-v1/`. Remote archive packaging and verification are pending; local bulk archives will not be retained.
