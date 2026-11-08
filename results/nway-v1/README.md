# Two versus five candidate classes

Completed 2026-11-08. Five-class tasks reduced the near-ceiling accuracy of the binary setting, but did not increase the learned conditioning advantage over strong shared selection. This is a focused extension of the existing study, rather than a new main result.

The experiment uses frozen CLIP ViT-B/16, pruning after block 6, 98 retained tokens, and seeds 2041–2043. C receives an unordered candidate-set descriptor; U allocates a similar parameter budget to image processing. Both task sizes use the same descriptor and supervision construction, which reduce to the previous definitions for two classes. All 24 selectors were newly trained with fixed update counts on the new task collections.

These are development-image results on candidate combinations not used for training. The [earlier reserved binary test](../final-test-v1/README.md) remains separate; its images, tasks and trained weights differ from this experiment.

## Classification

Accuracy is averaged equally across tasks and candidate classes, then across the three seeds. C−U differences are percentage points. Intervals below adjust over the four dataset/task-size comparisons; the [source table](analysis/contrasts.csv) also contains ordinary 95% intervals.

| Dataset | Candidate classes | Full CLIP (%) | C (%) | U (%) | C−U (pp) [adjusted interval] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pets | 2 | 99.321 | 99.378 | 99.287 | +0.091 [−0.057, +0.284] |
| Pets | 5 | 97.812 | 97.593 | 97.673 | −0.080 [−0.336, +0.160] |
| Flowers102 | 2 | 96.642 | 96.699 | 96.814 | −0.114 [−0.408, +0.159] |
| Flowers102 | 5 | 91.157 | 91.500 | 91.673 | −0.173 [−0.629, +0.288] |

All four C−U accuracy intervals include zero. The paired change in the C−U accuracy advantage from two to five classes is −0.171 pp on Pets and −0.059 pp on Flowers; both intervals also include zero. More difficult classification therefore did not produce a demonstrated accuracy benefit from conditioning in this experiment.

## Margin and task exchange

Flowers' C−U geometric-margin advantage decreased from 0.000969 with two candidates to 0.000056 with five. The paired change is −0.000914, with an adjusted interval of [−0.001470, −0.000391]. Pets' corresponding change is unresolved. See [within-setting contrasts](analysis/contrasts.csv) and [candidate-count interactions](analysis/way_interactions.csv).

Correct task descriptions still improve both raw and geometric margins over exchanged descriptions in all four settings. Geometric own-task minus exchanged-task differences are 0.001159 → 0.000308 on Pets and 0.002607 → 0.000352 on Flowers when moving from two to five candidates. These smaller exchange effects describe the observed task dependence; they do not by themselves explain why conditioning becomes less useful. Full exchange estimates are in [exchange.csv](analysis/exchange.csv).

The extension supports a narrower conclusion: relieving binary accuracy saturation is not sufficient to obtain a stronger C−U advantage with this selector and training setup. It leaves the existing distinction between useful task dependence and benefits over strong shared selection intact.

## Execution and evidence

- Pets: 2,944 training and 736 validation images; 148 training and 74 validation candidate sets at each task size. Flowers: 1,020 training and 1,020 validation images; 408 training and 204 validation sets.
- Training used 3,990 updates per Pets selector and 3,978 per Flowers selector. C has 348,033 parameters and U has 347,399.
- The two dataset pipelines ran concurrently on two RTX A6000 GPUs. Their preparation, training and evaluation stage wall times sum to 1,015.2 seconds; recorded optimizer-training time sums to 755.9 seconds. Previously generated CLIP prefixes and 512-mask responses were reused, so their original cost is not part of this extension's runtime.
- Preparation checks confirmed balanced class participation, image/task separation and binary-target equivalence. Independent reconstruction of all classification means, C−U contrasts and candidate-count interaction point estimates matched the analysis to numerical precision. See [preparation checks](preparation_audit.json) and [result reconstruction](result_audit.json).

The [analysis summary](analysis/summary.json), [classification table](analysis/classification.csv), [per-seed results](analysis/per_seed.csv), [figure and editable SVG](figures/), and [execution records](execution/) contain the complete small outputs. The [experiment protocol](../../docs/experiments/2026-11-08-nway.md) describes the task construction and training target.

Three archives totaling 555,008,216 bytes are retained in local `../artifacts/nway-v1/`, server `/data2/zhao.tong/artifacts/nway-v1/`, and the [GitHub release](https://github.com/NathanTong0921/ViT-token-compression/releases/tag/nway-v1). They preserve all 24 checkpoints, task panels and targets, predictions, selected positions, image features and analysis. All local archives passed CRC checks and all 24 checkpoints loaded successfully. Published asset sizes match the local copies; downloading the published analysis archive reproduced the classification tables, analysis summary and configurations. See [archive inventory](archive_summary.json), [release assets](release_assets.json), and [recovery record](recovery.json). Reusable CLIP prefix and mask-response caches remain in their existing server locations and earlier backups. Temporary local evaluation copies and transfer helpers were removed after verification.
