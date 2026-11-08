# Predicting when conditioning helps from prefix content

Completed 2026-11-08. **A lightweight readout of retained token content predicts part of the C−U classification-gap variation on new images and candidate pairs.** It separates groups favoring C from groups favoring U in three of four settings. The useful signal comes mainly from estimating how the two retained sets distinguish the candidate classes. A simple omitted-energy measure does not account for the predictive improvement. Automatic C/U selection has not established an accuracy advantage over both fixed endpoints.

## Experiment

The four existing development panels use frozen CLIP B/16, block 6, Pets/Flowers, 49/98 retained patches and seeds 2041–2043. Existing C/U predictions and masks are reused. Pets has 716 evaluated images and 66 candidate pairs; Flowers has 1,020 images and 204 pairs. Each benefit prediction holds out both the image group and candidate-pair group used for evaluation.

One affine readout maps the mean prefix patch representation to the full CLIP image embedding. It is fitted on the original 2,944 Pets and 1,020 Flowers training images, giving each dataset half the weight. This readout produces six content features describing retained/omitted candidate-direction energy, projected candidate-gap changes and prediction disagreement. The baseline has the original seven selection statistics plus the projected all-token absolute gap, so the main comparison adds content beyond an estimate of task difficulty.

All settings use fixed weighted ridge and shallow histogram-boosted predictors. The primary outcome is the unnormalized correct-minus-competing cosine-score gap; geometric margin and accuracy are also retained. Full-model signed difficulty is an additional diagnostic control. See the [protocol and execution record](../../docs/experiments/2026-11-08-conditioning-content.md).

## Available correction opportunity

The oracle chooses the correct endpoint whenever either C or U is correct. Its gain over the better fixed endpoint is calculated per training seed, then averaged. These are opportunities among the existing models, not achieved routing gains.

| Setting | Oracle accuracy headroom (pp) |
| --- | ---: |
| Pets49 | 0.270 |
| Pets98 | 0.064 |
| Flowers49 | 1.168 |
| Flowers98 | 0.498 |

Flowers49 has both C-only and U-only correct cases: 1.258% and 1.168% of weighted image/task observations, averaged across seeds. Correction identities also vary with training seed. Complete intervals, per-seed results and direction agreement are in [headroom](headroom/summary.json).

## Prediction on held-out images and tasks

R² is measured relative to the training-fold mean predictor. Negative values mean greater error than this mean. The table concerns score-gap prediction, not classification accuracy. Intervals for incremental R² adjust over the four settings.

| Setting | Old statistics R² | Context baseline R² | Content R² | Content minus context ΔR² [interval] |
| --- | ---: | ---: | ---: | ---: |
| Pets49 | 0.0263 | 0.0259 | 0.1092 | 0.0832 [0.0588, 0.1082] |
| Pets98 | −0.0030 | −0.0035 | −0.0021 | 0.0014 [−0.0032, 0.0064] |
| Flowers49 | −0.0013 | −0.0018 | 0.0571 | 0.0590 [0.0434, 0.0741] |
| Flowers98 | −0.0029 | −0.0026 | 0.0307 | 0.0333 [0.0226, 0.0441] |

The same three settings retain positive increments with the full-model difficulty control, within-image comparisons and the nonlinear predictor. Thus the signal includes differences between tasks on the same image, beyond a general estimate of image difficulty.

Using training-fold cutoffs, the linear predictor separates lower and upper thirds with the following actual C−U raw gaps (multiplied by 1,000):

| Setting | Low predicted-benefit group | High predicted-benefit group |
| --- | ---: | ---: |
| Pets49 | −7.451 | +1.882 |
| Pets98 | −0.734 | +0.212 |
| Flowers49 | −4.825 | +4.643 |
| Flowers98 | −1.346 | +3.551 |

For Pets49, Flowers49 and Flowers98, both group means have adjusted intervals excluding zero in their respective directions. Pets98 remains weak. Each group contains approximately one third of the weighted cases. See [all estimates](analysis/summary.csv) and [groups and transfer figure](figures/groups_and_transfer.png).

## Which content matters?

A follow-up component analysis keeps the same folds and ridge settings. It separates two energy features from three projected-gap/disagreement features, keeping the context baseline in every model.

Adding the gap group gives raw ΔR² of 0.0859, 0.0022, 0.0571 and 0.0326 on Pets49, Pets98, Flowers49 and Flowers98. The first, third and fourth increments have positive adjusted intervals and agree across all three seeds. Adding the energy group alone, or after the gap group, gives no established incremental improvement in any setting.

The evidence therefore supports a specific predictive result: **a coarse prefix readout can estimate some of the difference between the candidate discrimination of C's and U's retained sets.** The simpler idea that greater omitted task-direction energy explains when conditioning helps is not supported by the component comparison. Flowers has an univariate high/low omission association, but that is weaker evidence than an incremental predictive contribution. See [component results](components/summary.csv).

## Transfer and automatic selection

With the shared content readout fixed, fit the benefit rule on one dataset and apply it to the other. At 49 tokens, predicted high/low groups separate actual benefits in both directions. Absolute predictions transfer asymmetrically: Flowers→Pets has R² 0.0903, whereas Pets→Flowers has R² −0.0073 despite retaining useful ordering. At 98 tokens, Pets→Flowers gives R² 0.0156 and Flowers→Pets gives −0.0219. The common readout has seen both datasets' training images; this experiment transfers the benefit rule, not an entirely unseen-domain system.

The offline accuracy gate chooses C when its predicted accuracy gain is positive. Neither the linear nor nonlinear gate establishes a higher accuracy than both always-C and always-U in any within-dataset or transfer setting. For example, Flowers49's linear gate gains 0.106 pp over U, with adjusted interval [−0.266, +0.449], and 0.016 pp over C, with [−0.229, +0.257]. Predicting score-gap groups has been more successful than converting the prediction into additional correct answers. Full results are in [routing comparisons](analysis/routing_summary.csv).

## Readout quality, cost and saved evidence

Development embedding cosine similarity to full CLIP is 0.908 on Pets and 0.934 on Flowers; candidate-direction gap correlations are 0.918 and 0.939. The readout's binary accuracies are 95.09%/94.14%, compared with full CLIP's 99.26%/96.74%. Ordinary all-class accuracies are 50.95%/38.82%, compared with 87.23%/70.39%. It is a coarse content probe; direct downstream scores are still measured with the original CLIP suffix.

Feature/readout extraction took 74.47 seconds on four server CPU threads. Main prediction and analysis took 30.48 seconds locally; routing reused those predictions. No new CLIP or selector training, GPU inference or dataset download was needed. Independent checks reconstruct the content features, prediction statistics, groups and routing results from the source arrays.

The readout, feature arrays, fold predictions, tables, figures and small audit records are retained in this directory and ordinary Git. Original prefixes and selection outputs remain in the server runs and their earlier release backups. No new local bulk archive is required.

To reproduce after restoring the existing source arrays:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python scripts/extract_content_benefit_features.py --root . --output results/conditioning-content-v1/features
PYTHONPATH=src OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/analyze_content_benefit.py --features results/conditioning-content-v1/features --output results/conditioning-content-v1/analysis --risk-features u_omitted_contrast_energy
PYTHONPATH=src OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/analyze_content_benefit.py --features results/conditioning-content-v1/features --output results/conditioning-content-v1/analysis --routing-only
python scripts/plot_content_benefit.py
```
