# Fixed hybrid-target distillation

This follow-up completed all 18 new branch fits: three fixed seeds (2041–2043), two residual arms, and three dataset/budget settings. It reuses the 18 paired original-target fits and their exact frozen shared bases. The new supervision did not establish an advantage for the task-conditioned student over the same-size image-only student.

The useful distinction is between fitting the target distribution, reproducing its top-k mask, and preserving the teacher’s classification gain. These outcomes separated: at Flowers 49, the hybrid-target KL became much lower while mask overlap did not improve; at Flowers 98, the image-only student improved its margin over the shared base, but still fell short of the hybrid teacher and did not improve accuracy.

## Design and checks

- Frozen CLIP ViT-B/16, existing pruning position, shared U with 347,399 parameters; task correction 9,801 and image correction 9,850 trainable parameters. Both corrections start at zero.
- Hybrid teacher: keep the intersection of frozen C and U; replace half (rounded down) of the exclusive C tokens, removing the lowest C ranks and adding the highest U ranks. The total token count stays fixed. Candidate text is supplied without a true-class indicator.
- Both arms minimize the existing softmax KL loss. Only its target changes, from the original soft utility distribution to uniform probability over the hybrid mask. The original data order, updates, optimizer, and final-step evaluation are unchanged.
- Regenerated C, U, and hybrid masks exactly reproduce the historical matched-replacement teacher references. Frozen U masks and CLIP scores exactly reproduce the historical endpoint. Initial original-target losses and sample streams reproduce the paired old fits.
- Evaluation uses the existing held-out validation images and class combinations. The final test split remains unused. Paired intervals resample images after averaging all three seeds; per-seed effects remain in each summary.

## Primary comparison: task student minus image student

Margin is the geometric margin reported throughout the project, expressed below in units of 10⁻³. Accuracy differences are percentage points. Intervals in this table have the predeclared three-setting adjustment, separately for each metric.

| Setting | Margin difference ×10⁻³ | Accuracy difference, pp |
|---|---:|---:|
| Pets, 49 | -0.185 [-0.369, -0.002] | +0.052 [-0.039, +0.163] |
| Flowers, 49 | -0.519 [-0.892, -0.156] | -0.106 [-0.376, +0.139] |
| Flowers, 98 | -0.316 [-0.554, -0.106] | -0.106 [-0.237, +0.000] |

The task branch has lower margin in all three settings and all nine setting/seed combinations. The adjusted Pets margin interval ends very close to zero (−0.00000160 in unscaled units). No setting establishes an accuracy advantage for the task branch; the Flowers 98 adjusted accuracy interval touches zero.

## Did changing the target alter the conditioning advantage?

The planned secondary interaction is `(task − image) under hybrid target − (task − image) under original target`. The following intervals are ordinary paired 95% intervals.

| Setting | Margin interaction ×10⁻³ | Accuracy interaction, pp |
|---|---:|---:|
| Pets, 49 | -0.044 [-0.349, +0.243] | +0.054 [-0.098, +0.210] |
| Flowers, 49 | -0.114 [-0.504, +0.268] | -0.221 [-0.507, +0.057] |
| Flowers, 98 | -0.200 [-0.462, +0.045] | -0.090 [-0.253, +0.065] |

All interaction intervals include zero. This round does not support a reliable target-induced change in the task-versus-image advantage.

## What did the image-only student retain?

All entries are image student minus the named reference, with ordinary paired 95% intervals.

| Setting | Reference | Margin difference ×10⁻³ | Accuracy difference, pp |
|---|---|---:|---:|
| Pets, 49 | Frozen U | -1.508 [-1.872, -1.153] | -0.129 [-0.250, -0.010] |
| Pets, 49 | Original-target image student | -1.810 [-2.228, -1.394] | -0.130 [-0.275, +0.012] |
| Pets, 49 | Hybrid teacher | -0.649 [-1.091, -0.192] | -0.093 [-0.214, +0.024] |
| Flowers, 49 | Frozen U | +0.161 [-0.253, +0.590] | -0.139 [-0.376, +0.090] |
| Flowers, 49 | Original-target image student | -0.608 [-1.062, -0.147] | -0.065 [-0.302, +0.172] |
| Flowers, 49 | Hybrid teacher | -1.696 [-2.187, -1.215] | -0.343 [-0.588, -0.090] |
| Flowers, 98 | Frozen U | +0.389 [+0.142, +0.643] | -0.123 [-0.270, +0.016] |
| Flowers, 98 | Original-target image student | +0.512 [+0.214, +0.809] | -0.147 [-0.335, +0.033] |
| Flowers, 98 | Hybrid teacher | -1.236 [-1.495, -0.977] | -0.237 [-0.392, -0.090] |

Flowers 98 provides a limited positive transfer result: the image student improves margin over U and its original-target counterpart, while its accuracy is not improved. Both Flowers settings retain a clear margin gap between the student and the hybrid teacher. Pets is a contrasting setting: its hybrid teacher itself does not outperform U in margin.

## Same-target fit and mask transfer

Each KL entry below evaluates the **same hybrid target**, regardless of which target trained the student. Thus these are within-target comparisons, not comparisons of absolute KL values across distributions with different entropy. Overlap is intersection/k, balanced over tasks and classes; entries average all three seeds on validation.

| Setting | Student | Initial U overlap, % | Original-trained overlap, % | Hybrid-trained overlap, % | Original-trained hybrid KL | Hybrid-trained hybrid KL |
|---|---|---:|---:|---:|---:|---:|
| Pets, 49 | Task | 83.22 | 79.19 | 81.47 | 1.943 | 0.577 |
| Pets, 49 | Image | 83.22 | 81.89 | 81.92 | 1.835 | 0.568 |
| Flowers, 49 | Task | 73.10 | 70.30 | 70.04 | 2.341 | 0.862 |
| Flowers, 49 | Image | 73.10 | 71.59 | 70.83 | 2.165 | 0.846 |
| Flowers, 98 | Task | 80.95 | 78.64 | 79.76 | 0.822 | 0.470 |
| Flowers, 98 | Image | 80.95 | 78.80 | 80.14 | 0.806 | 0.463 |

At Flowers 49, training overlap also decreases relative to the original-trained counterparts (task: 71.97% → 71.40%; image: 73.52% → 72.12%), while training hybrid-target KL decreases strongly. The mask-transfer gap therefore cannot be described solely as a training-versus-validation gap. Optimizing the softmax distribution does not ensure better top-k membership under this fixed training recipe. A useful future diagnostic would separate probability calibration from ranking changes; it has not been run in this round.

## Absolute classification outcomes

| Method | Pets 49 accuracy, % | Pets 49 margin ×10⁻³ | Flowers 49 accuracy, % | Flowers 49 margin ×10⁻³ | Flowers 98 accuracy, % | Flowers 98 margin ×10⁻³ |
|---|---:|---:|---:|---:|---:|---:|
| Shared U | 99.017 | 117.217 | 96.732 | 90.209 | 96.912 | 106.136 |
| Conditioned C | 98.992 | 114.693 | 96.822 | 90.215 | 97.116 | 107.399 |
| Hybrid teacher | 98.981 | 116.358 | 96.936 | 92.066 | 97.026 | 107.760 |
| Original-target task | 99.016 | 117.378 | 96.773 | 90.573 | 96.920 | 105.896 |
| Original-target image | 99.018 | 117.519 | 96.658 | 90.978 | 96.936 | 106.012 |
| Hybrid-target task | 98.941 | 115.524 | 96.487 | 89.851 | 96.683 | 106.208 |
| Hybrid-target image | 98.888 | 115.709 | 96.593 | 90.370 | 96.789 | 106.524 |

## Task exchange

Correct-task minus exchanged-task scores for the task student, with ordinary 95% intervals. These diagnose use of task information; they are distinct from comparison with an image-only student.

| Setting | Margin difference ×10⁻³ | Accuracy difference, pp |
|---|---:|---:|
| Pets, 49 | -0.009 [-0.048, +0.031] | +0.053 [+0.018, +0.097] |
| Flowers, 49 | +0.174 [+0.111, +0.246] | +0.112 [+0.011, +0.215] |
| Flowers, 98 | +0.045 [+0.010, +0.080] | -0.038 [-0.106, +0.030] |

## Incremental recorded compute

The following seconds are summed across seeds. They exclude the already-completed CLIP feature preparation, original utility-supervision generation, and C/U teacher training. Cached features and a tiny residual branch make the new fits short; these times are not a complete project GPU-hour estimate or an end-to-end latency benchmark.

| Setting | Target generation, s | Hidden-feature cache, s | Six branch fits, s | Selector replay + suffix, s |
|---|---:|---:|---:|---:|
| Pets, 49 | 40.88 | 0.28 | 79.10 | 1.34 |
| Flowers, 49 | 7.58 | 0.19 | 81.17 | 2.31 |
| Flowers, 98 | 10.02 | 0.20 | 80.21 | 3.77 |

## Files

- `hybrid_distillation.png/.pdf`: paired classification contrasts and target interaction.
- `hybrid_target_fit.png/.pdf`: train/validation mask overlap for original- and hybrid-trained students.
- `classification_contrasts.csv`, `target_fit_table.csv`: plotting data.
- Each setting directory contains protocol, training/evaluation records, sample audit, target-fit diagnostics, validation image/task metadata, and small per-image analysis files.
- Complete branch checkpoints, target masks, logits, teacher references, and row-level raw outputs remain in the server run directory and are handled by the round’s recovery archive.

Regenerate figures from the repository root:

```bash
python3 scripts/plot_hybrid_distillation.py
```
