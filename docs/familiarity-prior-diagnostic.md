# Familiarity: candidate preference or image discrimination?

This CPU-only follow-up uses the already saved familiarity-switch features. It distinguishes a shift toward a candidate class from a change in how well scores separate images of the two classes. It does not train a selector or access final-test data.

## Algebra fixes the meaning of the previous margin statistic

Orient every binary score toward H0, the class held out in P0 and seen in P1:

`q = (cosine(image,H0) − cosine(image,H1)) / ||text_H0 − text_H1||`.

Let `y=+1` for H0 images and `−1` for H1 images. The true-class margin is `m=yq`. Let `r_p=q_real,p−mean(q_identity,p)` be the real-text advantage in this fixed direction. The previous seen-minus-held margin row is exactly

`y × (y r_1 − y r_0) = r_1 − r_0`.

This identity holds per row. With the existing equal task/class-side weights, its average equals the mean task midpoint shift. It does not equal the change in class separation. For each task, let `a` and `b` be the average change `r_1−r_0` on H0 and H1 images. Report `(a+b)/2` as the midpoint shift and `(a−b)/2` as the half-separation change, along with both original group means. A positive midpoint can arise when scores of both image classes move toward newly familiar H0. These formulas are definitions; the empirical questions concern their measured magnitude and whether an additive shift predicts changes on different images.

The same decomposition is reported against the wide image-only control. Raw within-task AUC toward H0 is also reported per model, then averaged over tasks/seeds/mappings. One constant added to every image of a task leaves that raw AUC unchanged. AUC phase changes are not the earlier oriented seen-minus-held contrast.

## Fixed correction and prediction experiment

Before examining these new results, fix five image folds with seed2301: sort unique image IDs, randomly permute them, and distribute round-robin. Every task and model occurrence of an image uses the same fold. No labels or class stratification select folds.

For each model, phase and seed, fit both of these corrections using only the other four folds:

- **Global:** the unweighted mean of `q_pruned−q_full` over calibration image/task rows.
- **Task-specific:** the unweighted mean of the same residual within each candidate task.

Subtract the fitted offset from held-image scores. The offset estimator receives scores, image/task IDs and folds, but no true labels or task/class weights. Candidate direction is reconstructed directly from saved image features and the original CLIP text dictionary, using the known training-group membership of candidate IDs. It is not reconstructed by passing true-class labels into calibration. Full CLIP is the fixed reference, so this is a diagnostic correction requiring reference outputs; no deployment or acceleration claim follows.

Ground-truth labels enter only afterward for accuracy, true-margin, AUC and the descriptive decomposition. Evaluate each model separately before averaging outcomes; do not classify averaged model scores. Report raw, global-corrected and task-corrected semantic role effects, absolute accuracy, and accuracy changes from raw. Both corrections are retained, without choosing a favorable one.

An average corrected margin can shrink partly through the mean-subtraction algebra, even under cross-fitting. The stronger correction evidence is therefore:

1. held-image accuracy effects after correction;
2. held-image mean-squared-error reduction when the fitted offsets predict the P1−P0 change in real-minus-identity directional score, against predicting zero change;
3. raw within-task AUC and separation changes, measured independently of correction.

Cross-fitted task offsets can differ across folds, so corrected aggregate AUC is not called offset-invariant. Unweighted calibration means can reflect image composition as well as a stable candidate preference. A successful correction supports an additive explanation of outputs; it does not uniquely identify an internal learned prior.

## Reporting and execution

Use all three training seeds and both fixed identity mappings. Preserve each seed's decomposition, task-level means, fold assignments, fitted offsets, reconstruction checks and compact score arrays. Image-bootstrap intervals use seed2302 and2,000 resamples, conditional on the fixed folds, fitted corrections, models and task panel. Corrections are not refitted inside bootstrap. New partition seeds2100/2101 use the same analysis without modification; the initial2099 analysis is exploratory and retained separately.

Code/protocol are committed before running the new analysis. Example commands from the server repository root:

```bash
.venv/bin/python scripts/analyze_familiarity_prior.py \
  --source runs/pets-familiarity-switch-evaluation-v1 \
  --output runs/familiarity-prior-v1/pets-p2099
.venv/bin/python scripts/analyze_familiarity_prior.py \
  --source runs/flowers-familiarity-switch-evaluation-v1 \
  --output runs/familiarity-prior-v1/flowers-p2099
```

For the new partitions, source names are `runs/{dataset}-familiarity-switch-partition-{2100,2101}-evaluation-v1`; use separate outputs `{dataset}-p{partition}`. `--plan-only` writes the protocol and fixed image folds without loading model predictions. No source artifacts are modified.
