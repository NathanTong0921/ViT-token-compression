# Can image content explain when conditioning helps?

## Question and starting evidence

The earlier benefit predictor used seven summaries of text separation, selector-score boundaries and selection disagreement. It found a small signal on Pets49 but no transferable rule. This experiment asks whether the *content* retained or omitted by a shared selector explains more of the C−U difference.

Use the existing development panels for Pets/Flowers, B/16, block 6, 49/98 patches, and seeds 2041–2043. The C and U weights remain fixed. No additional CLIP or selector training is required. The previously completed final-test results remain separate from this development extension.

## 1. Measure the available opportunity

For each trained C/U pair, count both-correct, both-wrong, C-only-correct and U-only-correct examples using the existing task/class weights. The accuracy of an oracle that selects a correct endpoint whenever possible is the union of their correct predictions. Its gain over the better fixed endpoint equals the smaller of the C-only and U-only proportions.

Report each seed, their mean, image-bootstrap intervals, and agreement of correction directions across seeds. Compute the analogous maximum-margin reference separately. This measures the opportunity among these fixed endpoints; the oracle uses observed outcomes.

The completed opportunity check motivates focusing interpretation on Flowers49 while retaining all four settings: mean per-seed oracle accuracy headroom is 1.168 pp on Flowers49, 0.498 on Flowers98, 0.270 on Pets49 and 0.064 on Pets98.

## 2. Read task-relevant content before selection

Fit one linear map from the mean of the prefix patch tokens to the full CLIP image embedding, using the original Pets and Flowers training images. Give each dataset half the fitting weight. Standardize the input using training statistics and use ridge penalty 0.01. A shared map keeps the content coordinates consistent between datasets. Evaluate embedding cosine similarity and candidate classification on the development images before interpreting the resulting features.

Apply this map to each prefix token and compare its response along the normalized difference of the two candidate text embeddings. The six content features are:

1. Fraction of centered candidate-direction energy outside U's retained set.
2. Difference between the fractions of that energy retained by C and U.
3. U's projected candidate-gap loss relative to the all-token projection, oriented by the all-token prediction.
4. C's projected candidate-gap recovery over U, with the same orientation.
5. Absolute candidate gap of the all-token projection.
6. Whether the projected C and U predictions disagree.

Pooled projected embeddings are normalized before forming candidate gaps. These are learned readouts of contextual token content, not labels for individual anatomical parts. True classes and actual post-pruning scores supply outcome targets only; they are not predictor inputs.

## 3. Test a content-based explanation

Primary outcome: C−U in the unnormalized correct-versus-competing cosine-score gap. Also report geometric margin and accuracy. Compare the old seven features, those features plus projected all-token task difficulty, and all content features together. The key incremental comparison is against the difficulty-enriched baseline. A diagnostic separately controls the actual full-model signed margin.

Use the existing five image folds and five candidate-pair folds. Each prediction holds out both its image group and its candidate-pair group. Pool the three training seeds while keeping repeated observations in the same folds. Fit the existing weighted ridge (penalty 0.001). Also compare the difficulty-enriched baseline and full content inputs with the same small nonlinear regressor: histogram gradient boosting, 100 iterations, 7 leaves, learning rate 0.05, minimum leaf size 30, L2 penalty 1 and seed 2361, with early stopping disabled. The nonlinear comparison tests whether the old linear readout limited prediction; it is run for raw gap and accuracy.

Report held-out prediction error, R-squared, improvement over baseline and within-image improvement. Form high/low thirds using training-fold thresholds, then measure actual C−U differences in the held-out groups. Repeat source-to-target transfer between datasets at the same budget with unchanged coefficients. Bootstrap images and retain all three seed results; intervals across the four settings accompany the primary comparisons.

If the content scores identify useful groups, use out-of-fold predicted accuracy gain above zero to choose C or U and compare with always-C, always-U and the training-selected fixed choice. This follows the observed opportunity check; it does not require a new selector architecture. Any routing cost must include the features and selectors it consumes.

## Outputs

Save the projection, its evaluation, feature definitions, aligned feature/outcome arrays, fold predictions, tables and figures under `results/conditioning-content-v1/`. Large source prefixes remain on the server. Record findings in the research insight log after the analysis completes.
