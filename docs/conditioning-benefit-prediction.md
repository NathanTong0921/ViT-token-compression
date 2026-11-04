# Predicting when candidate conditioning adds value

Fixed 2026-11-04 before fitting outcome predictors. This development analysis
follows I27–I29 and asks about gains over a strong shared selector, not task-mask
exchange losses. It uses existing Pets/Flowers49/98 outputs only. No selector,
CLIP model or supervision teacher is trained; the small ridge models below are
statistical probes. Reserved final-test images and tasks remain untouched.

## Outcomes and observable inputs

Primary: geometric classification-margin gain C_s−U_s for each of the three
fixed seeds. Secondary: CU_s,t−UU_s,t for all six distinct ordered seed pairs.
An additional secondary outcome subtracts the starting-endpoint gain:
`(CU−C)−(UU−U)`. This separates final mixture quality from its mixing increment.

For each outcome repeat the analysis using the unnormalized, signed difference
of correct-minus-competitor cosine scores. Geometric margin divides by the
candidate-text distance; this sensitivity prevents that denominator alone from
being interpreted as a discovered task condition. Labels and post-pruning scores
are used only to form outcomes, never as observable predictor inputs.

Observable baseline inputs, available before the suffix:

1. Candidate text distance.
2. Its reciprocal (explicit control for the margin denominator).
3. Shared U's kth-to-(k+1)th score gap divided by its row standard deviation.

The selection-aware model adds:

4. Fraction of C's retained tokens outside U's retained set.
5. C's normalized boundary gap.
6. One minus Spearman correlation of C/U token scores.
7. Disagreement fraction times U's normalized boundary gap.

The two mixture outcomes also add donor-disagreement excess:
`(|C_s minus U_t| − |U_s minus U_t|)/k`.
All score-derived features are positive-scale invariant, following the previous
diagnosis. There are no image/class/task IDs as inputs, no correct-class text
indicator, and no suffix scores. Each feature vector describes one actual seed
or selector pair; averaging ensembles is not required to form an input.

## Predictors, within-image analysis and validation

Fit weighted linear ridge with penalty0.001 on standardized predictors, an
unpenalized intercept, and weights normalized to sum1. Normalization is fitted
on training rows only. Pool the fixed seeds/pairs with row weight w/S; all repeats
of an image/task stay in the same split. No hyperparameter or threshold search.

Use five image groups (seed2351) and five task-pair groups (seed2352), produced
by shuffling sorted unique IDs and cyclic assignment. For each of25 cells,
predict its rows after excluding both its image group and its task group from
the probe's training data. Every row receives exactly one held-image/held-task
prediction. Original class-pair and image coverage is retained, including natural
gaps in training-fold class coverage.

Compare a training-fold constant mean, the observable baseline, and the
selection-aware probe. Primary predictive metrics are held-out weighted MSE,
R-squared relative to the training-fold constant prediction, and incremental
R-squared of selection inputs over the observable baseline. Pool all held-out
rows before computing ratios, rather than averaging cell-specific R-squared.

To isolate task variation, center outcomes and held-out predictions within each
(image,seed/pair) using original row weights. Report the corresponding within-image
predictive R-squared and increment over the baseline. Centering test outcomes is
only an evaluation operation, not an input or fitted test-image intercept.

Two diagnostic probes append the uncompressed signed geometric margin and its
square to the baseline and selection-aware inputs. They test whether selection
information adds predictive value beyond full-model difficulty. These label-informed,
full-forward controls are not used for routing or claimed to be available at pruning.

As a fixed secondary transfer check, train on all development rows of Pets and
evaluate Flowers at the same token budget, and vice versa. Apply the source-domain
normalization, coefficients and intercept without target-domain adjustment. No
dataset ID is supplied. This checks whether a low-dimensional explanation extends
beyond a separately fitted rule for each dataset.

## Practical and statistical summaries

For the direct and final-mixture outcomes only, an offline routing diagnostic uses
the selection-aware probe's predicted geometric gain >0 to choose C versus U,
or CU versus UU, separately for each seed/pair. Compare always-baseline,
always-alternative, and the fixed choice selected by the training fold's mean
geometric gain. Report accuracy as well as margins. No threshold is tuned on
held-out outcomes. Computing the features can require both selectors; this is
not a measured deployment-speed improvement.

Compute squared prediction errors separately for each seed/pair, then average
these errors before image-bootstrap summaries; this evaluates an individual
selector/pair, not prediction of an ensemble-averaged target. Use2000 draws,
seed2353, ordinary intervals plus four-setting adjustment within each outcome
and metric. Intervals condition on the fitted probes, fixed class-pair panels
and trained selectors; there is no repeated predictor refitting in bootstrap.
Keep every setting, seed/pair and transfer direction. Coefficients are predictive
associations among correlated inputs, not causal mechanism estimates.

Save feature/outcome arrays, folds, all held-out predictions, coefficients,
per-run effects, transfer predictions and plots. This round will establish how
much of the benefit is predictable from the specified inputs; it does not presume
a successful rule or automatically trigger a new selector-training round.
