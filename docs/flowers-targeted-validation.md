# Flowers102 targeted external validation

Protocol fixed on 2026-10-27 before inspecting Flowers102 model predictions. This study tests whether the 49/98-token contrast and H2 found on Oxford Pets recur on another dataset. Budget selection is informed by the completed Pets results: 49 had a negative conditioned-minus-image-only margin effect, 98 had a positive effect, and 147 had an uncertain effect. All three Pets budgets remain reported. Flowers102 uses the two contrasting budgets for targeted validation rather than repeating the full Pets matrix; this does not establish anything about Flowers102 at 147 tokens.

## Data, class names and task graph

Use the original Oxford Flowers102 images, `imagelabels.mat`, and `setid.mat`. Preserve the official image partitions: 1,020 training images, 1,020 validation images and 6,149 test images. Training and validation each contain ten images per class. Train selectors on the official training images only. The present external analysis uses the official validation images; official test images remain unused for predictions and training, as do the test-task pairs defined below.

Class names come from the authors' category page. Each named thumbnail contains an original image number; look up that number in `imagelabels.mat` to obtain its class ID. Do not treat the alphabetical category-page order as label order. Verify all 102 labels occur exactly once in the resulting mapping and retain the source mapping. Strip surrounding whitespace, retain the official category wording, and use the fixed prompt `a photo of a {class_name}, a type of flower.` No prompt search is included.

Select a balanced subset of candidate-class pairs before seeing model outcomes. Shuffle the 102 zero-based class IDs once with `random.Random(2027)`. For each index in the resulting circular ordering, connect it to the following offsets, canonicalizing each edge as an unordered class pair:

| Task split | Circular offsets | Pairs | Tasks involving each class |
| --- | --- | ---: | ---: |
| Training | 1 through 6 | 612 | 12 |
| Validation | 7 and 8 | 204 | 4 |
| Test | 9 and 10 | 204 | 4 |

The three edge sets are disjoint and cover every class. The remaining 4,131 of the 5,151 possible class pairs are unused. This is a fixed balanced task graph, not an 80/10/10 split of all possible pairs, and it is not selected for semantic similarity, CLIP difficulty or observed performance. It gives every validation image four unseen-combination tasks and six unordered exchanges: 4,080 image/task examples and 6,120 image/task-pair cases across all 1,020 validation images. Evaluation on the training combinations contains 12,240 image/task examples. These are new combinations of known classes, not unseen-class generalization.

## Fixed models and training

Retrain the two selectors on Flowers102; this is a replication of the study in a new dataset, not direct transfer of the Pets-trained selectors. Freeze the same OpenAI CLIP ViT-B/16, native preprocessing and text/image encoders. Prune once after block 6, retain the CLS token, and compare 49 and 98 of the 196 patch tokens. Verify that retaining every patch reproduces native CLIP features and predictions before using the pruned model.

For each budget generate the same type of budget-matched supervision on all 1,020 training and 1,020 validation images: 512 distinct random masks, base mask seed 2031 with the existing budget-specific derivation, and ridge 0.01. Training uses only training-image teachers; validation teachers supply diagnostic references and monitoring. Cache the frozen prefix once and reuse it across budgets. All teacher responses and fitted parameters remain available for checking and recovery.

Keep the selector architecture and learning objective unchanged: 348,033 parameters, hidden width 128, KL imitation of soft importance scores, temperature 1, AdamW learning rate 0.0003, weight decay 0.01 and batch size 128. For each budget train a conditioned selector and a capacity-matched image-only selector for seeds 2041, 2042 and 2043: twelve models total. Use the existing shared descriptor for the image-only model and identical paired initialization and sampling within each seed.

Use **3,978 updates**, with monitoring every 153 updates and the final update as the checkpoint. This gives 509,184 sample presentations and exactly 416 presentations of every training task/class side. It preserves the existing exactly balanced sampler with 612 training pairs; the update count is 0.3% below the Pets count of 3,990. All Flowers arms have identical updates and sampling. The different number of training images and classes means a between-dataset difference cannot be attributed solely to dataset appearance. No seed replacement, early stopping, checkpoint selection, new training objective or hyperparameter search is included.

## Classification and replication contrasts

At each budget evaluate full CLIP, random selection, attention, the image-only selector, the conditioned selector and the label-informed teacher. The teacher is a diagnostic and receives the true class label. Random selection uses seeds 2055/2056/2057 and a fixed per-image ordering shared across budgets, so the 49-token mask is nested in its 98-token counterpart. Store the complete 102-class scores, normalized 512-dimensional image features and selected positions needed to recompute comparisons and exchanges.

The primary population is validation images under the 204 validation task pairs. Use equal task weights, equal class-side weights within task and uniform images within class. Average the three fixed training seeds before paired image-cluster bootstrap (2,000 draws, seed 2052). Report accuracy and signed geometric margin, all per-seed results, and new errors/repairs relative to full CLIP. Intervals condition on the fixed task graph and seeds.

For descriptive exchange strata, freeze the 25th/50th/75th percentile cutoffs of native CLIP margins on training images and training tasks. Save these cutoffs before pruned outcomes and apply them unchanged to validation. H2 uses continuous difficulty controls, not these bins. The reused evaluation schema's old-panel flag is true for the whole Flowers panel because its source cache is the same complete feature cache; no old/new coverage comparison is inferred from that flag.

For each budget report the conditioned-minus-image-only effect with ordinary 95% intervals and **97.5% intervals adjusted across the two budgets**, separately for each metric. This adjustment does not jointly cover all metrics and auxiliary analyses. Prespecify the single direct budget contrast

`D_98 - D_49`, where `D_k = mean_margin(conditioned, k) - mean_margin(image_only, k)`.

Estimate this paired contrast with a 95% image-cluster interval. A positive interval supports replication of the relative budget contrast. Replicating the stronger Pets sign pattern additionally requires negative `D_49` and positive `D_98` under their two-budget adjusted intervals. If either condition is uncertain or reversed, report that explicitly; a positive contrast by itself does not establish a positive effect at 98 tokens. Accuracy is reported separately from margin. Seen combinations and teacher comparisons are secondary descriptions. Do not add the full 524-candidate shared-search study here: the matched trained image-only selector is the main shared baseline, with the task teacher as an offline reference.

## H2 replication and geometry

Use the same-image exchange intervention as in Pets: keep the actual classification task fixed and exchange the candidate-class descriptor used by the conditioned selector. The primary H2 population is the fixed 1,020-image/6,120-task-pair validation panel. Average seeds within each image/task pair, then give each image equal weight and its six task pairs equal weight.

Keep the same symmetric continuous difficulty controls: the smaller and larger uncompressed signed geometric margins, their squares and their product. Add the class-separating direction angle in units of ten degrees. Use image-cluster bootstrap with 2,000 draws and seed 2061. Report per-budget slopes, 95% intervals and two-budget-adjusted 97.5% intervals. Direct replication of the directional H2 association requires a positive adjusted slope with the adjusted interval above zero at both budgets.

Retain the prespecified checks: image fixed effects, per-seed slopes, and five-fold class-stratified held-image prediction with fold seed 2060. Fit preprocessing and regression only in the training folds. Report weighted MSE and the incremental out-of-fold R-squared from adding angle, including image-bootstrap intervals of the fixed out-of-fold predictions. These checks concern prediction on held-out images within the fixed validation task collection. Disagreement between pooled, same-image and predictive results limits the interpretation and remains in the report.

Preserve the full-dimensional identity

`L = 0.5 (z_A-z_B) dot (u_A-u_B) = ||z_A-z_B|| sin(theta/2) cos(phi)`.

Check numerical reconstruction and the exactly zero image-only exchange effect. Separate displacement magnitude, the deterministic angle factor and normalized alignment. Alignment diagnostics remain exploratory, with the existing nonzero-displacement thresholds 1e-12 and 1e-6 and explicit sample counts. The seed-mean displacement diagnoses mean loss; it is not a deployed ensemble. Neither the identity nor raw angle correlation alone establishes a new mechanism.

## Execution and evidence

Commit this protocol and experiment code before substantive model runs. Use one available server GPU; keep the same compute configuration for paired arms. Record supervision generation, selector training and evaluation costs separately from actual deployment latency. Complete both budgets and both main questions regardless of direction. Retain all outcomes, original predictions and checkpoints; preserve small tables/figures/audits in Git and larger scientific archives locally, on the server and in a GitHub Release. Remove transfer-only staging files after verifying the retained evidence.

This stage does not repeat all Pets explorations, tune the method on Flowers outcomes, evaluate official test images or substitute for the required ViT-L/14 and actual-latency studies. A failed replication narrows the current claim; it does not justify discarding the budget or adding an unreported seed search.

## Primary data references

- [Oxford Flowers102 dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- [Official category names and thumbnail identities](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/categories.html)
- [Official dataset README](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/README.txt)
- [Official image labels](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat)
- [Official image partitions](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat)
