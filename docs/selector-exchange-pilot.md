# Learned-selector task exchange

Protocol fixed on 2026-10-18 before inspecting crossed predictions. This extends the proposal's offline position-exchange experiment to the already trained selectors, connecting H1 to H3. No weights, training schedule, data split or selector hyperparameters change.

## Main question

For a fixed image, frozen checkpoint, pruning boundary and 49-patch budget, does supplying the evaluated task's own candidate-class descriptor improve classification relative to supplying another valid task descriptor for that same image?

Use the six final-epoch checkpoints from the first selector pilot. The primary task collection is the 66 unseen validation pairs; seen training pairs on the same new images form a secondary comparison. Enumerate every unordered pair of distinct tasks that both contain the image's true class, within each collection. Evaluate both directions. Alternative tasks are not chosen based on learned outcomes. Labels define the offline evaluation tasks and metrics but are never an input indicating the correct candidate to the selector.

The unseen exchange panel contains 124 eligible images and 868 unordered task pairs per seed (1,736 directed comparisons). Images with only one available task are excluded explicitly. It is a shared-anchor diagnostic panel, not the original balanced binary benchmark. Compute the learned image-only comparison on this same eligible panel. Seen/unseen differences remain descriptive because their task collections differ.

## Exact reuse of saved inference

The previous evaluation retained the compressed image's cosine scores against all 37 class texts for every supplied descriptor. Once positions have been selected, the CLIP suffix has no task input. Thus the saved score vector under descriptor B can be scored on task A exactly; new task-exchange outcomes do not require repeated full inference. Keep source row indices alongside derived cases so the four outcomes can be reconstructed.

For each image/task pair, form the four cells: descriptor A evaluated on A or B, and descriptor B evaluated on A or B. The primary effect is the mean of the two own-descriptor outcomes minus the mean of the two exchanged-descriptor outcomes, for signed geometric margin and accuracy. Positive values mean the correct task information helps. Compare own and exchanged outcomes against the matched image-only model. Its selections and score vectors should be invariant across descriptors, yielding a zero exchange effect.

Replay a small deterministic sample on the server from the saved prefix cache and final weights, using the same fixed batch size of 64 and stable top-k selection. Verify selected positions, logits, all-class scores and crossed correctness against the archived values. Record this replay's cost separately from CPU re-analysis.

## Analysis fixed in advance

Average pairs and seeds within each image, then images equally. Bootstrap images 2,000 times with seed 2051, keeping seed-level estimates separately. Intervals are conditional on the existing images' task collections and three trained checkpoints. Pair counts and training seeds do not multiply independent image counts.

Retain token overlap, changed predictions, new errors and recoveries. Use the original uncompressed margin boundaries from the pruning pilot: 0.09612737409770489, 0.12876027822494507, 0.15853078663349152. Assign each directed contrast by the evaluated task's own full-model margin; these are fixed reference bands, not new quartiles of the smaller panel. Also report the sensitivity subset in which both tasks fall in the same reference band. This is coarse difficulty restriction, not exact margin matching.

Separately inspect all prior conditioned-versus-image-only classification disagreements, by seed and image/task, without selecting only favorable examples. Summarize all seen/unseen rows within the same fixed margin bands, keeping the original full-task accuracy estimand distinct from the exchange-panel image-equal estimand.

Angles may be retained as descriptors. The raw angle/exchange-loss relationship has a known geometric scale factor; the saved 37 cosine scores do not recover full 512-dimensional embedding displacement. This round therefore does not claim a new normalized H2 mechanism.

## Decision and evidence

A positive exchange effect supports useful task-information dependence within these learned models; it does not establish superiority over the privileged shared search or a final-test generalization result. A near-zero effect despite changing masks would indicate that descriptor sensitivity alone does not produce useful specialization. Either outcome informs whether to prioritize data/training changes, stronger shared baselines, or conditional regimes before expanding the study.

Retain configuration, compact case arrays linked to original prediction rows, per-image/per-seed summaries, full error-transition records, figures and GPU replay evidence under results/pets-selector-exchange-v1. Original pilot artifacts and untouched official test data remain available for subsequent work.
