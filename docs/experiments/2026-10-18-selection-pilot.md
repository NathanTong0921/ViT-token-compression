# Offline task rankings, shared search and cross-task exchange

Completed 2026-10-18 using the [fixed selection protocol](../selection-pilot.md). Run code: `3f3c79a`; analysis code: `1948028`. All configurations were committed before their model runs.

## Execution and validation

- Model: frozen original OpenAI CLIP ViT-B/16; after block 6; 147/98/49 retained patches and CLS.
- GPU: one RTX A6000, physical GPU 2 (`CUDA_VISIBLE_DEVICES=2`). GPU 0 was occupied, so its workloads were not touched.
- Three runs: 36-image 2-by-2 deletion smoke panel; preselected 144-image expanded panel; 36-image single-patch deletion sensitivity panel. All use existing validation images and 66 validation tasks; no official test data.
- Expanded balanced evaluation: 144 images, 528 image/task examples per method and budget. Exchange evaluation: 124 eligible images across 31 anchors, 868 image/task-pair cases per budget, 2,604 across budgets.
- The first smoke attempt stopped before selection because changing the original 64-image encoder batches to a 36-image panel produced a maximum FP16 embedding difference of 0.00040594. Recomputing the original validation batches before extracting the panel restored exact equivalence. All successful runs have maximum native/saved feature and score differences 0 and no prediction disagreements. The failed run's log remains under `runs/pets-selection-smoke-batch-shape-check/` on the server.
- All subsequent suffix interventions use batches of 64; short batches repeat their last selection and discard padded outputs. This keeps batch shapes consistent for small deletion effects. Actual and requested forward-example counts are recorded separately.
- The primitive/evaluation tests and independent artifact reconstruction check rankings, positions, seeds, labels, winners and all exchange margins. On the expanded run, 166,656 candidate subsets and 6,336 single-task position rows were checked. Maximum saved task-feature-to-score discrepancy is 3.58e-7; exchange geometry residual is 1.82e-7.

## Main observations

The [expanded results](../../results/pets-selection-v1/README.md) contain all raw data and interval estimates.

1. **Proxy task specialization is reproducible within this development panel.** Mean exchange margin losses are 0.01481, 0.01652 and 0.00872 at 147, 98 and 49 patches. Their paired image-bootstrap intervals exclude zero. This evaluates actual compressed outputs using rankings derived from different, near-full deletion interventions.
2. **Specialization does not establish an advantage over the stronger shared reference.** Own proxy ranking minus shared-search mean margin is approximately +0.00012, -0.02104 and -0.06629. The first interval includes zero; the latter two favor shared search. On the common exchange panel, shared mean-search accuracy is 99.73%, 100%, 100%; own proxy accuracy is 99.73%, 99.68%, 93.72%.
3. **A local ranking proxy degrades under large joint deletion.** Balanced task-ranking accuracy is 99.81%, 99.62%, 93.37%, versus attention 99.05%, 98.86%, 97.92%. The same panel's original accuracy is 99.05%. At 49 patches the task ranking introduces 34 new errors and recovers 4; attention introduces 6 and recovers none. This pattern suggests a proxy/target-budget mismatch, but it does not by itself prove the cause.
4. **Search remains bounded.** At 49 patches, shared mean margin increases from 0.13660 to 0.14553 to 0.15072 across 16/32/64 candidates. This cannot be described as converged. Candidate-bank task-specific optimization gives extra margin by construction, with no extra accuracy at 49 or 98 patches in this panel; its positivity is not independent H1 evidence.
5. **H2 needs the geometric qualification.** Adding angle improves grouped prediction of raw exchange loss beyond baseline-margin controls, with MSE reductions of approximately 15.9%, 8.6%, 1.8% at 147/98/49 patches. For normalized directional alignment, the respective improvements are only about 1.1%, -0.3%, -1.0%. These exploratory results do not establish a mechanism beyond the known exchange identity, nor generalization to unseen task pairs.
6. **Single-patch sensitivity does not resolve the strong-sharing result.** On the same 36-image smoke panel, single-patch task ranking is 94.70% at 49 patches versus 93.18% for 2-by-2 ranking; at 98 patches it is 98.48% versus 100%. Shared search reaches 100% in both smoke variants. Block masking is a motivated proxy choice, not an experimentally established universal improvement here.

## Interpretation and next experiment

The data support task-dependent behavior of deletion-derived selections; they do not establish the desired consistent advantage over strong shared selection. Neither the shared search nor the task-specific search is deployable: both use labels, and the shared reference also receives both tasks. A learned conditioned selector has not been trained or tested.

The next useful experiment is a small, fixed-budget comparison of supervision quality: estimate task and shared importance in contexts already compressed near the target budget, or use a matched refinement of the same starting sets. Measure whether that improves the proxy-to-actual selection relationship, retaining the current strong shared references. If it does, begin matched conditional/unconditional selector trial training promptly; if the shared reference still suffices, investigate that limit rather than merely increasing search until a positive result appears. This is a proposed follow-up, not a result of this run.

## Cost and backup

Measured run wall times: 31.92 seconds (smoke), 69.36 seconds (expanded), 23.33 seconds (single-patch sensitivity). Expanded deletion/ranking took 9.88 seconds; final-subset evaluation took 50.91 seconds; model loading, full identity verification and file writes account for the remainder. Peak allocated GPU memory was about 756 MB. These are diagnostic computation costs, not deployment latencies. No selector training occurred.

The three result bundles total approximately 51 MB. Exact panels, raw deletion scores, all candidate positions/scores, selected task embeddings, statistical tables and figures are retained locally and in GitHub, with original run directories and curated results on the server. Temporary clones used to verify GitHub recovery are removed after checking. No data/model transfer archive or persistent prefix cache was created.
