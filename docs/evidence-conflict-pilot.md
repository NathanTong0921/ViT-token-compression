# Evidence conflict and contextual information: first screen

This implements direction A in [research-expansion.md](research-expansion.md). It asks whether the existing tasks share visual evidence, and whether a mid-layer contextual CLS already carries enough information to make patch selection less consequential. The pilot freezes OpenAI CLIP ViT-B/16 and trains no selector. It is a feasibility screen, not a new method comparison.

## Original-image diagnostic

Use two deterministically sampled validation images per class in Pets and Flowers102. At blocks 3 and 6, retain CLS alone or CLS plus 4, 16, 49, 98, or all 196 patches. Nontrivial patch budgets compare deterministic spatial coverage, four fixed random masks, and CLS-attention selection. The contextual CLS remains unchanged; CLS-only simply runs the remaining native blocks on a sequence of length one. Report full-class and preassigned validation binary-task accuracy, cosine margins, and native/full-retention agreement. This measures sufficiency under an intervention; high CLS-only accuracy would not establish that patches contain no additional information.

## Controlled two-source panel

Create 64 deterministic composites per dataset, using only validation images. Each image has two different source labels in equal half-image tiles. Horizontal versus vertical layout and source ordering are randomized. Aspect ratios are preserved by fitting each source into its tile and filling unused space with RGB(127,127,127); no source is cropped away. The complete PNG panel and transformations are saved.

Each source defines two binary tasks. Their distractors are the two text-nearest classes among its **preassigned validation candidate pairs**, excluding both source labels. Thus every task contains exactly one of the two designated source classes, and no reserved test pair is evaluated. An original natural image can contain incidental objects not exhaustively annotated by its dataset; the panel is a controlled source-label construction, not a verified segmentation benchmark.

Each composite gives two same-object task pairs and four different-object task pairs. Source image IDs, labels, candidate task IDs, layout, and transforms are retained. Sources can recur across composites, so these are exploratory estimates rather than independent-image confidence intervals.

## Common mask-library intervention

At blocks 3 and 6 with 16 and 49 retained patches, run the same 36-mask library for all four tasks: 32 fixed random masks, one spatial-coverage mask, one attention mask, and one spatial-coverage mask confined to each tile. All masks retain the original contextual CLS. Random masks are identical across images and layers at each budget; attention can change with image and layer. No task or answer is given to any bank mask generator. A second analysis excludes attention, leaving exactly the same 35 mask positions at both layers; this is the cleaner layer comparison because attention rankings can themselves change with depth.

Composite margins are geometric signed margins: divide the cosine-score difference by the norm of the two normalized class-text vectors' difference. Native and pruned margins use the same normalization; raw cosine scores and margins are also saved. Original-image summary explicitly labels its raw cosine margin. For each pair, report whether both tasks are individually achievable somewhere in the library, whether one common mask answers both correctly, and whether the first holds while the second fails. Also save the task-specific versus best-shared mean-margin and worst-task-margin gaps. A nonnegative optimized gap is guaranteed by construction and is **not** a discovery. The substantive comparison is how joint feasibility and the magnitude of conflict differ between same-object and different-object tasks and between pruning locations. These are label-informed finite-library diagnostics; finding a gap does not show a deployable selector can learn it, and finding no gap does not rule out a larger library.

Save every task's native and pruned scores, masks, and pair-level records. Report all pairs and the predefined subset where the native model answers both correctly. Summaries average within image before averaging images. Descriptive regressions include image indicators, the two ordered full-model margins and their squares, with a second version adding task-direction angle and its square. Coarse margin-bin matching and common margin-plus-angle cell counts expose support limitations; an adjusted coefficient with little common support is not evidence of an isolated object-conflict effect. These analyses do not assert causality or attach significance tests.

## Screen and commands

The default screen uses 128 composites total and roughly 18,432 composite suffix evaluations, plus fewer than 16,000 original-image diagnostic forwards. With batch size64 and no training it is intended to fit within one A6000 GPU-hour; elapsed time is saved rather than assumed. One dataset can run on each idle GPU if desired.

```bash
.venv/bin/python scripts/run_evidence_conflict_pilot.py \
  --dataset pets --device cuda:6 --output runs/evidence-conflict-pets-v1
.venv/bin/python scripts/run_evidence_conflict_pilot.py \
  --dataset flowers --device cuda:7 --output runs/evidence-conflict-flowers-v1
```

When running in parallel, use dataset-specific output roots to avoid both processes writing `protocol.json`, e.g. `runs/evidence-conflict-pets-v1` and `runs/evidence-conflict-flowers-v1`. Alternatively, one sequential command uses `--dataset both`.

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p test_evidence_conflict_pilot.py
```

First inspect native composite accuracy and source/layout coverage before interpreting pruning. If many tasks are already unanswerable for full CLIP, revise the synthetic task design openly and retain this first screen; do not rescue the claim by dropping difficult examples. If common masks remain nearly sufficient, this limits this panel's value for studying evidence conflict. If a same/different-object contrast survives comparable full-model difficulty and direction geometry, the next step is paired learning and a larger controlled panel, not declaring the bank optimum deployable. If CLS-only remains strong at block6 but weak at block3, that motivates a targeted representation-sharing intervention; it does not by itself prove why conditioning helps or fails.
