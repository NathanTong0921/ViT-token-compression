# Semantic text versus stable class identity: first development pilot

This experiment extends direction B in `research-expansion.md`. It asks whether selector generalization to new candidate combinations can be explained by stable class identity, and whether genuinely new classes change the value of semantic text. It is a development experiment; no official test images or reserved final task combinations are used.

## Fixed matrix

- Datasets/budgets: full-coverage Oxford Pets, 49 patches; Flowers102, 98 patches. Frozen CLIP ViT-B/16, block 6 pruning, existing KL teacher targets.
- Training regimes: all classes; and a fixed 25% class holdout (10/37 Pets classes, 26/102 Flowers classes), generated with NumPy PCG64 seed 2099. In the holdout regime, remove both training images of held classes and all training tasks containing a held class.
- Arms per regime: real text; fixed class/text permutation 2097; fixed class/text permutation 2098; wide image-only selector. Every arm is trained anew with weight/sampling seeds 2041, 2042, 2043: 48 models total across the two datasets.
- The three descriptor arms have identical architecture and initial weights within each seed. The image-only width is 199, following the existing approximately parameter-matched strong baseline. All arms receive identical examples, KL targets, optimizer settings and update counts within each regime.
- Keep the original 3990 Pets / 3978 Flowers updates and batch size 128. The all-class regime uses the original sampling stream exactly. In the holdout regime, distribute the fixed draw count across task/class sides with counts differing by at most one, then shuffle and draw within-class image quantiles.

The mapping changes only the text vectors passed to the selector. Teacher targets and final CLIP classification always use the original correct text dictionary and original labels. The permutations are global for the all-class regime; in the class-holdout regime they operate separately within seen and held class banks. Both arms therefore have the same training vector bank. Their actual sampled pair-descriptor distributions need not be identical. Fixed points are allowed and counted; no mapping is chosen after looking at results.

## Evaluation and primary comparisons

Use all original cached validation images and validation task combinations. These combinations are absent from selector training. For the class-holdout regime, report known/known, known/new and new/new candidate pairs separately. Within every stratum, give each task equal weight and each side of a task equal weight. New classes here means unseen during selector training, not unseen by CLIP pretraining.

The main effect is real-text accuracy/margin minus the average of both fixed permutations, averaged over all three paired training seeds. Preserve individual mapping and seed results. Also compare each with the newly trained wide reference. The main interaction is the real-minus-permuted effect on new/new pairs minus that on known/known pairs within the same class-holdout regime. Confidence intervals use paired image-cluster resampling, conditional on the fixed classes, task sets, mappings and trained models; this first pilot uses one class split. It does not estimate variation over all possible class partitions.

Record full-model outcomes on the same panel for difficulty context. Main summaries retain absolute arm accuracy/margins, paired effects, initial/final teacher loss, parameter counts and runtime. A fixed final checkpoint is evaluated; validation does not select a step or mapping.

## What the outcomes distinguish

If stable permuted identities support unseen combinations but real text has a stronger advantage on unseen classes, the two evaluation settings test different capabilities. If real text wins in both, semantic alignment contributes even for familiar classes. If neither wins, quantify how much useful conditioning remains relative to the wide image-only baseline. Similar performance of two arms is not by itself an equivalence result. A class-holdout weakness affects both access to training images and task coverage, so the causal comparison is real versus permuted within a regime, followed by the predeclared stratum interaction.

## Commands and outputs

Run from the repository root after committing and synchronizing this revision. The two datasets can use separate GPUs:

```bash
PYTHONPATH=src python scripts/run_semantic_identity_pilot.py \
  --config configs/pets_selector_scale.json \
  --device cuda:2 --output runs/pets-semantic-identity-v1

PYTHONPATH=src python scripts/run_semantic_identity_pilot.py \
  --config configs/flowers_selector_budget_98.json \
  --device cuda:6 --output runs/flowers-semantic-identity-v1

PYTHONPATH=src python -m unittest discover -s tests -p test_semantic_identity_pilot.py
```

Every output root contains the complete protocol and combined summary. Each regime stores the explicit mappings, selected training IDs/tasks, sampling counts, initial/final training records, validation panel metadata/full-model outcomes, all final checkpoints, and per-model compressed features, selected patches, margins and predictions. Small JSON summaries and protocols can be committed under `results/`; bulk checkpoint/prediction arrays remain server artifacts with local backups as appropriate. No new dataset download or teacher-query generation is required. Runtime should be measured on the first models; previous selector-only timings exclude the substantial frozen-suffix evaluation stage included here.
