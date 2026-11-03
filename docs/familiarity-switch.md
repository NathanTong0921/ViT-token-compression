# Familiarity switch on identical images and candidate tasks

This follow-up intervenes on the selector's training familiarity while retaining the exact evaluation image/task rows. It follows the Flowers mixed-task asymmetry and the difficulty adjustment; neither a new random split search nor a new dataset is used.

## Fixed training intervention

Generate the original class permutation with seed 2099. H0 is its first ceil(C/4) classes (the existing holdout), H1 is the next equally sized, disjoint block, and R contains the remaining classes. Original fit P0 sees H1+R and holds H0; switched fit P1 sees H0+R and holds H1. There are 10 classes per switched block in Pets and 26 in Flowers. Both training images and candidate tasks involving held classes are removed under the same rules. Use all available remaining training images, the original fixed update counts and seeds 2041–2043.

Keep the random identity control identical across fits: for each mapping seed 2097/2098, permute independently within H0, H1 and R. One resulting global class-code map is reused in P0 and P1. It preserves each fit's training vector bank and prevents code reassignment from accompanying the familiarity change.

For P0, reuse the exact previously trained real-text and wide-image checkpoints; retrain only the two three-block identity arms. For P1, fit all four arms. This is 18 new fits per dataset, 36 total. The old runs remain intact. New checkpoints/records identify reused sources and whether training occurred in the current run. Only the candidate descriptors are permuted; teacher targets and CLIP's final text dictionary remain correct.

## Identical evaluation panel

Use every original train/validation-manifest pair containing one H0 class and one H1 class, with the same original validation images. These candidate pairs are excluded from both P0 and P1 training. Do not read reserved final-task manifests or official test images. Re-evaluate both fits on exactly the same ordered image/task rows, with the same prefix, pruning position, patch budget, suffix implementation and task/class weighting.

When the true class is H0, P0→P1 changes it from held to seen; when the true class is H1, the direction reverses. In each task the distractor's familiarity also reverses. This is a controlled swap of the two candidate roles and the training class pool, not an isolated manipulation of the true class alone.

## Estimands fixed before the run

Let E_p(x,t) be real-text performance minus the mean of the two identity controls in fit p. For H0 images define D=E_1−E_0; for H1 images define D=E_0−E_1. The primary estimate averages D equally across tasks and both true-class sides: semantic advantage when seen minus when held on identical images/tasks. Report accuracy (pp) and geometric margin, all seed/mapping results, and paired image-bootstrap 95% intervals.

Also report the difference-in-differences `(mean(E1−E0 | H0) − mean(E1−E0 | H1))`. Its point estimate is twice the primary oriented average under the balanced two-side weighting; do not confuse these scales. Report the same comparisons against the wide-image baseline, absolute arm results, and the two true-class group changes separately. The full-model predictions are identical across fits by construction.

The JSON plan fixes both class groups, every class-code mapping, training configuration, seeds, reuse sources and evaluation policy before training. A positive effect would strengthen the familiarity interpretation; a null or reversed effect would favor class/task-specific differences over the earlier aggregate familiarity association.

## Execution

Run the plan-only commands first after committing/synchronizing code, using the same device arguments as the corresponding training commands:

```bash
PYTHONPATH=src python scripts/train_familiarity_switch.py \
  --config configs/pets_selector_scale.json \
  --original-references runs/pets-semantic-identity-v1 \
  --output runs/pets-familiarity-switch-training-v1 --device cuda:2 --plan-only

PYTHONPATH=src python scripts/train_familiarity_switch.py \
  --config configs/flowers_selector_budget_98.json \
  --original-references runs/flowers-semantic-identity-v1 \
  --output runs/flowers-familiarity-switch-training-v1 --device cuda:6 --plan-only
```

Repeat those commands without `--plan-only` to train both phases. `--phase original` and `--phase switched` can run phases separately; the default is both. Then evaluate:

```bash
PYTHONPATH=src python scripts/evaluate_familiarity_switch.py \
  --original-models runs/pets-familiarity-switch-training-v1/original \
  --switched-models runs/pets-familiarity-switch-training-v1/switched \
  --output runs/pets-familiarity-switch-evaluation-v1 --device cuda:2

PYTHONPATH=src python scripts/evaluate_familiarity_switch.py \
  --original-models runs/flowers-familiarity-switch-training-v1/original \
  --switched-models runs/flowers-familiarity-switch-training-v1/switched \
  --output runs/flowers-familiarity-switch-evaluation-v1 --device cuda:6

PYTHONPATH=src python -m unittest discover -s tests -p test_familiarity_switch.py
```

The existing semantic-identity runner defaults still produce the original experiment. Its only new capabilities are a disjoint holdout-block offset, explicitly supplied descriptor mappings, and exact reuse of the two reference arms. The separate familiarity wrapper fixes these options for this experiment.
