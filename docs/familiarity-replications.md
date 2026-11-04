# Fixed partition replications and interpretation of the role-switch effect

The 2099 Flowers98 result motivates two prospectively fixed new class partitions: **2100 and 2101**. Keep the original Pets49 and Flowers98 budgets, CLIP ViT-B/16, block-6 pruning, KL teacher, full remaining training-image coverage and update counts. For each partition, H0/H1 are consecutive disjoint ceil(C/4) blocks of that seed's class permutation; R is always seen. Fit all four arms in both roles, with training seeds 2041–2043 and identity mapping seeds 2097/2098. Each identity map preserves H0/H1/R separately and is identical across the two fits.

There is no checkpoint reuse for the new partitions: 24 fits per dataset/partition, **96 new fits total**. Keep historical2099 unchanged. No partition, mapping or checkpoint is chosen after inspecting outcomes. The same H0×H1 validation image/task rows are used for both fits; original training-manifest pairs in this panel are excluded from both fitted models. Reserved final tasks and official test images remain unused.

## What the replication can establish

Report both new partitions individually, an equal-weight mean of2100/2101 as the prospective replication result, and an additional historical-plus-new summary. The bootstrap uses the same resampled image IDs across partitions, since their panels overlap; it normalizes each partition separately before averaging. Retain individual training-seed results. This estimates image uncertainty conditional on the fixed partitions, not uncertainty over arbitrary class partitions.

The existing oriented margin contrast needs a precise interpretation. Orient a task score q toward H0, and let r=q_real−q_identity. For an H0 image the seen-minus-held semantic margin change is r1−r0. For an H1 image, both the true-class margin sign and the role direction reverse, producing the same r1−r0. The balanced average therefore measures a common candidate-direction score shift. It does not by itself measure improved separation of the two true image classes.

A separate CPU-only analysis of stored scores will decompose common offset and class separation, and examine image-cross-fitted offset correction relative to the full CLIP reference. That analysis is owned separately and requires no additional GPU fits. It will distinguish a preference toward the familiar candidate from useful visual discrimination; the raw replication remains intact.

## Execution

Run the following four commands with `--plan-only` first to save the split/mapping/configuration before training. Repeat without that flag to train both roles. Omitting `--original-references` means that all arms are fitted anew.

```bash
PYTHONPATH=src python scripts/train_familiarity_switch.py \
  --config configs/pets_selector_scale.json --partition-seed 2100 --device cuda:2 \
  --output runs/pets-familiarity-switch-partition-2100-training-v1 --plan-only
PYTHONPATH=src python scripts/train_familiarity_switch.py \
  --config configs/pets_selector_scale.json --partition-seed 2101 --device cuda:3 \
  --output runs/pets-familiarity-switch-partition-2101-training-v1 --plan-only
PYTHONPATH=src python scripts/train_familiarity_switch.py \
  --config configs/flowers_selector_budget_98.json --partition-seed 2100 --device cuda:6 \
  --output runs/flowers-familiarity-switch-partition-2100-training-v1 --plan-only
PYTHONPATH=src python scripts/train_familiarity_switch.py \
  --config configs/flowers_selector_budget_98.json --partition-seed 2101 --device cuda:7 \
  --output runs/flowers-familiarity-switch-partition-2101-training-v1 --plan-only
```

For each dataset/partition, evaluate its `training-v1/original` and `training-v1/switched` directories with the existing evaluator. Example:

```bash
PYTHONPATH=src python scripts/evaluate_familiarity_switch.py \
  --original-models runs/pets-familiarity-switch-partition-2100-training-v1/original \
  --switched-models runs/pets-familiarity-switch-partition-2100-training-v1/switched \
  --output runs/pets-familiarity-switch-partition-2100-evaluation-v1 --device cuda:2

python scripts/audit_familiarity_switch.py --runs-root runs --partition-seed 2100
python scripts/audit_familiarity_switch.py --runs-root runs --partition-seed 2101
python scripts/analyze_familiarity_replications.py --runs-root runs \
  --output results/familiarity-replications-v1/summary.json

PYTHONPATH=src python -m unittest discover -s tests -p test_familiarity_replications.py
```

The generalized trainer/evaluator preserve seed2099 defaults and the earlier explicit reference-reuse option. No shared selector implementation changes are needed.
