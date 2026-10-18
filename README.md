# ViT-token-compression

**Same Image, Different Evidence: When Does Token Compression Need Task Conditioning?**

This project studies whether changing the competing classes changes which visual tokens are useful for classification, and whether a lightweight task-conditioned selector can learn and generalize that advantage.

The main study uses frozen CLIP ViT-B/16 on Oxford-IIIT Pets, with controlled cross-task selection, strong shared-selection baselines, and matched conditional/unconditional selectors. Flowers102 provides validation; a targeted ViT-L/14 check is required in the updated execution plan.

`Project_Proposal.pdf` is the proposal for submission and the reference plan for project execution. 
Current status: a fixed-update comparison has expanded training coverage from 592 to 2,944 images, with 3,990 updates per selector and three paired seeds. Conditional teacher-imitation validation KL improves from 0.833 to 0.499, below the image-only model's 0.551, but actual unseen-combination accuracy is 98.9919% versus 98.9906%. The paired difference is +0.0013 percentage points (95% image-bootstrap interval: −0.2186 to +0.2141), and the image-only model retains a higher average geometric margin. The change in conditioning advantage also has an interval crossing zero. More image diversity improves supervision fitting without establishing superiority over the strong shared model. See the [current results](results/pets-selector-scale-v1/README.md), [run record](docs/experiments/2026-10-18-selector-data-coverage.md), and [fixed protocol](docs/selector-data-coverage.md).

Development evaluation now covers all 736 validation images; 716 participate in the 66 unseen validation class combinations. In a separate 616-image exchange diagnostic, correct task descriptors still improve margin over exchanged descriptors, while the accuracy effect is uncertain. Thus useful task dependence persists, but it does not establish a better selector than the separately trained image-only baseline. Official test images and 68 test task pairs remain untouched. The next stage is the full three-budget comparison and stricter difficulty/geometry analysis, followed by Flowers102, targeted ViT-L/14 validation, and actual inference timing.

The [preceding training pilot](results/pets-selector-pilot-v1/README.md) and [fixed-model exchange experiment](results/pets-selector-exchange-v1/README.md) retain their original results and scope. This coverage experiment retrains the small-data control with the same balanced sampling used for the large-data arm; differences from the old pilot cannot be attributed to evaluation-panel expansion alone.

## Working approach

- Follow the [project TODO](docs/TODO.md) for remaining experiments and deliverables; update it as evidence changes the next step.
- Keep code direct and focused on the current experiments. Avoid speculative abstractions, redundant checks, and silent fallbacks; retain checks that establish scientific correctness.
- Treat the proposal as the starting research plan. Raise new observations, hypotheses, and better experimental designs as they emerge, and record the evidence and tradeoffs behind changes to the plan.
- After each completed task, explain the result plainly, identify the corresponding proposal stage, distinguish any new insight from the original plan, and assess whether it advances the project goal. Report unexpected problems and useful observations during execution.
- Keep evolving findings in the [research insight log](docs/research-insights.md), with evidence links, their relationship to the proposal, and implications for the next experiment. Update the interpretation when new evidence changes it; retain dated revisions.

## Version management

- Keep `main` as the checked, reproducible baseline. Use short-lived `feat/...`, `fix/...`, or `exp/...` branches for substantial changes and experimental variants.
- Make each commit one coherent change, with a message explaining its purpose. Run relevant checks before committing and push completed milestones to this repository.
- Commit experiment code and configuration before a substantive run. Record the full commit SHA, command, configuration, seed, data split, model checkpoint, and environment alongside the results.
- Save small metrics, figures, split manifests, and experiment summaries in Git. Store datasets, feature caches, checkpoints, and raw run outputs separately, recording their locations and identifiers.
- Tag validated milestones when they exist, so the pilot, main experiments, and final report can be reproduced from known versions.

Use [the experiment record template](docs/experiment-log.md) for each substantive experiment. Negative and inconclusive results should retain the same provenance as positive results.

Paper evidence will be retained on the server, locally, and on GitHub: per-example results, figure source data, figures, configurations, and selected selector checkpoints. Small files belong in Git; larger essential artifacts belong in versioned GitHub Release attachments. See the [storage and backup plan](docs/storage-and-backup.md). Runtime records, the original baseline, pruning pilot and offline-selection diagnostics include locally retained scores, selection positions, statistical tables and plots. Transfer-only archives are removed after verification.

## File organization

As implementation begins, use `src/` for reusable code, `scripts/` for entry points, `configs/` for settings, `data/splits/` for small split manifests, `results/` for curated results, and `docs/` for research records.

Keep bulk artifacts in the ignored directories `data/raw/`, `data/processed/`, `cache/`, `checkpoints/`, `runs/`, or `outputs/`. Sample course reports and downloaded papers remain in the surrounding local project folder.
