# ViT-token-compression

**Same Image, Different Evidence: When Does Token Compression Need Task Conditioning?**

This project studies whether changing the competing classes changes which visual tokens are useful for classification, and whether a lightweight task-conditioned selector can learn and generalize that advantage.

The main study uses frozen CLIP ViT-B/16 on Oxford-IIIT Pets, with controlled cross-task selection, strong shared-selection baselines, and matched conditional/unconditional selectors. Flowers102 provides validation; a targeted ViT-L/14 check is planned if time permits.

`Project_Proposal.pdf` is the proposal for submission and the reference plan for project execution. 
Current status: exchanging task descriptors in the fixed trained selectors shows useful task dependence. On 124 eligible development images with unseen class combinations, correct descriptors improve image-equal accuracy by 0.533 percentage points (95% image-bootstrap interval: 0.125–1.030) and signed margin by 0.00416 (0.00322–0.00516) over another valid task descriptor for the same image. The image-only exchange effect is exactly zero; restricting both tasks to the same pre-existing difficulty band preserves the positive effect. This establishes useful conditioning within these models, while superiority over the separately trained image-only model remains uncertain. Official test images and 68 test task pairs remain untouched. See the [current results](results/pets-selector-exchange-v1/README.md), [run record](docs/experiments/2026-10-18-selector-exchange.md), and [fixed protocol](docs/selector-exchange-pilot.md).

The [preceding paired training pilot](results/pets-selector-pilot-v1/README.md) averaged 99.12% conditioned versus 98.74% image-only accuracy on its balanced 528-example unseen-combination benchmark (attention: 97.92%). That benchmark differs from the exchange panel. Each seed gained two net correct examples, but the improved cases differ across seeds. Teacher-imitation validation loss worsened during later training. The next training comparison should test whether more distinct training images improve generalization, rather than simply extending training.

## Working approach

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
