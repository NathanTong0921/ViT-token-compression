# ViT-token-compression

**Same Image, Different Evidence: When Does Token Compression Need Task Conditioning?**

This project studies whether changing the competing classes changes which visual tokens are useful for classification, and whether a lightweight task-conditioned selector can learn and generalize that advantage.

The main study uses frozen CLIP ViT-B/16 on Oxford-IIIT Pets, with controlled cross-task selection, strong shared-selection baselines, and matched conditional/unconditional selectors. Flowers102 provides validation; a targeted ViT-L/14 check is required in the updated execution plan.

`Project_Proposal.pdf` is the proposal for submission and the reference plan for project execution. 
Current status: the matched 49/98/147-token development comparison and H2 analysis are complete. Each budget uses full training coverage, the same 3,990 updates, and three fixed paired seeds. Conditioning has a small positive geometric-margin advantage over the image-only selector at 98 tokens, a negative advantage at 49, and an uncertain advantage at 147. No budget establishes an accuracy improvement. See the [current results](results/pets-selector-budgets-v1/README.md), [run record](docs/experiments/2026-10-27-selector-budgets-h2.md), and [fixed protocol](docs/selector-budgets-h2.md).

H2 receives development-set support: after adjusting for both uncompressed task margins, larger class-direction angles predict larger cross-task selection losses at all three budgets. Within-image checks agree; five-fold held-image incremental R-squared is 0.0240 / 0.0114 / 0.0062. This is a limited predictive association, not a causal mechanism or evidence of general superiority over shared selection. The exact geometric identity and normalized-alignment diagnostics are reported separately.

The main evaluation contains 716 development images and 66 unseen class combinations; the exchange analysis uses 616 eligible images and 4,315 task pairs. A privileged common-library search also finds a simultaneously correct shared mask for about 99.9% of image-weighted task pairs. Margin specialization and a necessity for different masks to answer correctly are therefore different claims in this setting. Official test images and 68 test pairs remain unused. Flowers102, targeted ViT-L/14 validation, actual inference timing and final-test confirmation remain required.

The [coverage experiment](results/pets-selector-scale-v1/README.md), [preceding training pilot](results/pets-selector-pilot-v1/README.md), and [fixed-model exchange experiment](results/pets-selector-exchange-v1/README.md) retain their original results and scope. The new comparison reuses the full-coverage 49-token models unchanged and trains budget-matched 98/147-token models. All seeds and both favorable and unfavorable results are retained.

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
