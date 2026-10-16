# ViT-token-compression

**Same Image, Different Evidence: When Does Token Compression Need Task Conditioning?**

This project studies whether changing the competing classes changes which visual tokens are useful for classification, and whether a lightweight task-conditioned selector can learn and generalize that advantage.

The main study uses frozen CLIP ViT-B/16 on Oxford-IIIT Pets, with controlled cross-task selection, strong shared-selection baselines, and matched conditional/unconditional selectors. Flowers102 provides validation; a targeted ViT-L/14 check is planned if time permits.

`Project_Proposal.pdf` is the proposal for submission and the reference plan for project execution. 
Current status: the server runtime is verified; Oxford Pets manifests and the frozen CLIP baseline are implemented. See the [Pets protocol and commands](docs/pets-protocol.md), [setup instructions](docs/server-setup.md), and [verified runtime records](results/setup/README.md). Pruning and selector experiments follow this baseline.

## Working approach

- Keep code direct and focused on the current experiments. Avoid speculative abstractions, redundant checks, and silent fallbacks; retain checks that establish scientific correctness.
- Treat the proposal as the starting research plan. Raise new observations, hypotheses, and better experimental designs as they emerge, and record the evidence and tradeoffs behind changes to the plan.

## Version management

- Keep `main` as the checked, reproducible baseline. Use short-lived `feat/...`, `fix/...`, or `exp/...` branches for substantial changes and experimental variants.
- Make each commit one coherent change, with a message explaining its purpose. Run relevant checks before committing and push completed milestones to this repository.
- Commit experiment code and configuration before a substantive run. Record the full commit SHA, command, configuration, seed, data split, model checkpoint, and environment alongside the results.
- Save small metrics, figures, split manifests, and experiment summaries in Git. Store datasets, feature caches, checkpoints, and raw run outputs separately, recording their locations and identifiers.
- Tag validated milestones when they exist, so the pilot, main experiments, and final report can be reproduced from known versions.

Use [the experiment record template](docs/experiment-log.md) for each substantive experiment. Negative and inconclusive results should retain the same provenance as positive results.

Paper evidence will be retained on the server, locally, and on GitHub: per-example results, figure source data, figures, configurations, and selected selector checkpoints. Small files belong in Git; larger essential artifacts belong in versioned GitHub Release attachments. See the [storage and backup plan](docs/storage-and-backup.md). The first runtime records have been copied back locally and committed; experiment-specific transfer scripts remain to be implemented.

## File organization

As implementation begins, use `src/` for reusable code, `scripts/` for entry points, `configs/` for settings, `data/splits/` for small split manifests, `results/` for curated results, and `docs/` for research records.

Keep bulk artifacts in the ignored directories `data/raw/`, `data/processed/`, `cache/`, `checkpoints/`, `runs/`, or `outputs/`. Sample course reports and downloaded papers remain in the surrounding local project folder.
