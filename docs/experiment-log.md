# Experiment records

Create one Markdown record in `docs/experiments/` for each substantive experiment, using a unique name such as `YYYY-MM-DD-short-description.md`. The fields below are a template, not a completed run. Pilot runs can be brief; final comparisons should include all applicable details.

## Run identity

- Run ID and date:
- Research question / hypothesis:
- Status (pilot, complete, failed, or inconclusive):
- Full code commit SHA:
- Configuration path and overrides:
- Exact command:
- Environment / dependency lockfile or exported environment path:
- Hardware, training time, and supervision-generation time:

## Inputs and protocol

- Dataset version, split manifest, and random seed(s):
- Candidate-set construction and train/test combination split:
- Frozen model checkpoint identifier / revision:
- Image preprocessing / augmentation and class-text prompt templates:
- Selector variant, pruning layer, and token budget:
- Baselines and matched settings:
- Cached supervision / feature artifact locations and hashes:
- Evaluation metrics and timing procedure:

## Results and interpretation

- Small result tables or figures committed under `results/`:
- Raw logs, checkpoints, and other artifact locations:
- Paper evidence bundle (per-example results, figure data, relevant images, and selected checkpoints):
- Verified backup locations (server, local, and GitHub commit/release); mark pending copies explicitly:
- Main observations and uncertainty:
- Failed runs or deviations from the planned protocol:
- Interpretation and next step:

The code SHA identifies the version that produced the result, even if this record is committed later. When an experiment changes a data split, checkpoint, configuration, or supervision cache, record the new identifier so results from different protocols remain distinguishable.
