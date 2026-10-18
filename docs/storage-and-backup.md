# Storage and backup plan

Status, 2026-10-18: the paired selector pilot, fixed-model task exchange and fixed-update image-coverage experiment are complete. Original baseline, pruning, offline-selection, budget-context and learned-selector evidence are retained locally and on the server; small evidence is versioned in Git. The latest coverage experiment adds about 1.11 GB of recomputable prefix caches on the server and two retained evidence archives totaling 1.38 GB locally and in Release storage. Dataset and CLIP weights remain server working assets. The capacity figures below are planning allowances, not current-use measurements.

The single-stage pruning pilot adds approximately 21.6 MB of scores, embeddings, retained positions, metrics and figure exports under `results/pets-pruning-v1/`. These are small enough for ordinary Git. Its approximately 223 MB prefix cache existed only in host RAM and was released after the run; no persistent prefix cache or selection-supervision cache was created.

## Capacity

For frozen CLIP ViT-B/16 on Oxford-IIIT Pets and Flowers102, budget 100 GB of available project storage on the server. Expected working use is approximately 30–60 GB with controlled cache retention. Use 150 GB if retaining additional model/layer caches or several environment copies. These figures exclude the operating-system image and large video-editing source files.

Locally, reserve 20–30 GB for development, dataset/model copies, key experiment artifacts, and report materials. Keeping the full reusable feature caches as well may require 40–50 GB. Ordinary Git history should remain small, preferably below 1 GB; larger essential artifacts go in GitHub Releases.

These are planning allowances, not disk-use measurements or upper bounds. Runtime packages/download caches can occupy 10–25 GB on the server; datasets and pretrained weights approximately 3–8 GB; ranking supervision, experiment outputs, selector checkpoints, and temporary files approximately 5–15 GB. Feature caches are additional.

## Cache assumptions

At 224-pixel resolution, approximately 15,600 images, one pruning boundary and one preprocessing view, a ViT-B/16 prefix cache in FP16 occupies approximately:

`15,600 images × 197 tokens × 768 dimensions × 2 bytes = 4.72 GB`.

Saving 196 masked global embeddings per image adds approximately:

`15,600 × 196 × 512 × 2 bytes = 3.13 GB`.

GB here denotes decimal gigabytes. Mask counts are illustrative, not a fixed experimental design. FP32 doubles these amounts; additional layers or views require additional storage. Check numerical consistency before adopting reduced-precision caches for the study.

Cache the unmasked prefix features and the necessary global embeddings/ranking targets. Do not retain full intermediate token tensors for every masking intervention: that example would exceed 900 GB. Reuse compatible caches across candidate sets and token budgets. The required targeted ViT-L/14 check should cache only its required subset unless broader coverage becomes necessary.

## What to retain

| Material | Server | Local | GitHub |
| --- | --- | --- | --- |
| Code, configurations, environment specification, image splits, candidate-set manifests, text prompts | Yes | Yes | Git |
| Per-example results supporting all reported comparisons, including baselines and negative findings | Yes | Yes | Git for small files; Release attachments for larger bundles |
| Seed-level metrics, timing samples, statistical summaries, figure source tables and plotting scripts | Yes | Yes | Git or Release attachments according to size |
| Final figures, token-selection visualizations, report/PPT source and exports | As needed | Yes | Git or Release attachments according to size |
| Images used for qualitative figures, with original IDs, preprocessing and source attribution | Yes | Yes | Relevant figure assets; source-image redistribution follows dataset permissions |
| Selected conditional and unconditional selector checkpoints for the reported seeds/configurations | Yes | Yes | Release attachments |
| Full public datasets and pretrained CLIP weights | Yes | Retain one copy when useful | Download sources/version information, not duplicate bulk files |
| Recomputable prefix caches, masked embeddings, temporary tensors and intermediate checkpoints | While useful | Optional | Normally omitted |

Per-example records must retain enough information to recalculate the reported comparisons: image/task identifiers, labels, candidate classes, model/selector configuration, seed, predictions or required class scores, full-model margins, retained token positions, and relevant cross-task selection outcomes. Store task-level geometry once where possible. Preserve source values at suitable precision rather than only rounded presentation tables.

## Transfer and verification

After each substantive completed experiment, collect its evidence bundle, copy it to local storage, and push the small results or attach the larger bundle to a versioned GitHub Release. Keep the exact code/configuration association in an artifact manifest. Retain meaningful failed or inconclusive runs as well.

Downloads staged locally only to reach the server belong in `cache/transfers/`. After verifying successful transfer and extraction/loading on the server, delete those local archives and temporary extracted copies promptly. Apply the same rule to temporary installation wheels and duplicate download caches created for the transfer. Keep local scientific evidence and selected paper assets according to the table above.

Local large-artifact archives are retained outside the Git checkout under `/Users/nathan_tong0921/Desktop/2012 Project/artifacts/`, in a directory for each experiment. The server checkout is configured in the ignored local `.env.server`. Existing ignored checkpoint/cache paths remain ignored. Release attachments provide the GitHub copy of selected model weights without committing them to ordinary Git history.

A backup is complete only after its contents have been checked at the destination. Verify at least one bundle by downloading it from GitHub, opening the records, loading a selected checkpoint, and reproducing a summary table or figure. Before deleting unique server results or releasing the instance, confirm that all required local and GitHub copies are complete. Recomputable bulk caches are optional backups.

## Source references

- [Oxford-IIIT Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- [CLIP ViT-B/16 configuration](https://huggingface.co/openai/clip-vit-base-patch16/blob/main/config.json)
- [GitHub large-file guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
- [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)

## Offline-selection development pilots

The three `results/pets-selection*` bundles add about 51 MB: fixed panels, deletion scores, full candidate indices/scores, the two task-ranking embeddings, derived metrics and figures. Each individual file is below 10 MB, so ordinary Git retains these scientific artifacts. Prefix states live only in RAM; no model/data archive was downloaded locally for these runs. Original outputs remain in server `runs/`, with curated tracked copies in `results/`.

## Budget-context development experiment

The three `results/pets-budget-ranking*` bundles add approximately 209.5 MB, including four contexts of complete mask/response evidence, ridge coefficients, independent-mask prediction records, two arms of candidate banks and selected positions, analyses and plots. The largest individual file is 20.2 MB; this milestone remains in ordinary Git under the planned repository-size allowance. Prefix tensors existed only in RAM. Original server outputs remain under `runs/`, and tracked copies retain the scientific evidence under `results/`. Local transfer archives were not needed.


## First learned-selector pilot

`results/pets-selector-pilot-v1/` adds about 45 MB of manifests, validation teacher, all predictions and selected positions, training losses, analyses and figures to ordinary Git. The 70.93 MB training teacher and all six final selector checkpoints are stored in `pets-selector-pilot-v1-artifacts.zip` in the local directory `/Users/nathan_tong0921/Desktop/2012 Project/artifacts/pets-selector-pilot-v1/`, with a copy in the [versioned release](https://github.com/NathanTong0921/ViT-token-compression/releases/tag/pets-selector-pilot-v1). The archive restores these files at their original repository-relative `runs/` paths; original server files remain in place. Prefix caches totaling 222.71 MB are retained only on the server and can be regenerated. The run record documents destination recovery checks. Temporary duplicate downloads/extractions are removed after those checks.

## Fixed-update image-coverage experiment

`results/pets-selector-scale-v1/` retains analysis tables, exchange case arrays, training curves, figures and independent audits in ordinary Git. Two ZIP archives in the local `artifacts/pets-selector-scale-v1/` directory retain the larger evidence: `pets-selector-scale-supervision-v1.zip` (441.90 MB; teachers, probe masks/responses and full-model evidence) and `pets-selector-scale-models-and-evaluation-v1.zip` (941.52 MB; all 12 final selectors, training histories, and per-example scores, full 512D features and selected positions). These scientific backups are retained; they are not transfer-only staging files. Original server outputs remain under the three matching `runs/pets-selector-scale-*` directories.

The [versioned release](https://github.com/NathanTong0921/ViT-token-compression/releases/tag/pets-selector-scale-v1) holds both archives. Destination recovery status and exact byte counts are recorded in `results/pets-selector-scale-v1/release_assets.json` and the backup verification records. Prefix caches are omitted from the archives because they can be regenerated from the fixed model, preprocessing and image manifests.
