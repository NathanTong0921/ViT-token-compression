# Storage and backup plan

Policy update, 2026-11-08: keep code, configurations, manifests, curated results, figure sources and writing materials locally. Keep complete experiment archives, selector checkpoints and bulk raw arrays remotely, using server working outputs and versioned GitHub Release attachments. Retrieve individual archives when an analysis needs them, then remove the temporary local copies. This policy supersedes the local-retention descriptions in the milestone records below, which document their original backup stage.

Local archive offload is complete. All 55 ZIPs matched their uploaded GitHub assets and their local copies were removed, freeing 26.99 GB. The project folder now occupies approximately 1.32 GB; small archive metadata remains local. The repository, including its approximately 0.79 GB of curated results and 0.51 GB of Git history, stays local. Asset locations and cleanup details are recorded in [the offload manifest](../results/storage-offload-v1/manifest.json).

The single-stage pruning pilot adds approximately 21.6 MB of scores, embeddings, retained positions, metrics and figure exports under `results/pets-pruning-v1/`. These are small enough for ordinary Git. Its approximately 223 MB prefix cache existed only in host RAM and was released after the run; no persistent prefix cache or selection-supervision cache was created.

## Capacity

For frozen CLIP ViT-B/16 on Oxford-IIIT Pets and Flowers102, budget 100 GB of available project storage on the server. Expected working use is approximately 30–60 GB with controlled cache retention. Use 150 GB if retaining additional model/layer caches or several environment copies. These figures exclude the operating-system image and large video-editing source files.

The current local working set is approximately 1.32 GB after archive offload, plus space for report/PPT/video materials and any temporary downloads. Full datasets, pretrained CLIP weights, feature caches and selector checkpoints do not need permanent local copies. Ordinary Git history should remain small, preferably below 1 GB; larger essential artifacts go in GitHub Releases.

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
| Per-example results supporting all reported comparisons, including baselines and negative findings | Yes | Curated analysis arrays; full arrays on demand | Git for small files; Release attachments for larger bundles |
| Seed-level metrics, timing samples, statistical summaries, figure source tables and plotting scripts | Yes | Yes | Git or Release attachments according to size |
| Final figures, token-selection visualizations, report/PPT source and exports | As needed | Yes | Git or Release attachments according to size |
| Images used for qualitative figures, with original IDs, preprocessing and source attribution | Yes | Yes | Relevant figure assets; source-image redistribution follows dataset permissions |
| Selected conditional and unconditional selector checkpoints for the reported seeds/configurations | Yes | On demand | Release attachments |
| Full public datasets and pretrained CLIP weights | Yes | On demand | Download sources/version information, not duplicate bulk files |
| Recomputable prefix caches, masked embeddings, temporary tensors and intermediate checkpoints | While useful | Temporary only when needed | Normally omitted |

Per-example records must retain enough information to recalculate the reported comparisons: image/task identifiers, labels, candidate classes, model/selector configuration, seed, predictions or required class scores, full-model margins, retained token positions, and relevant cross-task selection outcomes. Store task-level geometry once where possible. Preserve source values at suitable precision rather than only rounded presentation tables.

## Transfer and verification

After each substantive completed experiment, retain the working outputs on the server, copy the curated results and figure sources to the local repository, and push them to GitHub. Attach the complete larger evidence bundle to a versioned GitHub Release. Keep its remote location and code/configuration association in an artifact manifest. Retain meaningful failed or inconclusive runs as well.

Downloads staged locally only to reach the server belong in `cache/transfers/`. After verifying successful transfer and extraction/loading on the server, delete those local archives and temporary extracted copies promptly. Apply the same rule to temporary installation wheels and duplicate download caches created for the transfer. Keep local scientific evidence and selected paper assets according to the table above.

The project-parent `/Users/nathan_tong0921/Desktop/2012 Project/artifacts/` directory is an on-demand staging location, rather than a permanent local replica. Small archive inventories and remote asset references remain local. Download a needed bundle there, extract the required files, and remove the temporary copies when the analysis is complete. Existing figures and summary-based plots use the retained repository results. The server checkout is configured in the ignored local `.env.server`; ignored checkpoint/cache paths remain ignored.

Check uploaded assets against the archive inventory before removing local ZIPs, and retain the destination recovery records alongside the results. Full-array reanalysis can run on the server or use the corresponding Release attachment. Before deleting unique server results or releasing the instance, confirm that the required evidence and checkpoints are available in GitHub Releases. Recomputable bulk caches are optional backups.

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

The [versioned release](https://github.com/NathanTong0921/ViT-token-compression/releases/tag/pets-selector-scale-v1) holds both archives. Destination recovery passed: both published archives were downloaded, all 34 members matched original server files, all 12 restored checkpoints loaded, and primary results were independently reconstructed. Exact byte counts and recovery evidence are recorded in `results/pets-selector-scale-v1/release_assets.json` and `github_recovery_verification.json`. Temporary recovery downloads/extractions and server transfer-only ZIP copies were removed. Local scientific archives remain retained. Original server analysis outputs are retained in `runs/pets-selector-scale-analysis-v1/`; tracked curated results remain under `results/`. Prefix caches are omitted from the archives because they can be regenerated from the fixed model, preprocessing and image manifests.


## Three-budget comparison and H2

`results/pets-selector-budgets-v1/` retains the main and exchange statistics, H2 case-level data and cross-fitted predictions, shared-search selected sets, four figures in PNG/PDF, execution records and independent numerical audits. Five scientific archives live in the project parent directory `artifacts/pets-selector-budgets-v1/` and the [versioned release](https://github.com/NathanTong0921/ViT-token-compression/releases/tag/pets-selector-budgets-v1):

- `pets-selector-budget-98-supervision-v1.zip` (450.63 MB)
- `pets-selector-budget-98-models-and-evaluation-v1.zip` (743.57 MB)
- `pets-selector-budget-147-supervision-v1.zip` (449.13 MB)
- `pets-selector-budget-147-models-and-evaluation-v1.zip` (741.53 MB)
- `pets-selector-budget-49-and-shared-supplement-v1.zip` (184.34 MB)

Total: 2,569,188,841 bytes. The supplement contains the three random-mask runs at 49 tokens, all three shared-search results, regenerated probe responses for the 144 older validation images, and execution logs. Reused 49-token weights/predictions and the other teacher responses remain in `pets-selector-scale-v1`; restoring this milestone requires that earlier release as well. Public data and the frozen CLIP model plus the preserved preprocessing/manifests/code can regenerate prefix caches. Prefix tensors are not present in either release.

The recovery record distinguishes actual destination downloads from remote content-integrity verification. This avoids claiming that every archive was downloaded again. Small recovery details and the final transfer cleanup record are retained in the result directory; long-term scientific archives are kept locally.


## Flowers102 targeted replication

`results/flowers-selector-budgets-v1/` contains all two-budget statistics, exchange/geometry cases, cross-fitted predictions, figures, execution metadata and independent scientific checks. The five local archives in `../artifacts/flowers-selector-budgets-v1/` are also retained in [Release `flowers-selector-budgets-v1`](https://github.com/NathanTong0921/ViT-token-compression/releases/tag/flowers-selector-budgets-v1):

| Archive | Bytes |
| --- | ---: |
| flowers-selector-49-supervision-v1.zip | 678,179,424 |
| flowers-selector-49-eval-v1.zip | 515,734,236 |
| flowers-selector-98-supervision-v1.zip | 677,318,981 |
| flowers-selector-98-eval-v1.zip | 517,396,959 |
| flowers-selector-models-and-records-v1.zip | 23,628,778 |

Total: **2,412,258,378 bytes**. Supervision archives contain both image splits, full-model evidence, probe masks/responses and ridge teachers. Evaluation archives retain all reference, learned and random predictions, full embeddings and retained positions. The small models archive contains all twelve checkpoints, training histories, native-feature evidence and pipeline records. Prefix tensors are intentionally omitted; they can be regenerated from the public data/model plus the saved manifests and code. Flowers archives do not require the earlier Pets Releases for restoration.

All five local archives passed full CRC/content checks and were matched against the uploaded assets. The models-and-records archive was additionally downloaded from GitHub: all twelve checkpoints passed strict loading and forward checks, training summaries were reconstructed, and the native CLIP unseen-task baseline was recovered as 3947/4080. Recovery details distinguish these actual downloads from the large archives' upload-content checks. Original server runs and local scientific archives remain retained; local dataset-download staging and duplicate archive/recovery transfers were removed after verification. The result directory holds the asset, verification and cleanup records.


## Patch Ranking preservation external baseline

`results/patch-ranking-v1/` retains the classification tables, per-example CSVs, training histories, latency measurements and figures. The 248,018,584-byte archive in `../artifacts/patch-ranking-v1/` and `/data2/zhao.tong/artifacts/patch-ranking-v1/` contains 80 files: all six predictors, preservation rankings and deletion class scores, selected positions, image embeddings, analyses and timing records. The [versioned release](https://github.com/NathanTong0921/ViT-token-compression/releases/tag/patch-ranking-v1) provides the remote artifact backup. Frozen prefixes are reused from earlier milestones.

The checkpoint follow-up is retained separately in `results/patch-ranking-checkpoints-v1/`, with its 85,655,688-byte archive in the corresponding local and server artifact directories and [versioned release](https://github.com/NathanTong0921/ViT-token-compression/releases/tag/patch-ranking-checkpoints-v1). Its 78 files include six final and six validation-selected checkpoints, predictions, and analyses; reused teachers remain in the original PR release. Recovery downloaded the selected Pets checkpoint and both analysis summaries through HTTP byte ranges, loaded the checkpoint, and matched the classification tables. The original epoch-50 comparison remains available.

## Reserved-test confirmation

`results/final-test-v1/` retains the seven-setting tables, figures, execution summaries and independent point-estimate checks. Six archives totaling 5,536,287,748 bytes are stored in local `../artifacts/final-test-v1/`, server `/data2/zhao.tong/artifacts/final-test-v1/`, and the [versioned release](https://github.com/NathanTong0921/ViT-token-compression/releases/tag/final-test-v1). Their 251 members preserve full predictions, selected patches, image features, paired analysis arrays and the fixed manifests/configuration. Recomputable prefix caches are omitted; existing model checkpoints are listed as dependencies in `archive_summary.json`.

All local archives passed CRC checks and match their published assets. Byte-range recovery downloaded selected members from every archive and reconstructed a classification table for each of the four dataset/backbone groups. It did not download all six archives again. `release_assets.json` and `recovery.json` record the verified copies; transfer helpers and temporary response files were removed after verification.

## Two- and five-class extension

`results/nway-v1/` retains summary tables, figures, candidate manifests, training curves, costs and independent reconstruction records. Three archives totaling 555,008,216 bytes are stored in local `../artifacts/nway-v1/`, server `/data2/zhao.tong/artifacts/nway-v1/`, and the [versioned release](https://github.com/NathanTong0921/ViT-token-compression/releases/tag/nway-v1). Their 179 members include all 24 selectors, prepared task panels and supervision, raw predictions, retained positions, image features and analysis files. Existing prefix and 512-mask response caches are referenced rather than duplicated.

All local archives passed CRC checks and all 24 checkpoint payloads loaded. GitHub reports all three assets uploaded with matching sizes. The analysis archive was downloaded through the asset API and its tables, summary and configurations matched the working copies. These checks are recorded in `results/nway-v1/recovery.json`; local evaluation-transfer copies and temporary helpers were removed.

## CUB candidate-set pilot

`results/cub-pilot-v1/` retains the two- and five-class tables, paired analysis arrays, figures, task manifests, training histories, costs and independent result checks. Two archives totaling 402,539,580 bytes preserve all twelve selectors, task panels and supervision targets, raw predictions, selected patches, image features and analysis files in server `/data2/zhao.tong/artifacts/cub-pilot-v1/` and the [versioned release](https://github.com/NathanTong0921/ViT-token-compression/releases/tag/cub-pilot-v1). Prefix tensors and 512-mask response caches remain on the server and can be regenerated from the fixed manifests and code.

Both archives passed server CRC checks and match their published content digests. Recovery downloaded the small analysis archive, checked all 63 members and matched its summary, classification table, configuration and protocol to the working copies. The raw-result audit independently reconstructed the metrics and loaded all twelve checkpoints. Details are in `results/cub-pilot-v1/release_assets.json`, `recovery.json` and `result_audit.json`. Local dataset-transfer files were removed after extraction; no CUB checkpoint archive or recovery ZIP is retained locally.

## Content-based benefit prediction

`results/conditioning-content-v1/` adds approximately 33 MB of readout coefficients, content features, fold predictions, component analyses, routing decisions, figures and source tables. These small scientific artifacts are retained locally, in ordinary Git and on the server. Existing source prefixes and C/U outputs are reused from earlier runs and backups; no new dataset, CLIP checkpoint or bulk archive was downloaded. The fitted tree settings and seeds are recorded, and fold fits can be regenerated from the saved feature/outcome arrays and code.

## Direct classification training

`results/direct-classification-v1/` retains approximately 3.3 MB of paired outcomes, tables, figures, training records, costs and numerical checks. Two archives totaling 314,202,634 bytes are stored in server `/data2/zhao.tong/artifacts/direct-classification-v1/` and the [versioned release](https://github.com/NathanTong0921/ViT-token-compression/releases/tag/direct-classification-v1). They preserve 72 saved continuation checkpoints, six original C/U checkpoints, native evaluation scores and selected positions, source configuration and analysis. Existing Flowers49 prefix/teacher caches and the CLIP checkpoint are dependencies.

Both archives passed CRC checks and match the uploaded assets. All twelve final checkpoints loaded from the run archive. The small analysis archive was recovered into memory; the summary, seed table, paired outcomes, configuration and protocol matched local files. The working tree retains no ZIP or selector checkpoint copy from this run. Inventory and recovery records are in the result directory.

## Expanded direct-classification evaluation

On 2026-11-10, the user specified data4 for new outputs because data2 was nearly full. Code and reused data caches remain in `/data2/zhao.tong/experiments`; the new eighteen-endpoint evaluation is in `/data4/zhao.tong/experiments/direct-classification-panel-v1/`.

Curated outcomes, tables, figures and checks occupy approximately 10 MB locally and in Git. Two archives totaling 123,773,199 bytes are stored at `/data4/zhao.tong/artifacts/direct-classification-panel-v1/` and the [versioned release](https://github.com/NathanTong0921/ViT-token-compression/releases/tag/direct-classification-panel-v1). They preserve all raw evaluations and the analysis/code/configuration; existing checkpoints are dependencies from `direct-classification-v1`.

CRC and uploaded-content checks passed. The analysis archive was recovered from GitHub into memory and its key scientific files matched the working copies. Local transfer ZIPs and temporary helpers were removed. Inventory, asset URLs and recovery details are saved in `results/direct-classification-panel-v1/`.
