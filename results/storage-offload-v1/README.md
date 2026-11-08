# Local archive offload

Completed 2026-11-08. Removed 55 local ZIP copies totaling 26.99 GB after matching every archive's size and content digest to an uploaded GitHub Release asset. The project retains code, configurations, curated results, figure inputs and exports, experiment records, proposal and reference papers locally. Server working data was not changed.

The [manifest](manifest.json) maps each former local archive path to its GitHub asset, release and download location. Some releases are drafts and require the repository owner's authenticated account. `depth-intervention-training-v1.zip` is already backed up as `depth-intervention-training-v1-server.zip`; its contents match, so no additional upload was needed.

For new analyses requiring complete raw arrays or checkpoints, use the server working copy or retrieve the relevant archive listed in the manifest. Download only the required bundle, inspect its members before extraction, and remove the temporary local download when finished. Existing summary-based figures can be regenerated from the repository's `results/` and plotting scripts.

Small original archive inventories and recovery records are preserved in [source-manifests](source-manifests/). Their local paths describe the earlier backup locations; the current local-retention status is recorded in this directory and the [storage plan](../../docs/storage-and-backup.md).
