# Server setup

Develop code locally, commit and push it, then pull the same version into the server checkout. The server checkout is named `experiments` under the user-selected data directory. Connection details belong in the ignored local `.env.server` file; no password is stored in the repository.

## Environment

The initial server has Ubuntu 22.04, Python 3.10 and NVIDIA driver 550.120. Use one RTX A6000 for the initial pipeline. PyTorch 2.6.0 and torchvision 0.21.0 support the CUDA 12.4 runtime; the CLIP source revision is fixed in `requirements.txt`.

Create `.venv` in the checkout and install the requirements. The selected server lacks the Ubuntu `ensurepip` component, so its environment was initialized with `python3 -m venv --without-pip .venv`, followed by the official [pip bootstrap script](https://bootstrap.pypa.io/get-pip.py) inside that environment. No system packages are changed.

The selected server can reach the Aliyun PyPI mirror, while direct connections to PyPI and the PyTorch wheel index failed during setup. The mirror works for small packages:

```bash
.venv/bin/python -m pip install --no-cache-dir \
  --index-url https://mirrors.aliyun.com/pypi/simple -r requirements.txt
```

Large dependencies were prepared locally as Linux x86-64, CPython 3.10 wheels and transferred into `cache/setup/wheels`. The Linux-only CUDA and Triton requirements were taken from PyTorch 2.6.0's published package metadata: a cross-platform download on macOS does not automatically include dependencies guarded by Linux environment markers. The CLIP wheel was built from the pinned revision in `requirements.txt`.

Once this wheel directory is complete, install without downloading dependencies again:

```bash
.venv/bin/python -m pip install --no-index --find-links cache/setup/wheels \
  torch==2.6.0 torchvision==0.21.0 numpy==1.26.4 packaging clip
.venv/bin/python -m pip check
```

This installation and `pip check` completed successfully on 2026-10-16. The temporary server wheel directory was then removed. Local transfer copies of the wheels and checkpoint, together with matching newly created pip download-cache copies, were also removed (approximately 6.5 GB total). The installed environment and server B/16 checkpoint remain available.

When the checkout was renamed to lowercase `experiments`, the environment's activation scripts and executable entry points were updated to the new absolute path. Both pip and a CUDA tensor operation passed afterwards.

Install this repository's small Python package after pulling the experiment code:

```bash
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
```

## Runtime check

Check current GPU activity before choosing a device. The following uses one GPU and one synthetic image to verify that CUDA and the pretrained CLIP model load and execute. It does not measure classification accuracy.

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/check_runtime.py \
  --output runs/setup/runtime-check.json
.venv/bin/python -m pip freeze --all > runs/setup/requirements-server.txt
```

Copy these two records back to local `results/setup/` for review and version management. Generated server files stay under the ignored `runs/` directory so that subsequent code pulls do not conflict with locally curated results.

Model downloads go under `cache/clip`. If the official checkpoint download is unavailable on the server, download that checkpoint locally and transfer it into this directory before running the check.

Remove local transfer-only downloads once the remote copy is verified. Preserve scientific outputs, split manifests and selected figure assets; they are backups, not disposable transfer intermediates.

## Disk and results

At connection time the shared data filesystem had approximately 79 GiB available. This is shared free space, not an allocation reserved for this project. Start with the environment and B/16 checkpoint; measure disk use before generating feature caches.

Keep datasets in `data/raw`, feature caches in `cache`, selector weights in `checkpoints`, and raw run outputs in `runs` or `outputs`. Curated results go in `results` and are copied back locally before committing. Larger selected artifacts will use the [storage and backup plan](storage-and-backup.md).
