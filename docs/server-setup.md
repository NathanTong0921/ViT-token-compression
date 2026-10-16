# Server setup

Develop code locally, commit and push it, then pull the same version into the server checkout. The server checkout is named `Experiments` under the user-selected data directory. Connection details belong in the ignored local `.env.server` file; no password is stored in the repository.

## Environment

The initial server has Ubuntu 22.04, Python 3.10 and NVIDIA driver 550.120. Use one RTX A6000 for the initial pipeline. PyTorch 2.6.0 and torchvision 0.21.0 support the CUDA 12.4 runtime; the CLIP source revision is fixed in `requirements.txt`.

Create `.venv` in the checkout and install the requirements. The selected server lacks the Ubuntu `ensurepip` component, so its environment was initialized with `python3 -m venv --without-pip .venv`, followed by the official [pip bootstrap script](https://bootstrap.pypa.io/get-pip.py) inside that environment. No system packages are changed.

The selected server can reach the Aliyun PyPI mirror, while direct connections to PyPI and the PyTorch wheel index failed during setup. Install from the reachable mirror:

```bash
.venv/bin/python -m pip install --no-cache-dir \
  --index-url https://mirrors.aliyun.com/pypi/simple -r requirements.txt
```

## Runtime check

Check current GPU activity before choosing a device. The following uses one GPU and one synthetic image to verify that CUDA and the pretrained CLIP model load and execute. It does not measure classification accuracy.

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/check_runtime.py \
  --output results/setup/runtime-check.json
.venv/bin/python -m pip freeze > results/setup/requirements-server.txt
```

Model downloads go under `cache/clip`. If the official checkpoint download is unavailable on the server, download that checkpoint locally and transfer it into this directory before running the check.

## Disk and results

At connection time the shared data filesystem had approximately 79 GiB available. This is shared free space, not an allocation reserved for this project. Start with the environment and B/16 checkpoint; measure disk use before generating feature caches.

Keep datasets in `data/raw`, feature caches in `cache`, selector weights in `checkpoints`, and raw run outputs in `runs` or `outputs`. Curated results go in `results` and are copied back locally before committing. Larger selected artifacts will use the [storage and backup plan](storage-and-backup.md).
