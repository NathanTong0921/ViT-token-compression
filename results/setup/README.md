# Runtime verification, 2026-10-16

The pretrained CLIP ViT-B/16 model loaded and completed image/text inference on one NVIDIA RTX A6000. The input was a uniform synthetic 224-pixel RGB image and two class prompts. The output had the expected shape and finite values. Dataset accuracy, token pruning, and the project hypotheses will be evaluated in subsequent experiments.

- `runtime-check.json`: the measured Python, PyTorch, torchvision, CUDA and GPU information, with the smoke-check outcome.
- `requirements-server.txt`: all installed Python package versions.
- Code version used for the check: `2fdfc220667e945847c6e4c1c58f12eef237a03f`.
- CLIP source revision: fixed in the root `requirements.txt`; the installed `clip==1.0` wheel was built from that revision.
- Installation validation: `pip check` reported no broken requirements.

Command executed from the server checkout:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/check_runtime.py \
  --output runs/setup/runtime-check.json
```

The generated records remain under the server's `runs/setup/`. These copies were retrieved into local `results/setup/` and checked before committing to GitHub. The checkpoint and environment are stored outside Git under the ignored `cache/clip/` and `.venv/` directories.
