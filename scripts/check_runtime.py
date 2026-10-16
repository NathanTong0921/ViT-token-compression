"""Check CUDA and pretrained CLIP inference; this is not an accuracy experiment."""

import argparse
import json
import platform
from pathlib import Path

import clip
import torch
import torchvision
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-cache", type=Path, default=Path("cache/clip"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    gpu_name = torch.cuda.get_device_name(device)
    model, preprocess = clip.load(
        "ViT-B/16", device=device, jit=False, download_root=str(args.model_cache)
    )
    model.eval()
    image = preprocess(Image.new("RGB", (224, 224), (120, 120, 120)))
    image = image.unsqueeze(0).to(device)
    text = clip.tokenize(["a photo of a dog.", "a photo of a cat."]).to(device)
    with torch.inference_mode():
        logits, _ = model(image, text)
    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all().item()

    result = {
        "status": "passed",
        "scope": "CUDA and CLIP runtime only; synthetic image, no accuracy evaluation",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu_name,
        "device": str(device),
        "model": "ViT-B/16",
        "logits_shape": list(logits.shape),
        "finite_logits": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
