"""Create Flowers102 manifests from the official labels, splits and category page."""

import argparse
import json
from pathlib import Path

from token_compression.flowers import prepare_manifests


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits/flowers-v1"))
    parser.add_argument("--task-seed", type=int, default=2027)
    args = parser.parse_args()
    summary = prepare_manifests(args.data_root, args.output_dir, args.task_seed)
    print(json.dumps({key: summary[key] for key in (
        "image_counts", "task_counts", "unused_pairs", "validation_image_task_examples",
        "validation_exchange_examples")}, indent=2))
