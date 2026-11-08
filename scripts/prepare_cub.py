"""Create fixed CUB development-image manifests from official metadata."""

import argparse
import json
from pathlib import Path

from token_compression.cub import prepare_manifests


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits/cub-pilot-v1"))
    parser.add_argument("--seed", type=int, default=2615)
    parser.add_argument("--train-per-class", type=int, default=20)
    parser.add_argument("--val-per-class", type=int, default=5)
    args = parser.parse_args()
    summary = prepare_manifests(args.data_root, args.output_dir, args.seed,
                                args.train_per_class, args.val_per_class)
    print(json.dumps({key: summary[key] for key in
                      ("class_count", "image_counts", "image_seed", "annotation_files")}, indent=2))
