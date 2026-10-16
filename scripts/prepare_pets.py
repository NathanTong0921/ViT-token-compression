"""Create Oxford Pets image splits and candidate-class task manifests."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from token_compression.pets import prepare_manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--task-seed", type=int, default=2027)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()
    summary = prepare_manifests(
        args.data_root, args.output_dir, args.seed, args.task_seed, args.val_fraction
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
