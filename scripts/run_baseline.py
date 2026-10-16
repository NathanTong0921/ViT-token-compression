"""Run the frozen CLIP baseline from the repository root."""

import argparse
import json

from token_compression.baseline import run_baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pets_baseline.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.config) as handle:
        config = json.load(handle)
    summary = run_baseline(config, args.output)
    print(json.dumps({key: summary[key] for key in (
        "image_count", "accuracy_37class", "binary", "image_extraction_wall_seconds"
    )}, indent=2))


if __name__ == "__main__":
    main()
