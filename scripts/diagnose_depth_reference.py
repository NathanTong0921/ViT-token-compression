"""Measure final teacher fit for the existing block-6 C/U checkpoints."""

import argparse
import json
import os
from pathlib import Path

from evaluate_core_protection import SETTINGS
from run_depth_training import final_fit_diagnostics


def run(dataset, device):
    spec = SETTINGS[f"{dataset}_b16_49"]
    config = json.loads(Path(spec["config"]).read_text())
    root = Path("runs/depth-intervention-v1") / f"{dataset}_b16_d6_49"
    training = root / "training"
    config.update(device=device, training_dir=str(training),
                  selector_arms=["conditioned", "image_only_wide"])
    for seed in config["training_seeds"]:
        folder = training / "full" / f"seed_{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        for arm, source in (("conditioned", spec["conditioned_dir"]),
                            ("image_only_wide", spec["wide_dir"])):
            path = Path(source) / "full" / f"seed_{seed}" / f"{arm}.pt"
            target = folder / f"{arm}.pt"
            target.symlink_to(os.path.relpath(path.resolve(), folder.resolve()))
    (root / "diagnostic_config.json").write_text(json.dumps(config, indent=2) + "\n")
    return final_fit_diagnostics(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("pets", "flowers"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.dataset, args.device)
