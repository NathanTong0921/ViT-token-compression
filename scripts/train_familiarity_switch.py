"""Fixed disjoint class-role switch with identical random identity codes."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from semantic_identity_protocol import class_partition, class_permutation


def make_plan(class_count, config, original_references=None, partition_seed=2099):
    h0 = class_partition(class_count, partition_seed, holdout_block=0)[1]
    h1 = class_partition(class_count, partition_seed, holdout_block=1)[1]
    remaining = np.setdiff1d(np.arange(class_count), np.r_[h0, h1])
    mappings = {f"permuted_{seed}": class_permutation(class_count, seed, (h0, h1, remaining)).tolist()
                for seed in (2097, 2098)}
    return dict(class_count=class_count, partition_seed=partition_seed, H0=h0.tolist(), H1=h1.tolist(),
                always_seen=remaining.tolist(), partition_blocks=[0, 1],
                training_seeds=[2041, 2042, 2043], permutation_seeds=[2097, 2098], mappings=mappings,
                mapping_policy="One fixed class-code map per seed, preserving H0, H1 and always-seen classes separately; identical in both fits.",
                original_references=str(original_references) if original_references is not None else None, config=config,
                execution={"original": "Reuse real_text and image_only_wide exactly; fit both fixed-code arms for 3 seeds."
                           if original_references is not None else "Fit all 4 arms for 3 seeds.",
                           "switched": "Fit all 4 arms for 3 seeds."},
                new_fits=18 if original_references is not None else 24,
                final_test_used=False, official_test_used=False,
                evaluation="Identical validation image/task rows for H0-by-H1 candidate pairs from original train/val manifests only. Both candidate roles reverse; this is not an isolated intervention on the true class alone.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--original-references", type=Path,
                        help="Optional matching P0 references; omit to train all arms in both phases.")
    parser.add_argument("--partition-seed", type=int, default=2099)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--phase", choices=["original", "switched", "both"], default="both")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.device:
        config["device"] = args.device
    with np.load(Path(config["cache_dir"]) / "train" / "evidence.npz") as saved:
        class_count = len(saved["text_features"])
    plan = make_plan(class_count, config, args.original_references, args.partition_seed)
    plan["code_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert config["training_seeds"] == plan["training_seeds"]
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "frozen_plan.json"
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan, "Run configuration differs from the saved fixed plan"
    else:
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"plan": str(plan_path), "partition_seed": plan["partition_seed"],
                      "new_fits": plan["new_fits"], "H0": plan["H0"], "H1": plan["H1"]}), flush=True)
    if args.plan_only:
        return
    from run_semantic_identity_pilot import run
    phases = ["original", "switched"] if args.phase == "both" else [args.phase]
    for phase in phases:
        run(config, args.output / phase, ["class_holdout"], plan["permutation_seeds"], plan["partition_seed"],
            args.bootstrap_replicates, holdout_block=0 if phase == "original" else 1,
            fixed_mappings=plan["mappings"], reuse_references=args.original_references if phase == "original" else None)


if __name__ == "__main__":
    main()
