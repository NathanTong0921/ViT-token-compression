"""Repeat the fixed PR fits and compare validation-selected and final weights."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def run(config_path, output):
    config = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    common = ["--config", str(config_path)]
    steps = [
        ("train", "train_patch_ranking.py", ["--output", str(output / "training"), "--select-best-val"]),
        ("eval_final", "evaluate_patch_ranking.py", ["--models-dir", str(output / "training"),
            "--output", str(output / "eval_final")]),
        ("eval_best", "evaluate_patch_ranking.py", ["--models-dir", str(output / "training/best_val_kl"),
            "--output", str(output / "eval_best")]),
        ("analyze", "analyze_patch_ranking_checkpoints.py", ["--run-dir", str(output),
            "--output", str(output / "analysis")]),
    ]
    status = {"dataset": config["dataset"], "config": str(config_path), "steps": {}}
    for name, script, extra in steps:
        status["active_stage"] = name
        (output / "pipeline_status.json").write_text(json.dumps(status, indent=2) + "\n")
        command = [sys.executable, "scripts/" + script, *common, *extra]
        start = time.perf_counter()
        with (output / f"{name}.log").open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        status["steps"][name] = {"returncode": result.returncode,
            "seconds": time.perf_counter() - start, "command": command}
        (output / "pipeline_status.json").write_text(json.dumps(status, indent=2) + "\n")
        if result.returncode:
            raise SystemExit(result.returncode)
    status["active_stage"] = "complete"
    (output / "pipeline_status.json").write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)
