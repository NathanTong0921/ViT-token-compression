"""Prepare, train and evaluate one dataset's two/five-way development study."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def run(config_path, output):
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text())
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    status = {"dataset": config["dataset"],
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "code_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], text=True).strip(),
              "steps": []}
    started = time.perf_counter()
    commands = [
        ("prepare", ["scripts/prepare_nway.py", "--config", str(config_path),
                     "--output", str(output)]),
        ("train", ["scripts/train_nway_selectors.py", "--config", str(config_path),
                   "--prepared", str(output), "--output", str(output / "training")]),
        ("evaluate", ["scripts/evaluate_nway_selectors.py", "--config", str(config_path),
                      "--prepared", str(output), "--models", str(output / "training"),
                      "--output", str(output / "eval")]),
    ]
    for stage, command in commands:
        status["stage"] = stage
        status["updated_utc"] = datetime.now(timezone.utc).isoformat()
        (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps({"stage": stage, "dataset": config["dataset"]}), flush=True)
        before = time.perf_counter()
        completed = subprocess.run([sys.executable, *command])
        status["steps"].append({"stage": stage, "seconds": time.perf_counter() - before,
                                "returncode": completed.returncode})
        if completed.returncode:
            status["stage"] = "failed"
            (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
            raise SystemExit(completed.returncode)
    status.update(stage="complete", wall_seconds=time.perf_counter() - started,
                  updated_utc=datetime.now(timezone.utc).isoformat())
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)
