"""Run one fixed backbone/dataset group of the final evaluation."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from evaluate_final_test import run as evaluate
from prepare_final_test_cache import run as prepare


def run(config, group, skip_cache=False):
    root = Path("runs/final-test-v1")
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / f"{group}_status.json"
    status = {"group": group, "started_utc": datetime.now(timezone.utc).isoformat(),
              "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "completed_settings": []}

    def update(stage):
        status["stage"] = stage
        status["updated_utc"] = datetime.now(timezone.utc).isoformat()
        status_path.write_text(json.dumps(status, indent=2) + "\n")

    cache = next(c for c in config["feature_caches"] if c["id"] == group)
    update("features")
    if not skip_cache:
        prepare({**config, **cache}, Path(cache["output"]))
    for setting in config["settings"]:
        if setting["group"] == group:
            update(setting["id"])
            evaluate(config, setting)
            status["completed_settings"].append(setting["id"])
    update("complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--skip-cache", action="store_true")
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.group, args.skip_cache)
