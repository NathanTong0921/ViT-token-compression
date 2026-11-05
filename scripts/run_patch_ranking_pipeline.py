"""Run one dataset's preparation, predictor fits and external comparison."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main(config_path):
    config = json.loads(config_path.read_text())
    output = Path(config["preservation_dir"]).parent
    output.mkdir(parents=True, exist_ok=True)
    common = ["--config", str(config_path)]
    steps = [
        ("prepare", "prepare_patch_ranking.py", []),
        ("train", "train_patch_ranking.py", ["--output", str(output / "training")]),
        ("evaluate", "evaluate_patch_ranking.py", ["--models-dir", str(output / "training"),
                                                  "--output", str(output / "evaluation")]),
        ("analyze", "analyze_patch_ranking.py", ["--source", str(output / "evaluation"),
                                                "--output", str(output / "analysis"), "--skip-plot"]),
    ]
    status = {"dataset": config["dataset"], "config": str(config_path), "steps": {}}
    for name, script, args in steps:
        command = [sys.executable, "scripts/" + script, *common, *args]
        status["active_stage"] = name
        (output / "pipeline_status.json").write_text(json.dumps(status, indent=2)+"\n")
        start = time.perf_counter()
        with (output / f"{name}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        status["steps"][name] = {"returncode": result.returncode,
                                  "seconds": time.perf_counter()-start, "command": command}
        (output / "pipeline_status.json").write_text(json.dumps(status, indent=2)+"\n")
        if result.returncode:
            raise SystemExit(result.returncode)
    status["active_stage"] = "complete"
    (output / "pipeline_status.json").write_text(json.dumps(status, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    main(parser.parse_args().config)
