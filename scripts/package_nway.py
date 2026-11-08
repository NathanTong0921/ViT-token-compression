"""Archive the candidate-count experiment without copying reusable CLIP caches."""

import argparse
import json
from pathlib import Path
import zipfile


def run(source, output, results):
    output.mkdir(parents=True, exist_ok=True)
    archives = []
    for dataset in ("pets", "flowers"):
        folder = source / dataset
        status = json.loads((folder / "status.json").read_text())
        assert status["stage"] == "complete"
        files = sorted(path for path in folder.rglob("*") if path.is_file())
        log = source / f"{dataset}.log"
        if log.exists():
            files.append(log)
        target = output / f"nway-v1-{dataset}.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in files:
                archive.write(path, path.as_posix())
        with zipfile.ZipFile(target) as archive:
            assert archive.testzip() is None
        archives.append({"name": target.name, "bytes": target.stat().st_size,
                         "members": len(files), "uncompressed_bytes": sum(p.stat().st_size for p in files)})
    files = sorted(path for path in results.rglob("*") if path.is_file())
    files += [Path("configs") / f"nway_{d}.json" for d in ("pets", "flowers")]
    files += [Path("docs/experiments/2026-11-08-nway.md")]
    files += [Path("src/token_compression/nway.py")]
    files += sorted(Path("scripts").glob("*nway*.py"))
    files += sorted(Path("tests").glob("test_nway*.py"))
    target = output / "nway-v1-analysis.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.as_posix())
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
    archives.append({"name": target.name, "bytes": target.stat().st_size, "members": len(files),
                     "uncompressed_bytes": sum(p.stat().st_size for p in files)})
    summary = {"archives": archives,
               "scope": "Task panels and targets, all selectors, predictions, retained positions, features, costs, summaries and figures.",
               "reused_dependencies": [
                   "runs/pets-selector-budget-98-supervision-v1",
                   "runs/flowers-selector-budget-98-supervision-v1"],
               "source_code": "Included nway files supplement the versioned repository and its shared modules."}
    (output / "archive_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runs/nway-v1"))
    parser.add_argument("--results", type=Path, default=Path("results/nway-v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output, args.results)
