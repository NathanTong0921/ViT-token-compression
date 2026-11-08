"""Archive the candidate-count experiment without copying reusable CLIP caches."""

import argparse
import json
from pathlib import Path
import zipfile


def run(source, output, results, inputs=None, archive_prefix="nway-v1",
        protocol=Path("docs/experiments/2026-11-08-nway.md")):
    output.mkdir(parents=True, exist_ok=True)
    roots = inputs if inputs is not None else {dataset: source / dataset for dataset in ("pets", "flowers")}
    archives, dependencies = [], []
    for dataset, folder in roots.items():
        status = json.loads((folder / "status.json").read_text())
        assert status["stage"] == "complete"
        config = json.loads((folder / "config.json").read_text())
        dependencies.append(config["cache_dir"])
        files = sorted(path for path in folder.rglob("*") if path.is_file())
        log = folder.parent / f"{folder.name}.log"
        if log.exists():
            files.append(log)
        target = output / f"{archive_prefix}-{dataset}.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in files:
                archive.write(path, path.as_posix())
        with zipfile.ZipFile(target) as archive:
            assert archive.testzip() is None
        archives.append({"name": target.name, "bytes": target.stat().st_size,
                         "members": len(files), "uncompressed_bytes": sum(p.stat().st_size for p in files)})
    files = sorted(path for path in results.rglob("*") if path.is_file())
    files += [Path("configs") / f"nway_{d}.json" for d in roots]
    files += [protocol]
    files += [Path("src/token_compression/nway.py")]
    files += sorted(Path("scripts").glob("*nway*.py"))
    files += sorted(Path("tests").glob("test_nway*.py"))
    if "cub" in roots:
        files += [Path("src/token_compression/cub.py")]
        files += sorted(Path("scripts").glob("*cub*.py"))
        files += sorted(Path("tests").glob("test_cub*.py"))
    target = output / f"{archive_prefix}-analysis.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.as_posix())
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
    archives.append({"name": target.name, "bytes": target.stat().st_size, "members": len(files),
                     "uncompressed_bytes": sum(p.stat().st_size for p in files)})
    summary = {"archives": archives,
               "scope": "Task panels and targets, all selectors, predictions, retained positions, features, costs, summaries and figures.",
               "reused_dependencies": sorted(set(dependencies)),
               "source_code": "Included nway files supplement the versioned repository and its shared modules."}
    (output / "archive_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runs/nway-v1"))
    parser.add_argument("--results", type=Path, default=Path("results/nway-v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", nargs="+", metavar="DATASET=ROOT")
    parser.add_argument("--archive-prefix", default="nway-v1")
    parser.add_argument("--protocol", type=Path, default=Path("docs/experiments/2026-11-08-nway.md"))
    args = parser.parse_args()
    roots = ({name: Path(path) for name, path in (item.split("=", 1) for item in args.inputs)}
             if args.inputs else None)
    run(args.source, args.output, args.results, roots, args.archive_prefix, args.protocol)
