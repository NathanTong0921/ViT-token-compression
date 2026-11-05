"""Package the Pets/Flowers Patch Ranking runs with their replayable outputs."""

import argparse
import json
from pathlib import Path
import zipfile


def package(source, output):
    source, output = source.resolve(), output.resolve()
    files = []
    datasets = {}
    for dataset in ("pets", "flowers"):
        folder = source / dataset
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        selected = [path for path in sorted(folder.rglob("*")) if path.is_file()
                    and not any("smoke" in part.lower() or part in ("cache", "caches", "__pycache__")
                                for part in path.relative_to(source).parts)
                    and path.name != "prefix_tokens.npy"]
        files.extend(selected)
        datasets[dataset] = {"files": len(selected),
                             "source_bytes": sum(path.stat().st_size for path in selected)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as archive:
        for path in files:
            member = f"runs/{source.name}/{path.relative_to(source).as_posix()}"
            compression = zipfile.ZIP_STORED if path.suffix in (".npz", ".pt") else zipfile.ZIP_DEFLATED
            archive.write(path, arcname=member, compress_type=compression)
    summary = {"source": str(source), "archive": str(output), "file_count": len(files),
               "source_bytes": sum(item["source_bytes"] for item in datasets.values()),
               "archive_bytes": output.stat().st_size, "datasets": datasets}
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runs/patch-ranking-v1"))
    parser.add_argument("--output", type=Path, default=Path(
        "/data2/zhao.tong/artifacts/patch-ranking-v1/patch-ranking-v1-artifacts.zip"))
    args = parser.parse_args()
    print(json.dumps(package(args.source, args.output), indent=2))
