"""Archive reserved-test evidence by dataset/backbone, excluding prefix caches."""

import argparse
import json
from pathlib import Path
import zipfile


STORED_SUFFIXES = {".npz", ".pt", ".png", ".pdf", ".zip"}


def evidence_files(folder):
    return [path for path in sorted(folder.rglob("*")) if path.is_file()
            and path.name != "prefix_tokens.npy" and "__pycache__" not in path.parts]


def split_members(files, limit):
    """Bound parts by source bytes; compressed files are stored without recompression."""
    parts, current, total = [], [], 0
    for path in sorted(set(files)):
        size = path.stat().st_size + 1024
        if size > limit:
            raise ValueError(f"One file exceeds archive part budget: {path}")
        if current and total + size > limit:
            parts.append(current)
            current, total = [], 0
        current.append(path)
        total += size
    if current:
        parts.append(current)
    return parts


def write_archive(files, archive_path, repo):
    members = []
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as archive:
        for path in files:
            name = path.relative_to(repo).as_posix()
            compression = zipfile.ZIP_STORED if path.suffix in STORED_SUFFIXES else zipfile.ZIP_DEFLATED
            archive.write(path, arcname=name, compress_type=compression)
        for info in archive.infolist():
            members.append({"path": info.filename, "source_bytes": info.file_size,
                            "compressed_bytes": info.compress_size})
    return {"archive": archive_path.name, "archive_bytes": archive_path.stat().st_size,
            "file_count": len(members), "source_bytes": sum(row["source_bytes"] for row in members),
            "members": members}


def package(config_path, source, output, repo=None, part_bytes=1_900_000_000):
    repo = (Path.cwd() if repo is None else Path(repo)).resolve()
    config_path = Path(config_path).resolve()
    source, output = Path(source).resolve(), Path(output).resolve()
    config = json.loads(config_path.read_text())
    analysis = source / "analysis"
    if not (analysis / "analysis.json").is_file():
        raise FileNotFoundError(analysis / "analysis.json")
    output.mkdir(parents=True, exist_ok=True)
    archives, dependencies, owned_files = [], [], set()
    for cache in config["feature_caches"]:
        group = cache["id"]
        settings = [setting for setting in config["settings"] if setting["group"] == group]
        status_path = source / f"{group}_status.json"
        status = json.loads(status_path.read_text())
        assert status["stage"] == "complete", f"{group} is not complete"
        assert set(status["completed_settings"]) == {setting["id"] for setting in settings}
        cache_dir = (repo / cache["output"]).resolve()
        files = evidence_files(cache_dir)
        if not (cache_dir / "test/evidence.npz").is_file():
            raise FileNotFoundError(cache_dir / "test/evidence.npz")
        for setting in settings:
            eval_dir = (repo / setting["eval_dir"]).resolve()
            setting_analysis = analysis / setting["id"]
            if not (eval_dir / "evaluation_summary.json").is_file():
                raise FileNotFoundError(eval_dir / "evaluation_summary.json")
            if not (setting_analysis / "analysis.json").is_file():
                raise FileNotFoundError(setting_analysis / "analysis.json")
            files += evidence_files(eval_dir) + evidence_files(setting_analysis)
            for method, template in setting["checkpoints"].items():
                for seed in setting.get("seeds", config["seeds"]):
                    dependencies.append({"group": group, "setting": setting["id"],
                                         "method": method, "seed": seed,
                                         "checkpoint": template.format(seed=seed)})
        files += [path for path in source.glob(f"{group}*") if path.is_file()]
        owned_files.update(files)
        parts = split_members(files, part_bytes)
        for index, part in enumerate(parts, start=1):
            suffix = f"-part{index:02d}" if len(parts) > 1 else ""
            name = f"{source.name}-{group}{suffix}-artifacts.zip"
            entry = write_archive(part, output / name, repo)
            entry.update(group=group, part=index, parts=len(parts),
                         settings=[setting["id"] for setting in settings])
            archives.append(entry)
            print(json.dumps({"archive": name, "bytes": entry["archive_bytes"],
                              "files": entry["file_count"]}), flush=True)

    common = [config_path]
    common += [path for path in analysis.iterdir() if path.is_file()]
    common += [path for path in source.iterdir() if path.is_file() and path not in owned_files]
    common += [path for path in owned_files if path.suffix == ".json"
               and (path.name.endswith("summary.json") or path.name.endswith("status.json")
                    or path.name == "analysis.json")]
    experiment_note = repo / "docs/experiments/2026-11-08-final-test.md"
    if experiment_note.is_file():
        common.append(experiment_note)
    for manifest_dir in {cache["manifest_dir"] for cache in config["feature_caches"]}:
        common += evidence_files(repo / manifest_dir)
    common_parts = split_members(common, part_bytes)
    for index, part in enumerate(common_parts, start=1):
        suffix = f"-part{index:02d}" if len(common_parts) > 1 else ""
        name = f"{source.name}-common{suffix}-artifacts.zip"
        entry = write_archive(part, output / name, repo)
        entry.update(group="common", part=index, parts=len(common_parts))
        archives.append(entry)
        print(json.dumps({"archive": name, "bytes": entry["archive_bytes"],
                          "files": entry["file_count"]}), flush=True)
    summary = {"source": str(source), "repository": str(repo), "config": config_path.relative_to(repo).as_posix(),
               "part_source_byte_limit": part_bytes, "archives": archives,
               "archive_count": len(archives), "total_archive_bytes": sum(a["archive_bytes"] for a in archives),
               "file_count": sum(a["file_count"] for a in archives),
               "checkpoint_dependencies": dependencies,
               "omitted": "Regenerable prefix_tokens.npy caches and previously backed-up selector/PR weights.",
               "restoration": "Archive members retain repository-relative paths. Restore all group and common archives alongside the listed checkpoint dependencies."}
    (output / "archive_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/final_test_v1.json"))
    parser.add_argument("--source", type=Path, default=Path("runs/final-test-v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--part-bytes", type=int, default=1_900_000_000)
    args = parser.parse_args()
    result = package(args.config, args.source, args.output, part_bytes=args.part_bytes)
    print(json.dumps({key: result[key] for key in ("archive_count", "total_archive_bytes", "file_count")}))
