"""Archive depth results and replayable training evidence without full prefix caches."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np


DATASETS = ("pets", "flowers")
DEPTHS = (3, 6, 9)
NEW_DEPTHS = (3, 9)
ARMS = ("conditioned", "image_only_wide")
SEEDS = (2041, 2042, 2043)
TEACHER_FIELDS = ("coefficients", "intercepts", "masks", "image_ids")
DEPENDENCIES = {
    "block6_evaluation_and_checkpoint_bundles": {
        "backup_manifest": "results/hybrid-followup-v1/backup_manifest.json",
        "archives": ["hybrid-specificity-v1.zip", "hybrid-distillation-v1.zip"],
    },
    "block6_original_core_replay": {
        "backup_manifest": "results/parallel-research-pilots-v1/backup_manifest.json",
        "archive": "core-protection-v1.zip",
    },
    "scope": "Block-6 evaluation rows are included for the paired depth analysis. Old block-6 checkpoints and teacher caches are not duplicated; use their existing backup manifests.",
}


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Archive:
    """Stream ordinary files and write small derived arrays directly into ZIP."""

    def __init__(self, path):
        self.path = Path(path)
        self.zip = zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED,
                                   compresslevel=6, allowZip64=True)
        self.members = []

    def add_file(self, path, member):
        path = Path(path)
        info = zipfile.ZipInfo.from_file(path, arcname=member)
        info.compress_type = (zipfile.ZIP_STORED if path.suffix in (".npz", ".pt")
                              else zipfile.ZIP_DEFLATED)
        digest, size = hashlib.sha256(), 0
        with path.open("rb") as source, self.zip.open(info, "w", force_zip64=True) as target:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                target.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        self.members.append(dict(path=member, bytes=size, sha256=digest.hexdigest()))

    def add_bytes(self, member, data):
        compression = zipfile.ZIP_STORED if member.endswith(".npz") else zipfile.ZIP_DEFLATED
        self.zip.writestr(member, data, compress_type=compression)
        self.members.append(dict(path=member, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))

    def add_json(self, member, value):
        self.add_bytes(member, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())

    def add_npz(self, member, **arrays):
        with io.BytesIO() as buffer:
            np.savez_compressed(buffer, **arrays)
            self.add_bytes(member, buffer.getvalue())

    def finish(self):
        self.zip.close()
        return dict(file=self.path.name, path=str(self.path.resolve()),
                    bytes=self.path.stat().st_size, sha256=file_digest(self.path),
                    member_count=len(self.members),
                    uncompressed_member_bytes=sum(row["bytes"] for row in self.members),
                    members=self.members)


def add_prefix_sample(archive, folder, split, member_root):
    features = folder / "features" / split
    prefix = np.load(features / "prefix_tokens.npy", mmap_mode="r")
    with np.load(features / "evidence.npz") as evidence:
        image_id = str(evidence["image_ids"][0])
        label = int(evidence["labels"][0])
        text = evidence["text_features"].copy()
    with (folder / "features" / f"{split}_tasks.csv").open() as handle:
        tasks = [row for row in csv.DictReader(handle)
                 if label in (int(row["class_a"]), int(row["class_b"]))]
    pairs = np.asarray([[int(row["class_a"]), int(row["class_b"])] for row in tasks])
    sample_path = f"{member_root}/features/{split}/prefix_replay_sample.npz"
    archive.add_npz(sample_path, prefix_tokens=np.asarray(prefix[:1]),
                    image_ids=np.asarray([image_id]), labels=np.asarray([label]),
                    text_features=text, candidate_pairs=pairs,
                    task_ids=np.asarray([row["task_id"] for row in tasks]))
    metadata = dict(split=split, source_row=0, image_id=image_id, class_id=label,
                    source_prefix=str(features / "prefix_tokens.npy"),
                    full_cache_shape=list(prefix.shape), prefix_shape=list(prefix[:1].shape),
                    dtype=str(prefix.dtype), matching_task_count=len(tasks),
                    archive_sample=sample_path,
                    scope="First cached image only. Use bundled task manifests for the training prior and paired checkpoint configurations for selector replay.")
    archive.add_json(f"{member_root}/features/{split}/prefix_replay_sample.json", metadata)
    return metadata


def package(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    eval_names = [f"{dataset}_d{depth}" for dataset in DATASETS for depth in DEPTHS]
    training_names = [f"{dataset}_b16_d{depth}_49" for dataset in DATASETS for depth in NEW_DEPTHS]
    reference_names = [f"{dataset}_b16_d6_49" for dataset in DATASETS]
    reference_files = ("diagnostic_config.json", "training/final_teacher_diagnostics.json")
    required_training = ("training_summary.json", "training_history.csv", "sampling_audit.json",
                         "completed_runs.json", "final_teacher_diagnostics.json")
    configs = {}
    for name in eval_names:
        for file in ("examples.csv", "evidence.npz", "protocol.json", "execution.json"):
            assert (source / name / file).is_file(), str(source / name / file)
        for s in SEEDS:
            for t in SEEDS:
                if s != t:
                    assert (source / name / f"pair_{s}_{t}.npz").is_file()
    for name in training_names:
        folder = source / name
        summary = json.loads((folder / "training/training_summary.json").read_text())
        config = configs[name] = summary["config"]
        assert tuple(config["training_seeds"]) == SEEDS
        assert tuple(config["selector_arms"]) == ARMS
        for file in required_training:
            assert (folder / "training" / file).is_file(), str(folder / "training" / file)
        for seed in SEEDS:
            for arm in ARMS:
                assert (folder / "training/full" / f"seed_{seed}" / f"{arm}.pt").is_file()
        for split in ("train", "val"):
            for file in (folder / "supervision" / split / "teacher.npz",
                         folder / "features" / split / "prefix_tokens.npy",
                         folder / "features" / split / "evidence.npz"):
                assert file.is_file(), str(file)
    for name in reference_names:
        for file in reference_files:
            assert (source / name / file).is_file(), str(source / name / file)
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"runs/{source.name}"
    evaluation = Archive(output / "depth-intervention-evaluation-v1.zip")
    for name in eval_names:
        for path in sorted((source / name).rglob("*")):
            if path.is_file():
                evaluation.add_file(path, f"{prefix}/{path.relative_to(source).as_posix()}")
    panel_names = set(eval_names + training_names + reference_names)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if (path.is_file() and relative.parts[0] not in panel_names
                and (path.suffix in (".json", ".log", ".sh", ".txt", ".py", ".csv")
                     or path.name.startswith("launch"))):
            evaluation.add_file(path, f"{prefix}/{relative.as_posix()}")
    evaluation.add_json("artifact_metadata/dependencies.json", DEPENDENCIES)
    evaluation_record = evaluation.finish()

    training = Archive(output / "depth-intervention-training-v1.zip")
    samples, teachers, checkpoint_count = [], [], 0
    for name in reference_names:
        for file in reference_files:
            training.add_file(source / name / file, f"{prefix}/{name}/{file}")
    for name in training_names:
        folder, member_root = source / name, f"{prefix}/{name}"
        training.add_json(f"{member_root}/configuration.json", configs[name])
        for file in required_training:
            training.add_file(folder / "training" / file, f"{member_root}/training/{file}")
        for seed in SEEDS:
            for arm in ARMS:
                relative = f"training/full/seed_{seed}/{arm}.pt"
                training.add_file(folder / relative, f"{member_root}/{relative}")
                checkpoint_count += 1
        for section in ("features", "supervision"):
            for path in sorted((folder / section).rglob("*")):
                if path.is_file() and path.suffix in (".json", ".csv"):
                    training.add_file(path, f"{member_root}/{path.relative_to(folder).as_posix()}")
        for split in ("train", "val"):
            evidence = folder / "features" / split / "evidence.npz"
            training.add_file(evidence, f"{member_root}/features/{split}/evidence.npz")
            with np.load(folder / "supervision" / split / "teacher.npz") as teacher:
                arrays = {key: teacher[key] for key in TEACHER_FIELDS}
                training.add_npz(f"{member_root}/supervision/{split}/teacher.npz", **arrays)
                teachers.append(dict(setting=name, split=split,
                    fields=list(TEACHER_FIELDS), images=len(arrays["image_ids"]),
                    coefficient_shape=list(arrays["coefficients"].shape),
                    response_arrays_included=False))
            samples.append(dict(setting=name, **add_prefix_sample(training, folder, split, member_root)))
    training.add_json("artifact_metadata/dependencies.json", DEPENDENCIES)
    training_record = training.finish()
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), source=str(source),
        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        archives=[evaluation_record, training_record],
        evaluation_panels=eval_names, newly_trained_settings=training_names,
        block6_diagnostic_settings=reference_names,
        final_checkpoint_count=checkpoint_count, teacher_arrays=teachers,
        prefix_replay_samples=samples, dependencies=DEPENDENCIES,
        full_prefix_caches_included=False, raw_teacher_responses_included=False,
        server_raw_runs_retained=True,
        restore_scope="All six evaluated panels, 24 new final selectors, complete new teacher coefficient/mask arrays, full-model evidence, training records and eight one-image prefix samples. Full prefixes and raw teacher responses remain at the source on the server.")
    (output / "package_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runs/depth-intervention-v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = package(args.source, args.output)
    print(json.dumps({"checkpoints": result["final_checkpoint_count"],
        "archives": [{key: archive[key] for key in ("file", "bytes", "member_count")}
                     for archive in result["archives"]]}, indent=2))
