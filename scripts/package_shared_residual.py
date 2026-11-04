"""Bundle frozen bases and one saved online-forward example per finished setting."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import torch


SETTINGS = ("pets_b16_49", "flowers_b16_49", "flowers_b16_98")
ARMS = ("task_residual", "image_residual")


def package(root):
    root = Path(root).resolve()
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "root": str(root), "checkpoint_records": [], "verification_examples": [],
                "restore_scope": "Frozen base plus learned branch; one cached validation CLIP-prefix/text input and expected selector outputs per setting. Full classification evidence remains in the setting outputs.",
                "restore_tolerance": "An approximately 1e-5 float32 CPU/GPU forward tolerance is a reconstruction check, separate from reported accuracy evaluation."}
    for setting in SETTINGS:
        folder = root / setting
        protocol = json.loads((folder / "protocol.json").read_text())
        summary = json.loads((folder / "analysis/summary.json").read_text())
        assert summary["training_seeds"] == protocol["training_seeds"]
        assert len(summary["training_records"]) == len(ARMS) * len(protocol["training_seeds"])
        assert len(summary["evaluation_records"]) == len(protocol["training_seeds"])
    for setting in SETTINGS:
        folder = root / setting
        protocol = json.loads((folder / "protocol.json").read_text())
        destination = root / "base_checkpoints" / setting
        destination.mkdir(parents=True, exist_ok=True)
        seeds = protocol["training_seeds"]
        for seed in seeds:
            branches = {arm: folder / f"seed_{seed}" / f"{arm}.pt" for arm in ARMS}
            sources = {arm: Path(torch.load(path, map_location="cpu", weights_only=True)["shared_checkpoint"])
                       for arm, path in branches.items()}
            assert sources[ARMS[0]] == sources[ARMS[1]]
            source = sources[ARMS[0]]
            copied = destination / f"seed_{seed}.pt"
            shutil.copy2(source, copied)
            assert source.read_bytes() == copied.read_bytes()
            manifest["checkpoint_records"].append({"setting": setting, "seed": seed,
                "original_source": str(source), "bundled_base": str(copied.relative_to(root)),
                "branch_checkpoints": {arm: str(path.relative_to(root)) for arm, path in branches.items()},
                "size_bytes": copied.stat().st_size, "exact_bytes_verified": True})
        with (folder / "examples.csv").open() as handle:
            first = next(csv.DictReader(handle))
        image_index = int(first["image_index"])
        pair = np.asarray([[int(first["class_a"]), int(first["class_b"])]])
        prefix_path = Path(protocol["config"]["cache_dir"]) / "val/prefix_tokens.npy"
        prefixes = np.load(prefix_path, mmap_mode="r")
        with np.load(folder / "evidence.npz") as saved:
            candidate_text = saved["text_features"][pair]
            assert str(saved["image_ids"][image_index]) == first["image_id"]
        logits, positions = [], []
        for seed in seeds:
            with np.load(folder / f"seed_{seed}.npz") as saved:
                np.testing.assert_array_equal(saved["methods"], protocol["methods"])
                logits.append(saved["selector_logits"][:, :1])
                positions.append(saved["patch_indices"][:, :1])
        example = destination / "verification_input.npz"
        np.savez_compressed(example, prefix_tokens=np.asarray(prefixes[image_index:image_index + 1]),
                            candidate_text=candidate_text, seeds=np.asarray(seeds), methods=np.asarray(protocol["methods"]),
                            expected_selector_logits=np.stack(logits), expected_patch_indices=np.stack(positions),
                            image_id=np.asarray(first["image_id"]), task_id=np.asarray(first["task_id"]), class_pair=pair)
        manifest["verification_examples"].append({"setting": setting, "path": str(example.relative_to(root)),
            "image_id": first["image_id"], "task_id": first["task_id"], "source_example_row": 0,
            "prefix_shape": list(prefixes[image_index:image_index + 1].shape),
            "expected_logit_shape": list(np.stack(logits).shape), "size_bytes": example.stat().st_size})
    manifest["total_checkpoint_bytes"] = sum(r["size_bytes"] for r in manifest["checkpoint_records"])
    manifest["total_verification_bytes"] = sum(r["size_bytes"] for r in manifest["verification_examples"])
    path = root / "base_checkpoints/manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/shared-residual-v1"))
    args = parser.parse_args()
    result = package(args.root)
    print(json.dumps({"base_checkpoints": len(result["checkpoint_records"]),
                      "verification_examples": len(result["verification_examples"]),
                      "total_checkpoint_bytes": result["total_checkpoint_bytes"],
                      "total_verification_bytes": result["total_verification_bytes"]}))
