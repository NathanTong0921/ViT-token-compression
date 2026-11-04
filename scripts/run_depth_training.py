"""Prepare depth-specific CLIP teachers and train the fixed C / wide-U pair.

The source_cache_dir field is used by the existing trainer only for the old
training-image IDs and task manifest. Prefixes and supervision used in training
always come from cache_dir, rebuilt at the requested depth. The block-6 reference
cache is read separately to compare unchanged full-model evidence and teacher
scales; its prefixes and teachers never enter the new-depth optimizer.
"""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check_full_reference(config):
    """Depth changes the cached prefix, while full-model predictions stay fixed."""
    current = Path(config["feature_cache_dir"])
    reference = Path(config["reference_cache_dir"])
    old = json.loads((reference / "prep_summary.json").read_text())["config"]
    for field in ("model", "prompt_template", "image_batch_size"):
        assert config[field] == old[field], field
    assert old["prune_after"] == 6
    checks = {}
    for split in ("train", "val"):
        assert read_rows(current / f"{split}_tasks.csv") == read_rows(reference / f"{split}_tasks.csv")
        with np.load(current / split / "evidence.npz") as new, np.load(reference / split / "evidence.npz") as saved:
            for field in ("image_ids", "labels"):
                np.testing.assert_array_equal(new[field], saved[field])
            errors = {}
            for field in ("text_features", "full_features", "full_scores"):
                np.testing.assert_allclose(new[field], saved[field], rtol=0,
                                           atol=config["identity_atol"])
                errors[field] = float(np.max(np.abs(new[field] - saved[field])))
            disagreements = int(np.sum(new["full_scores"].argmax(1) != saved["full_scores"].argmax(1)))
            assert disagreements == 0
            checks[split] = {"images": len(new["image_ids"]), "max_abs_error": errors,
                             "multiclass_prediction_disagreements": disagreements}
    result = {"depth": config["prune_after"], "reference_depth": 6,
              "reference_cache": str(reference), "splits": checks}
    write_json(current / "depth_reference_check.json", result)
    return result


def check_training_cache(config):
    """Check that the trainer receives the requested depth and mask experiment."""
    from token_compression.budget_ranking import sample_masks

    cache, features = Path(config["cache_dir"]), Path(config["feature_cache_dir"])
    fields = ("model", "prune_after", "keep_patch_tokens", "probe_count", "mask_seed", "ridge", "prompt_template")
    for folder in (cache, features):
        saved = json.loads((folder / "prep_summary.json").read_text())["config"]
        for field in fields:
            assert saved[field] == config[field], (str(folder), field)
    mask_seed = int(np.random.SeedSequence(
        [config["mask_seed"], config["keep_patch_tokens"], 0]).generate_state(1)[0])
    for split in ("train", "val"):
        prefix = cache / split / "prefix_tokens.npy"
        assert prefix.resolve() == (features / split / "prefix_tokens.npy").resolve()
        patches = np.load(prefix, mmap_mode="r").shape[1] - 1
        expected_masks = sample_masks(patches, config["keep_patch_tokens"], config["probe_count"], mask_seed)
        with np.load(cache / split / "teacher.npz") as teacher:
            np.testing.assert_array_equal(teacher["masks"], expected_masks)


def raw_utility_scales(coefficients, labels, pairs, text):
    """Return per-example geometric utility SD before the target's 1e-6 clamp."""
    image_indices, pair_indices = np.nonzero((labels[:, None, None] == pairs[None]).any(-1))
    scales = np.empty(len(image_indices), dtype=np.float32)
    for image_index in np.unique(image_indices):
        rows = np.flatnonzero(image_indices == image_index)
        label = labels[image_index]
        candidates = pairs[pair_indices[rows]]
        competitors = np.where(candidates[:, 0] == label, candidates[:, 1], candidates[:, 0])
        coeff = coefficients[image_index].astype(np.float32)
        utility = coeff[:, label][None, :] - coeff[:, competitors].T
        utility /= np.linalg.norm(text[label] - text[competitors], axis=-1)[:, None]
        utility -= utility.mean(axis=1, keepdims=True)
        scales[rows] = utility.std(axis=1, ddof=0)
    return scales, image_indices, pair_indices


def teacher_scale_diagnostics(config):
    from token_compression.selector_scale import balanced_example_weights

    records = []
    for role, cache in (("current", Path(config["cache_dir"])),
                        ("block6_reference", Path(config["reference_cache_dir"]))):
        for split in ("train", "val"):
            tasks = read_rows(cache / f"{split}_tasks.csv")
            pairs = np.array([[int(row["class_a"]), int(row["class_b"])] for row in tasks])
            with np.load(cache / split / "evidence.npz") as evidence:
                ids, labels = evidence["image_ids"].copy(), evidence["labels"].copy()
                text = evidence["text_features"].astype(np.float32)
            with np.load(cache / split / "teacher.npz") as teacher:
                np.testing.assert_array_equal(ids, teacher["image_ids"])
                scales, ii, ti = raw_utility_scales(teacher["coefficients"], labels, pairs, text)
            weights = balanced_example_weights(labels, pairs, ii, ti)
            records.append({"role": role, "cache": str(cache), "split": split,
                            "examples": len(scales), "balanced_mean_std": float(weights @ scales),
                            "unweighted_std_quantiles": np.quantile(scales, [0, .1, .25, .5, .75, .9, 1]).tolist(),
                            "near_zero_threshold": 1e-6,
                            "balanced_near_zero_fraction": float(weights @ (scales < 1e-6)),
                            "near_zero_examples": int(np.sum(scales < 1e-6))})
    result = {"definition": "SD across patches of (true-class minus competitor ridge coefficient) / text-direction norm, before utility standardization; float32 as in training.",
              "near_zero_definition": "Utility SD below the existing 1e-6 target-normalization clamp.",
              "weighting": "Equal task weight, then equal class weight, then equal image weight.",
              "records": records}
    write_json(Path(config["cache_dir"]) / "teacher_scale_diagnostics.json", result)
    return result


def final_fit_diagnostics(config):
    import torch
    from evaluate_selector_scale import selector_from_checkpoint
    from train_selector_scale import (load_split, evaluation_view,
                                      evaluate_teacher_diagnostics)
    from token_compression.selector import shared_descriptor

    before = time.perf_counter()
    device = torch.device(config["device"])
    cache = Path(config["cache_dir"])
    data = {split: load_split(cache, split, device, config["temperature"], config["batch_size"])
            for split in ("train", "val")}
    views = {split: evaluation_view(value) for split, value in data.items()}
    prior = shared_descriptor(data["train"]["text"], data["train"]["pairs"])
    records = []
    for seed in config["training_seeds"]:
        for arm in config["selector_arms"]:
            checkpoint_path = Path(config["training_dir"]) / "full" / f"seed_{seed}" / f"{arm}.pt"
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            assert checkpoint["seed"] == seed
            assert checkpoint["config"]["prune_after"] == config["prune_after"]
            assert checkpoint["config"]["keep_patch_tokens"] == config["keep_patch_tokens"]
            assert checkpoint["step"] == config["training_steps"]
            model = selector_from_checkpoint(checkpoint, prior, arm, data["train"]["tokens"].shape[-1], device)
            scores = {split: evaluate_teacher_diagnostics(model, value, views[split], config["batch_size"],
                                                          config["temperature"], config["keep_patch_tokens"])
                      for split, value in data.items()}
            records.append({"seed": seed, "arm": arm, "checkpoint": str(checkpoint_path), **scores})
    result = {"depth": config["prune_after"], "keep_patch_tokens": config["keep_patch_tokens"],
              "policy": "Final fixed-step checkpoints; diagnostics only, with no optimizer updates or checkpoint selection.",
              "wall_seconds": time.perf_counter() - before, "records": records}
    write_json(Path(config["training_dir"]) / "final_teacher_diagnostics.json", result)
    return result


def run(config, stage):
    if stage in ("features", "all"):
        from prepare_frozen_features import run as prepare_features
        prepare_features(config, Path(config["feature_cache_dir"]))
        check_full_reference(config)
    if stage in ("teacher", "all"):
        from prepare_selector_budget import run as prepare_teacher
        prepare_teacher(config, Path(config["cache_dir"]))
        teacher_scale_diagnostics(config)
    if stage in ("train", "all"):
        from train_selector_scale import train
        check_training_cache(config)
        summary = train(config, Path(config["training_dir"]))
        summary.update(run_type="fixed_update_depth_intervention",
                       depth=config["prune_after"], reference_depth=6,
                       capacity_comparison="The unchanged conditioned network (hidden dimension 128) and strong shared image-only-wide network (hidden dimension 199); identical task/class/image stream and fixed optimizer settings.",
                       source_cache_use="Read legacy training-image IDs and task manifest only; all optimized targets and prefixes are from the depth-specific cache_dir.")
        write_json(Path(config["training_dir"]) / "training_summary.json", summary)
        final_fit_diagnostics(config)
    if stage == "diagnostics":
        check_training_cache(config)
        teacher_scale_diagnostics(config)
        final_fit_diagnostics(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("features", "teacher", "train", "diagnostics", "all"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configuration = json.loads(args.config.read_text())
    if args.dry_run:
        print(json.dumps({"stage": args.stage, "config": configuration}, indent=2))
    else:
        run(configuration, args.stage)
