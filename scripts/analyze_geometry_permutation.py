"""Compare existing task alignment with within-image assignment permutations.

The default extracts compact development panels from saved model predictions.
Use --from-panels to rerun without the large prediction archives. No model is
loaded, trained or evaluated by this script.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from token_compression.geometry_permutation import (
    STATISTIC_NAMES, prepare_image, prepare_statistics, run_permutations,
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def selected_predictions(path, method, seeds=None):
    """Load only fields needed for the fixed unseen development-task panel."""
    with np.load(path) as saved:
        chosen = (saved["task_split"] == "unseen") & (saved["method"] == method)
        if seeds is not None:
            chosen &= np.isin(saved["seed"], seeds)
        fields = ("image_ids", "task_ids", "labels", "class_pairs", "seed",
                  "image_features", "row_id", "keep_patch_tokens")
        result = {key: saved[key][chosen].copy() for key in fields}
        result["text_features"] = saved["text_features"].copy()
    return result


def validate_development_rows(rows, manifest_dir):
    """Verify selected rows against existing development images and task pairs."""
    manifest_dir = Path(manifest_dir)
    images = {row["image_id"]: int(row["class_id"])
              for row in read_rows(manifest_dir / "images.csv") if row["split"] == "val"}
    tasks = {row["task_id"]: (int(row["class_a"]), int(row["class_b"]))
             for row in read_rows(manifest_dir / "tasks.csv") if row["split"] == "val"}
    for image_id, task_id, label, pair in zip(
            rows["image_ids"], rows["task_ids"], rows["labels"], rows["class_pairs"]):
        assert str(image_id) in images, f"Image {image_id} is outside the development manifest"
        assert images[str(image_id)] == label
        assert str(task_id) in tasks, f"Task {task_id} is outside the development manifest"
        assert tuple(pair) == tasks[str(task_id)]
        assert label in pair
    return {"development_image_count_in_manifest": len(images),
            "development_task_count_in_manifest": len(tasks),
            "selected_rows_match_development_manifests": True}


def build_panel(setting, seeds):
    source = Path(setting["source"])
    conditioned_path = source / "full" / "predictions.npz"
    reference_path = source / "references" / "predictions.npz"
    conditioned = selected_predictions(conditioned_path, "conditioned", seeds)
    reference = selected_predictions(reference_path, "full_model")
    np.testing.assert_array_equal(conditioned["text_features"], reference["text_features"])
    scope_a = validate_development_rows(conditioned, setting["manifest_dir"])
    validate_development_rows(reference, setting["manifest_dir"])
    assert np.all(conditioned["keep_patch_tokens"] == setting["keep_patch_tokens"])
    np.testing.assert_array_equal(np.unique(conditioned["seed"]), sorted(seeds))

    ordered = []
    for seed in seeds:
        indices = np.flatnonzero(conditioned["seed"] == seed)
        order = np.lexsort((conditioned["task_ids"][indices], conditioned["image_ids"][indices]))
        ordered.append(indices[order])
    first = ordered[0]
    for indices in ordered[1:]:
        for field in ("image_ids", "task_ids", "labels", "class_pairs"):
            np.testing.assert_array_equal(conditioned[field][indices], conditioned[field][first])
    keys = list(zip(conditioned["image_ids"][first], conditioned["task_ids"][first]))
    assert len(set(keys)) == len(keys), "Repeated image/task rows for a training seed"
    unique, inverse, counts = np.unique(conditioned["image_ids"][first], return_inverse=True, return_counts=True)
    eligible = counts[inverse] >= 2
    ordered = [indices[eligible] for indices in ordered]
    first = ordered[0]

    reference_keys = list(zip(reference["image_ids"], reference["task_ids"]))
    assert len(set(reference_keys)) == len(reference_keys)
    reference_lookup = {key: i for i, key in enumerate(reference_keys)}
    ref = np.asarray([reference_lookup[key] for key in zip(
        conditioned["image_ids"][first], conditioned["task_ids"][first])], dtype=np.int64)
    for field in ("labels", "class_pairs"):
        np.testing.assert_array_equal(conditioned[field][first], reference[field][ref])
    image_ids, image_first, image_index = np.unique(conditioned["image_ids"][first],
                                                  return_index=True, return_inverse=True)
    full_features = reference["image_features"][ref[image_first]]
    np.testing.assert_array_equal(reference["image_features"][ref], full_features[image_index])
    labels, pairs = conditioned["labels"][first], conditioned["class_pairs"][first]
    competitors = np.where(pairs[:, 0] == labels, pairs[:, 1], pairs[:, 0])
    text = conditioned["text_features"].astype(np.float64)
    direction = text[labels] - text[competitors]
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    seed_features = np.stack([conditioned["image_features"][i] for i in ordered])
    np.testing.assert_allclose(np.linalg.norm(seed_features.astype(float), axis=2), 1., rtol=0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(full_features.astype(float), axis=1), 1., rtol=0, atol=1e-5)
    panel = {
        "training_seeds": np.asarray(seeds, dtype=np.int64),
        "image_ids": conditioned["image_ids"][first], "task_ids": conditioned["task_ids"][first],
        "labels": labels, "class_pairs": pairs, "competitors": competitors,
        "seed_features": seed_features, "task_directions": direction,
        "full_margins": np.einsum("ij,ij->i", full_features[image_index].astype(float), direction),
        "unique_image_ids": image_ids, "image_index": image_index,
        "full_image_features": full_features, "text_features": conditioned["text_features"],
        "source_rows_per_seed": np.stack([conditioned["row_id"][i] for i in ordered]),
        "reference_rows": reference["row_id"][ref],
    }
    metadata = {
        "schema_version": 1, "setting": setting, "training_seeds": list(seeds),
        "task_split": "unseen", "image_split": "val", "final_test_used": False,
        "sources": {"conditioned": str(conditioned_path), "full_model": str(reference_path)},
        "source_row_field": "row_id", "feature_axes": ["training_seed", "image_task", "feature"],
        "feature_aggregation": "Convert saved normalized feature vectors to float64, then average seeds; do not renormalize the average.",
        "direction_definition": "normalize(text_features[true_class] - text_features[competing_class]) in float64",
        "full_margin_definition": "dot(saved full-model image feature, oriented unit task direction) in float64",
        "image_count": len(image_ids), "image_task_count": len(first),
        "feature_dimension": seed_features.shape[-1],
        "single_task_image_count_excluded": int(np.sum(counts == 1)),
        "selected_image_count_before_pair_eligibility": len(unique),
        "array_bytes_uncompressed": sum(array.nbytes for array in panel.values()),
        **scope_a,
    }
    return panel, metadata


def panel_images(panel, setting, seed_index=None):
    """Recover per-image complete task graphs from the compact flattened panel."""
    images = []
    for image_index, image_id in enumerate(panel["unique_image_ids"]):
        rows = np.flatnonzero(panel["image_index"] == image_index)
        features = panel["seed_features"][:, rows]
        if seed_index is not None:
            features = features[seed_index:seed_index + 1]
        images.append(prepare_image(image_id, panel["task_ids"][rows], features,
                                    panel["task_directions"][rows], panel["full_margins"][rows],
                                    namespace=setting["dataset"]))
    return images


def observed_pair_cases(panel, images, budget):
    parts = []
    for index, image in enumerate(images):
        rows = np.flatnonzero(panel["image_index"] == index)
        rows = rows[np.argsort(panel["task_ids"][rows])]
        a, b = image.edges.T
        alignment = image.alignment()
        parts.append({
            "image_ids": np.repeat(image.image_id, len(a)),
            "task_a": image.task_ids[a], "task_b": image.task_ids[b],
            "labels": panel["labels"][rows[a]],
            "competitor_a": panel["competitors"][rows[a]], "competitor_b": panel["competitors"][rows[b]],
            "keep_patch_tokens": np.full(len(a), budget),
            "base_margin_a": image.full_margins[a], "base_margin_b": image.full_margins[b],
            "angle_degrees": image.angle_degrees, "displacement_norm": image.feature_norm,
            "direction_displacement_norm": image.direction_norm, "alignment": alignment,
            "exchange_loss": image.feature_norm * image.direction_norm * alignment / 2,
            "seed_count": np.full(len(a), len(panel["training_seeds"])),
        })
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def audit_existing_geometry(setting, cases, observed, statistics):
    root = Path(setting["geometry_source"])
    budget = str(setting["keep_patch_tokens"])
    path = root / f"budget_{budget}" / "seed_mean_cases.npz"
    with np.load(path) as saved:
        previous = {key: saved[key].copy() for key in saved.files}
    keys = ("image_ids", "task_a", "task_b")
    lookup = {key: i for i, key in enumerate(zip(*[previous[k] for k in keys]))}
    order = np.asarray([lookup[key] for key in zip(*[cases[k] for k in keys])])
    assert len(order) == len(lookup), "Task-pair panel changed from the existing H2 analysis"
    for key in (*keys, "labels", "competitor_a", "competitor_b", "keep_patch_tokens", "seed_count"):
        np.testing.assert_array_equal(cases[key], previous[key][order])
    errors = {}
    for key in ("base_margin_a", "base_margin_b", "angle_degrees", "displacement_norm", "alignment", "exchange_loss"):
        np.testing.assert_allclose(cases[key], previous[key][order], rtol=0, atol=1e-10)
        errors[key] = float(np.max(np.abs(cases[key] - previous[key][order])))
    old = json.loads((root / "analysis.json").read_text())["budgets"][budget]
    alignment = next(item for item in old["normalized_alignment_diagnostics"]
                     if item["minimum_displacement_norm"] == 1e-12)
    old_slopes = np.asarray([alignment["difficulty_adjusted"]["angle_slope"],
                             alignment["image_fixed_effects"]["angle_slope"]])
    np.testing.assert_allclose(observed[:2], old_slopes, rtol=0, atol=1e-10)
    exchange_slopes = cases["exchange_loss"] @ statistics.coefficients[:2].T
    old_exchange_slopes = np.asarray([old["primary_difficulty_adjusted"]["angle_slope"],
                                     old["image_fixed_effects_sensitivity"]["angle_slope"]])
    np.testing.assert_allclose(exchange_slopes, old_exchange_slopes, rtol=0, atol=1e-10)
    return {"existing_pair_source": str(path), "existing_summary_source": str(root / "analysis.json"),
            "same_image_task_pairs": True, "pair_field_max_abs_errors": errors,
            "alignment_slope_max_abs_error": float(np.max(np.abs(observed[:2] - old_slopes))),
            "exchange_loss_slope_max_abs_error": float(np.max(np.abs(exchange_slopes - old_exchange_slopes)))}


def write_cases(path, cases):
    np.savez_compressed(path.with_suffix(".npz"), **cases)
    with path.with_suffix(".csv").open("w") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(cases)
        writer.writerows(zip(*cases.values()))


def run(config, output, panel_root, from_panels=False, prepare_only=False):
    output, panel_root = Path(output), Path(panel_root)
    output.mkdir(parents=True, exist_ok=True)
    panel_root.mkdir(parents=True, exist_ok=True)
    assert config["family_size"] == len(config["settings"])
    assert config["final_test_used"] is False
    seeds = config["training_seeds"]
    summary = {
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config, "final_test_used": False, "from_compact_panels": from_panels,
        "status": "preparing", "primary_statistic": STATISTIC_NAMES[0],
        "scope": "Unseen validation class combinations on development images; all complete within-image task graphs.",
        "interpretation": "Exploratory within-image assignment reference, conditional on saved feature vectors, task directions, development panel and three fixed training seeds.",
        "multiplicity": "One-sided upper-tail reference probability with Bonferroni correction across all seven primary settings. Secondary statistics are descriptive.",
        "seed_aggregation": "Average normalized feature vectors in float64 without renormalization; calculate alignment after averaging.",
        "weighting": "Equal total weight per image; equal weight per unordered task pair within image.",
        "settings": {},
    }
    started = time.perf_counter()
    prepared, dataset_keys = {}, {}
    # Every source panel and its original H2 statistics must agree before the
    # first permutation outcome is calculated.
    for setting in config["settings"]:
        name = setting["id"]
        path = panel_root / f"{name}.npz"
        if from_panels:
            with np.load(path) as saved:
                panel = {key: saved[key].copy() for key in saved.files}
            metadata = json.loads(path.with_suffix(".json").read_text())
            assert metadata["setting"] == setting
        else:
            panel, metadata = build_panel(setting, seeds)
            np.savez_compressed(path, **panel)
            write_json(path.with_suffix(".json"), metadata)
        np.testing.assert_array_equal(panel["training_seeds"], seeds)
        keys = np.column_stack([panel["image_ids"], panel["task_ids"]])
        if setting["dataset"] in dataset_keys:
            np.testing.assert_array_equal(keys, dataset_keys[setting["dataset"]])
        dataset_keys[setting["dataset"]] = keys
        images = panel_images(panel, setting)
        statistics = prepare_statistics(images)
        observed = statistics.evaluate([image.alignment() for image in images])
        cases = observed_pair_cases(panel, images, setting["keep_patch_tokens"])
        audit = audit_existing_geometry(setting, cases, observed, statistics)
        per_seed = []
        for index, seed in enumerate(seeds):
            seed_images = panel_images(panel, setting, seed_index=index)
            values = statistics.evaluate([image.alignment() for image in seed_images])
            per_seed.append({"training_seed": seed, **dict(zip(STATISTIC_NAMES, values.tolist()))})
        folder = output / name
        folder.mkdir(exist_ok=True)
        write_cases(folder / "observed_pair_cases", cases)
        summary["settings"][name] = {
            "setting": setting, "panel_path": str(path), "panel_metadata": metadata,
            "panel_file_bytes": path.stat().st_size, "input_reproduction_audit": audit,
            "observed_statistics": dict(zip(STATISTIC_NAMES, observed.tolist())),
            "observed_statistics_per_seed": per_seed, "design": statistics.metadata,
        }
        prepared[name] = images
        print(json.dumps({"stage": "input_verified", "setting": name,
                          "images": len(images), "pairs": len(cases["image_ids"])}), flush=True)
    summary["status"] = "inputs_verified"
    write_json(output / "analysis.json", summary)
    if prepare_only:
        return summary
    for setting in config["settings"]:
        name = setting["id"]
        before = time.perf_counter()
        result = run_permutations(prepared[name], draws=config["draws"], seed=config["permutation_seed"])
        folder = output / name
        np.savez_compressed(folder / "permutation_statistics.npz",
                            statistic_names=np.asarray(result["statistic_names"]),
                            observed=result["observed"], null_statistics=result["null_statistics"],
                            upper_tail_p=result["upper_tail_p"],
                            draw_index=np.arange(config["draws"]),
                            permutation_seed=np.asarray(config["permutation_seed"]))
        statistics = {}
        for j, statistic in enumerate(result["statistic_names"]):
            null = result["null_statistics"][:, j]
            statistics[statistic] = {
                "observed": float(result["observed"][j]), "upper_tail_p": float(result["upper_tail_p"][j]),
                "exceedance_count": int(np.count_nonzero(null >= result["observed"][j])),
                "null_mean": float(null.mean()), "null_std": float(null.std(ddof=1)),
                "null_quantiles_025_50_975": np.quantile(null, [.025, .5, .975]).tolist(),
                "observed_minus_null_mean": float(result["observed"][j] - null.mean()),
                "role": "primary" if j == 0 else "descriptive",
            }
        primary = statistics[STATISTIC_NAMES[0]]
        primary["bonferroni_upper_tail_p"] = min(1., primary["upper_tail_p"] * config["family_size"])
        primary["above_assignment_reference_after_seven_settings"] = (
            primary["bonferroni_upper_tail_p"] <= .05 and primary["observed"] > 0)
        summary["settings"][name].update(statistics=statistics,
                                            permutation_wall_seconds=time.perf_counter() - before)
        write_json(output / "analysis.json", summary)
        print(json.dumps({"stage": "permutations_complete", "setting": name,
                          "primary": primary}), flush=True)
    summary["status"] = "complete"
    summary["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["total_wall_seconds"] = time.perf_counter() - started
    write_json(output / "analysis.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, default=Path("runs/geometry-permutation-v1/panels"))
    parser.add_argument("--from-panels", action="store_true", help="Read saved compact panels instead of extracting raw predictions")
    parser.add_argument("--prepare-only", action="store_true", help="Extract/reload panels and reproduce existing H2 values, then stop")
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output, args.panel_root, args.from_panels, args.prepare_only)
