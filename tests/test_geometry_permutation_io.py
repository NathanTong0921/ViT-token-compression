import csv
import importlib.util
import json
from itertools import combinations
from pathlib import Path
import tempfile
import unittest

import numpy as np

from token_compression.selector_geometry import adjusted_angle_association


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_script("analyze_geometry_permutation")
old_geometry = load_script("analyze_selector_geometry")


def write_csv(path, rows):
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fixture(root):
    """Small shuffled predictions with repeated seeds, unused rows and one singleton."""
    rng = np.random.default_rng(512)
    text = rng.normal(size=(4, 8))
    text = (text / np.linalg.norm(text, axis=1, keepdims=True)).astype(np.float32)
    labels = np.arange(37) % 4
    full = rng.normal(size=(37, 8)) + 2 * text[labels]
    full = (full / np.linalg.norm(full, axis=1, keepdims=True)).astype(np.float32)
    seeds = [11, 12, 13]
    manifest = root / "manifests"
    manifest.mkdir()
    image_rows = [{"image_id": f"image_{i:03d}", "class_id": label, "split": "val"}
                  for i, label in enumerate(labels)]
    image_rows.append({"image_id": "reserved_image", "class_id": 0, "split": "test"})
    write_csv(manifest / "images.csv", image_rows)
    write_csv(manifest / "tasks.csv", [
        {"task_id": f"{a}_{b}", "class_a": a, "class_b": b, "split": "val"}
        for a, b in combinations(range(4), 2)])
    source = root / "raw"
    for group in ("references", "full"):
        jobs = [(-1, "full_model")] if group == "references" else [
            (seed, method) for seed in seeds for method in ("conditioned", "image_only")]
        records = []
        for seed, method in jobs:
            for index, label in enumerate(labels):
                tasks = [pair for pair in combinations(range(4), 2) if label in pair]
                if index == 36:
                    tasks = tasks[:1]
                for pair in tasks:
                    competitor = pair[1] if pair[0] == label else pair[0]
                    z = full[index].astype(float)
                    if method == "conditioned":
                        z += (.04 + .01 * (seed - 11)) * text[competitor]
                    elif method == "image_only":
                        z += .03 * text[label]
                    z = (z / np.linalg.norm(z)).astype(np.float32) if method != "full_model" else full[index]
                    records.append({
                        "seed": seed, "method": method, "task_split": "unseen",
                        "image_ids": f"image_{index:03d}", "labels": label,
                        "task_ids": f"{pair[0]}_{pair[1]}", "class_pairs": pair,
                        "image_features": z, "cosine_scores": z @ text.T,
                        "patch_indices": np.array([0, 2, -1, -1]),
                        "keep_patch_tokens": 4 if method == "full_model" else 2,
                    })
        unused = dict(records[0], image_ids="reserved_image", task_split="test")
        records.extend([unused, dict(records[0], task_split="seen")])
        rng.shuffle(records)
        packed = {key: np.asarray([row[key] for row in records]) for key in records[0]}
        packed["row_id"] = np.arange(len(records))
        packed["text_features"] = text
        (source / group).mkdir(parents=True)
        np.savez_compressed(source / group / "predictions.npz", **packed)
    setting = {"id": "fixture", "dataset": "synthetic", "model": "synthetic",
               "keep_patch_tokens": 2, "source": str(source), "manifest_dir": str(manifest),
               "geometry_source": str(root / "old_geometry")}
    return setting, seeds


def save_existing_geometry(setting, seeds):
    predictions = old_geometry.load_predictions(Path(setting["source"]), seeds)
    cases, _ = old_geometry.construct_geometry(predictions, seeds)
    average = old_geometry.average_seeds(cases, seeds)
    root = Path(setting["geometry_source"])
    (root / "budget_2").mkdir(parents=True)
    np.savez_compressed(root / "budget_2" / "seed_mean_cases.npz", **old_geometry.scalar_cases(average))
    ordinary = adjusted_angle_association(average, outcome="alignment", replicates=2, family_size=1)
    fixed = adjusted_angle_association(average, outcome="alignment", replicates=2,
                                       image_fixed_effects=True, family_size=1)
    exchange = adjusted_angle_association(average, replicates=2, family_size=1)
    exchange_fixed = adjusted_angle_association(average, replicates=2,
                                                image_fixed_effects=True, family_size=1)
    (root / "analysis.json").write_text(json.dumps({"budgets": {"2": {
        "primary_difficulty_adjusted": exchange, "image_fixed_effects_sensitivity": exchange_fixed,
        "normalized_alignment_diagnostics": [{"minimum_displacement_norm": 1e-12,
                                               "difficulty_adjusted": ordinary,
                                               "image_fixed_effects": fixed}]}}}))


class GeometryPermutationIOTests(unittest.TestCase):
    def test_compact_panel_preserves_exact_seed_vectors_and_development_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            setting, seeds = fixture(Path(directory))
            panel, metadata = runner.build_panel(setting, seeds)
            self.assertEqual(metadata["image_count"], 36)
            self.assertEqual(metadata["image_task_count"], 108)
            self.assertEqual(metadata["single_task_image_count_excluded"], 1)
            self.assertEqual(panel["seed_features"].shape, (3, 108, 8))
            self.assertEqual(panel["seed_features"].dtype, np.float32)
            self.assertNotIn("reserved_image", panel["image_ids"])
            with np.load(Path(setting["source"]) / "full" / "predictions.npz") as raw:
                for seed_index, seed in enumerate(seeds):
                    rows = panel["source_rows_per_seed"][seed_index]
                    np.testing.assert_array_equal(raw["seed"][rows], np.full(108, seed))
                    np.testing.assert_array_equal(raw["image_features"][rows], panel["seed_features"][seed_index])
                    np.testing.assert_array_equal(raw["task_ids"][rows], panel["task_ids"])
            expected = panel["text_features"][panel["labels"]].astype(float)
            expected -= panel["text_features"][panel["competitors"]]
            expected /= np.linalg.norm(expected, axis=1, keepdims=True)
            np.testing.assert_array_equal(expected, panel["task_directions"])
            images = runner.panel_images(panel, setting)
            np.testing.assert_array_equal(images[0].mean_features, panel["seed_features"][:, :3].astype(float).mean(axis=0))
            self.assertTrue(np.any(np.abs(np.linalg.norm(images[0].mean_features, axis=1) - 1.) > 1e-7))

    def test_selected_reserved_image_is_rejected_by_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            setting, seeds = fixture(Path(directory))
            source = Path(setting["source"]) / "full" / "predictions.npz"
            with np.load(source) as saved:
                packed = {key: saved[key].copy() for key in saved.files}
            packed["task_split"][packed["image_ids"] == "reserved_image"] = "unseen"
            np.savez_compressed(source, **packed)
            with self.assertRaisesRegex(AssertionError, "outside the development manifest"):
                runner.build_panel(setting, seeds)

    def test_one_undefined_seed_pair_does_not_change_primary_mean_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            setting, seeds = fixture(Path(directory))
            panel, _ = runner.build_panel(setting, seeds)
            # Exactly one task pair has identical vectors in the middle seed.
            # Other seeds still provide a valid complete seed-mean graph.
            panel["seed_features"][1, 1] = panel["seed_features"][1, 0]
            images = runner.panel_images(panel, setting)
            cases = runner.observed_pair_cases(panel, images, 2)
            self.assertEqual(len(cases["image_ids"]), 108)
            self.assertTrue(np.all(cases["displacement_norm"] > 1e-12))
            diagnostics = runner.per_seed_statistics(panel, cases)
            self.assertEqual([row["excluded_pair_count"] for row in diagnostics], [0, 1, 0])
            middle = diagnostics[1]
            self.assertEqual(middle["total_pair_count"], 108)
            self.assertEqual(middle["eligible_pair_count"], 107)
            self.assertEqual(middle["eligible_image_count"], 36)
            self.assertEqual(middle["excluded_pairs"], [{"image_id": "image_000", "task_a": "0_1", "task_b": "0_2"}])
            # Reconstruct the seed target independently and compare the original
            # regression routine on exactly the valid support.
            lookup = dict(zip(zip(panel["image_ids"], panel["task_ids"]), range(108)))
            targets, valid = [], []
            for image, a, b in zip(cases["image_ids"], cases["task_a"], cases["task_b"]):
                i, j = lookup[image, a], lookup[image, b]
                delta = panel["seed_features"][1, i].astype(float) - panel["seed_features"][1, j]
                direction = panel["task_directions"][i] - panel["task_directions"][j]
                keep = np.linalg.norm(delta) > 1e-12
                valid.append(keep)
                if keep:
                    targets.append(np.dot(delta, direction) / (np.linalg.norm(delta) * np.linalg.norm(direction)))
            subset = {key: values[np.asarray(valid)] for key, values in cases.items()}
            subset["alignment"] = np.asarray(targets)
            for fixed, key in ((False, "difficulty_adjusted_slope"), (True, "image_fixed_effects_slope")):
                expected = adjusted_angle_association(subset, outcome="alignment", replicates=2,
                                                      image_fixed_effects=fixed, family_size=1)
                self.assertAlmostEqual(middle[key], expected["angle_slope"], places=12)
            per_image = [np.mean(subset["alignment"][subset["image_ids"] == image])
                         for image in np.unique(subset["image_ids"])]
            self.assertAlmostEqual(middle["mean_alignment"], np.mean(per_image), places=12)

    def test_full_runner_reproduces_h2_and_reruns_only_from_saved_panels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            setting, seeds = fixture(root)
            save_existing_geometry(setting, seeds)
            config = {"settings": [setting], "training_seeds": seeds, "draws": 11,
                      "permutation_seed": 71, "family_size": 1, "final_test_used": False}
            panels = root / "panels"
            first = runner.run(config, root / "result", panels)
            source = Path(setting["source"])
            source.rename(source.with_name("not_available"))
            second = runner.run(config, root / "rerun", panels, from_panels=True)
            self.assertEqual(first["status"], "complete")
            self.assertEqual(first["settings"]["fixture"]["statistics"], second["settings"]["fixture"]["statistics"])
            audit = first["settings"]["fixture"]["input_reproduction_audit"]
            self.assertLess(audit["alignment_slope_max_abs_error"], 1e-10)
            with np.load(root / "result" / "fixture" / "permutation_statistics.npz") as saved:
                self.assertEqual(saved["null_statistics"].shape, (11, 3))
                np.testing.assert_array_equal(saved["draw_index"], np.arange(11))
            with np.load(root / "result" / "fixture" / "observed_pair_cases.npz") as saved:
                self.assertEqual(len(saved["image_ids"]), 108)
                np.testing.assert_allclose(saved["exchange_loss"], saved["displacement_norm"] *
                                           saved["direction_displacement_norm"] * saved["alignment"] / 2)


if __name__ == "__main__":
    unittest.main()
