import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("ablation_audit", SCRIPTS / "audit_selector_ablations.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
import analyze_selector_ablations as analysis


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture(root):
    examples = [(f"a{i}", "01", 0, 0, 1) for i in range(3)] + [("b0", "01", 1, 0, 1)]
    examples += [(f"a{i}", "02", 0, 0, 2) for i in range(3)] + [(f"c{i}", "02", 2, 0, 2) for i in range(2)]
    sources = {source: {} for source in audit.SOURCE_METHODS}
    for budget in (49, 98):
        config = {"model": "ViT-B/16", "prune_after": 6, "keep_patch_tokens": budget,
                  "prompt_template": "a photo of a {class_name}", "manifest_dir": "fixture/splits",
                  "training_seeds": list(audit.SEEDS), "hidden_dim": 128, "training_steps": 3990,
                  "batch_size": 128, "temperature": 1., "learning_rate": .0003, "weight_decay": .01,
                  "cache_dir": f"fixture/teacher_{budget}", "probe_count": 512, "mask_seed": 2031, "ridge": .01}
        for source, methods in audit.SOURCE_METHODS.items():
            path, train = root / f"{source}_{budget}", root / f"{source}_{budget}_train"
            sources[source][budget] = path
            config_run = dict(config)
            if source == "ablations":
                config_run.update(selector_arms=list(methods), wide_hidden_dim=199, objective="kl")
            write_json(path / "eval_summary.json", {"models_dir": str(train), "config": config_run})
            runs, samples = [], []
            for seed in audit.SEEDS:
                samples.append({"seed": seed, "total_draws": 510720, "task_side_count_min": 480,
                                "task_side_count_max": 480, "coverages": {"full": {"available_images": 6,
                                "first_batch": {"image_ids": ["a0", "b0"]}, "image_draw_counts": {"a0": 4, "b0": 4}}}})
                for method in methods:
                    runs.append({"seed": seed, "arm": method, "coverage": "full",
                                 "parameter_count": 347399 if method == "image_only_wide" else 348033,
                                 "training_steps": 3990, "training_draws": 510720, "available_training_images": 6,
                                 "initial_train_kl": 1.2, "initial_val_kl": 1.1, "final_train_kl": .5, "final_val_kl": .8,
                                 "initial_state_exactly_matches_shared_seed_state": method != "image_only_wide",
                                 "initial_state_exactly_matches_own_architecture_seed_state": True,
                                 "image_draws_reused_between_arms": True, "clip_parameters_in_optimizer": 0})
            write_json(train / "training_summary.json", {"config": config_run, "runs": runs, "sampling_audit": "sampling.json"})
            write_json(train / "sampling.json", samples)
            rows, scores, positions = [], [], []
            for si, seed in enumerate(audit.SEEDS):
                for method in methods:
                    for image, task, label, a, b in examples:
                        image_index = int(image[-1])
                        value = np.ones(3, dtype=np.float32) * .9
                        value[label] = .83 + .03 * image_index + .01 * si
                        strength = {"conditioned": .12, "mean_only": .08, "difference_only": .04,
                                    "image_only": 0., "image_only_wide": 0.}[method]
                        competitor = b if a == label else a
                        value[competitor] -= strength
                        margin = float((value[label] - value[competitor]) / np.linalg.norm(np.eye(3)[label] - np.eye(3)[competitor]))
                        predicted = (a, b)[np.argmax(value[[a, b]])]
                        rows.append({"image_id": image, "task_id": task, "class_id": label, "class_a": a,
                                     "class_b": b, "method": method, "seed": seed, "task_split": "unseen",
                                     "keep_patch_tokens": budget, "correct": int(predicted == label), "margin": margin})
                        scores.append(value)
                        offset = int(task == "02") if method not in ("image_only", "image_only_wide") else 0
                        pos = np.full(196, -1, dtype=np.int16)
                        pos[:budget] = np.arange(budget) + offset
                        positions.append(pos)
            write_csv(path / "full/predictions.csv", rows)
            arrays = {"row_id": np.arange(len(rows)), "image_ids": [r["image_id"] for r in rows],
                      "task_ids": [r["task_id"] for r in rows], "labels": [r["class_id"] for r in rows],
                      "class_pairs": [[r["class_a"], r["class_b"]] for r in rows],
                      "cosine_scores": scores, "patch_indices": positions}
            arrays.update({key: [r[key] for r in rows] for key in ("seed", "method", "task_split", "keep_patch_tokens", "correct", "margin")})
            permutation = np.random.default_rng(8).permutation(len(rows))
            arrays = {key: np.asarray(value)[permutation] for key, value in arrays.items()}
            np.savez_compressed(path / "full/predictions.npz", **arrays, text_features=np.eye(3, dtype=np.float32))
    with contextlib.redirect_stdout(io.StringIO()):
        analysis.run("pets", sources, root / "results")
    return root / "results/analysis.json", sources


class AblationAuditTests(unittest.TestCase):
    def test_complete_fixture_reconstructs_all_methods_and_exchanges(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = fixture(root)
            result = audit.audit(path, root / "audit.json")
            self.assertEqual(result["status"], "passed")
            for budget in ("49", "98"):
                self.assertTrue(result["budgets"][budget]["exchange"]["both_image_only_variants_invariant"])
                self.assertEqual(result["budgets"][budget]["training_controls"]["parameters"]["image_only_wide"], 347399)

    def test_raw_score_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, sources = fixture(Path(tmp))
            p = sources["ablations"][49] / "full/predictions.npz"
            with np.load(p) as f:
                a = {k: f[k] for k in f.files}
            a["cosine_scores"][0, a["labels"][0]] += 1
            np.savez(p, **a)
            with self.assertRaises(AssertionError):
                audit.read_raw(sources["ablations"][49], audit.SOURCE_METHODS["ablations"], 49)

    def test_duplicated_prediction_identity_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, sources = fixture(Path(tmp))
            p = sources["ablations"][49] / "full/predictions.csv"
            rows = audit.read_csv(p)
            write_csv(p, rows + [rows[0]])
            with self.assertRaises(AssertionError):
                audit.read_raw(sources["ablations"][49], audit.SOURCE_METHODS["ablations"], 49)

    def test_false_wide_initialization_claim_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, sources = fixture(Path(tmp))
            p = Path(tmp) / "ablations_49_train/training_summary.json"
            d = audit.read_json(p)
            next(r for r in d["runs"] if r["arm"] == "image_only_wide")["initial_state_exactly_matches_shared_seed_state"] = True
            write_json(p, d)
            with self.assertRaises(AssertionError):
                audit.verify_training({s: paths[49] for s, paths in sources.items()}, 49)

    def test_adjusted_interval_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = fixture(Path(tmp))
            data = audit.read_json(path)
            data["results"]["49"]["primary_contrasts"]["conditioned_minus_mean_only"]["margin"]["bonferroni_family_ci95"][0] += .01
            write_json(path, data)
            with self.assertRaises(AssertionError):
                audit.audit(path, Path(tmp) / "audit.json")

    def test_bootstrap_clusters_repeated_task_rows_by_image(self):
        ids = np.array(["a", "a", "b", "c"])
        weights = np.array([.1, .1, .3, .5])
        values = np.array([[0., .2], [0., .2], [1., .1], [0., -.1]])
        point, intervals = audit.bootstrap(ids, weights, values, seed=2091, replicates=100)
        point2, intervals2 = audit.bootstrap(ids[[0, 2, 3]], np.array([.2, .3, .5]), values[[0, 2, 3]], seed=2091, replicates=100)
        np.testing.assert_allclose(point, [.3, .02], atol=1e-15)
        np.testing.assert_allclose(point, point2)
        for family in (1, 2, 4):
            np.testing.assert_allclose(intervals[family], intervals2[family])


if __name__ == "__main__":
    unittest.main()
