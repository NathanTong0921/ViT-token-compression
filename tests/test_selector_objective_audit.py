import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("objective_audit", SCRIPTS / "audit_selector_objectives.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
from analyze_selector_objectives import analyze_budget


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture(root):
    examples = [(f"a{i}", "01", 0, 0, 1) for i in range(3)] + [("b0", "01", 1, 0, 1)]
    examples += [(f"a{i}", "02", 0, 0, 2) for i in range(3)] + [(f"c{i}", "02", 2, 0, 2) for i in range(2)]
    summary = {"dataset": "pets", "budgets": [49, 98], "training_seeds": list(audit.SEEDS),
               "bootstrap_replicates": 2000, "bootstrap_seed": 2085,
               "sources": {o: {} for o in audit.OBJECTIVES}, "results": {}}
    for budget in summary["budgets"]:
        config = {"model": "ViT-B/16", "prune_after": 6, "keep_patch_tokens": budget,
                  "prompt_template": "a photo of a {class_name}", "manifest_dir": "fixture/splits",
                  "training_seeds": list(audit.SEEDS), "hidden_dim": 128, "training_steps": 3990,
                  "batch_size": 128, "temperature": 1., "learning_rate": .0003, "weight_decay": .01,
                  "cache_dir": f"fixture/teacher_{budget}", "probe_count": 512, "mask_seed": 2031, "ridge": .01}
        rows_by_objective = {}
        for objective in audit.OBJECTIVES:
            source = root / f"{objective}_{budget}"
            train_dir = root / f"{objective}_{budget}_training"
            summary["sources"][objective][str(budget)] = str(source)
            run_config = {**config, **({"objective": "topk_rank"} if objective == "rank" else {})}
            write_json(source / "eval_summary.json", {"config": run_config, "models_dir": str(train_dir)})
            runs, sampling = [], []
            for seed in audit.SEEDS:
                full = {"available_images": 6, "unique_images_drawn": 6,
                        "first_batch": {"image_ids": ["a0", "b0"], "task_ids": ["01", "01"], "sides": [0, 1]},
                        "image_draw_counts": {"a0": 4, "b0": 4}}
                sampling.append({"seed": seed, "total_draws": 510720, "task_side_count_min": 480,
                                 "task_side_count_max": 480, "coverages": {"full": full}})
                for arm in audit.ARMS:
                    runs.append({"seed": seed, "arm": arm, "coverage": "full", "parameter_count": 348033,
                                 "training_steps": 3990, "training_draws": 510720, "available_training_images": 6,
                                 "initial_train_kl": 1.2, "initial_val_kl": 1.1,
                                 "initial_state_exactly_matches_shared_seed_state": True,
                                 "image_draws_reused_between_arms": True, "clip_parameters_in_optimizer": 0})
            if objective == "kl" and budget == 49:
                # Historical baseline includes small-coverage runs; only full is paired.
                runs += [{**r, "coverage": "small", "available_training_images": 2} for r in runs]
                for record in sampling:
                    record["coverages"]["small"] = {"available_images": 2}
            write_json(train_dir / "training_summary.json", {"config": run_config, "runs": runs,
                                                              "sampling_audit": "sampling_audit.json"})
            write_json(train_dir / "sampling_audit.json", sampling)
            rows, scores = [], []
            for si, seed in enumerate(audit.SEEDS):
                for arm in audit.ARMS:
                    for ei, (image, task, label, a, b) in enumerate(examples):
                        margin = -.06 + .015 * ei + .004 * si
                        if arm == "conditioned":
                            margin += .01
                        if objective == "rank":
                            margin += .01 + (.03 if arm == "conditioned" and image == "a0" else 0)
                        pair_scores = np.zeros(3, dtype=np.float32)
                        pair_scores[label] = margin * np.sqrt(2)
                        actual_margin = float(pair_scores[label] / np.float32(np.sqrt(2)))
                        predicted = (a, b)[int(np.argmax(pair_scores[[a, b]]))]
                        rows.append({"image_id": image, "task_id": task, "class_id": label,
                                     "class_a": a, "class_b": b, "seed": seed, "method": arm,
                                     "task_split": "unseen", "keep_patch_tokens": budget,
                                     "correct": int(predicted == label), "margin": actual_margin})
                        scores.append(pair_scores)
            write_csv(source / "full" / "predictions.csv", rows)
            arrays = {"image_ids": np.asarray([r["image_id"] for r in rows]),
                      "task_ids": np.asarray([r["task_id"] for r in rows]),
                      "labels": np.asarray([r["class_id"] for r in rows]),
                      "class_pairs": np.asarray([[r["class_a"], r["class_b"]] for r in rows]),
                      "cosine_scores": np.asarray(scores)}
            for field in ("seed", "method", "task_split", "keep_patch_tokens", "correct", "margin"):
                arrays[field] = np.asarray([r[field] for r in rows])
            order = np.random.default_rng(6).permutation(len(rows))
            arrays = {k: v[order] for k, v in arrays.items()}
            np.savez(source / "full" / "predictions.npz", **arrays, text_features=np.eye(3, dtype=np.float32))
            rows_by_objective[objective] = rows
        result, cases, packed, *_ = analyze_budget(rows_by_objective, list(audit.SEEDS), budget, 2000, 2085)
        summary["results"][str(budget)] = result
        folder = root / "results" / f"budget_{budget}"
        case_rows = [dict(zip(cases, values)) for values in zip(*cases.values())]
        write_csv(folder / "paired_cases.csv", case_rows)
        np.savez(folder / "paired_cases.npz", **packed)
    path = root / "results" / "analysis.json"
    write_json(path, summary)
    return path


class SelectorObjectiveAuditTests(unittest.TestCase):
    def test_end_to_end_raw_score_pairing_and_reconstructed_intervals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = fixture(root)
            result = audit.audit(summary, root / "audit.json")
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["budgets"]["49"]["image_count"], 6)
            self.assertEqual(result["budgets"]["49"]["raw_unseen_rows_verified_per_objective"], 54)
            self.assertTrue(result["budgets"]["49"]["training_controls"]["recorded_full_sampling_matches"])
            self.assertGreater(result["budgets"]["49"]["primary_interaction"]["margin"]["mean"], 0)

    def test_audit_rejects_modified_effect_interval_and_cached_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = fixture(root)
            original = audit.read_json(path)
            for field in ("mean", "bonferroni_family_ci95"):
                changed = copy.deepcopy(original)
                target = changed["results"]["49"]["primary_interaction"]["margin"]
                if field == "mean":
                    target[field] += .01
                else:
                    target[field][0] += .01
                write_json(path, changed)
                with self.subTest(field=field), self.assertRaises(AssertionError):
                    audit.audit(path, root / "audit.json")
            write_json(path, original)
            cases = root / "results" / "budget_49" / "paired_cases.csv"
            rows = audit.read_csv(cases)
            rows[0]["task_class_weight"] = .9
            write_csv(cases, rows)
            with self.assertRaises(AssertionError):
                audit.audit(path, root / "audit.json")

    def test_raw_score_reconstruction_rejects_falsified_correctness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            path = root / "rank_49" / "full" / "predictions.csv"
            rows = audit.read_csv(path)
            rows[0]["correct"] = 1 - int(rows[0]["correct"])
            write_csv(path, rows)
            with self.assertRaises(AssertionError):
                audit.load_predictions(root / "rank_49", 49)

    def test_full_sampling_or_teacher_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            sources = {o: root / f"{o}_49" for o in audit.OBJECTIVES}
            metadata = {o: audit.read_json(sources[o] / "eval_summary.json") for o in audit.OBJECTIVES}
            broken = copy.deepcopy(metadata)
            broken["rank"]["config"]["cache_dir"] = "different_teacher"
            with self.assertRaises(AssertionError):
                audit.audit_training(broken, 49)
            path = root / "rank_49_training" / "sampling_audit.json"
            records = audit.read_json(path)
            records[0]["coverages"]["full"]["image_draw_counts"]["a0"] += 1
            write_json(path, records)
            with self.assertRaises(AssertionError):
                audit.audit_training(metadata, 49)

    def test_pairing_rejects_duplicated_missing_and_relabelled_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            rows = {o: audit.read_csv(root / f"{o}_49" / "full" / "predictions.csv") for o in audit.OBJECTIVES}
            for change in ("duplicate", "missing", "label"):
                altered = copy.deepcopy(rows)
                if change == "duplicate":
                    altered["rank"].append(dict(altered["rank"][0]))
                elif change == "missing":
                    altered["rank"].pop()
                else:
                    altered["rank"][0]["class_id"] = 1
                with self.subTest(change=change), self.assertRaises(AssertionError):
                    audit.align(altered, 49)

    def test_bootstrap_matches_direct_cluster_resampling_with_unequal_weights(self):
        ids = np.asarray(["a", "b", "c", "d", "a", "e", "f"])
        weights = np.asarray([.25, 1/12, 1/12, 1/12, .25, .125, .125])
        values = np.column_stack((np.arange(7), np.arange(7) ** 2))
        point, ci95, ci975 = audit.bootstrap(ids, weights, values, 200, 2085)
        rng, images, direct = np.random.default_rng(2085), np.unique(ids), []
        for _ in range(200):
            draw = images[rng.integers(len(images), size=len(images))]
            rows = np.concatenate([np.flatnonzero(ids == image) for image in draw])
            direct.append(np.average(values[rows], axis=0, weights=weights[rows]))
        np.testing.assert_allclose(point, np.average(values, axis=0, weights=weights))
        np.testing.assert_allclose(ci95, np.quantile(direct, [.025, .975], axis=0))
        np.testing.assert_allclose(ci975, np.quantile(direct, [.0125, .9875], axis=0))


if __name__ == "__main__":
    unittest.main()
