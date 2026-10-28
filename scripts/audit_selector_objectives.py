"""Independent CPU audit of the matched KL/ranking selector experiment.

Raw prediction CSVs and NPZ class scores are the inputs. No project analysis
helper is imported. The audit recomputes weighted effects and image-bootstrap
intervals, and checks the recorded training controls for full coverage only.
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np


SEEDS = (2041, 2042, 2043)
ARMS = ("conditioned", "image_only")
OBJECTIVES = ("kl", "rank")
METRICS = ("accuracy", "margin")
CONFIG_KEYS = ("model", "prune_after", "keep_patch_tokens", "prompt_template",
               "manifest_dir", "training_seeds", "hidden_dim", "training_steps",
               "batch_size", "temperature", "learning_rate", "weight_decay",
               "cache_dir", "probe_count", "mask_seed", "ridge")


def close(actual, expected, atol=2e-10):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_predictions(source, budget):
    """Check the CSV against independent score-based outcomes in the raw NPZ."""
    rows = [r for r in read_csv(source / "full" / "predictions.csv")
            if r["task_split"] == "unseen" and r["method"] in ARMS]
    keys = [(r["image_id"], r["task_id"], r["method"], int(r["seed"])) for r in rows]
    assert rows and len(keys) == len(set(keys)), "Duplicate or empty prediction rows"
    with np.load(source / "full" / "predictions.npz") as raw:
        take = (raw["task_split"] == "unseen") & np.isin(raw["method"], ARMS)
        arrays = {k: raw[k][take] for k in ("image_ids", "task_ids", "method", "seed",
                  "labels", "class_pairs", "cosine_scores", "correct", "margin", "keep_patch_tokens")}
        text = raw["text_features"].astype(float)
    npz_keys = list(zip(arrays["image_ids"], arrays["task_ids"], arrays["method"], arrays["seed"]))
    assert len(npz_keys) == len(set(npz_keys)) and set(npz_keys) == set(keys)
    indices = {k: i for i, k in enumerate(npz_keys)}
    order = np.asarray([indices[k] for k in keys])
    arrays = {k: v[order] for k, v in arrays.items()}
    labels, pairs = arrays["labels"], arrays["class_pairs"]
    assert np.all((pairs == labels[:, None]).any(axis=1))
    assert np.all(arrays["keep_patch_tokens"] == budget)
    np.testing.assert_array_equal(labels, [int(r["class_id"]) for r in rows])
    np.testing.assert_array_equal(pairs, [[int(r["class_a"]), int(r["class_b"])] for r in rows])
    score_rows = np.arange(len(rows))
    scores = arrays["cosine_scores"].astype(float)
    predicted = pairs[score_rows, scores[score_rows[:, None], pairs].argmax(axis=1)]
    correct = predicted == labels
    competitor = np.where(labels == pairs[:, 0], pairs[:, 1], pairs[:, 0])
    margin = (scores[score_rows, labels] - scores[score_rows, competitor]) / np.linalg.norm(
        text[labels] - text[competitor], axis=1)
    assert np.isfinite(margin).all()
    np.testing.assert_array_equal(correct, arrays["correct"])
    np.testing.assert_array_equal(correct, [int(r["correct"]) for r in rows])
    close(margin, arrays["margin"], atol=2e-7)
    close(arrays["margin"], [float(r["margin"]) for r in rows])
    return rows


def align(rows_by_objective, budget):
    """Identity-align every arm/seed and independently construct task weights."""
    grouped, reference = {}, None
    for objective in OBJECTIVES:
        rows = rows_by_objective[objective]
        assert {int(r["seed"]) for r in rows} == set(SEEDS)
        for arm in ARMS:
            pieces = []
            for seed in SEEDS:
                rows_for_run = sorted([r for r in rows if r["method"] == arm and int(r["seed"]) == seed],
                                      key=lambda r: (r["image_id"], r["task_id"]))
                metadata = [(r["image_id"], r["task_id"], int(r["class_id"]),
                             int(r["class_a"]), int(r["class_b"])) for r in rows_for_run]
                assert metadata and len({m[:2] for m in metadata}) == len(metadata)
                assert all(int(r["keep_patch_tokens"]) == budget for r in rows_for_run)
                if reference is None:
                    reference = metadata
                else:
                    assert metadata == reference, "Prediction identities/labels changed across arms or seeds"
                pieces.append([[int(r["correct"]), float(r["margin"])] for r in rows_for_run])
            grouped[f"{objective}_{arm}"] = np.asarray(pieces, float)
    count = Counter((row[1], row[2]) for row in reference)
    tasks = {row[1] for row in reference}
    for task in tasks:
        labels = {row[2] for row in reference if row[1] == task}
        pair = {row[3:] for row in reference if row[1] == task}
        assert len(pair) == 1 and labels == set(next(iter(pair))) and len(labels) == 2
    weights = np.asarray([1 / (2 * len(tasks) * count[row[1], row[2]]) for row in reference])
    close(weights.sum(), 1)
    assert all(np.isfinite(v).all() and np.isin(v[:, :, 0], [0, 1]).all() for v in grouped.values())
    return reference, grouped, weights


def effects(groups):
    values = dict(groups)
    for objective in OBJECTIVES:
        values[f"{objective}_conditioned_minus_image_only"] = (
            groups[f"{objective}_conditioned"] - groups[f"{objective}_image_only"])
    for arm in ARMS:
        values[f"{arm}_rank_minus_kl"] = groups[f"rank_{arm}"] - groups[f"kl_{arm}"]
    values["interaction"] = (values["rank_conditioned_minus_image_only"]
                             - values["kl_conditioned_minus_image_only"])
    return values


def bootstrap(ids, weights, values, replicates, seed):
    """Draw whole images with fixed row weights and a resampled denominator."""
    images = sorted(set(ids))
    index = {image: i for i, image in enumerate(images)}
    numerator = np.zeros((len(images), values.shape[1]))
    denominator = np.zeros(len(images))
    for image, weight, value in zip(ids, weights, values):
        numerator[index[image]] += weight * value
        denominator[index[image]] += weight
    draws = np.random.default_rng(seed)
    repeated = np.empty((replicates, values.shape[1]))
    for i in range(replicates):
        sampled = draws.integers(len(images), size=len(images))
        repeated[i] = numerator[sampled].sum(axis=0) / denominator[sampled].sum()
    return numerator.sum(axis=0) / denominator.sum(), np.quantile(repeated, [.025, .975], axis=0), np.quantile(
        repeated, [.0125, .9875], axis=0)


def audit_training(metadata, budget):
    configs = {o: metadata[o]["config"] for o in OBJECTIVES}
    for key in CONFIG_KEYS:
        assert configs["kl"][key] == configs["rank"][key], f"Training control changed: {key}"
    assert configs["kl"]["keep_patch_tokens"] == budget
    assert configs["kl"].get("objective", "kl") == "kl"
    assert configs["rank"]["objective"] == "topk_rank"
    records, samples, paths = {}, {}, {}
    for objective in OBJECTIVES:
        folder = Path(metadata[objective]["models_dir"])
        summary = read_json(folder / "training_summary.json")
        for key in CONFIG_KEYS:
            assert summary["config"][key] == configs[objective][key]
        full = [r for r in summary["runs"] if r["coverage"] == "full"]
        records[objective] = {(r["seed"], r["arm"]): r for r in full}
        assert len(full) == len(records[objective]) == 6
        assert set(records[objective]) == {(s, a) for s in SEEDS for a in ARMS}
        for row in full:
            assert row["training_steps"] == configs[objective]["training_steps"]
            assert row["training_draws"] == row["training_steps"] * configs[objective]["batch_size"]
            assert row["initial_state_exactly_matches_shared_seed_state"]
            assert row["image_draws_reused_between_arms"] and row["clip_parameters_in_optimizer"] == 0
        sampling_path = folder / summary["sampling_audit"]
        sample_rows = read_json(sampling_path)
        samples[objective] = {r["seed"]: {"total_draws": r["total_draws"],
            "task_side_count_min": r["task_side_count_min"], "task_side_count_max": r["task_side_count_max"],
            "full": r["coverages"]["full"]} for r in sample_rows}
        assert len(sample_rows) == len(samples[objective]) == 3 and set(samples[objective]) == set(SEEDS)
        paths[objective] = {"training_summary": str(folder / "training_summary.json"),
                            "sampling_audit": str(sampling_path)}
    assert samples["kl"] == samples["rank"], "Recorded full-coverage sampling changed"
    for key in records["kl"]:
        old, new = records["kl"][key], records["rank"][key]
        for field in ("parameter_count", "training_steps", "training_draws", "available_training_images"):
            assert old[field] == new[field], field
        close([old["initial_train_kl"], old["initial_val_kl"]],
              [new["initial_train_kl"], new["initial_val_kl"]], atol=1e-6)
    return {"sources": paths, "same_teacher_cache": configs["kl"]["cache_dir"],
            "matched_full_coverage_runs_per_objective": 6,
            "matched_controls": list(CONFIG_KEYS), "initial_common_kl_agrees": True,
            "recorded_full_sampling_matches": True,
            "scope": "Checks recorded within-arm initialization assertions and cross-objective initial KL, first batch and per-image draw counts. Historical initial tensors and full ordered sample streams were not saved, so this is not a new tensor/step-by-step replay. Old small-coverage runs are excluded."}


def audit(summary_path, output):
    summary_path, output = Path(summary_path), Path(output)
    summary = read_json(summary_path)
    assert summary["budgets"] == [49, 98] and summary["training_seeds"] == list(SEEDS)
    assert summary["bootstrap_replicates"] == 2000 and summary["bootstrap_seed"] == 2085
    result = {"status": "passed", "dataset": summary["dataset"], "analysis_source": str(summary_path),
              "bootstrap_replicates": 2000, "bootstrap_seed": 2085,
              "scope": "Independent raw CSV/NPZ classification reconstruction, strict paired identities, weighted main/secondary effects, per-seed effects, image-bootstrap intervals and recorded training controls. Reference predictions are not read. Common teacher diagnostic metrics are not independently recomputed by this audit.",
              "budgets": {}}
    previous = None
    for budget in summary["budgets"]:
        sources = {o: Path(summary["sources"][o][str(budget)]) for o in OBJECTIVES}
        metadata = {o: read_json(sources[o] / "eval_summary.json") for o in OBJECTIVES}
        training = audit_training(metadata, budget)
        rows = {o: load_predictions(sources[o], budget) for o in OBJECTIVES}
        reference, grouped, weights = align(rows, budget)
        if previous is not None:
            assert reference == previous, "Budgets use different image/task panels"
        previous = reference
        ids = np.asarray([r[0] for r in reference])
        contrasts = effects(grouped)
        names = list(contrasts)
        values = np.concatenate([contrasts[name].mean(axis=0) for name in names], axis=1)
        point, ci95, ci975 = bootstrap(ids, weights, values, 2000, 2085)
        saved = summary["results"][str(budget)]
        assert saved["image_count"] == len(set(ids)) and saved["image_task_examples"] == len(ids)
        assert saved["task_count"] == len({r[1] for r in reference})
        for i, name in enumerate(names):
            if name in grouped:
                objective, arm = name.split("_", 1)
                target = next(r for r in saved["table"] if r["objective"] == objective and r["arm"] == arm)
            else:
                target = saved["primary_interaction"] if name == "interaction" else saved["secondary_contrasts"][name]
            for j, metric in enumerate(METRICS):
                close(point[2 * i + j], target[metric]["mean"])
                close(ci95[:, 2 * i + j], target[metric]["ci95"])
                if name == "interaction":
                    close(ci975[:, 2 * i + j], target[metric]["bonferroni_family_ci95"])
                    assert target[metric]["individual_interval_level"] == .975
            for si, seed in enumerate(SEEDS):
                per_seed = (next(r for r in target["per_seed"] if r["seed"] == seed) if name in grouped else
                            next(r for r in saved["per_seed"] if r["seed"] == seed)["contrasts"][name])
                close(weights @ contrasts[name][si], [per_seed[m] for m in METRICS])
        folder = summary_path.parent / f"budget_{budget}"
        with np.load(folder / "paired_cases.npz") as packed:
            for column, column_index in (("image_id", 0), ("task_id", 1), ("class_id", 2), ("class_a", 3), ("class_b", 4)):
                np.testing.assert_array_equal(packed[column], [r[column_index] for r in reference])
            close(packed["task_class_weight"], weights)
            np.testing.assert_array_equal(packed["training_seeds"], SEEDS)
            for name, data in contrasts.items():
                if name in grouped:
                    close(packed[f"{name}_per_seed"], data)
                for j, metric in enumerate(METRICS):
                    close(packed[f"{name}_{metric}"], data.mean(axis=0)[:, j])
        cases = read_csv(folder / "paired_cases.csv")
        assert [(r["image_id"], r["task_id"]) for r in cases] == [r[:2] for r in reference]
        close([float(r["task_class_weight"]) for r in cases], weights)
        for name, data in contrasts.items():
            for j, metric in enumerate(METRICS):
                close([float(r[f"{name}_{metric}"]) for r in cases], data.mean(axis=0)[:, j])
        result["budgets"][str(budget)] = {"image_count": len(set(ids)), "image_task_rows_per_arm_seed": len(ids),
            "raw_unseen_rows_verified_per_objective": len(ids) * 6,
            "primary_interaction": saved["primary_interaction"], "training_controls": training,
            "classification_points_intervals_and_per_seed": "passed", "paired_case_csv_and_npz": "passed"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    outcome = audit(args.summary, args.output)
    print(json.dumps({"status": outcome["status"], "dataset": outcome["dataset"],
                      "budgets": list(outcome["budgets"])}))
