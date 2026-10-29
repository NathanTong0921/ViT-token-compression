"""Reconstruct ablation results from raw predictions and training records.

This audit imports no analysis code. Class scores independently verify stored
outcomes; paired effects, bootstrap intervals and exchanges are then rebuilt.
"""

import argparse
from collections import Counter, defaultdict
import csv
from itertools import combinations
import json
from pathlib import Path

import numpy as np

SEEDS = (2041, 2042, 2043)
METHODS = ("conditioned", "image_only", "image_only_wide", "mean_only", "difference_only")
SOURCE_METHODS = {"baseline": METHODS[:2], "ablations": METHODS[2:]}
METRICS = ("accuracy", "margin")
FAMILIES = {"conditioned_minus_image_only_wide": 2,
            "conditioned_minus_mean_only": 4, "conditioned_minus_difference_only": 4}
CONFIG_KEYS = ("model", "prune_after", "keep_patch_tokens", "prompt_template", "manifest_dir",
               "training_seeds", "hidden_dim", "training_steps", "batch_size", "temperature",
               "learning_rate", "weight_decay", "cache_dir", "probe_count", "mask_seed", "ridge")


def close(actual, expected, atol=2e-10):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def score_outcomes(scores, labels, competitors, text):
    rows = np.arange(len(labels))
    correct_scores, competitor_scores = scores[rows, labels], scores[rows, competitors]
    correct = (correct_scores > competitor_scores) | ((correct_scores == competitor_scores) & (labels < competitors))
    margin = (correct_scores - competitor_scores) / np.linalg.norm(text[labels] - text[competitors], axis=1)
    return np.column_stack((correct, margin))


def read_raw(source, methods, budget):
    rows = [r for r in read_csv(source / "full/predictions.csv")
            if r["task_split"] == "unseen" and r["method"] in methods]
    keys = [(r["image_id"], r["task_id"], r["method"], int(r["seed"])) for r in rows]
    assert rows and len(keys) == len(set(keys))
    with np.load(source / "full/predictions.npz") as f:
        take = (f["task_split"] == "unseen") & np.isin(f["method"], methods)
        a = {k: f[k][take] for k in ("row_id", "image_ids", "task_ids", "method", "seed", "labels",
              "class_pairs", "cosine_scores", "correct", "margin", "keep_patch_tokens", "patch_indices")}
        text = f["text_features"].astype(float)
    raw_keys = list(zip(a["image_ids"], a["task_ids"], a["method"], a["seed"]))
    assert len(raw_keys) == len(set(raw_keys)) and set(raw_keys) == set(keys)
    lookup = {key: i for i, key in enumerate(raw_keys)}
    order = np.array([lookup[key] for key in keys])
    a = {k: value[order] for k, value in a.items()}
    labels, pairs = a["labels"], a["class_pairs"]
    np.testing.assert_array_equal(labels, [int(r["class_id"]) for r in rows])
    np.testing.assert_array_equal(pairs, [[int(r["class_a"]), int(r["class_b"])] for r in rows])
    assert np.all((pairs == labels[:, None]).any(axis=1)) and np.all(pairs[:, 0] < pairs[:, 1])
    assert np.all(a["keep_patch_tokens"] == budget)
    assert np.all(a["patch_indices"][:, :budget] >= 0) and np.all(a["patch_indices"][:, budget:] == -1)
    assert np.all(np.diff(a["patch_indices"][:, :budget], axis=1) > 0)
    competitors = np.where(labels == pairs[:, 0], pairs[:, 1], pairs[:, 0])
    reconstructed = score_outcomes(a["cosine_scores"].astype(float), labels, competitors, text)
    np.testing.assert_array_equal(reconstructed[:, 0], a["correct"])
    close(reconstructed[:, 1], a["margin"], atol=2e-7)
    np.testing.assert_array_equal(a["correct"], [int(r["correct"]) for r in rows])
    close(a["margin"], [float(r["margin"]) for r in rows])
    return rows, a, text


def aligned_sources(sources, budget):
    reference, text, outcomes, raw = None, None, {}, {}
    for source, methods in SOURCE_METHODS.items():
        rows, arrays, current_text = read_raw(sources[source], methods, budget)
        if text is None:
            text = current_text
        else:
            np.testing.assert_array_equal(text, current_text)
        for method in methods:
            pieces, scores, positions, source_rows = [], [], [], []
            assert {int(r["seed"]) for r in rows if r["method"] == method} == set(SEEDS)
            for seed in SEEDS:
                take = sorted([i for i, r in enumerate(rows) if r["method"] == method and int(r["seed"]) == seed],
                              key=lambda i: (rows[i]["image_id"], rows[i]["task_id"]))
                metadata = [(rows[i]["image_id"], rows[i]["task_id"], int(rows[i]["class_id"]),
                             int(rows[i]["class_a"]), int(rows[i]["class_b"])) for i in take]
                assert metadata and len(metadata) == len({m[:2] for m in metadata})
                if reference is None:
                    reference = metadata
                else:
                    assert reference == metadata, "Unpaired identities across methods or seeds"
                pieces.append([[int(rows[i]["correct"]), float(rows[i]["margin"])] for i in take])
                scores.append(arrays["cosine_scores"][take].astype(float))
                positions.append(arrays["patch_indices"][take])
                source_rows.append(arrays["row_id"][take])
            outcomes[method] = np.asarray(pieces)
            raw[method] = {"scores": np.asarray(scores), "positions": np.asarray(positions), "rows": np.asarray(source_rows)}
    counts = Counter((r[1], r[2]) for r in reference)
    tasks = {r[1] for r in reference}
    for task in tasks:
        rows = [r for r in reference if r[1] == task]
        assert len({r[3:] for r in rows}) == 1 and {r[2] for r in rows} == set(rows[0][3:])
    weights = np.array([1 / (2 * len(tasks) * counts[r[1], r[2]]) for r in reference])
    close(weights.sum(), 1)
    return reference, outcomes, raw, text, weights


def contrasts(outcomes):
    d = {"conditioned_minus_" + m: outcomes["conditioned"] - outcomes[m] for m in METHODS[1:]}
    d["image_only_wide_minus_image_only"] = outcomes["image_only_wide"] - outcomes["image_only"]
    d["mean_only_minus_difference_only"] = outcomes["mean_only"] - outcomes["difference_only"]
    return d


def bootstrap(ids, weights, values, seed, replicates=2000):
    images, inverse = np.unique(ids, return_inverse=True)
    numerator = np.zeros((len(images), values.shape[1]))
    denominator = np.zeros(len(images))
    np.add.at(numerator, inverse, weights[:, None] * values)
    np.add.at(denominator, inverse, weights)
    draws = np.random.default_rng(seed)
    sample = np.empty((replicates, values.shape[1]))
    for i in range(replicates):
        selected = draws.integers(len(images), size=len(images))
        sample[i] = numerator[selected].sum(axis=0) / denominator[selected].sum()
    intervals = {family: np.quantile(sample, [.025 / family, 1 - .025 / family], axis=0) for family in (1, 2, 4)}
    return numerator.sum(axis=0) / denominator.sum(), intervals


def verify_estimate(target, point, intervals, offset=0, family=1):
    for j, metric in enumerate(METRICS):
        close(target[metric]["mean"], point[offset + j])
        close(target[metric]["ci95"], intervals[1][:, offset + j])
        if family > 1:
            close(target[metric]["bonferroni_family_ci95"], intervals[family][:, offset + j])
            close(target[metric]["individual_interval_level"], 1 - .05 / family)


def verify_training(sources, budget):
    metadata = {s: read_json(p / "eval_summary.json") for s, p in sources.items()}
    configs = {s: d["config"] for s, d in metadata.items()}
    for key in CONFIG_KEYS:
        assert configs["baseline"][key] == configs["ablations"][key], key
    assert configs["baseline"]["keep_patch_tokens"] == budget
    assert all(c.get("objective", "kl") == "kl" for c in configs.values())
    assert set(configs["ablations"]["selector_arms"]) == set(METHODS[2:])
    assert configs["ablations"]["wide_hidden_dim"] == 199
    records, sampling = {}, {}
    for source, methods in SOURCE_METHODS.items():
        folder = Path(metadata[source]["models_dir"])
        saved = read_json(folder / "training_summary.json")
        for key in CONFIG_KEYS:
            assert saved["config"][key] == configs[source][key]
        full = [r for r in saved["runs"] if r["coverage"] == "full"]
        records[source] = {(r["seed"], r["arm"]): r for r in full}
        assert len(full) == len(records[source]) == 3 * len(methods)
        assert set(records[source]) == {(s, m) for s in SEEDS for m in methods}
        for r in full:
            assert r["training_steps"] == configs[source]["training_steps"]
            assert r["training_draws"] == r["training_steps"] * configs[source]["batch_size"]
            assert r["image_draws_reused_between_arms"] and r["clip_parameters_in_optimizer"] == 0
            assert r["parameter_count"] == (347399 if r["arm"] == "image_only_wide" else 348033)
            assert r["initial_state_exactly_matches_shared_seed_state"] == (r["arm"] != "image_only_wide")
            if source == "ablations":
                assert r["initial_state_exactly_matches_own_architecture_seed_state"]
            assert np.isfinite([r[k] for k in ("initial_train_kl", "initial_val_kl", "final_train_kl", "final_val_kl")]).all()
        sample = read_json(folder / saved["sampling_audit"])
        sampling[source] = {r["seed"]: {"total_draws": r["total_draws"],
            "task_side_count_min": r["task_side_count_min"], "task_side_count_max": r["task_side_count_max"],
            "full": r["coverages"]["full"]} for r in sample}
        assert len(sample) == len(sampling[source]) == 3 and set(sampling[source]) == set(SEEDS)
    assert sampling["baseline"] == sampling["ablations"], "Training sampling changed"
    for seed in SEEDS:
        count = records["baseline"][seed, "conditioned"]["available_training_images"]
        assert all(records["ablations"][seed, m]["available_training_images"] == count for m in METHODS[2:])
    return {"matched_controls": list(CONFIG_KEYS), "recorded_sampling_matches": True,
            "parameters": {m: 347399 if m == "image_only_wide" else 348033 for m in METHODS},
            "initialization": "Descriptor arms match original seed template; wider image-only matches its own seeded architecture. Checked saved assertions and sampling summaries, not historical initial tensors."}


def verify_exchange(reference, raw, text, budget, saved, folder):
    groups = defaultdict(list)
    for i, r in enumerate(reference):
        groups[r[0]].append(i)
    pairs = [pair for image in sorted(groups) for pair in combinations(groups[image], 2)]
    a, b = np.asarray(pairs).T
    labels = np.array([r[2] for r in reference])
    competitors = np.array([r[4] if r[2] == r[3] else r[3] for r in reference])
    ids = np.array([reference[i][0] for i in a])
    counts = Counter(ids)
    weights = np.array([1 / (len(counts) * counts[i]) for i in ids])
    assert saved["image_count"] == len(counts) and saved["image_task_pairs"] == len(pairs)
    effects = {}
    with np.load(folder / "exchange_cases.npz") as archive:
        np.testing.assert_array_equal(archive["image_id"], ids)
        close(archive["image_pair_weight"], weights)
        for method in METHODS:
            own, swapped, overlap = [], [], []
            for si, seed in enumerate(SEEDS):
                scores, pos = raw[method]["scores"][si], raw[method]["positions"][si]
                own.append((score_outcomes(scores[a], labels[a], competitors[a], text) +
                            score_outcomes(scores[b], labels[b], competitors[b], text)) / 2)
                swapped.append((score_outcomes(scores[b], labels[a], competitors[a], text) +
                                score_outcomes(scores[a], labels[b], competitors[b], text)) / 2)
                overlap.append([len(set(pos[i, :budget]) & set(pos[j, :budget])) / budget for i, j in pairs])
                if method in ("image_only", "image_only_wide"):
                    np.testing.assert_array_equal(scores[a], scores[b])
                    np.testing.assert_array_equal(pos[a], pos[b])
            own, swapped, overlap = np.asarray(own), np.asarray(swapped), np.asarray(overlap)
            effects[method] = own - swapped
            record = next(r for r in saved["table"] if r["method"] == method)
            for name, values in (("own", own), ("swapped", swapped), ("own_minus_swapped", effects[method])):
                point, intervals = bootstrap(ids, weights, values.mean(axis=0), 2092)
                verify_estimate(record[name], point, intervals)
                close(archive[f"{method}_{name}_per_seed"], values)
            close(archive[f"{method}_overlap_per_seed"], overlap)
            close(record["token_overlap"]["mean"], weights @ overlap.mean(axis=0))
            for si, seed in enumerate(SEEDS):
                per_seed = next(r for r in record["per_seed"] if r["seed"] == seed)
                close([per_seed[m] for m in METRICS], weights @ effects[method][si])
    for method in ("mean_only", "difference_only"):
        point, intervals = bootstrap(ids, weights, (effects["conditioned"] - effects[method]).mean(axis=0), 2092)
        verify_estimate(saved["secondary_effect_contrasts"][f"conditioned_minus_{method}"], point, intervals)
    return {"image_count": len(counts), "task_pairs": len(pairs), "score_reconstruction": "passed",
            "own_exchanged_points_intervals_per_seed": "passed", "both_image_only_variants_invariant": True}


def audit(summary_path, output):
    summary_path, output = Path(summary_path), Path(output)
    summary = read_json(summary_path)
    assert summary["budgets"] == [49, 98] and summary["training_seeds"] == list(SEEDS)
    assert summary["bootstrap_seed"] == 2091 and summary["bootstrap_replicates"] == 2000
    assert summary["exchange_bootstrap_seed"] == 2092
    result = {"status": "passed", "dataset": summary["dataset"], "analysis_source": str(summary_path),
              "scope": "Independent raw score reconstruction, identities, weighted effects and paired image bootstrap; all exchange effects and training records. No model inference or new test images.", "budgets": {}}
    previous = None
    for budget in summary["budgets"]:
        sources = {s: Path(p) for s, p in summary["sources"][str(budget)].items()}
        controls = verify_training(sources, budget)
        reference, outcomes, raw, text, weights = aligned_sources(sources, budget)
        if previous is not None:
            assert reference == previous
        previous = reference
        saved = summary["results"][str(budget)]
        ids = np.array([r[0] for r in reference])
        assert saved["image_count"] == len(set(ids)) and saved["image_task_examples"] == len(ids)
        assert saved["task_count"] == len({r[1] for r in reference})
        diffs = contrasts(outcomes)
        values = {**outcomes, **diffs}
        names = list(values)
        point, intervals = bootstrap(ids, weights, np.concatenate([v.mean(axis=0) for v in values.values()], axis=1), 2091)
        folder = summary_path.parent / f"budget_{budget}"
        with np.load(folder / "paired_cases.npz") as archive:
            np.testing.assert_array_equal(archive["image_id"], ids)
            np.testing.assert_array_equal(archive["task_id"], [r[1] for r in reference])
            close(archive["task_class_weight"], weights)
            for i, name in enumerate(names):
                if name in outcomes:
                    target = next(r for r in saved["table"] if r["method"] == name)
                    close(archive[f"{name}_per_seed"], values[name])
                else:
                    target = saved["primary_contrasts" if name in FAMILIES else "secondary_contrasts"][name]
                verify_estimate(target, point, intervals, i * 2, FAMILIES.get(name, 1))
                for j, metric in enumerate(METRICS):
                    close(archive[f"{name}_{metric}"], values[name].mean(axis=0)[:, j])
                for si, seed in enumerate(SEEDS):
                    row = (next(r for r in target["per_seed"] if r["seed"] == seed) if name in outcomes else
                           next(r for r in saved["per_seed"] if r["seed"] == seed)["contrasts"][name])
                    close([row[m] for m in METRICS], weights @ values[name][si])
        csv_rows = read_csv(folder / "paired_cases.csv")
        assert [(r["image_id"], r["task_id"]) for r in csv_rows] == [r[:2] for r in reference]
        close([float(r["task_class_weight"]) for r in csv_rows], weights)
        for name, value in values.items():
            for j, metric in enumerate(METRICS):
                close([float(r[f"{name}_{metric}"]) for r in csv_rows], value.mean(axis=0)[:, j])
        exchange = verify_exchange(reference, raw, text, budget, saved["exchange"], folder)
        result["budgets"][str(budget)] = {"image_count": len(set(ids)), "task_classification_rows": len(ids),
            "primary_contrasts": saved["primary_contrasts"], "classification_points_intervals_per_seed": "passed",
            "paired_case_csv_npz": "passed", "training_controls": controls, "exchange": exchange}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.summary, args.output)
    print(json.dumps({"status": result["status"], "dataset": result["dataset"], "budgets": list(result["budgets"])}))
