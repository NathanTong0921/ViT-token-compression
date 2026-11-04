"""Independently reconstruct the fixed pruning-depth study from raw outputs.

This module imports no project analysis code, runs no models, and fits nothing.
It checks source identities/masks, reconstructs task-balanced outcomes, and
independently repeats the paired image bootstrap used in the saved analysis.
"""

import argparse
from collections import Counter
import csv
from itertools import permutations
import json
from pathlib import Path

import numpy as np


DATASETS = ("pets", "flowers")
DEPTHS = (3, 6, 9)
SEEDS = (2041, 2042, 2043)
PAIRS = tuple(permutations(SEEDS, 2))
METHODS = ("conditioned", "shared_recipient", "shared_donor", "cu_half", "uu_half")
METRICS = ("accuracy", "geom_margin", "raw_gap")
FIXED_CONFIG = {
    "model": "ViT-B/16", "keep_patch_tokens": 49,
    "hidden_dim": 128, "wide_hidden_dim": 199,
    "batch_size": 128, "temperature": 1.0,
    "learning_rate": 0.0003, "weight_decay": 0.01,
    "probe_count": 512, "mask_seed": 2031, "ridge": 0.01,
    "image_batch_size": 64, "teacher_batch_size": 64,
}


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


class NumericalChecks:
    def __init__(self):
        self.comparisons = 0
        self.max_absolute_error = 0.0

    def compare(self, actual, expected, name, atol=2e-10):
        actual, expected = np.asarray(actual), np.asarray(expected)
        assert actual.shape == expected.shape, name
        np.testing.assert_allclose(actual, expected, rtol=0, atol=atol, err_msg=name)
        self.comparisons += actual.size
        if actual.size:
            self.max_absolute_error = max(self.max_absolute_error,
                float(np.max(np.abs(actual.astype(float) - expected.astype(float)))))


def outcomes(scores, labels, competitors, text):
    """Read candidate scores directly; exact ties favor the lower class ID."""
    rows = np.arange(len(labels))
    a = scores[..., rows, labels]
    b = scores[..., rows, competitors]
    gap = a - b
    correct = (a > b) | ((a == b) & (labels < competitors))
    distance = np.sqrt(np.square(text[labels] - text[competitors]).sum(axis=1))
    assert np.all(distance > 0)
    return np.stack((correct.astype(float), gap / distance, gap), axis=-1)


def read_panel(folder, dataset, depth, numeric):
    protocol = read_json(folder / "protocol.json")
    config = protocol["config"]
    assert protocol["setting"] == f"{dataset}_d{depth}"
    assert config["prune_after"] == depth
    for key, value in FIXED_CONFIG.items():
        assert config[key] == value, (folder, key, config[key], value)
    assert config["training_steps"] == (3990 if dataset == "pets" else 3978)
    assert config.get("objective", "kl") == "kl"
    assert tuple(sorted(config["training_seeds"])) == SEEDS
    assert tuple(sorted(protocol["training_seeds"])) == SEEDS
    assert tuple(sorted(map(tuple, protocol["primary_seed_pairs"]))) == PAIRS
    assert protocol["final_test_used"] is False
    assert tuple(protocol["methods"]) == METHODS

    rows = read_csv(folder / "examples.csv")
    keys = [(r["image_id"], r["task_id"]) for r in rows]
    assert rows and len(keys) == len(set(keys))
    order = sorted(range(len(rows)), key=keys.__getitem__)
    rows = [rows[i] for i in order]
    ids = np.array([r["image_id"] for r in rows])
    labels = np.array([int(r["class_id"]) for r in rows])
    image_indices = np.array([int(r["image_index"]) for r in rows])
    candidates = np.array([[int(r["class_a"]), int(r["class_b"])] for r in rows])
    assert np.all(candidates[:, 0] < candidates[:, 1])
    assert np.all((candidates == labels[:, None]).any(axis=1))
    assert all(r["task_split"] == "unseen" for r in rows)
    competitors = np.array([b if y == a else a for y, (a, b) in zip(labels, candidates)])
    metadata = [(r["image_id"], r["task_id"], int(r["class_id"]),
                 int(r["class_a"]), int(r["class_b"])) for r in rows]
    task_classes = {}
    for _, task, label, a, b in metadata:
        if task not in task_classes:
            task_classes[task] = (a, b)
        assert task_classes[task] == (a, b)
    counts = Counter((r[1], r[2]) for r in metadata)
    for task, pair in task_classes.items():
        assert all(counts[task, label] > 0 for label in pair)
    weights = np.array([1 / (2 * len(task_classes) * counts[task, label])
                        for _, task, label, _, _ in metadata])
    numeric.compare(weights, [float(r["task_class_weight"]) for r in rows],
                    f"{folder}: independent task/class weights", atol=1e-12)
    numeric.compare(weights.sum(), 1.0, f"{folder}: weight sum", atol=1e-12)
    with np.load(folder / "evidence.npz") as saved:
        np.testing.assert_array_equal(saved["image_ids"][image_indices], ids)
        np.testing.assert_array_equal(saved["labels"][image_indices], labels)
        text = saved["text_features"].astype(float)
        full_scores = saved["full_scores"][image_indices].astype(float)
    assert np.isfinite(text).all() and np.isfinite(full_scores).all()

    image_groups = [np.flatnonzero(ids == image) for image in sorted(set(ids))]
    endpoint_scores, endpoint_masks, mixtures, donors = {}, {}, [], []
    mask_count = 0
    for s, t in PAIRS:
        with np.load(folder / f"pair_{s}_{t}.npz") as saved:
            assert int(saved["recipient_seed"]) == s
            assert int(saved["donor_seed"]) == t
            np.testing.assert_array_equal(saved["methods"], METHODS)
            scores = saved["cosine_scores"][:, order].astype(float)
            masks = saved["patch_indices"][:, order]
            changed = saved["changed_tokens"][:, order]
        assert scores.shape == (5, len(rows), len(text))
        assert masks.shape == (5, len(rows), 49)
        assert np.issubdtype(masks.dtype, np.integer)
        assert np.isfinite(scores).all()
        assert np.all((masks >= 0) & (masks < 196))
        assert np.all(np.diff(masks, axis=-1) > 0), "Masks must be unique and spatially sorted"
        mask_count += masks.shape[0] * masks.shape[1]
        for offset, recipient in enumerate((0, 1)):
            remaining = (masks[3 + offset, :, :, None] == masks[recipient, :, None, :]).any(-1)
            np.testing.assert_array_equal(changed[offset], 49 - remaining.sum(-1))
        for method in (1, 2, 4):
            for group in image_groups:
                np.testing.assert_array_equal(masks[method, group],
                    np.repeat(masks[method, group[:1]], len(group), axis=0))
                numeric.compare(scores[method, group],
                    np.repeat(scores[method, group[:1]], len(group), axis=0),
                    f"{folder}/{s}_{t}: image-only method {method} task invariance", atol=1e-6)
        if s in endpoint_scores:
            numeric.compare(scores[:2], endpoint_scores[s],
                            f"{folder}: repeated recipient {s}", atol=1e-6)
            np.testing.assert_array_equal(masks[:2], endpoint_masks[s])
        else:
            endpoint_scores[s], endpoint_masks[s] = scores[:2], masks[:2]
        donors.append((t, scores[2], masks[2]))
        mixtures.append(outcomes(scores[3:5], labels, competitors, text))
    for seed, scores, masks in donors:
        numeric.compare(scores, endpoint_scores[seed][1],
                        f"{folder}: donor {seed} equals shared recipient", atol=1e-6)
        np.testing.assert_array_equal(masks, endpoint_masks[seed][1])
    endpoint_values = np.array([outcomes(endpoint_scores[s], labels, competitors, text)
                               for s in SEEDS])
    return dict(metadata=metadata, image_ids=ids, weights=weights, text=text,
                full_scores=full_scores, config=config, mask_count=mask_count,
                endpoints=endpoint_values, mixtures=np.array(mixtures),
                full_reference=outcomes(full_scores, labels, competitors, text))


def independent_bootstrap(ids, weights, values, replicates, seed):
    """Cluster rows manually; jointly resample image counts for every column."""
    image_rows = [np.flatnonzero(ids == image) for image in sorted(set(ids))]
    mass = np.array([weights[r].sum() for r in image_rows])
    total = np.array([(weights[r, None] * values[r]).sum(axis=0) for r in image_rows])
    point = total.sum(axis=0) / mass.sum()
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, values.shape[1]))
    for i in range(replicates):
        sample = rng.integers(0, len(image_rows), size=len(image_rows))
        multiplicity = np.bincount(sample, minlength=len(image_rows))
        draws[i] = (multiplicity @ total) / (multiplicity @ mass)
    return point, {family: np.quantile(draws, [.025 / family, 1 - .025 / family], axis=0)
                   for family in (1, 4)}


def audit_dataset(sources, analysis, dataset, replicates, seed, numeric):
    panels = {d: read_panel(sources / f"{dataset}_d{d}", dataset, d, numeric) for d in DEPTHS}
    base = panels[6]
    for depth, panel in panels.items():
        assert panel["metadata"] == base["metadata"], (dataset, depth, "row identities")
        np.testing.assert_array_equal(panel["weights"], base["weights"])
        numeric.compare(panel["text"], base["text"], f"{dataset}/{depth}: text", atol=1e-8)
        numeric.compare(panel["full_scores"], base["full_scores"],
                        f"{dataset}/{depth}: full-model scores", atol=1e-6)
        numeric.compare(panel["full_reference"], base["full_reference"],
                        f"{dataset}/{depth}: full-model outcomes", atol=1e-6)
        for key in (*FIXED_CONFIG, "training_steps", "manifest_dir", "prompt_template"):
            assert panel["config"][key] == base["config"][key], (dataset, depth, key)
    saved = read_json(analysis / dataset / "summary.json")
    assert saved["depths"] == list(DEPTHS)
    assert saved["training_seeds"] == list(SEEDS)
    assert saved["primary_seed_pairs"] == [list(p) for p in PAIRS]
    assert saved["bootstrap_replicates"] == replicates and saved["bootstrap_seed"] == seed
    assert saved["family_size"] == 4
    assert saved["image_count"] == len(set(base["image_ids"]))
    assert saved["row_count"] == len(base["metadata"])
    assert saved["task_count"] == len({r[1] for r in base["metadata"]})

    jobs, direct, mix = [], {}, {}
    for depth in DEPTHS:
        panel = panels[depth]
        table = {row["method"]: row for row in saved["by_depth"][str(depth)]["table"]}
        for j, method in enumerate(("conditioned", "shared")):
            jobs.append((f"d{depth}/{method}", panel["endpoints"][:, j], SEEDS, table[method], 1))
        for j, method in enumerate(("cu_half", "uu_half")):
            jobs.append((f"d{depth}/{method}", panel["mixtures"][:, j], PAIRS, table[method], 1))
        direct[depth] = panel["endpoints"][:, 0] - panel["endpoints"][:, 1]
        mix[depth] = panel["mixtures"][:, 0] - panel["mixtures"][:, 1]
        for name, values, runs in (("conditioned_minus_shared", direct[depth], SEEDS),
                                   ("cu_minus_uu", mix[depth], PAIRS)):
            jobs.append((f"d{depth}/{name}", values, runs,
                         saved["by_depth"][str(depth)]["contrasts"][name], 1))
    changes = {}
    for name, values, runs in (("conditioned_minus_shared", direct, SEEDS), ("cu_minus_uu", mix, PAIRS)):
        for depth in (3, 9):
            key = f"depth_{depth}_minus_6"
            delta = values[depth] - values[6]
            changes[f"{name}_{key}"] = delta
            jobs.append((f"{name}/{key}", delta, runs,
                         saved["depth_changes"][name][key], 4))
    jobs.append(("full_reference", base["full_reference"][None], (-1,), saved["full_reference"], 1))
    mean_values = np.concatenate([values.mean(axis=0) for _, values, _, _, _ in jobs], axis=1)
    point, intervals = independent_bootstrap(base["image_ids"], base["weights"], mean_values,
                                           replicates, seed)
    checked = []
    for ji, (name, values, runs, target, family) in enumerate(jobs):
        for mi, metric in enumerate(METRICS):
            column = ji * len(METRICS) + mi
            numeric.compare(target[metric]["mean"], point[column], f"{dataset}/{name}/{metric}: point")
            numeric.compare(target[metric]["ci95"], intervals[1][:, column],
                            f"{dataset}/{name}/{metric}: interval")
            if family == 4:
                numeric.compare(target[metric]["bonferroni_family_ci95"], intervals[4][:, column],
                                f"{dataset}/{name}/{metric}: family interval")
                assert target[metric]["individual_interval_level"] == .9875
        if "per_run" in target:
            assert len(target["per_run"]) == len(runs)
            for ri, run in enumerate(runs):
                assert target["per_run"][ri]["run_id"] == np.asarray(run).tolist()
                run_point = np.sum(base["weights"][:, None] * values[ri], axis=0) / base["weights"].sum()
                numeric.compare([target["per_run"][ri][m] for m in METRICS], run_point,
                                f"{dataset}/{name}: per-run {run}")
        checked.append(dict(name=name, family_size=family,
                            mean=dict(zip(METRICS, point[ji*3:ji*3+3].tolist()))))

    with np.load(analysis / dataset / "paired_cases.npz") as arrays:
        for key, expected in (("depths", DEPTHS), ("metric_names", METRICS),
                              ("training_seeds", SEEDS), ("seed_pairs", PAIRS),
                              ("image_id", base["image_ids"]),
                              ("task_id", [r[1] for r in base["metadata"]]),
                              ("class_id", [r[2] for r in base["metadata"]]),
                              ("class_a", [r[3] for r in base["metadata"]]),
                              ("class_b", [r[4] for r in base["metadata"]])):
            np.testing.assert_array_equal(arrays[key], expected)
        numeric.compare(arrays["weights"], base["weights"], f"{dataset}: saved weights")
        expected_arrays = dict(endpoint_outcomes=np.array([panels[d]["endpoints"] for d in DEPTHS]),
            mixture_outcomes=np.array([panels[d]["mixtures"] for d in DEPTHS]),
            conditioned_minus_shared=np.array([direct[d] for d in DEPTHS]),
            cu_minus_uu=np.array([mix[d] for d in DEPTHS]),
            full_reference=base["full_reference"], **changes)
        for key, expected in expected_arrays.items():
            numeric.compare(arrays[key], expected, f"{dataset}: saved {key}")
    return dict(images=len(set(base["image_ids"])), rows=len(base["metadata"]),
                tasks=len({r[1] for r in base["metadata"]}),
                masks_checked=sum(p["mask_count"] for p in panels.values()),
                checked_estimates=checked, all_raw_and_saved_arrays_match=True)


def audit(sources, analysis, output=None, replicates=2000, seed=2361):
    sources, analysis = Path(sources), Path(analysis)
    summary = read_json(analysis / "summary.json")
    assert summary["reference_depth"] == 6 and summary["compared_depths"] == [3, 9]
    assert summary["primary_family_size"] == 4
    assert summary["bootstrap_replicates"] == replicates and summary["bootstrap_seed"] == seed
    for dataset in DATASETS:
        assert summary["datasets"][dataset] == read_json(analysis / dataset / "summary.json")
    numeric = NumericalChecks()
    result = dict(status="passed", sources=str(sources.resolve()), analysis=str(analysis.resolve()),
        scope="Independent raw-score/mask/metadata reconstruction; no analysis-code imports, model execution, training, or new inferential comparison.",
        bootstrap=dict(replicates=replicates, seed=seed,
            method="Average fixed runs per row, aggregate task/class-weighted contributions by image, then resample image multiplicities jointly for all endpoints and depths.",
            intervals="Ordinary 95%; four-comparison 98.75% intervals for each predeclared interaction family."),
        configured_selector_parameters=dict(conditioned=348033, image_only_wide=347399,
            verification_scope="Widths and fixed configuration checked; checkpoint weights are audited separately."),
        datasets={dataset: audit_dataset(sources, analysis, dataset, replicates, seed, numeric)
                  for dataset in DATASETS})
    export = read_csv(analysis / "depth_changes.csv")
    expected_keys = {(dataset, endpoint, f"depth_{depth}_minus_6", metric)
                     for dataset in DATASETS
                     for endpoint in ("conditioned_minus_shared", "cu_minus_uu")
                     for depth in (3, 9) for metric in METRICS}
    keys = [(r["dataset"], r["endpoint"], r["contrast"], r["metric"]) for r in export]
    assert len(keys) == len(set(keys)) and set(keys) == expected_keys
    for row, (dataset, endpoint, contrast, metric) in zip(export, keys):
        estimate = summary["datasets"][dataset]["depth_changes"][endpoint][contrast][metric]
        numeric.compare([float(row[k]) for k in
                         ("mean", "ci95_low", "ci95_high", "family_ci95_low", "family_ci95_high")],
                        [estimate["mean"], *estimate["ci95"], *estimate["bonferroni_family_ci95"]],
                        f"exported depth_changes.csv: {dataset}/{endpoint}/{contrast}/{metric}")
    result.update(numeric_values_compared=numeric.comparisons,
                  max_absolute_numeric_difference=numeric.max_absolute_error,
                  exported_comparison_rows_checked=len(export))
    destination = Path(output) if output else analysis / "independent_check.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.sources, args.analysis, args.output)
    print(json.dumps({key: result[key] for key in
                      ("status", "numeric_values_compared", "max_absolute_numeric_difference")}, indent=2))
