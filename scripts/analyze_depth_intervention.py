"""Paired changes in conditioning benefits when pruning after blocks 3, 6 or 9."""

import argparse
from itertools import permutations
import json
from pathlib import Path

import numpy as np

from analyze_selector_budgets import weighted_estimates
from analyze_selector_scale import panel_weights, read_rows
from evaluate_selectors import write_rows


DEPTHS = (3, 6, 9)
DATASETS = ("pets", "flowers")
METHODS = ("conditioned", "shared_recipient", "shared_donor", "cu_half", "uu_half")
METRICS = ("accuracy", "geom_margin", "raw_gap")
ENDPOINT_METHODS = ("conditioned", "shared")
MIXTURE_METHODS = ("cu_half", "uu_half")


def binary_outcomes(scores, labels, competitors, text):
    """Score before averaging runs; classification ties use the lower class ID."""
    scores, text = np.asarray(scores, float), np.asarray(text, float)
    rows = np.arange(len(labels))
    own, other = scores[..., rows, labels], scores[..., rows, competitors]
    raw = own - other
    distance = np.linalg.norm(text[labels] - text[competitors], axis=1)
    correct = (own > other) | ((own == other) & (labels < competitors))
    return np.stack((correct.astype(float), raw / distance, raw), axis=-1)


def read_depth(source):
    """Read an evaluator panel and canonicalize image/task and run identities."""
    source = Path(source)
    protocol = json.loads((source / "protocol.json").read_text())
    rows = read_rows(source / "examples.csv")
    keys = [(row["image_id"], row["task_id"]) for row in rows]
    assert len(keys) == len(set(keys))
    order = np.asarray(sorted(range(len(keys)), key=keys.__getitem__))
    rows = [rows[i] for i in order]
    image_ids = np.asarray([row["image_id"] for row in rows])
    task_ids = np.asarray([row["task_id"] for row in rows])
    image_indices = np.asarray([int(row["image_index"]) for row in rows])
    labels = np.asarray([int(row["class_id"]) for row in rows])
    candidates = np.asarray([[int(row["class_a"]), int(row["class_b"])] for row in rows])
    competitors = np.where(labels == candidates[:, 0], candidates[:, 1], candidates[:, 0])
    weights = panel_weights(rows, "all")
    np.testing.assert_allclose(weights, [float(row["task_class_weight"]) for row in rows],
                               atol=1e-12, rtol=0)
    with np.load(source / "evidence.npz") as data:
        text = data["text_features"].astype(float)
        full = data["full_scores"][image_indices].astype(float)
        np.testing.assert_array_equal(data["image_ids"][image_indices], image_ids)
        np.testing.assert_array_equal(data["labels"][image_indices], labels)
    seeds = sorted(protocol["training_seeds"])
    pairs = sorted(map(tuple, protocol["primary_seed_pairs"]))
    assert pairs == sorted(permutations(seeds, 2))
    endpoints, mixtures, donor_endpoints = {}, [], []
    for s, t in pairs:
        with np.load(source / f"pair_{s}_{t}.npz") as data:
            np.testing.assert_array_equal(data["methods"][:5], METHODS)
            scores = data["cosine_scores"][:5, order].astype(float)
        if s in endpoints:
            np.testing.assert_allclose(scores[:2], endpoints[s], atol=1e-6, rtol=0)
        else:
            endpoints[s] = scores[:2]
        donor_endpoints.append((t, scores[2]))
        mixtures.append(scores[3:5])
    for donor, scores in donor_endpoints:
        np.testing.assert_allclose(scores, endpoints[donor][1], atol=1e-6, rtol=0)
    return dict(source=str(source.resolve()), protocol=protocol, image_id=image_ids,
                task_id=task_ids, class_id=labels, class_a=candidates[:, 0],
                class_b=candidates[:, 1], weights=weights, text_features=text,
                training_seeds=np.asarray(seeds), seed_pairs=np.asarray(pairs),
                endpoints=binary_outcomes(np.stack([endpoints[s] for s in seeds]),
                                          labels, competitors, text),
                mixtures=binary_outcomes(np.stack(mixtures), labels, competitors, text),
                full_reference=binary_outcomes(full, labels, competitors, text))


def summarize_runs(ids, weights, values, run_ids, replicates, bootstrap_seed,
                   family_size=1):
    """Average fixed runs within rows, then resample whole paired images."""
    result = weighted_estimates(ids, weights, values.mean(axis=0), METRICS,
                                replicates, bootstrap_seed, family_size)
    points = np.einsum("n,rnm->rm", weights, values) / weights.sum()
    result["per_run"] = [dict(run_id=np.asarray(run).tolist(),
                              **dict(zip(METRICS, point.tolist())))
                         for run, point in zip(run_ids, points)]
    return result


def analyze_dataset(sources, output, dataset, replicates=2000, bootstrap_seed=2361):
    sources, output = Path(sources), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    panels = [read_depth(sources / f"{dataset}_d{depth}") for depth in DEPTHS]
    base = panels[1]
    for depth, panel in zip(DEPTHS, panels):
        assert int(panel["protocol"]["config"]["prune_after"]) == depth
        for key in ("image_id", "task_id", "class_id", "class_a", "class_b",
                    "training_seeds", "seed_pairs", "weights"):
            np.testing.assert_array_equal(panel[key], base[key])
        np.testing.assert_allclose(panel["text_features"], base["text_features"], atol=1e-8, rtol=0)
        np.testing.assert_allclose(panel["full_reference"], base["full_reference"], atol=1e-6, rtol=0)
        for key in ("model", "keep_patch_tokens"):
            assert panel["protocol"]["config"][key] == base["protocol"]["config"][key]
    endpoints = np.stack([panel["endpoints"] for panel in panels])
    mixtures = np.stack([panel["mixtures"] for panel in panels])
    direct = endpoints[:, :, 0] - endpoints[:, :, 1]
    mix = mixtures[:, :, 0] - mixtures[:, :, 1]
    ids, weights = base["image_id"], base["weights"]
    seeds, pairs = base["training_seeds"], base["seed_pairs"]
    by_depth, image_values = {}, {}
    for i, depth in enumerate(DEPTHS):
        table = []
        for values, names, runs in ((endpoints[i], ENDPOINT_METHODS, seeds),
                                    (mixtures[i], MIXTURE_METHODS, pairs)):
            for j, name in enumerate(names):
                table.append(dict(method=name, **summarize_runs(
                    ids, weights, values[:, j], runs, replicates, bootstrap_seed)))
                image_values[f"d{depth}_{name}"] = values[:, j].mean(axis=0)
        contrasts = {}
        for name, values, runs in (("conditioned_minus_shared", direct[i], seeds),
                                   ("cu_minus_uu", mix[i], pairs)):
            contrasts[name] = summarize_runs(ids, weights, values, runs, replicates, bootstrap_seed)
            image_values[f"d{depth}_{name}"] = values.mean(axis=0)
        by_depth[str(depth)] = dict(table=table, contrasts=contrasts)
    changes, change_arrays = {}, {}
    for name, values, runs in (("conditioned_minus_shared", direct, seeds),
                               ("cu_minus_uu", mix, pairs)):
        changes[name] = {}
        for i, depth in ((0, 3), (2, 9)):
            label = f"depth_{depth}_minus_6"
            delta = values[i] - values[1]
            changes[name][label] = summarize_runs(ids, weights, delta, runs, replicates,
                                                  bootstrap_seed, family_size=4)
            change_arrays[f"{name}_{label}"] = delta
            image_values[f"{name}_{label}"] = delta.mean(axis=0)
    metadata = {key: base[key] for key in
                ("image_id", "task_id", "class_id", "class_a", "class_b", "weights")}
    np.savez_compressed(output / "paired_cases.npz", **metadata, depths=DEPTHS,
                        metric_names=METRICS, training_seeds=seeds, seed_pairs=pairs,
                        endpoint_methods=ENDPOINT_METHODS, mixture_methods=MIXTURE_METHODS,
                        endpoint_outcomes=endpoints, mixture_outcomes=mixtures,
                        conditioned_minus_shared=direct, cu_minus_uu=mix,
                        full_reference=base["full_reference"], **change_arrays)
    unique, inverse = np.unique(ids, return_inverse=True)
    columns = dict(image_id=unique, weight=np.bincount(inverse, weights=weights))
    for name, values in image_values.items():
        for j, metric in enumerate(METRICS):
            columns[f"{name}_{metric}_weighted_sum"] = np.bincount(
                inverse, weights=weights * values[:, j])
    write_rows(output / "image_contributions.csv",
               [dict(zip(columns, row)) for row in zip(*columns.values())])
    result = dict(dataset=dataset, depths=list(DEPTHS),
        sources={str(depth): panel["source"] for depth, panel in zip(DEPTHS, panels)},
        model=base["protocol"]["config"]["model"],
        keep_patch_tokens=base["protocol"]["config"]["keep_patch_tokens"],
        training_seeds=seeds.tolist(), primary_seed_pairs=pairs.tolist(),
        image_count=len(unique), task_count=len(set(base["task_id"])), row_count=len(ids),
        bootstrap_replicates=replicates, bootstrap_seed=bootstrap_seed,
        family_size=4,
        primary="Change in C minus U geometric margin at depth 3 or 9 relative to depth 6.",
        secondary="Change in CU minus UU; accuracy and raw cosine gap accompany geometric margin.",
        aggregation="Match images, tasks and training seeds or ordered seed pairs across depths. Average fixed runs within each row, then paired image bootstrap with original equal-task/equal-class weights.",
        uncertainty="All estimates use the same image resamples across depths and methods. Intervals condition on these tasks and fixed runs. Adjust the two depth changes across two datasets separately for each endpoint family and metric.",
        by_depth=by_depth, depth_changes=changes,
        full_reference=weighted_estimates(ids, weights, base["full_reference"], METRICS,
                                          replicates, bootstrap_seed))
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def analyze(sources, output, replicates=2000, bootstrap_seed=2361):
    output = Path(output)
    datasets = {dataset: analyze_dataset(sources, output / dataset, dataset,
                                         replicates, bootstrap_seed) for dataset in DATASETS}
    rows = []
    for dataset, result in datasets.items():
        for endpoint, changes in result["depth_changes"].items():
            for contrast, stats in changes.items():
                for metric in METRICS:
                    estimate = stats[metric]
                    rows.append(dict(dataset=dataset, endpoint=endpoint, contrast=contrast,
                        metric=metric, mean=estimate["mean"],
                        ci95_low=estimate["ci95"][0], ci95_high=estimate["ci95"][1],
                        family_ci95_low=estimate["bonferroni_family_ci95"][0],
                        family_ci95_high=estimate["bonferroni_family_ci95"][1]))
    write_rows(output / "depth_changes.csv", rows)
    result = dict(datasets=datasets, metrics=list(METRICS), reference_depth=6,
                  compared_depths=[3, 9], primary_family_size=4,
                  bootstrap_replicates=replicates, bootstrap_seed=bootstrap_seed)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2361)
    args = parser.parse_args()
    result = analyze(args.sources, args.output, args.replicates, args.bootstrap_seed)
    print(json.dumps({name: values["depth_changes"] for name, values in result["datasets"].items()}, indent=2))


if __name__ == "__main__":
    main()
