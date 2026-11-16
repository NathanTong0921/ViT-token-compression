"""Paired C/U capacity curves with fixed large-minus-small interactions."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_direct_classification import METRICS, SEEDS, reconstruct_outcomes, summarize, write_csv

DATASETS = ("pets", "flowers")
CAPACITIES = ("small", "medium", "large")
ARMS = ("conditioned", "image_only_wide")
METHODS = tuple(f"{capacity}_{arm}" for capacity in CAPACITIES for arm in ("C", "U"))
PRIMARY = "gap_large_minus_small"
CONTRASTS = (
    "small_C_minus_U", "medium_C_minus_U", "large_C_minus_U",
    PRIMARY, "gap_medium_minus_small", "gap_large_minus_medium",
    "C_large_minus_small", "U_large_minus_small",
)


def capacity_contrasts(values):
    """Input [seed, six methods, row, metric], preserving paired rows."""
    index = {method: i for i, method in enumerate(METHODS)}
    gaps = {capacity: values[:, index[f"{capacity}_C"]] - values[:, index[f"{capacity}_U"]]
            for capacity in CAPACITIES}
    c_change = values[:, index["large_C"]] - values[:, index["small_C"]]
    u_change = values[:, index["large_U"]] - values[:, index["small_U"]]
    interaction = gaps["large"] - gaps["small"]
    np.testing.assert_allclose(interaction, c_change - u_change, atol=1e-14, rtol=1e-12)
    return np.stack((gaps["small"], gaps["medium"], gaps["large"], interaction,
                     gaps["medium"] - gaps["small"], gaps["large"] - gaps["medium"],
                     c_change, u_change), axis=1)


def read_panel(source, seeds):
    with (source / "examples.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    image_ids = np.asarray([row["image_id"] for row in rows])
    task_ids = np.asarray([row["task_id"] for row in rows])
    labels = np.asarray([int(row["class_id"]) for row in rows])
    pairs = np.asarray([[int(row["class_a"]), int(row["class_b"])] for row in rows])
    weights = np.asarray([float(row["task_class_weight"]) for row in rows])
    assert len(set(zip(image_ids, task_ids))) == len(rows)
    np.testing.assert_allclose(weights.sum(), 1., atol=1e-12, rtol=0.)
    groups = np.asarray([f"{task}:{label}" for task, label in zip(task_ids, labels)])
    unique, group_inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    assert len(unique) == 2 * len(set(task_ids))
    np.testing.assert_allclose(weights, 1. / (len(unique) * counts[group_inverse]), atol=1e-15, rtol=0.)
    with np.load(source / "evidence.npz") as evidence:
        image_indices = np.asarray([int(row["image_index"]) for row in rows])
        np.testing.assert_array_equal(evidence["image_ids"][image_indices], image_ids)
        np.testing.assert_array_equal(evidence["labels"][image_indices], labels)
        text, scale = evidence["text_features"], float(evidence["logit_scale"])
    execution = json.loads((source / "evaluation_summary.json").read_text())
    jobs = {(int(job["seed"]), job["capacity"], job["arm"]): job for job in execution["jobs"]}
    values, checks, metadata = [], [], {}
    for seed in seeds:
        by_method = []
        for capacity in CAPACITIES:
            for arm, name in zip(ARMS, ("C", "U")):
                job = jobs[seed, capacity, arm]
                method = f"{capacity}_{name}"
                current = dict(width=int(job["width"]), parameter_count=int(job["parameter_count"]))
                if method in metadata:
                    assert metadata[method] == current
                metadata[method] = current
                with np.load(source / f"seed_{seed}" / capacity / (arm + ".npz")) as raw:
                    selected = raw["patch_indices"]
                    assert selected.shape == (len(rows), 49)
                    assert np.all((selected >= 0) & (selected < 196))
                    assert np.all(np.diff(np.sort(selected, axis=1), axis=1) > 0)
                    outcomes = reconstruct_outcomes(raw["cosine_scores"], labels, pairs, text, scale)
                    errors = {}
                    for qi, metric in enumerate(METRICS):
                        np.testing.assert_allclose(outcomes[:, qi], raw[metric], atol=3e-5, rtol=1e-5)
                        errors[metric] = float(np.max(np.abs(outcomes[:, qi] - raw[metric])))
                by_method.append(outcomes)
                checks.append(dict(seed=int(seed), method=method, reconstruction_errors=errors))
        values.append(np.stack(by_method))
    return dict(values=np.stack(values), image_ids=image_ids, task_ids=task_ids, labels=labels,
                pairs=pairs, weights=weights, metadata=metadata, checks=checks, logit_scale=scale)


def analyze_panel(data, dataset, panel, output, seeds, replicates=2000, bootstrap_seed=2721):
    output.mkdir(parents=True, exist_ok=True)
    methods, method_seeds, _, _ = summarize(data["values"], data["weights"], data["image_ids"],
                                           METHODS, replicates, bootstrap_seed)
    differences = capacity_contrasts(data["values"])
    # summarize uses 1.25%/98.75% quantiles for a named primary contrast.
    # Each dataset contributes one predeclared interaction to the two-dataset family.
    contrasts, contrast_seeds, contributions, totals = summarize(
        differences, data["weights"], data["image_ids"], CONTRASTS,
        replicates, bootstrap_seed, primary=(PRIMARY,))
    summary = dict(dataset=dataset, panel=panel, seeds=list(seeds), methods=methods,
        method_metadata=data["metadata"], contrasts=contrasts, primary_contrast=PRIMARY,
        primary_metric="accuracy", primary_family_size=2,
        family="Pets and Flowers large-minus-small C-U accuracy interactions, separately for each evaluation panel.",
        main_panel="test; development is a training/generalization comparison",
        metrics=list(METRICS), rows=len(data["weights"]), images=len(set(data["image_ids"])),
        tasks=len(set(data["task_ids"])), keep_patch_tokens=49,
        bootstrap_replicates=replicates, bootstrap_seed=bootstrap_seed,
        weighting="Original equal-task/equal-class-side weights; fixed seed mean, image-cluster resampling.",
        interval_scope="Primary large-small interaction adjusted across two datasets; secondary intervals are ordinary 95%.",
        positive_direction="Larger C-U favors conditioning for accuracy/gaps; smaller C-U favors conditioning for CE.")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    method_rows, contrast_rows, seed_rows = [], [], []
    for method, metrics in methods.items():
        for metric, value in metrics.items():
            method_rows.append(dict(method=method, **data["metadata"][method], metric=metric,
                                    mean=value["mean"], low=value["ci95"][0], high=value["ci95"][1]))
    for name, metrics in contrasts.items():
        for metric, value in metrics.items():
            adjusted = value.get("two_primary_adjusted_ci95", (None, None))
            contrast_rows.append(dict(contrast=name, metric=metric, mean=value["mean"],
                low=value["ci95"][0], high=value["ci95"][1],
                primary_adjusted_low=adjusted[0], primary_adjusted_high=adjusted[1]))
    for kind, names, by_seed in (("method", METHODS, method_seeds),
                                 ("contrast", CONTRASTS, contrast_seeds)):
        for si, seed in enumerate(seeds):
            for ni, name in enumerate(names):
                seed_rows.append(dict(seed=seed, kind=kind, name=name,
                    **{metric: float(by_seed[si, ni, qi]) for qi, metric in enumerate(METRICS)}))
    write_csv(output / "methods.csv", method_rows)
    write_csv(output / "contrasts.csv", contrast_rows)
    write_csv(output / "per_seed.csv", seed_rows)
    (output / "audit.json").write_text(json.dumps(dict(task_class_weights_verified=True,
        paired_identity_verified=True, checks=data["checks"]), indent=2) + "\n")
    np.savez_compressed(output / "paired_outcomes.npz", outcomes=data["values"], methods=METHODS,
        metrics=METRICS, seeds=seeds, weights=data["weights"], image_ids=data["image_ids"],
        task_ids=data["task_ids"], labels=data["labels"], pairs=data["pairs"],
        contrast_image_contributions=contributions, image_weights=totals, contrasts=CONTRASTS)
    return summary


def summarize_training(source, output, seeds):
    """Retain final teacher-fit values and existing curves for all capacities."""
    rows, curves = [], []
    for dataset in DATASETS:
        for capacity in CAPACITIES:
            folder = source / dataset / "training" / capacity
            summary = json.loads((folder / "training_summary.json").read_text())
            for run in summary["runs"]:
                if run["seed"] not in seeds or run["arm"] not in ARMS or run["coverage"] != "full":
                    continue
                rows.append(dict(dataset=dataset, capacity=capacity, seed=run["seed"], arm=run["arm"],
                    reused=capacity == "medium", parameter_count=run["parameter_count"],
                    training_steps=run["training_steps"], final_train_kl=run["final_train_kl"],
                    final_val_kl=run["final_val_kl"],
                    val_minus_train_kl=run["final_val_kl"]-run["final_train_kl"]))
            with (folder / "training_history.csv").open() as handle:
                for row in csv.DictReader(handle):
                    if int(row["seed"]) in seeds and row["arm"] in ARMS and row["coverage"] == "full":
                        curves.append(dict(dataset=dataset, capacity=capacity, **row))
    assert len(rows) == len(DATASETS) * len(CAPACITIES) * len(ARMS) * len(seeds)
    means = []
    for dataset in DATASETS:
        for capacity in CAPACITIES:
            for arm in ARMS:
                group = [row for row in rows if (row["dataset"],row["capacity"],row["arm"]) == (dataset,capacity,arm)]
                assert sorted(row["seed"] for row in group) == sorted(seeds)
                assert len({row["parameter_count"] for row in group}) == 1
                means.append(dict(dataset=dataset, capacity=capacity, arm=arm,
                    parameter_count=group[0]["parameter_count"], training_steps=group[0]["training_steps"],
                    **{key:float(np.mean([row[key] for row in group]))
                       for key in ("final_train_kl","final_val_kl","val_minus_train_kl")}))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "training_fit.csv", rows)
    write_csv(output / "training_fit_summary.csv", means)
    write_csv(output / "training_history.csv", curves)


def main(args):
    summarize_training(args.source, args.output, args.seeds)
    summaries = {}
    for panel in args.panels:
        summaries[panel] = {}
        for dataset in DATASETS:
            source = args.source / dataset / "evaluation" / panel
            data = read_panel(source, args.seeds)
            summary = analyze_panel(data, dataset, panel, args.output / panel / dataset,
                                    args.seeds, args.replicates, args.bootstrap_seed)
            summaries[panel][dataset] = summary
            print(json.dumps(dict(dataset=dataset, panel=panel,
                primary=summary["contrasts"][PRIMARY]["accuracy"])), flush=True)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--panels", nargs="+", choices=("dev", "test"), default=["dev", "test"])
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--replicates", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=2721)
    main(p.parse_args())
