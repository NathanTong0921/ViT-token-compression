"""Measure C/U routing headroom from existing development predictions."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SETTINGS = ("pets_b16_49", "pets_b16_98", "flowers_b16_49", "flowers_b16_98")
METRICS = ("accuracy", "geometric_margin", "raw_gap")


def endpoint_summary(conditioned, shared, weights):
    """Oracle is the best outcome per row, separately for each metric and seed."""
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    cmean = np.einsum("snm,n->sm", conditioned, weights)
    umean = np.einsum("snm,n->sm", shared, weights)
    oracle = np.einsum("snm,n->sm", np.maximum(conditioned, shared), weights)
    return dict(conditioned=cmean, shared=umean, oracle=oracle,
                better_fixed=np.maximum(cmean, umean),
                oracle_gain=oracle - np.maximum(cmean, umean))


def correctness_indicators(conditioned, shared):
    c, u = conditioned[:, :, 0] == 1, shared[:, :, 0] == 1
    return dict(both_correct=c & u, both_wrong=~c & ~u,
                conditioned_only=c & ~u, shared_only=~c & u)


def stability_indicators(conditioned, shared):
    """Each column is an image/task row, repeated across three trained seeds."""
    indicators = correctness_indicators(conditioned, shared)
    ccount = indicators["conditioned_only"].sum(axis=0)
    ucount = indicators["shared_only"].sum(axis=0)
    seeds = len(conditioned)
    return dict(conditioned_only_any=ccount > 0,
                shared_only_any=ucount > 0,
                conditioned_only_majority=ccount > seeds / 2,
                shared_only_majority=ucount > seeds / 2,
                conditioned_only_all=ccount == seeds,
                shared_only_all=ucount == seeds,
                direction_reverses=(ccount > 0) & (ucount > 0),
                any_disagreement=(ccount + ucount) > 0)


def aggregate_image_values(values, weights, image_ids):
    """Retain the original task/class weights and cluster repeated tasks by image."""
    unique, inverse = np.unique(image_ids, return_inverse=True)
    aggregate = np.zeros((len(unique), *values.shape[1:]), dtype=float)
    np.add.at(aggregate, inverse, values * weights.reshape((-1,) + (1,) * (values.ndim - 1)))
    return unique, aggregate, np.bincount(inverse, weights=weights)


def bootstrap_headroom(conditioned, shared, weights, image_ids, replicates, seed):
    # Rows are image/task pairs. Seeds are paired measurements, never resampled.
    packed = np.concatenate((conditioned, shared, np.maximum(conditioned, shared)), axis=2)
    _, sums, totals = aggregate_image_values(packed.transpose(1, 0, 2), weights, image_ids)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        chosen = rng.integers(len(totals), size=len(totals))
        means = sums[chosen].sum(axis=0) / totals[chosen].sum()
        c, u, oracle = np.split(means, 3, axis=-1)
        samples.append((oracle - np.maximum(c, u)).mean(axis=0))
    samples = np.asarray(samples)
    return dict(zip(METRICS, np.quantile(samples, [.025, .975], axis=0).T.tolist()))


def summarize_setting(source, output, replicates=2000, seed=2711):
    with np.load(source / "inputs.npz", allow_pickle=False) as data:
        c, u = data["primary_alternative"], data["primary_base"]
        weights, ids, tasks = data["weights"], data["image_ids"], data["task_ids"]
        seeds = data["primary_run_ids"]
    weights = weights / weights.sum()
    endpoints = endpoint_summary(c, u, weights)
    indicators = correctness_indicators(c, u)
    quadrants = {name: values @ weights for name, values in indicators.items()}
    stability = stability_indicators(c, u)
    intervals = bootstrap_headroom(c, u, weights, ids, replicates, seed)
    per_seed = []
    for si, training_seed in enumerate(seeds):
        row = {"setting": source.name, "seed": int(training_seed)}
        row.update({name + "_percent": float(value[si] * 100) for name, value in quadrants.items()})
        for mi, metric in enumerate(METRICS):
            multiplier = 100 if metric == "accuracy" else 1
            for name, values in endpoints.items():
                row[metric + "_" + name] = float(values[si, mi] * multiplier)
        per_seed.append(row)
    summary = {"setting": source.name, "rows": len(weights), "images": len(np.unique(ids)),
               "tasks": len(np.unique(tasks)), "training_seeds": seeds.tolist(),
               "correctness_percent": {name: float(values.mean() * 100) for name, values in quadrants.items()},
               "stability_percent": {name: float(weights @ values * 100) for name, values in stability.items()},
               "stability_row_counts": {name: int(values.sum()) for name, values in stability.items()},
               "metrics": {}}
    for mi, metric in enumerate(METRICS):
        multiplier = 100 if metric == "accuracy" else 1
        mean_c, mean_u, mean_oracle = (endpoints[k][:, mi].mean() for k in ("conditioned", "shared", "oracle"))
        summary["metrics"][metric] = {
            "unit": "percent_or_percentage_points" if metric == "accuracy" else "score",
            "conditioned": float(mean_c * multiplier), "shared": float(mean_u * multiplier),
            "oracle": float(mean_oracle * multiplier),
            "mean_per_seed_oracle_gain": float(endpoints["oracle_gain"][:, mi].mean() * multiplier),
            "mean_per_seed_oracle_gain_ci95": [x * multiplier for x in intervals[metric]],
            "oracle_gain_over_better_seed_average": float((mean_oracle - max(mean_c, mean_u)) * multiplier),
        }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "correctness.npz", image_ids=ids, task_ids=tasks, weights=weights,
                        training_seeds=seeds, conditioned_correct=c[:, :, 0].astype(np.int8),
                        shared_correct=u[:, :, 0].astype(np.int8))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary, per_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/conditioning-benefit-v1"))
    parser.add_argument("--output", type=Path, default=Path("results/conditioning-content-v1/headroom"))
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2711)
    args = parser.parse_args()
    summaries, per_seed = [], []
    for setting in SETTINGS:
        summary, rows = summarize_setting(args.source / setting, args.output / setting,
                                          args.replicates, args.seed)
        summaries.append(summary)
        per_seed.extend(rows)
    records = []
    for summary in summaries:
        acc = summary["metrics"]["accuracy"]
        lo, hi = acc["mean_per_seed_oracle_gain_ci95"]
        records.append({"setting": summary["setting"], "C_accuracy_percent": acc["conditioned"],
                        "U_accuracy_percent": acc["shared"], "oracle_accuracy_percent": acc["oracle"],
                        "oracle_gain_pp": acc["mean_per_seed_oracle_gain"],
                        "oracle_gain_ci95_lower_pp": lo, "oracle_gain_ci95_upper_pp": hi,
                        **summary["correctness_percent"], **summary["stability_percent"]})
    for name, rows in (("summary.csv", records), ("per_seed.csv", per_seed)):
        with (args.output / name).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    record = {"source": str(args.source), "settings": summaries,
              "bootstrap_replicates": args.replicates, "bootstrap_seed": args.seed,
              "interpretation": "Label-informed per-example oracle for these fixed C/U models. Accuracy and margin oracles are computed separately; no deployable gate or new predictor is evaluated.",
              "intervals": "Ordinary 95% image-cluster percentile intervals, conditional on the three trained seeds. Original task/class weights are retained and normalized in each resample."}
    (args.output / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
