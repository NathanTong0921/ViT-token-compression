"""Paired image-level comparison of direct-training effects across token budgets."""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_direct_classification import METHODS, METRICS, summarize, write_csv


NAMES = ("direct_gap_high_minus_low", "objective_interaction_high_minus_low")


def paired_budget_effects(low, high):
    """Subtract per-example effects only after matching the entire paired panel."""
    for key in ("image_ids", "task_ids", "seeds", "methods", "metrics"):
        np.testing.assert_array_equal(low[key], high[key], err_msg=f"Budget pairing: {key}")
    np.testing.assert_allclose(low["weights"], high["weights"], atol=1e-15, rtol=0.)
    names = list(map(str, low["methods"]))
    assert names == list(METHODS)
    assert list(map(str, low["metrics"])) == list(METRICS)
    index = {name: i for i, name in enumerate(names)}

    def effects(data):
        x = data["outcomes"]
        direct = x[:, index["direct_C"]] - x[:, index["direct_U"]]
        kl = x[:, index["kl_C"]] - x[:, index["kl_U"]]
        return np.stack((direct, direct-kl), axis=1)

    lower, upper = effects(low), effects(high)
    return upper-lower, lower, upper


def load(path):
    with np.load(path) as saved:
        return {key: saved[key].copy() for key in saved.files}


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    low, high = load(args.low), load(args.high)
    delta, lower, upper = paired_budget_effects(low, high)
    weights = low["weights"] / low["weights"].sum()
    results, by_seed, contributions, totals = summarize(delta, weights, low["image_ids"], NAMES,
        args.replicates, args.bootstrap_seed, primary=NAMES)
    lower_points = np.einsum("n,smnq->smq", weights, lower)
    upper_points = np.einsum("n,smnq->smq", weights, upper)
    summary = dict(low_budget=args.low_budget, high_budget=args.high_budget,
        direction="Higher token budget minus lower token budget, paired on the same image/task decisions.",
        rows=len(weights), images=len(set(low["image_ids"])), tasks=len(set(low["task_ids"])),
        seeds=low["seeds"].tolist(), metrics=list(METRICS), contrasts=results,
        primary_metrics=["accuracy"], primary_contrasts=list(NAMES),
        interval_scope="Adjustment over the two budget comparisons, separately per metric; raw gap, geometric margin and CE are accompanying metrics.",
        bootstrap_replicates=args.replicates, bootstrap_seed=args.bootstrap_seed,
        sources=dict(low=str(args.low), high=str(args.high)),
        pairing=dict(image_task_order_equal=True, seeds_methods_metrics_equal=True,
                     maximum_weight_difference=float(np.max(np.abs(low["weights"]-high["weights"])))))
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    rows=[]
    for ni, name in enumerate(NAMES):
        for mi, metric in enumerate(METRICS):
            value=results[name][metric]
            rows.append(dict(contrast=name, metric=metric, low_budget_mean=float(lower_points[:,ni,mi].mean()),
                high_budget_mean=float(upper_points[:,ni,mi].mean()), difference=value["mean"],
                adjusted_low=value["two_primary_adjusted_ci95"][0],
                adjusted_high=value["two_primary_adjusted_ci95"][1]))
    write_csv(args.output/"summary.csv", rows)
    per_seed=[]
    for si, seed in enumerate(low["seeds"]):
        for ni, name in enumerate(NAMES):
            per_seed.append(dict(seed=int(seed), contrast=name,
                **{metric:float(by_seed[si,ni,mi]) for mi,metric in enumerate(METRICS)}))
    write_csv(args.output/"per_seed.csv", per_seed)
    np.savez_compressed(args.output/"paired_budget_contrasts.npz", differences=delta,
        low_effects=lower, high_effects=upper, names=NAMES, metrics=METRICS,
        image_ids=low["image_ids"], task_ids=low["task_ids"], weights=weights, seeds=low["seeds"],
        image_contributions=contributions, image_weights=totals)
    print(json.dumps({name:results[name]["accuracy"] for name in NAMES}), flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low", type=Path, required=True)
    parser.add_argument("--high", type=Path, required=True)
    parser.add_argument("--low-budget", type=int, default=49)
    parser.add_argument("--high-budget", type=int, default=98)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2717)
    main(parser.parse_args())
