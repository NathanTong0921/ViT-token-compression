"""Replay the fixed replication analysis from verified local directional scores.

The original run folders are not reconstructed or overwritten.  This entry point
uses the compact score export of the independent familiarity-prior analysis and
the unchanged image-bootstrap implementation of the frozen replication plan.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_familiarity_replications import CONTRASTS, equal_partition_cluster_summary
from analyze_familiarity_switch import summarize_switch


def load_partition(root, dataset, partition_seed, replicates):
    folder = root / f"{dataset}-p{partition_seed}"
    protocol = json.loads((folder / "protocol.json").read_text())
    prior = json.loads((folder / "summary.json").read_text())
    assert protocol["partition_seed"] == partition_seed
    assert not protocol["final_test_used"]
    with np.load(folder / "analysis_arrays.npz") as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    q, true_h0 = arrays["q"], arrays["true_h0"]
    sign = np.where(true_h0, 1., -1.)
    predict_h0 = (q > 0) | ((q == 0) & (arrays["h0_class"] < arrays["h1_class"]))
    correct = predict_h0 == true_h0
    margin = q * sign
    full_q = arrays["full_q"]
    full_predict = (full_q > 0) | ((full_q == 0) & (arrays["h0_class"] < arrays["h1_class"]))
    panel = dict(image_ids=arrays["image_ids"], task_ids=arrays["task_ids"],
                 weights=arrays["weights"], true_original_held=true_h0,
                 full_correct=full_predict == true_h0, full_margin=full_q * sign)
    arms, seeds = arrays["arms"].tolist(), arrays["seeds"].tolist()
    predictions = {(phase, seed, arm): dict(correct=correct[p, s, a], margin=margin[p, s, a])
                   for p, phase in enumerate(("original", "switched"))
                   for s, seed in enumerate(seeds) for a, arm in enumerate(arms)}
    summary = summarize_switch(panel, predictions, arms, seeds, replicates)
    per_seed, cases, checks = {}, {}, {}
    indices = {arm: arms.index(arm) for arm in arms}
    mapping_indices = [i for i, arm in enumerate(arms) if arm.startswith("permuted_")]
    for metric, values in (("correct", correct.astype(float) * 100.), ("margin", margin)):
        real, wide = values[:, :, indices["real_text"]], values[:, :, indices["image_only_wide"]]
        identity = values[:, :, mapping_indices].mean(2)
        contrasts = np.stack((real - identity, real - wide, identity - wide), axis=-1)
        oriented = (contrasts[1] - contrasts[0]) * sign[None, :, None]
        cases[metric] = dict(image_ids=panel["image_ids"], weights=panel["weights"], values=oriented.mean(0))
        per_seed[metric] = np.average(oriented, axis=1, weights=panel["weights"])
        prior_metric = "accuracy_pp" if metric == "correct" else "true_margin"
        expected = prior["correction"]["raw"]["contrasts"][prior_metric]["real_minus_identity"]["seen_minus_held"]["estimate"]
        actual = float(per_seed[metric][:, 0].mean())
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        checks[metric] = dict(reconstructed=actual, prior_diagnostic=expected,
                              absolute_difference=abs(actual - expected))
    brief = dict(partition_seed=partition_seed, source=str(folder), source_run=protocol["source"],
                 tasks=summary["tasks"], images=summary["images"], rows=summary["rows"],
                 training_seeds=seeds, reconstruction_checks=checks,
                 metrics={metric: {key: summary["metrics"][metric]["contrasts"][key] for key in CONTRASTS}
                          for metric in ("correct", "margin")})
    return cases, per_seed, brief


def analyze(root, dataset, replicates):
    loaded = {seed: load_partition(root, dataset, seed, replicates) for seed in (2099, 2100, 2101)}
    result = dict(per_partition={str(seed): value[2] for seed, value in loaded.items()}, aggregates={})
    for name, partitions in (("new_partitions_only", (2100, 2101)), ("historical_plus_new", (2099, 2100, 2101))):
        group = dict(partition_seeds=list(partitions), metrics={})
        for metric in ("correct", "margin"):
            stats = equal_partition_cluster_summary([loaded[seed][0][metric] for seed in partitions], replicates)
            train_seed_values = np.mean([loaded[seed][1][metric] for seed in partitions], axis=0)
            group["metrics"][metric] = dict(
                unit="percentage points" if metric == "correct" else "geometric margin",
                images=stats["images"], rows=stats["examples"],
                contrasts={contrast: dict(estimate=stats["estimates"][index], ci95=stats["ci95"][index],
                    per_training_seed=train_seed_values[:, index].tolist(),
                    per_partition_estimates=[float(loaded[seed][1][metric][:, index].mean()) for seed in partitions])
                    for index, contrast in enumerate(CONTRASTS)})
        result["aggregates"][name] = group
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-root", type=Path, default=Path("results/familiarity-prior-v1"))
    parser.add_argument("--output", type=Path, default=Path("results/familiarity-replications-v1/summary.json"))
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    result = dict(
        partition_seeds=[2099, 2100, 2101], replication_seeds=[2100, 2101],
        input_source="Independently verified compact q/image/task exports from familiarity-prior-v1; original prediction folders are not recreated.",
        input_verification=str(args.prior_root / "verification.json"),
        partition_weighting="Equal partition weight; within each partition equal task and class-side weight.",
        uncertainty="Frozen joint image bootstrap (2000 draws, seed2102): shared image IDs resampled once across partitions; conditional on fixed partitions, mappings and fitted models.",
        primary_replication="Prospectively fixed new partitions2100/2101; historical2099 included separately for context.",
        metric_interpretation="Role accuracy equals relative H0-choice frequency change; role margin equals candidate-direction shift. Neither alone measures ordinary accuracy improvement.",
        final_test_used=False,
        datasets={dataset: analyze(args.prior_root, dataset, args.bootstrap_replicates) for dataset in ("pets", "flowers")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)
