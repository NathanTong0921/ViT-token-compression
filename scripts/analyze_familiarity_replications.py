"""Aggregate fixed familiarity partitions while preserving shared-image dependence."""

import argparse
import json
from pathlib import Path

import numpy as np


CONTRASTS = ("real_minus_permuted_mean", "real_minus_wide", "permuted_mean_minus_wide")


def evaluation_root(root, dataset, partition_seed):
    stem = f"{dataset}-familiarity-switch"
    if partition_seed != 2099:
        stem += f"-partition-{partition_seed}"
    return root / f"{stem}-evaluation-v1"


def equal_partition_cluster_summary(cases, replicates=2000, seed=2102):
    """Resample shared image IDs once, then average separately normalized partitions."""
    image_ids = np.unique(np.concatenate([case["image_ids"] for case in cases]))
    p, n, k = len(cases), len(image_ids), cases[0]["values"].shape[1]
    numerator = np.zeros((p, n, k))
    denominator = np.zeros((p, n))
    for partition, case in enumerate(cases):
        ix = np.searchsorted(image_ids, case["image_ids"])
        np.add.at(denominator[partition], ix, case["weights"])
        np.add.at(numerator[partition], ix, case["weights"][:, None] * case["values"])
    point = (numerator.sum(1) / denominator.sum(1)[:, None]).mean(0)
    rng = np.random.default_rng(seed)
    boot = np.empty((replicates, k))
    for start in range(0, replicates, 100):
        ix = rng.integers(n, size=(min(100, replicates - start), n))
        part_num = numerator[:, ix, :].sum(2)
        part_den = denominator[:, ix].sum(2)
        boot[start:start + len(ix)] = (part_num / part_den[:, :, None]).mean(0)
    intervals = np.percentile(boot, [2.5, 97.5], axis=0)
    return dict(estimates=point.tolist(), ci95=intervals.T.tolist(), images=n,
                partitions=p, examples=sum(len(case["weights"]) for case in cases))


def load_partition(root, dataset, partition_seed):
    folder = evaluation_root(root, dataset, partition_seed)
    protocol = json.loads((folder / "protocol.json").read_text())
    assert protocol["partition_seed"] == partition_seed
    summary = json.loads((folder / "summary.json").read_text())
    with np.load(folder / "evaluation_panel.npz") as saved:
        panel = {key: saved[key].copy() for key in saved.files}
    seeds = protocol["training_seeds"]
    mappings = [arm for arm in protocol["arms"] if arm.startswith("permuted_")]
    parts = {}
    for phase in ("original", "switched"):
        parts[phase] = {}
        for metric in ("correct", "margin"):
            arrays = {}
            for arm in protocol["arms"]:
                values = []
                for seed in seeds:
                    with np.load(folder / phase / f"seed_{seed}_{arm}.npz") as saved:
                        values.append(saved[metric].astype(float))
                arrays[arm] = np.stack(values)
            permuted = np.mean([arrays[arm] for arm in mappings], axis=0)
            parts[phase][metric] = np.stack((arrays["real_text"] - permuted,
                                           arrays["real_text"] - arrays["image_only_wide"],
                                           permuted - arrays["image_only_wide"]), axis=-1)
    orientation = np.where(panel["true_original_held"], 1., -1.)
    cases, seed_estimates = {}, {}
    for metric in ("correct", "margin"):
        scale = 100. if metric == "correct" else 1.
        per_seed = (parts["switched"][metric] - parts["original"][metric]) * orientation[None, :, None] * scale
        cases[metric] = dict(image_ids=panel["image_ids"], weights=panel["weights"], values=per_seed.mean(0))
        seed_estimates[metric] = np.average(per_seed, axis=1, weights=panel["weights"])
        expected = [summary["metrics"][metric]["contrasts"][key]["seen_minus_held"]["estimate"] for key in CONTRASTS]
        np.testing.assert_allclose(seed_estimates[metric].mean(0), expected, atol=1e-12)
    brief = dict(partition_seed=partition_seed, source=str(folder), tasks=summary["tasks"],
                 images=summary["images"], rows=summary["rows"], training_seeds=seeds,
                 metrics={metric: {key: summary["metrics"][metric]["contrasts"][key]
                                   for key in CONTRASTS} for metric in ("correct", "margin")})
    return cases, seed_estimates, brief


def analyze(root, dataset, replicates):
    loaded = {seed: load_partition(root, dataset, seed) for seed in (2099, 2100, 2101)}
    result = dict(per_partition={str(seed): value[2] for seed, value in loaded.items()}, aggregates={})
    for name, partitions in (("new_partitions_only", (2100, 2101)), ("historical_plus_new", (2099, 2100, 2101))):
        group = dict(partition_seeds=list(partitions), metrics={})
        for metric in ("correct", "margin"):
            stats = equal_partition_cluster_summary([loaded[seed][0][metric] for seed in partitions], replicates)
            per_train_seed = np.mean([loaded[seed][1][metric] for seed in partitions], axis=0)
            group["metrics"][metric] = {
                "unit": "percentage points" if metric == "correct" else "geometric margin",
                "images": stats["images"], "rows": stats["examples"],
                "contrasts": {contrast: {"estimate": stats["estimates"][index], "ci95": stats["ci95"][index],
                    "per_training_seed": per_train_seed[:, index].tolist(),
                    "per_partition_estimates": [float(loaded[seed][1][metric][:, index].mean()) for seed in partitions]}
                    for index, contrast in enumerate(CONTRASTS)}}
        result["aggregates"][name] = group
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    result = dict(partition_seeds=[2099, 2100, 2101], replication_seeds=[2100, 2101],
        partition_weighting="Equal partition weight; within each partition equal task and class-side weight.",
        uncertainty="One paired image resample shared across all partitions, conditional on fixed partition choices, mappings and fitted models; training seed and partition estimates also retained.",
        primary_replication="The two prospectively fixed new partitions 2100/2101; historical2099 included separately for context.",
        margin_interpretation="The oriented margin role contrast is a candidate-direction score shift; it is not an offset-invariant class-separation measure. See the separate bias/decomposition analysis.",
        final_test_used=False,
        datasets={dataset: analyze(args.runs_root, dataset, args.bootstrap_replicates) for dataset in ("pets", "flowers")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
