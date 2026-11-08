"""Paired analysis of direct task-loss fine-tuning and KL continuation."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = ("accuracy", "raw_gap", "geometric_margin", "ce")
ARMS = ("conditioned", "image_only_wide")
METHODS = ("initial_C", "initial_U", "kl_C", "kl_U", "direct_C", "direct_U")
SEEDS = (2041, 2042, 2043)
PRIMARY = ("direct_C_minus_direct_U", "direct_gap_minus_kl_gap")
CONTRASTS = {
    "direct_C_minus_direct_U": {"direct_C": 1, "direct_U": -1},
    "direct_gap_minus_kl_gap": {"direct_C": 1, "direct_U": -1, "kl_C": -1, "kl_U": 1},
    "initial_C_minus_initial_U": {"initial_C": 1, "initial_U": -1},
    "kl_C_minus_kl_U": {"kl_C": 1, "kl_U": -1},
    "direct_C_minus_initial_C": {"direct_C": 1, "initial_C": -1},
    "direct_U_minus_initial_U": {"direct_U": 1, "initial_U": -1},
    "direct_C_minus_kl_C": {"direct_C": 1, "kl_C": -1},
    "direct_U_minus_kl_U": {"direct_U": 1, "kl_U": -1},
    "kl_C_minus_initial_C": {"kl_C": 1, "initial_C": -1},
    "kl_U_minus_initial_U": {"kl_U": 1, "initial_U": -1},
}


def reconstruct_outcomes(scores, labels, pairs, text_features, logit_scale):
    """Compute binary outcomes directly from class cosine scores.

    Ties use the lower class index, independently of candidate presentation order.
    Cross-entropy is softplus of the negative scaled true-versus-rival gap.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels, pairs = np.asarray(labels), np.asarray(pairs)
    assert np.all((pairs == labels[:, None]).any(axis=1))
    other = np.where(pairs[:, 0] == labels, pairs[:, 1], pairs[:, 0])
    rows = np.arange(len(labels))
    raw = scores[rows, labels] - scores[rows, other]
    correct = (raw > 0) | ((raw == 0) & (labels < other))
    text = np.asarray(text_features, dtype=np.float64)
    distance = np.linalg.norm(text[labels] - text[other], axis=1)
    return np.column_stack((correct.astype(float), raw, raw / distance,
                            np.logaddexp(0., -float(logit_scale) * raw)))


def contrast_values(outcomes, methods=METHODS):
    """Input [seeds, methods, rows, metrics]; output [seeds, contrasts, rows, metrics]."""
    index = {name: i for i, name in enumerate(methods)}
    return np.stack([sum(coef * outcomes[:, index[method]] for method, coef in formula.items())
                     for formula in CONTRASTS.values()], axis=1)


def image_contributions(values, weights, image_ids):
    """Average fixed trained seeds, then aggregate weighted repeated tasks by image."""
    unique, inverse = np.unique(image_ids, return_inverse=True)
    per_image = np.zeros((len(unique), values.shape[1], values.shape[-1]), dtype=float)
    np.add.at(per_image, inverse, weights[:, None, None] * values.mean(axis=0).transpose(1, 0, 2))
    totals = np.bincount(inverse, weights=weights)
    return unique, per_image, totals


def summarize(values, weights, image_ids, names, replicates=2000, bootstrap_seed=2703,
              primary=()):
    """Resample image clusters, retaining the original task/class weighting."""
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    per_seed = np.einsum("n,smnq->smq", weights, values)
    point = per_seed.mean(axis=0)
    unique, sums, totals = image_contributions(values, weights, image_ids)
    rng = np.random.default_rng(bootstrap_seed)
    draws = np.empty((replicates, len(names), len(METRICS)))
    for start in range(0, replicates, 100):
        count = min(100, replicates - start)
        sampled = rng.integers(len(unique), size=(count, len(unique)))
        draws[start:start+count] = sums[sampled].sum(axis=1) / totals[sampled].sum(axis=1)[:, None, None]
    result = {}
    for ni, name in enumerate(names):
        result[name] = {}
        for qi, metric in enumerate(METRICS):
            record = dict(mean=float(point[ni, qi]),
                          ci95=np.quantile(draws[:, ni, qi], [.025, .975]).tolist())
            if name in primary:
                record["two_primary_adjusted_ci95"] = np.quantile(
                    draws[:, ni, qi], [.0125, .9875]).tolist()
            result[name][metric] = record
    return result, per_seed, sums, totals


def load_evaluation(source, step, seeds):
    folder = source / "evaluation"
    with (folder / "examples.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    image_ids = np.asarray([row["image_id"] for row in rows])
    labels = np.asarray([int(row["class_id"]) for row in rows])
    pairs = np.asarray([[int(row["class_a"]), int(row["class_b"])] for row in rows])
    weights = np.asarray([float(row["task_class_weight"]) for row in rows])
    with np.load(folder / "evidence.npz") as evidence:
        text = evidence["text_features"]
        scale = float(evidence["logit_scale"])
        image_index = np.asarray([int(row["image_index"]) for row in rows])
        np.testing.assert_array_equal(evidence["image_ids"][image_index], image_ids)
        np.testing.assert_array_equal(evidence["labels"][image_index], labels)
    values, checks, masks = [], [], []
    for seed in seeds:
        by_method, by_mask = [], []
        for stage in ("initial", f"kl_step_{step:04d}", f"direct_step_{step:04d}"):
            for arm in ARMS:
                path = folder / f"seed_{seed}" / arm / (stage + ".npz")
                with np.load(path) as data:
                    outcome = reconstruct_outcomes(data["cosine_scores"], labels, pairs, text, scale)
                    selected = data["patch_indices"].copy()
                    assert selected.shape == (len(rows), 49)
                    assert np.all(np.diff(np.sort(selected, axis=1), axis=1) > 0)
                    error = {}
                    for qi, metric in enumerate(METRICS):
                        reference = data[metric]
                        error[metric] = float(np.max(np.abs(outcome[:, qi] - reference)))
                        np.testing.assert_allclose(outcome[:, qi], reference, atol=3e-5, rtol=1e-5)
                checks.append(dict(seed=seed, stage=stage, arm=arm, maximum_error=error))
                by_method.append(outcome)
                by_mask.append(selected)
        values.append(np.stack(by_method))
        masks.append(np.stack(by_mask))
    return dict(outcomes=np.stack(values), masks=np.stack(masks), weights=weights,
                image_ids=image_ids, task_ids=np.asarray([row["task_id"] for row in rows]),
                labels=labels, pairs=pairs, logit_scale=scale, checks=checks)


def write_csv(path, rows):
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_evaluation(args.source, args.step, args.seeds)
    weights = data["weights"] / data["weights"].sum()
    methods, method_seeds, _, _ = summarize(data["outcomes"], weights, data["image_ids"], METHODS,
        args.replicates, args.bootstrap_seed)
    differences = contrast_values(data["outcomes"])
    contrasts, contrast_seeds, image_sums, image_weights = summarize(
        differences, weights, data["image_ids"], tuple(CONTRASTS), args.replicates,
        args.bootstrap_seed, PRIMARY)
    summary = dict(step=args.step, training_seeds=args.seeds, rows=len(weights),
                   images=len(set(data["image_ids"])), tasks=len(set(data["task_ids"])),
                   logit_scale=data["logit_scale"], methods=methods, contrasts=contrasts,
                   primary_contrasts=list(PRIMARY), metrics=list(METRICS),
                   interval_scope="Two-primary adjustment separately per metric; other intervals are ordinary 95%.",
                   bootstrap_replicates=args.replicates, bootstrap_seed=args.bootstrap_seed,
                   positive_direction="C or direct preferred for accuracy/gaps; lower CE is preferred.")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    rows = []
    for kind, results in (("method", methods), ("contrast", contrasts)):
        for name, metrics in results.items():
            for metric, value in metrics.items():
                adjusted = value.get("two_primary_adjusted_ci95", [None, None])
                rows.append(dict(kind=kind, name=name, metric=metric, mean=value["mean"],
                                 low=value["ci95"][0], high=value["ci95"][1],
                                 primary_adjusted_low=adjusted[0], primary_adjusted_high=adjusted[1]))
    write_csv(args.output / "summary.csv", rows)
    per_seed = []
    for si, seed in enumerate(args.seeds):
        for kind, names, values in (("method", METHODS, method_seeds),
                                    ("contrast", tuple(CONTRASTS), contrast_seeds)):
            for ni, name in enumerate(names):
                per_seed.append(dict(seed=seed, kind=kind, name=name,
                                     **{metric: float(values[si, ni, qi]) for qi, metric in enumerate(METRICS)}))
    write_csv(args.output / "per_seed.csv", per_seed)
    (args.output / "outcome_reconstruction.json").write_text(json.dumps(dict(
        passed=True, comparisons=data["checks"]), indent=2)+"\n")
    np.savez_compressed(args.output / "paired_outcomes.npz", outcomes=data["outcomes"],
        methods=METHODS, contrasts=tuple(CONTRASTS), metrics=METRICS, seeds=args.seeds,
        weights=weights, image_ids=data["image_ids"], task_ids=data["task_ids"],
        image_contributions=image_sums, image_weights=image_weights)
    print(json.dumps({name: contrasts[name]["accuracy"] for name in PRIMARY}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, default=1200)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2703)
    main(parser.parse_args())
