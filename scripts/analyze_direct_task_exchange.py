"""Re-score fixed-image task-description exchanges from saved full-class outputs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_direct_classification import (
    ARMS, METHODS, METRICS, SEEDS, reconstruct_outcomes, summarize, write_csv,
)


STAGES = ("initial", "kl", "direct")
CONTEXTS = ("own", "swapped")
CONTRAST_NAMES = (
    "initial_mapping", "kl_mapping", "direct_mapping",
    "mapping_change_direct_vs_kl", "own_C_change_direct_vs_kl",
    "swapped_C_change_direct_vs_kl", "U_change_direct_vs_kl",
    "direct_C_minus_U", "kl_C_minus_U", "CU_change_direct_vs_kl",
    "swapped_remaining_change",
)


def donor_rows(image_ids, task_ids, labels, expected_tasks=4):
    """Each target gets every other task of the same image, exactly once."""
    donors = np.empty((len(image_ids), expected_tasks - 1), dtype=np.int64)
    unique, inverse = np.unique(image_ids, return_inverse=True)
    for image_index in range(len(unique)):
        rows = np.flatnonzero(inverse == image_index)
        assert len(rows) == expected_tasks, "Every image must have four candidate tasks."
        assert len(set(task_ids[rows])) == expected_tasks, "Image/task rows must be unique."
        assert len(set(labels[rows])) == 1, "A fixed image must retain its ground-truth class."
        for row in rows:
            donors[row] = rows[rows != row]
    return donors


def exchange_outcomes(scores, labels, pairs, text, scale, donors):
    """Score each donor output against target candidates, then average outcomes."""
    own = reconstruct_outcomes(scores, labels, pairs, text, scale)
    swapped = np.mean([
        reconstruct_outcomes(scores[donors[:, i]], labels, pairs, text, scale)
        for i in range(donors.shape[1])
    ], axis=0)
    return np.stack((own, swapped), axis=0)


def selection_overlap(selected, donors, patch_count=196):
    """Mean fraction of target retained patches also retained by each donor."""
    masks = np.zeros((len(selected), patch_count), dtype=bool)
    masks[np.arange(len(selected))[:, None], selected] = True
    return np.mean([(masks & masks[donors[:, i]]).sum(axis=1) / selected.shape[1]
                    for i in range(donors.shape[1])], axis=0)


def exchange_contrasts(values):
    """Input [seed, six methods, own/swapped, row, metric]."""
    index = {name: i for i, name in enumerate(METHODS)}
    own = {name: values[:, i, 0] for name, i in index.items()}
    swapped = {name: values[:, i, 1] for name, i in index.items()}
    mapping = {stage: own[stage + "_C"] - swapped[stage + "_C"] for stage in STAGES}
    own_change = own["direct_C"] - own["kl_C"]
    swapped_change = swapped["direct_C"] - swapped["kl_C"]
    u_change = own["direct_U"] - own["kl_U"]
    direct_gap = own["direct_C"] - own["direct_U"]
    kl_gap = own["kl_C"] - own["kl_U"]
    mapping_change = mapping["direct"] - mapping["kl"]
    remaining = swapped_change - u_change
    np.testing.assert_allclose(direct_gap - kl_gap, mapping_change + remaining,
                               atol=1e-13, rtol=1e-12)
    return np.stack((mapping["initial"], mapping["kl"], mapping["direct"],
                     mapping_change, own_change, swapped_change, u_change,
                     direct_gap, kl_gap, direct_gap - kl_gap, remaining), axis=1)


def read_panel(source, budget, step, seeds):
    with (source / "evaluation/examples.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    ids = np.asarray([row["image_id"] for row in rows])
    tasks = np.asarray([row["task_id"] for row in rows])
    labels = np.asarray([int(row["class_id"]) for row in rows])
    pairs = np.asarray([[int(row["class_a"]), int(row["class_b"])] for row in rows])
    weights = np.asarray([float(row["task_class_weight"]) for row in rows])
    weights /= weights.sum()
    donors = donor_rows(ids, tasks, labels)
    with np.load(source / "evaluation/evidence.npz") as saved:
        text, scale = saved["text_features"], float(saved["logit_scale"])
        indices = np.asarray([int(row["image_index"]) for row in rows])
        np.testing.assert_array_equal(saved["image_ids"][indices], ids)
        np.testing.assert_array_equal(saved["labels"][indices], labels)
    values, checks, overlaps = [], [], []
    with np.load(source / "analysis/paired_outcomes.npz") as original:
        np.testing.assert_array_equal(original["image_ids"], ids)
        np.testing.assert_array_equal(original["task_ids"], tasks)
        np.testing.assert_array_equal(original["seeds"], seeds)
        np.testing.assert_array_equal(original["methods"], METHODS)
        np.testing.assert_allclose(original["weights"], weights, atol=1e-15, rtol=0)
        for si, seed in enumerate(seeds):
            methods = []
            for stage, short in (("initial", "initial"),
                                 (f"kl_step_{step}", "kl"), (f"direct_step_{step}", "direct")):
                for arm, letter in zip(ARMS, ("C", "U")):
                    method = short + "_" + letter
                    with np.load(source / f"evaluation/seed_{seed}" / arm / (stage + ".npz")) as raw:
                        assert raw["patch_indices"].shape == (len(rows), budget)
                        scores = raw["cosine_scores"]
                        result = exchange_outcomes(scores, labels, pairs, text, scale, donors)
                        original_values = original["outcomes"][si, METHODS.index(method)]
                        np.testing.assert_allclose(result[0], original_values, atol=1e-14, rtol=1e-12)
                        check = dict(seed=seed, method=method,
                                     own_max_error=float(np.max(np.abs(result[0] - original_values))))
                        if letter == "C":
                            overlap = selection_overlap(raw["patch_indices"], donors)
                            overlaps.append(dict(seed=seed, stage=short,
                                retained_set_overlap=float(weights @ overlap)))
                        if letter == "U":
                            score_difference = max(float(np.max(np.abs(scores - scores[donors[:, i]])))
                                                   for i in range(donors.shape[1]))
                            check["U_max_same_image_score_difference"] = score_difference
                            check["U_max_exchange_difference_by_metric"] = dict(zip(
                                METRICS, np.max(np.abs(result[0] - result[1]), axis=0).tolist()))
                            np.testing.assert_allclose(scores, scores[donors[:, 0]], atol=2e-6, rtol=1e-5)
                    methods.append(result)
                    checks.append(check)
            values.append(np.stack(methods))
    return dict(values=np.stack(values), image_ids=ids, task_ids=tasks, labels=labels,
                pairs=pairs, weights=weights, donors=donors, checks=checks,
                overlaps=overlaps, logit_scale=scale)


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    budgets = (49, 98)
    panels = [read_panel(source, budget, args.step, args.seeds)
              for source, budget in zip((args.source49, args.source98), budgets)]
    for key in ("image_ids", "task_ids", "labels", "pairs", "donors"):
        np.testing.assert_array_equal(panels[0][key], panels[1][key])
    np.testing.assert_allclose(panels[0]["weights"], panels[1]["weights"], atol=1e-15, rtol=0)
    base = panels[0]
    method_names = tuple(f"budget{budget}_{method}_{context}"
                         for budget in budgets for method in METHODS for context in CONTEXTS)
    method_values = np.concatenate([panel["values"].reshape(
        len(args.seeds), len(METHODS) * len(CONTEXTS), len(base["weights"]), len(METRICS))
        for panel in panels], axis=1)
    contrast_names = tuple(f"budget{budget}_{name}" for budget in budgets for name in CONTRAST_NAMES)
    contrast_values = np.concatenate([exchange_contrasts(panel["values"]) for panel in panels], axis=1)
    primary = tuple(f"budget{budget}_mapping_change_direct_vs_kl" for budget in budgets)
    methods, method_seeds, _, _ = summarize(method_values, base["weights"], base["image_ids"],
        method_names, args.replicates, args.bootstrap_seed)
    contrasts, contrast_seeds, contributions, totals = summarize(contrast_values, base["weights"],
        base["image_ids"], contrast_names, args.replicates, args.bootstrap_seed, primary=primary)
    summary = dict(budgets=budgets, seeds=args.seeds, rows=len(base["weights"]),
        images=len(set(base["image_ids"])), tasks=len(set(base["task_ids"])), donors_per_target=3,
        methods=methods, contrasts=contrasts, metrics=METRICS, primary_contrasts=primary,
        primary_metric="accuracy", bootstrap_replicates=args.replicates, bootstrap_seed=args.bootstrap_seed,
        interval_scope="Adjustment across two budgets for primary mapping-change comparisons; other intervals are ordinary 95%.",
        exchange_definition="Keep image and target candidates fixed. Re-score each of the other three task-conditioned outputs against the target candidates, then average the three outcomes.",
        decomposition="C-U objective increment = mapping-effect change + (swapped-C objective change - U objective change). The latter is the remaining change after exchange.",
        direction="Larger own-minus-swapped accuracy/gaps favor correct mapping; lower CE is preferred.",
        weighting="Original equal-task/equal-class-side weights, with all task rows clustered by image.")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for kind, results in (("methods", methods), ("contrasts", contrasts)):
        records = []
        for name, metrics in results.items():
            for metric, value in metrics.items():
                adjusted = value.get("two_primary_adjusted_ci95", (None, None))
                records.append(dict(name=name, metric=metric, mean=value["mean"],
                    low=value["ci95"][0], high=value["ci95"][1],
                    primary_adjusted_low=adjusted[0], primary_adjusted_high=adjusted[1]))
        write_csv(args.output / (kind + ".csv"), records)
    seed_rows = []
    for kind, names, values in (("method", method_names, method_seeds),
                                ("contrast", contrast_names, contrast_seeds)):
        for si, seed in enumerate(args.seeds):
            for ni, name in enumerate(names):
                seed_rows.append(dict(seed=seed, kind=kind, name=name,
                    **{metric: float(values[si, ni, qi]) for qi, metric in enumerate(METRICS)}))
    write_csv(args.output / "per_seed.csv", seed_rows)
    overlap_rows = []
    for budget, panel in zip(budgets, panels):
        overlap_rows.extend(dict(budget=budget, **row) for row in panel["overlaps"])
        for stage in STAGES:
            overlap_rows.append(dict(budget=budget, seed="mean", stage=stage,
                retained_set_overlap=float(np.mean([row["retained_set_overlap"]
                    for row in panel["overlaps"] if row["stage"] == stage]))))
    write_csv(args.output / "selection_overlap.csv", overlap_rows)
    audit = dict(donor_count=3, all_images_have_four_distinct_tasks=True,
        budget_row_alignment=True, own_outcomes_match_existing_analysis=True,
        decomposition_identity_passed=True, checks={str(b):p["checks"] for b,p in zip(budgets,panels)})
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    np.savez_compressed(args.output / "paired_exchange_outcomes.npz", outcomes=method_values,
        methods=method_names, contrasts=contrast_names, metrics=METRICS, seeds=args.seeds,
        weights=base["weights"], image_ids=base["image_ids"], task_ids=base["task_ids"],
        labels=base["labels"], pairs=base["pairs"], donors=base["donors"],
        image_contributions=contributions, image_weights=totals)
    print(json.dumps({name:contrasts[name]["accuracy"] for name in primary}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source49", type=Path, required=True)
    parser.add_argument("--source98", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, default=1200)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2717)
    main(parser.parse_args())
