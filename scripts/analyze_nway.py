"""Analyze the paired two/five-candidate development experiment."""

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from analyze_selector_budgets import weighted_estimates


METRICS = ("accuracy", "margin", "geometric_margin")
ROW_KEYS = ("image_ids", "task_ids", "candidates", "labels", "weights")


def outcomes(scores, candidates, labels, text):
    """Reconstruct accuracy and both multiclass margins from all-class scores."""
    scores, text = np.asarray(scores, float), np.asarray(text, float)
    candidates, labels = np.asarray(candidates), np.asarray(labels)
    candidates = np.sort(candidates, axis=1)
    rows = np.arange(len(labels))
    included = candidates == labels[:, None]
    assert np.all(included.sum(axis=1) == 1)
    candidate_scores = np.take_along_axis(scores, candidates, axis=1)
    predicted = candidates[rows, candidate_scores.argmax(axis=1)]
    competing = np.where(included, -np.inf, candidate_scores)
    raw_margin = scores[rows, labels] - competing.max(axis=1)
    norms = np.linalg.norm(text[labels, None] - text[candidates], axis=-1)
    norms[included] = 1.
    geometric = np.where(included, np.inf,
                         (scores[rows, labels, None] - candidate_scores) / norms).min(axis=1)
    return np.column_stack([predicted == labels, raw_margin, geometric])


def load_setting(folder):
    """Align all seeds and methods by image/task, independent of file row order."""
    files = [folder / "references.npz", *sorted(folder.glob("seed_*.npz"))]
    records, text, base = {}, None, None
    for path in files:
        with np.load(path) as saved:
            if text is None:
                text = saved["text_features"].copy()
            else:
                np.testing.assert_array_equal(text, saved["text_features"])
            for method in np.unique(saved["method"]):
                for seed in np.unique(saved["seed"][saved["method"] == method]):
                    selected = np.flatnonzero((saved["method"] == method) & (saved["seed"] == seed))
                    order = np.lexsort((saved["task_ids"][selected], saved["image_ids"][selected]))
                    selected = selected[order]
                    panel = {key: saved[key][selected].copy() for key in ROW_KEYS}
                    if base is None:
                        base = panel
                    else:
                        for key in ROW_KEYS:
                            np.testing.assert_array_equal(panel[key], base[key])
                    values = outcomes(saved["cosine_scores"][selected], panel["candidates"],
                                      panel["labels"], text)
                    np.testing.assert_array_equal(values[:, 0], saved["correct"][selected])
                    for column, name in enumerate(METRICS[1:], 1):
                        np.testing.assert_allclose(values[:, column], saved[name][selected],
                                                   rtol=1e-5, atol=1e-7)
                    key = str(method), int(seed)
                    assert key not in records
                    records[key] = {"outcomes": values,
                                    "scores": saved["cosine_scores"][selected].copy()}
    assert base is not None
    identities = list(zip(base["image_ids"], base["task_ids"]))
    assert len(set(identities)) == len(identities)
    _, groups, counts = np.unique(np.column_stack([base["task_ids"], base["labels"].astype(str)]),
                                  axis=0, return_inverse=True, return_counts=True)
    task_count, way = len(np.unique(base["task_ids"])), base["candidates"].shape[1]
    assert len(counts) == task_count * way
    np.testing.assert_allclose(base["weights"], 1 / (task_count * way * counts[groups]),
                               atol=1e-12, rtol=0)
    np.testing.assert_allclose(base["weights"].sum(), 1., atol=1e-12)
    runs = {method: sorted(seed for m, seed in records if m == method)
            for method in sorted({key[0] for key in records})}
    assert runs["conditioned"] == runs["image_only_wide"]
    arrays = {method: np.stack([records[method, seed]["outcomes"] for seed in seeds])
              for method, seeds in runs.items()}
    return {"panel": base, "text": text, "records": records, "seeds": runs, "arrays": arrays}


def interval_stats(point, bootstrap, names, family_size):
    quantiles = np.quantile(bootstrap, [.025, .975, .025 / family_size, 1 - .025 / family_size], axis=0)
    return {name: {"mean": float(point[j]), "ci95": quantiles[:2, j].tolist(),
                   "bonferroni_family_ci95": quantiles[2:, j].tolist(),
                   "individual_interval_level": 1 - .05 / family_size}
            for j, name in enumerate(names)}


def image_totals(ids, weights, values):
    unique, inverse = np.unique(ids, return_inverse=True)
    numerator = np.zeros((len(unique), values.shape[1]))
    np.add.at(numerator, inverse, weights[:, None] * values)
    denominator = np.bincount(inverse, weights=weights, minlength=len(unique))
    return unique, numerator, denominator


def way_interaction(low, high, replicates=2000, seed=2613, family_size=2):
    """Pair images across candidate counts; tasks need not be identical."""
    quantities = []
    for setting in (low, high):
        panel = setting["panel"]
        delta = setting["arrays"]["conditioned"].mean(axis=0) - setting["arrays"]["image_only_wide"].mean(axis=0)
        quantities.append(image_totals(panel["image_ids"], panel["weights"], delta))
    ids, lower_n, lower_d = quantities[0]
    higher_ids, higher_n, higher_d = quantities[1]
    np.testing.assert_array_equal(ids, higher_ids)
    point = higher_n.sum(axis=0) / higher_d.sum() - lower_n.sum(axis=0) / lower_d.sum()
    rng = np.random.default_rng(seed)
    boot = np.empty((replicates, len(METRICS)))
    for start in range(0, replicates, 100):
        draws = rng.integers(len(ids), size=(min(100, replicates - start), len(ids)))
        boot[start:start + len(draws)] = (
            higher_n[draws].sum(axis=1) / higher_d[draws].sum(axis=1)[:, None]
            - lower_n[draws].sum(axis=1) / lower_d[draws].sum(axis=1)[:, None])
    return interval_stats(point, boot, METRICS, family_size)


def exchange_losses(setting):
    """Exchange C's selections while retaining each actual classification task."""
    panel, text, records = setting["panel"], setting["text"], setting["records"]
    ids, inverse = np.unique(panel["image_ids"], return_inverse=True)
    first, second, image_index = [], [], []
    for image in range(len(ids)):
        selected = np.flatnonzero(inverse == image)
        for a, b in combinations(selected, 2):
            first.append(a)
            second.append(b)
            image_index.append(image)
    first, second, image_index = [np.asarray(v, dtype=np.int64) for v in (first, second, image_index)]
    np.testing.assert_array_equal(panel["labels"][first], panel["labels"][second])
    counts = np.bincount(image_index, minlength=len(ids))
    eligible = counts > 0
    by_seed, shared_difference = [], 0.
    for seed in setting["seeds"]["conditioned"]:
        own = records["conditioned", seed]["outcomes"]
        scores = records["conditioned", seed]["scores"]
        ab = outcomes(scores[first], panel["candidates"][second], panel["labels"][second], text)
        ba = outcomes(scores[second], panel["candidates"][first], panel["labels"][first], text)
        losses = .5 * (own[first] + own[second] - ab - ba)
        totals = np.zeros((len(ids), len(METRICS)))
        np.add.at(totals, image_index, losses)
        by_seed.append(totals[eligible] / counts[eligible, None])
        shared = records["image_only_wide", seed]["scores"]
        shared_difference = max(shared_difference, float(np.max(np.abs(shared[first] - shared[second]))))
    return {"image_ids": ids[eligible], "seeds": np.array(setting["seeds"]["conditioned"]),
            "per_seed": np.stack(by_seed), "pair_counts": counts[eligible],
            "pair_count": len(first), "shared_score_exchange_max_difference": shared_difference}


def flatten_estimates(dataset, way, contrast, estimates, family_size):
    rows = []
    for metric, stats in estimates.items():
        scale = 100 if metric == "accuracy" else 1
        family = stats.get("bonferroni_family_ci95", stats["ci95"])
        rows.append({"dataset": dataset, "way": way, "contrast": contrast, "metric": metric,
                     "unit": "percentage_points" if metric == "accuracy" else "cosine_margin",
                     "mean": scale * stats["mean"], "ci95_low": scale * stats["ci95"][0],
                     "ci95_high": scale * stats["ci95"][1], "family_size": family_size,
                     "family_ci95_low": scale * family[0], "family_ci95_high": scale * family[1]})
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(inputs, output, replicates=2000, seed=2613):
    output.mkdir(parents=True, exist_ok=True)
    family = 2 * len(inputs)
    summary = {"scope": "Development images and unseen candidate combinations; two/five-way, B/16, block 6, 98 tokens.",
               "final_test_used": False, "bootstrap_replicates": replicates, "bootstrap_seed": seed,
               "classification_weighting": "Equal tasks and classes within tasks, then equal class images; average seeds before image-cluster bootstrap.",
               "exchange_weighting": "Average task pairs and fixed seeds within each image, then equal images.",
               "intervals": "Image-cluster percentile bootstrap; conditions on task collection and trained seeds. Adjusted intervals cover the dataset/way settings per metric and named contrast.",
               "settings": {}, "way_interactions": {}}
    tables, contrasts, interactions, exchanges, seed_rows = [], [], [], [], []
    for dataset, root in inputs.items():
        retained = {}
        for way in (2, 5):
            name = f"{dataset}_{way}way"
            data = load_setting(root / "eval" / f"{way}way")
            retained[way] = data
            panel = data["panel"]
            ids, weights, arrays = panel["image_ids"], panel["weights"], data["arrays"]
            full = arrays["full_model"][0, :, 0]
            result = {"dataset": dataset, "way": way, "image_count": len(np.unique(ids)),
                      "task_count": len(np.unique(panel["task_ids"])), "image_task_count": len(ids),
                      "seeds": data["seeds"]["conditioned"], "table": []}
            for method, values in arrays.items():
                stats = weighted_estimates(ids, weights, values.mean(axis=0), METRICS, replicates, seed)
                repairs = ((values[:, :, 0] == 1) & (full[None] == 0)).mean(axis=0)
                errors = ((values[:, :, 0] == 0) & (full[None] == 1)).mean(axis=0)
                row = {"method": method, **stats, "repairs_vs_full": float(weights @ repairs),
                       "new_errors_vs_full": float(weights @ errors)}
                result["table"].append(row)
                tables.append({"dataset": dataset, "way": way, "method": method,
                               "accuracy_percent": 100 * stats["accuracy"]["mean"],
                               "accuracy_ci95_low_percent": 100 * stats["accuracy"]["ci95"][0],
                               "accuracy_ci95_high_percent": 100 * stats["accuracy"]["ci95"][1],
                               "margin": stats["margin"]["mean"],
                               "geometric_margin": stats["geometric_margin"]["mean"],
                               "repairs_vs_full_pp": 100 * row["repairs_vs_full"],
                               "new_errors_vs_full_pp": 100 * row["new_errors_vs_full"]})
                for s, values_s in zip(data["seeds"][method], values):
                    point = weights @ values_s
                    seed_rows.append({"dataset": dataset, "way": way, "method": method, "seed": s,
                                      **{metric: float(point[j]) * (100 if metric == "accuracy" else 1)
                                         for j, metric in enumerate(METRICS)}})
            delta = arrays["conditioned"].mean(axis=0) - arrays["image_only_wide"].mean(axis=0)
            result["conditioned_minus_image_only_wide"] = weighted_estimates(
                ids, weights, delta, METRICS, replicates, seed, family)
            contrasts.extend(flatten_estimates(dataset, way, "C_minus_U", result["conditioned_minus_image_only_wide"], family))
            exchange = exchange_losses(data)
            exchange_values = exchange["per_seed"].mean(axis=0)
            exchange_stats = weighted_estimates(exchange["image_ids"], np.ones(len(exchange_values)),
                                                exchange_values, METRICS, replicates, seed, family)
            result["exchange_own_minus_swapped"] = exchange_stats
            result["exchange_image_count"] = len(exchange_values)
            result["exchange_pair_count"] = exchange["pair_count"]
            result["shared_score_exchange_max_difference"] = exchange["shared_score_exchange_max_difference"]
            exchanges.extend(flatten_estimates(dataset, way, "own_minus_swapped", exchange_stats, family))
            np.savez_compressed(output / f"{name}_exchange.npz", image_ids=exchange["image_ids"],
                                seeds=exchange["seeds"], per_seed=exchange["per_seed"],
                                pair_counts=exchange["pair_counts"], metrics=np.asarray(METRICS))
            np.savez_compressed(output / f"{name}_paired_outcomes.npz", **panel,
                                **{f"{method}_per_seed": values for method, values in arrays.items()})
            summary["settings"][name] = result
            print(json.dumps({"setting": name, "C_minus_U": result["conditioned_minus_image_only_wide"]}), flush=True)
        interaction = way_interaction(retained[2], retained[5], replicates, seed, len(inputs))
        summary["way_interactions"][dataset] = interaction
        interactions.extend(flatten_estimates(dataset, "5_minus_2", "C_minus_U_interaction", interaction, len(inputs)))
    for name, rows in (("classification", tables), ("contrasts", contrasts), ("way_interactions", interactions),
                       ("exchange", exchanges), ("per_seed", seed_rows)):
        write_csv(output / f"{name}.csv", rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, metavar="DATASET=ROOT")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2613)
    args = parser.parse_args()
    roots = {name: Path(path) for name, path in (item.split("=", 1) for item in args.inputs)}
    run(roots, args.output, args.bootstrap_replicates, args.seed)
