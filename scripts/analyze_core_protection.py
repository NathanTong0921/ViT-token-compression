"""Paired classification and task-exchange analysis of core-protection replay."""

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from analyze_selector_ablations import binary_scores
from analyze_selector_budgets import weighted_estimates
from analyze_selector_scale import panel_weights, read_rows
from evaluate_selectors import write_rows


METRICS = ("accuracy", "margin")


def grouped_variants(values, variants):
    """Average random-core outcomes, not logits or predictions before scoring."""
    output = {name: values[:, variants.index(name)] for name in ("conditioned", "wide", "top_50", "top_75")}
    for percentage in (50, 75):
        indices = [i for i, name in enumerate(variants) if name.startswith(f"random_{percentage}_")]
        output[f"random_{percentage}"] = values[:, indices].mean(axis=1)
    return output


def exchange_outcomes(scores, labels, competitors, text, left, right):
    """Score each task with its own selected feature and the other task's feature."""
    shape = scores.shape
    flat = scores.reshape(-1, shape[-2], shape[-1])
    own = (binary_scores(flat[:, left], labels[left], competitors[left], text)
           + binary_scores(flat[:, right], labels[right], competitors[right], text)) / 2
    swapped = (binary_scores(flat[:, right], labels[left], competitors[left], text)
               + binary_scores(flat[:, left], labels[right], competitors[right], text)) / 2
    return own.reshape(*shape[:-2], len(left), 2), swapped.reshape(*shape[:-2], len(left), 2)


def selection_diagnostics(masks, variants, left, right):
    """Count actual replacements: protecting q need not change q original tokens."""
    membership = np.zeros((*masks.shape[:-1], int(masks.max()) + 1), dtype=bool)
    np.put_along_axis(membership, masks.astype(int), True, axis=-1)
    k = masks.shape[-1]
    replacements = np.stack([k - (membership & membership[:, variants.index(name):variants.index(name) + 1]).sum(axis=-1)
                             for name in ("conditioned", "wide")], axis=-1)
    disagreement = k - (membership[:, :, left] & membership[:, :, right]).sum(axis=-1)
    return grouped_variants(replacements, variants), grouped_variants(disagreement[..., None], variants)


def summarize(ids, weights, values, seeds, replicates, bootstrap_seed, primary=False):
    table = []
    for name, value in values.items():
        points = np.einsum("n,snm->sm", weights, value) / weights.sum()
        table.append({"method": name, **weighted_estimates(ids, weights, value.mean(axis=0), METRICS,
                                                            replicates, bootstrap_seed),
                      "per_seed": [{"seed": seed, **dict(zip(METRICS, p.tolist()))}
                                   for seed, p in zip(seeds, points)]})
    comparisons = [("top_50", "random_50"), ("top_75", "random_75"), ("wide", "conditioned")]
    comparisons += [(f"{kind}_{p}", baseline) for kind in ("top", "random")
                    for p in (50, 75) for baseline in ("conditioned", "wide")]
    contrasts, differences = {}, {}
    for a, b in comparisons:
        name = f"{a}_minus_{b}"
        delta = values[a] - values[b]
        family = 4 if primary and name in ("top_50_minus_random_50", "top_75_minus_random_75") else 1
        contrasts[name] = weighted_estimates(ids, weights, delta.mean(axis=0), METRICS,
                                              replicates, bootstrap_seed, family)
        points = np.einsum("n,snm->sm", weights, delta) / weights.sum()
        contrasts[name]["per_seed"] = [{"seed": seed, **dict(zip(METRICS, point.tolist()))}
                                        for seed, point in zip(seeds, points)]
        differences[name] = delta
    return {"table": table, "contrasts": contrasts}, differences


def save_cases(output, name, metadata, values, seeds, weights):
    """Keep paired row outcomes and image numerator/denominator contributions."""
    arrays = {**metadata, "weight": weights, "training_seeds": np.asarray(seeds)}
    columns = {**metadata, "weight": weights}
    for method, value in values.items():
        arrays[f"{method}_per_seed"] = value
        for j, metric in enumerate(METRICS):
            columns[f"{method}_{metric}"] = value.mean(axis=0)[:, j]
    np.savez_compressed(output / f"{name}_cases.npz", **arrays)
    write_rows(output / f"{name}_cases.csv", [dict(zip(columns, row)) for row in zip(*columns.values())])
    unique, inverse = np.unique(metadata["image_id"], return_inverse=True)
    denominator = np.bincount(inverse, weights=weights)
    per_image = {"image_id": unique, "weight": denominator}
    for key, value in columns.items():
        if key not in metadata and key != "weight":
            per_image[f"{key}_weighted_sum"] = np.bincount(inverse, weights=weights * value)
    write_rows(output / f"{name}_images.csv", [dict(zip(per_image, row)) for row in zip(*per_image.values())])


def analyze(source, output, replicates=2000):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((source / "summary.json").read_text())
    seeds, variants = protocol["training_seeds"], protocol["variants"]
    base = read_rows(source / "examples.csv")
    ix = np.asarray([int(r["image_index"]) for r in base])
    ids = np.asarray([r["image_id"] for r in base])
    tasks = np.asarray([r["task_id"] for r in base])
    labels = np.asarray([int(r["class_id"]) for r in base])
    pairs = np.asarray([[int(r["class_a"]), int(r["class_b"])] for r in base])
    competitors = np.where(labels == pairs[:, 0], pairs[:, 1], pairs[:, 0])
    weights = panel_weights(base, "all")
    np.testing.assert_allclose(weights, [float(r["task_class_weight"]) for r in base], rtol=0, atol=1e-12)
    with np.load(source / "evidence.npz") as saved:
        text, full_scores = saved["text_features"].astype(float), saved["full_scores"][ix].astype(float)
        np.testing.assert_array_equal(saved["labels"][ix], labels)
        np.testing.assert_array_equal(saved["image_ids"][ix], ids)
    score_runs, masks = [], []
    for seed in seeds:
        with np.load(source / f"seed_{seed}.npz") as saved:
            np.testing.assert_array_equal(saved["variants"], variants)
            score_runs.append(saved["cosine_scores"].astype(float))
            masks.append(saved["patch_indices"].copy())
    scores, masks = np.stack(score_runs), np.stack(masks)
    assert scores.shape[2] == len(base) and np.isfinite(scores).all()
    raw = binary_scores(scores.reshape(-1, len(base), len(text)), labels, competitors, text)
    raw = raw.reshape(len(seeds), len(variants), len(base), 2)
    values = grouped_variants(raw, variants)
    classification, differences = summarize(ids, weights, values, seeds, replicates, 2217, primary=True)
    full_values = binary_scores(full_scores[None], labels, competitors, text)[0]
    classification["full_model_reference"] = weighted_estimates(ids, weights, full_values, METRICS, replicates, 2217)
    metadata = {"image_id": ids, "task_id": tasks, "class_id": labels,
                "class_a": pairs[:, 0], "class_b": pairs[:, 1]}
    save_cases(output, "classification", metadata, {**values, **differences}, seeds, weights)
    np.savez_compressed(output / "raw_classification_outcomes.npz", training_seeds=seeds,
                        variants=variants, outcomes=raw, metric_names=METRICS)

    by_image = defaultdict(list)
    for i, image_id in enumerate(ids):
        by_image[image_id].append(i)
    pair_indices = [pair for image_id in sorted(by_image) for pair in combinations(by_image[image_id], 2)]
    left, right = np.asarray(pair_indices, dtype=int).T
    np.testing.assert_array_equal(labels[left], labels[right])
    exchange_ids = ids[left]
    _, counts = np.unique(exchange_ids, return_counts=True)
    counts_by_id = dict(zip(np.unique(exchange_ids), counts))
    exchange_weights = np.asarray([1 / (len(counts) * counts_by_id[i]) for i in exchange_ids])
    own_raw, swapped_raw = exchange_outcomes(scores, labels, competitors, text, left, right)
    own, swapped = grouped_variants(own_raw, variants), grouped_variants(swapped_raw, variants)
    effects = {name: own[name] - swapped[name] for name in own}
    # The wide endpoint is task-independent by construction and exact feature reuse.
    np.testing.assert_array_equal(effects["wide"], np.zeros_like(effects["wide"]))
    np.testing.assert_array_equal(masks[:, variants.index("wide"), left], masks[:, variants.index("wide"), right])
    exchange, exchange_differences = summarize(exchange_ids, exchange_weights, effects, seeds, replicates, 2218)
    exchange["correct_descriptor_table"] = summarize(exchange_ids, exchange_weights, own, seeds, replicates, 2218)[0]["table"]
    exchange["swapped_descriptor_table"] = summarize(exchange_ids, exchange_weights, swapped, seeds, replicates, 2218)[0]["table"]
    exchange["image_count"], exchange["task_pair_count"] = len(counts), len(left)
    exchange_meta = {"image_id": exchange_ids, "task_a": tasks[left], "task_b": tasks[right], "class_id": labels[left]}
    exchange_values = {**effects, **exchange_differences,
                       **{f"{name}_own": v for name, v in own.items()},
                       **{f"{name}_swapped": v for name, v in swapped.items()}}
    save_cases(output, "exchange", exchange_meta, exchange_values, seeds, exchange_weights)
    np.savez_compressed(output / "raw_exchange_outcomes.npz", training_seeds=seeds, variants=variants,
                        own=own_raw, swapped=swapped_raw, left_rows=left, right_rows=right, metric_names=METRICS)
    replacements, disagreement = selection_diagnostics(masks, variants, left, right)
    token_diagnostics = {"classification_replacements": [], "exchange_replacements": []}
    token_rows = []
    for name, value in replacements.items():
        names = ("replaced_tokens_vs_conditioned", "replaced_tokens_vs_wide")
        token_diagnostics["classification_replacements"].append({"method": name,
            **weighted_estimates(ids, weights, value.mean(axis=0), names, replicates, 2217)})
        token_rows.extend({"image_id": image_id, "task_id": task, "method": name,
                           **dict(zip(names, row.tolist()))}
                          for image_id, task, row in zip(ids, tasks, value.mean(axis=0)))
        token_diagnostics["exchange_replacements"].append({"method": name,
            **weighted_estimates(exchange_ids, exchange_weights, disagreement[name].mean(axis=0),
                                 ("changed_tokens_between_tasks",), replicates, 2218)})
    write_rows(output / "token_replacements.csv", token_rows)
    np.savez_compressed(output / "token_replacements.npz", training_seeds=seeds,
                        **{f"{name}_classification": value for name, value in replacements.items()},
                        **{f"{name}_exchange": value for name, value in disagreement.items()})
    result = {"setting": protocol["setting"], "source": str(source.resolve()),
              "evaluation_git_commit": protocol["git_commit"], "training_seeds": seeds,
              "random_core_seeds": protocol["random_core_seeds"], "protected_counts": protocol["protected_counts"],
              "classification_image_count": len(set(ids)), "classification_task_count": len(set(tasks)),
              "classification_image_task_count": len(base), "bootstrap_replicates": replicates,
              "aggregation": "Classification: equal tasks, equal class sides, equal images within each side. Exchange: equal images then equal within-image task pairs. Random-core outcomes are averaged across mask seeds before fixed training seeds; paired bootstrap resamples images.",
              "primary": "Top-ranked core minus size-matched random core at 50% and 75%, across Pets49 and Flowers98: four contrasts per metric. Margin is the primary mechanism metric; accuracy is secondary. All fixed seeds and both settings are retained.",
              "uncertainty": "Intervals condition on the observed validation tasks and training seeds. Primary contrasts include Bonferroni-adjusted intervals for the four-setting/fraction family separately for each metric.",
              "classification": classification, "exchange": exchange, "token_diagnostics": token_diagnostics}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()
    result = analyze(args.source, args.output, args.replicates)
    print(json.dumps({"setting": result["setting"], "contrasts": result["classification"]["contrasts"]}, indent=2))


if __name__ == "__main__":
    main()
