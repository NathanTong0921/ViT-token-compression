"""Analyze equal-displacement shared-ranking interventions on validation tasks."""

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from analyze_core_protection import (METRICS, exchange_outcomes, grouped_variants,
                                     save_cases, selection_diagnostics, summarize)
from analyze_selector_ablations import binary_scores
from analyze_selector_budgets import weighted_estimates
from analyze_selector_scale import panel_weights, read_rows
from evaluate_selectors import write_rows


def validate_interventions(saved):
    """Confirm actual displacement and identical removals on the saved masks."""
    variants = list(saved["variants"])
    masks = saved["patch_indices"]
    membership = np.zeros((*masks.shape[:-1], int(masks.max()) + 1), dtype=bool)
    np.put_along_axis(membership, masks.astype(int), True, axis=-1)
    c, u = membership[variants.index("conditioned")], membership[variants.index("wide")]
    common = c & u
    np.testing.assert_array_equal(membership.sum(axis=-1), np.full(masks.shape[:2], masks.shape[-1]))
    assert (membership | ~common[None]).all()
    for pi, percentage in enumerate(saved["percentages"]):
        expected = saved["replacement_counts"][pi]
        names = (["conditioned"] if percentage == 0 else ["wide"] if percentage == 100 else
                 [name for name in variants if name.startswith((f"top_{percentage}", f"random_{percentage}_"))])
        expected_removed = np.zeros_like(c)
        positions = saved["removed_indices"][pi]
        row, col = np.nonzero(positions >= 0)
        expected_removed[row, positions[row, col]] = True
        for name in names:
            vi = variants.index(name)
            np.testing.assert_array_equal(c & ~membership[vi], expected_removed)
            np.testing.assert_array_equal((membership[vi] & ~c).sum(axis=-1), expected)
            np.testing.assert_array_equal((c & ~membership[vi]).sum(axis=-1), expected)
            positions = saved["added_indices"][vi]
            expected_added = np.zeros_like(c)
            row, col = np.nonzero(positions >= 0)
            expected_added[row, positions[row, col]] = True
            np.testing.assert_array_equal(membership[vi] & ~c, expected_added)
    np.testing.assert_array_equal(saved["replacement_counts"][-1], (u & ~c).sum(axis=-1))
    return True


def analyze(source, output, replicates=2000):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((source / "summary.json").read_text())
    seeds, variants = protocol["training_seeds"], protocol["variants"]
    base = read_rows(source / "examples.csv")
    ix = np.asarray([int(r["image_index"]) for r in base])
    ids, tasks = np.asarray([r["image_id"] for r in base]), np.asarray([r["task_id"] for r in base])
    labels = np.asarray([int(r["class_id"]) for r in base])
    pairs = np.asarray([[int(r["class_a"]), int(r["class_b"])] for r in base])
    competitors = np.where(labels == pairs[:, 0], pairs[:, 1], pairs[:, 0])
    weights = panel_weights(base, "all")
    np.testing.assert_allclose(weights, [float(r["task_class_weight"]) for r in base], atol=1e-12, rtol=0)
    with np.load(source / "evidence.npz") as saved:
        text, full_scores = saved["text_features"].astype(float), saved["full_scores"][ix].astype(float)
        np.testing.assert_array_equal(saved["image_ids"][ix], ids)
        np.testing.assert_array_equal(saved["labels"][ix], labels)
    score_runs, mask_runs, count_runs = [], [], []
    for seed in seeds:
        with np.load(source / f"seed_{seed}.npz") as saved:
            np.testing.assert_array_equal(saved["variants"], variants)
            validate_interventions(saved)
            score_runs.append(saved["cosine_scores"].astype(float))
            mask_runs.append(saved["patch_indices"].copy())
            count_runs.append(saved["replacement_counts"].copy())
    scores, masks, replacement_counts = np.stack(score_runs), np.stack(mask_runs), np.stack(count_runs)
    raw = binary_scores(scores.reshape(-1, len(base), len(text)), labels, competitors, text)
    raw = raw.reshape(len(seeds), len(variants), len(base), 2)
    values = grouped_variants(raw, variants)
    classification, differences = summarize(ids, weights, values, seeds, replicates, 2241)
    # Eight predeclared comparisons: four dataset/budget settings x two fractions.
    for name in ("top_50_minus_random_50", "top_75_minus_random_75"):
        classification["contrasts"][name].update(weighted_estimates(
            ids, weights, differences[name].mean(axis=0), METRICS, replicates, 2241, family_size=8))
    classification["full_model_reference"] = weighted_estimates(ids, weights,
        binary_scores(full_scores[None], labels, competitors, text)[0], METRICS, replicates, 2241)
    metadata = {"image_id": ids, "task_id": tasks, "class_id": labels,
                "class_a": pairs[:, 0], "class_b": pairs[:, 1]}
    save_cases(output, "classification", metadata, {**values, **differences}, seeds, weights)
    np.savez_compressed(output / "raw_classification_outcomes.npz", training_seeds=seeds,
                        variants=variants, outcomes=raw, metric_names=METRICS)

    by_image = defaultdict(list)
    for row, image_id in enumerate(ids):
        by_image[image_id].append(row)
    pair_indices = [pair for image_id in sorted(by_image) for pair in combinations(by_image[image_id], 2)]
    left, right = np.asarray(pair_indices, dtype=int).T
    np.testing.assert_array_equal(labels[left], labels[right])
    exchange_ids = ids[left]
    unique_ids, counts = np.unique(exchange_ids, return_counts=True)
    count_lookup = dict(zip(unique_ids, counts))
    exchange_weights = np.asarray([1 / (len(unique_ids) * count_lookup[i]) for i in exchange_ids])
    own_raw, swapped_raw = exchange_outcomes(scores, labels, competitors, text, left, right)
    own, swapped = grouped_variants(own_raw, variants), grouped_variants(swapped_raw, variants)
    effects = {name: own[name] - swapped[name] for name in own}
    np.testing.assert_array_equal(effects["wide"], np.zeros_like(effects["wide"]))
    exchange, exchange_differences = summarize(exchange_ids, exchange_weights, effects, seeds, replicates, 2242)
    exchange["correct_descriptor_table"] = summarize(exchange_ids, exchange_weights, own, seeds, replicates, 2242)[0]["table"]
    exchange["swapped_descriptor_table"] = summarize(exchange_ids, exchange_weights, swapped, seeds, replicates, 2242)[0]["table"]
    exchange.update(image_count=len(unique_ids), task_pair_count=len(left))
    exchange_meta = {"image_id": exchange_ids, "task_a": tasks[left], "task_b": tasks[right], "class_id": labels[left]}
    exchange_values = {**effects, **exchange_differences,
                       **{f"{name}_own": v for name, v in own.items()},
                       **{f"{name}_swapped": v for name, v in swapped.items()}}
    save_cases(output, "exchange", exchange_meta, exchange_values, seeds, exchange_weights)
    np.savez_compressed(output / "raw_exchange_outcomes.npz", training_seeds=seeds, variants=variants,
                        own=own_raw, swapped=swapped_raw, left_rows=left, right_rows=right, metric_names=METRICS)
    replacements, disagreement = selection_diagnostics(masks, variants, left, right)
    token_diagnostics = {"classification_replacements": [], "exchange_replacements": []}
    for name in replacements:
        token_diagnostics["classification_replacements"].append({"method": name,
            **weighted_estimates(ids, weights, replacements[name].mean(axis=0),
                ("replaced_tokens_vs_conditioned", "replaced_tokens_vs_wide"), replicates, 2241)})
        token_diagnostics["exchange_replacements"].append({"method": name,
            **weighted_estimates(exchange_ids, exchange_weights, disagreement[name].mean(axis=0),
                                 ("changed_tokens_between_tasks",), replicates, 2242)})
    for percentage in (50, 75):
        np.testing.assert_array_equal(replacements[f"top_{percentage}"], replacements[f"random_{percentage}"])
    count_rows = [{"image_id": image_id, "task_id": task, "seed": seed,
                   **{f"d_{p}": int(replacement_counts[si, pi, row])
                      for pi, p in enumerate(protocol["replacement_percentages"])}}
                  for si, seed in enumerate(seeds) for row, (image_id, task) in enumerate(zip(ids, tasks))]
    write_rows(output / "actual_replacement_counts.csv", count_rows)
    np.savez_compressed(output / "token_replacements.npz", training_seeds=seeds,
                        replacement_counts=replacement_counts,
                        **{f"{name}_classification": value for name, value in replacements.items()},
                        **{f"{name}_exchange": value for name, value in disagreement.items()})
    result = {"setting": protocol["setting"], "source": str(source.resolve()),
              "evaluation_git_commit": protocol["git_commit"], "training_seeds": seeds,
              "random_addition_seeds": protocol["random_addition_seeds"],
              "replacement_percentages": protocol["replacement_percentages"],
              "classification_image_count": len(set(ids)), "classification_task_count": len(set(tasks)),
              "classification_image_task_count": len(base), "bootstrap_replicates": replicates,
              "primary_family_size": 8, "intervention_audit": "Every saved mask has fixed k; C intersection U is retained; each fraction has identical removed positions and actual displacement for top and every random arm. Endpoints match C and U.",
              "aggregation": "Classification: equal tasks and class sides, then equal images within side. Exchange: equal images, then within-image task pairs. Average outcomes over random additions and fixed training seeds before paired image bootstrap.",
              "primary": "Top-ranked minus random additions at 50% and 75% of the disagreement set, all four Pets/Flowers49/98 settings: eight contrasts. Margin primary; accuracy secondary. Ordinary and Bonferroni-adjusted intervals separately for each metric.",
              "uncertainty": "Intervals condition on these validation tasks, cached selectors, training seeds, and random-addition seeds. Equal-displacement follow-up motivated by the earlier core-protection pilot.",
              "classification": classification, "exchange": exchange, "token_diagnostics": token_diagnostics}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()
    result = analyze(args.source, args.output, args.replicates)
    print(json.dumps({"setting": result["setting"], "primary_family_size": result["primary_family_size"],
                      "contrasts": result["classification"]["contrasts"]}, indent=2))
