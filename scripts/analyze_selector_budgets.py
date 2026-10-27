"""Paired development-set comparisons across matched token budgets."""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_selector_scale import (EXCHANGE_KEYS, named_estimates, panel_weights,
                                    read_rows, read_unseen_predictions)
from token_compression.selection_analysis import cluster_summary
from token_compression.selector_exchange import build_exchange_cases, summarize_exchange


METHODS = ("full_model", "random", "attention", "image_only", "conditioned", "label_informed_teacher")
METRICS = ("accuracy", "margin")


def weighted_estimates(ids, weights, values, names, replicates=2000, seed=2052, family_size=1):
    """Resample whole images, retaining task/class weights in a ratio estimator."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if values.ndim == 1:
        values = values[:, None]
    unique, inverse = np.unique(ids, return_inverse=True)
    denominator = np.bincount(inverse, weights=weights, minlength=len(unique))
    numerator = np.zeros((len(unique), values.shape[1]))
    np.add.at(numerator, inverse, weights[:, None] * values)
    point = numerator.sum(axis=0) / denominator.sum()
    rng = np.random.default_rng(seed)
    boot = np.empty((replicates, values.shape[1]))
    for start in range(0, replicates, 100):
        draws = rng.integers(len(unique), size=(min(100, replicates - start), len(unique)))
        boot[start:start + len(draws)] = numerator[draws].sum(axis=1) / denominator[draws].sum(axis=1)[:, None]
    low, high = np.quantile(boot, [.025, .975], axis=0)
    output = {name: {"mean": float(point[j]), "ci95": [float(low[j]), float(high[j])]}
              for j, name in enumerate(names)}
    if family_size > 1:
        lower, upper = np.quantile(boot, [.025 / family_size, 1 - .025 / family_size], axis=0)
        for j, name in enumerate(names):
            output[name]["bonferroni_family_ci95"] = [float(lower[j]), float(upper[j])]
            output[name]["individual_interval_level"] = 1 - .05 / family_size
    return output


def aligned_runs(rows, method, split, keys, expected_seeds=None):
    """Align every run by image/task identity before averaging seeds or masks."""
    selected = [r for r in rows if r["method"] == method and r["task_split"] == split]
    seeds = sorted({int(r["seed"]) for r in selected})
    assert seeds
    if expected_seeds is not None:
        assert seeds == sorted(expected_seeds)
    arrays = []
    for seed in seeds:
        run = [r for r in selected if int(r["seed"]) == seed]
        lookup = {(r["image_id"], r["task_id"]): r for r in run}
        assert len(run) == len(lookup) == len(keys)
        assert set(lookup) == set(keys)
        arrays.append([[float(lookup[key][m]) for m in ("correct", "margin")] for key in keys])
    return seeds, np.asarray(arrays, dtype=float)


def analyze_rows(rows_by_budget, training_seeds, random_seeds, replicates=2000, seed=2052):
    budgets = sorted(rows_by_budget)
    result = {
        "primary": "Conditioned minus image-only performance at each predeclared budget on unseen validation class combinations.",
        "budgets": budgets, "training_seeds": training_seeds, "random_mask_seeds": random_seeds,
        "aggregation": "Equal tasks and equal class sides within each task, then equal images within class. Average fixed training seeds or random-mask seeds before image resampling.",
        "uncertainty": f"Paired image-cluster bootstrap of weighted numerator/denominator totals. Intervals condition on these tasks and seeds. Bonferroni intervals cover the {len(budgets)} predeclared budgets separately for each metric, not all metrics jointly.",
        "bootstrap_replicates": replicates, "bootstrap_seed": seed, "task_splits": {}}
    for split in ("seen", "unseen"):
        base = [r for r in rows_by_budget[budgets[0]] if r["method"] == "full_model" and r["task_split"] == split]
        keys = [(r["image_id"], r["task_id"]) for r in base]
        assert len(set(keys)) == len(keys)
        ids = np.asarray([r["image_id"] for r in base])
        weights = panel_weights(base, "all")
        assert np.all(weights > 0)
        budget_results, differences, base_values = {}, {}, None
        for budget in budgets:
            rows = rows_by_budget[budget]
            arrays, run_ids = {}, {}
            for method in METHODS:
                expected = (training_seeds if method in ("conditioned", "image_only") else
                            random_seeds if method == "random" else [-1])
                run_ids[method], arrays[method] = aligned_runs(rows, method, split, keys, expected)
            if base_values is None:
                base_values = arrays["full_model"][0]
            else:
                np.testing.assert_allclose(arrays["full_model"][0], base_values, atol=1e-7, rtol=0)
            means = {method: data.mean(axis=0) for method, data in arrays.items()}
            table = []
            for method, data in arrays.items():
                estimates = weighted_estimates(ids, weights, means[method], METRICS, replicates, seed)
                full_correct = base_values[:, 0]
                repairs = (data[:, :, 0] == 1) & (full_correct[None] == 0)
                errors = (data[:, :, 0] == 0) & (full_correct[None] == 1)
                points = (weights[None, :, None] * data).sum(axis=1)
                table.append({
                    "method": method, "accuracy": estimates["accuracy"], "margin": estimates["margin"],
                    "weighted_repairs_vs_full_model": float((repairs * weights).sum(axis=1).mean()),
                    "weighted_new_errors_vs_full_model": float((errors * weights).sum(axis=1).mean()),
                    "run_seed_type": "training" if method in ("conditioned", "image_only") else "mask" if method == "random" else "deterministic",
                    "per_seed": [{"seed": s, "accuracy": float(point[0]), "margin": float(point[1])}
                                 for s, point in zip(run_ids[method], points)]})
            delta = arrays["conditioned"] - arrays["image_only"]
            differences[budget] = delta.mean(axis=0)
            primary = weighted_estimates(ids, weights, differences[budget], METRICS, replicates, seed, len(budgets))
            other = {f"{a}_minus_{b}": weighted_estimates(ids, weights, means[a] - means[b], METRICS, replicates, seed)
                     for a, b in (("conditioned", "attention"), ("image_only", "attention"), ("conditioned", "random"))}
            counts = []
            for i, s in enumerate(training_seeds):
                repaired = (arrays["conditioned"][i, :, 0] == 1) & (arrays["image_only"][i, :, 0] == 0)
                new_error = (arrays["conditioned"][i, :, 0] == 0) & (arrays["image_only"][i, :, 0] == 1)
                points = (weights[:, None] * delta[i]).sum(axis=0)
                counts.append({"seed": s, "accuracy_difference": float(points[0]), "margin_difference": float(points[1]),
                               "repair_image_task_cases": int(repaired.sum()), "new_error_image_task_cases": int(new_error.sum()),
                               "weighted_repair_fraction": float(np.dot(weights, repaired)),
                               "weighted_new_error_fraction": float(np.dot(weights, new_error))})
            budget_results[str(budget)] = {"table": table, "conditioned_minus_image_only": primary,
                                            "secondary_contrasts": other, "paired_seed_results": counts}
        interactions = {f"{budget}_minus_{budgets[0]}": weighted_estimates(
            ids, weights, differences[budget] - differences[budgets[0]], METRICS, replicates, seed)
            for budget in budgets[1:]}
        result["task_splits"][split] = {"image_count": len(set(ids)), "image_task_examples": len(keys),
                                         "task_count": len({key[1] for key in keys}),
                                         "budgets": budget_results, "secondary_change_in_conditioning_advantage": interactions}
    return result


def analyze_exchange(sources, output, training_seeds, margin_edges, replicates=2000, seed=2052):
    summaries, cases_by_budget = {}, {}
    for budget, source in sorted(sources.items()):
        full = read_unseen_predictions(source / "references" / "predictions.npz", full_only=True)
        learned = read_unseen_predictions(source / "full" / "predictions.npz")
        np.testing.assert_array_equal(full["text_features"], learned["text_features"])
        merged = {k: np.concatenate([full[k], learned[k]]) for k in EXCHANGE_KEYS}
        merged["text_features"] = full["text_features"]
        merged["row_id"] = np.arange(len(merged["seed"]))
        cases = build_exchange_cases(merged, training_seeds, ["unseen"])
        assert np.all(cases["keep_patch_tokens"] == budget)
        np.savez_compressed(output / f"budget_{budget}_exchange_cases.npz", **cases)
        cases_by_budget[budget] = cases
        summaries[str(budget)] = summarize_exchange(cases, margin_edges, replicates, seed)
    budgets = sorted(cases_by_budget)
    baseline = cases_by_budget[budgets[0]]
    def effect(cases, metric):
        return (cases[f"{metric}_aa"].astype(float) + cases[f"{metric}_bb"]
                - cases[f"{metric}_ba"] - cases[f"{metric}_ab"]) / 2
    changes = {}
    for budget in budgets[1:]:
        cases = cases_by_budget[budget]
        for key in ("image_ids", "task_a", "task_b", "seed"):
            np.testing.assert_array_equal(cases[key], baseline[key])
        values = np.column_stack([effect(cases, metric) - effect(baseline, metric) for metric in ("correct", "margin")])
        estimates = cluster_summary(cases["image_ids"], values, replicates, seed)
        changes[f"{budget}_minus_{budgets[0]}"] = named_estimates(estimates, METRICS)
    return {"scope": "Secondary descriptor-exchange diagnostic; average task pairs and seeds within each image, then weight images equally. Distinct from task/class-balanced main classification.",
            "budgets": summaries, "change_in_exchange_effect": changes}


def plot(result, exchange, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    budgets = result["budgets"]
    main = result["task_splits"]["unseen"]["budgets"]
    labels = {"full_model": "Full CLIP", "random": "Random", "attention": "Attention",
              "image_only": "Image-only", "conditioned": "Conditioned", "label_informed_teacher": "Label-informed diagnostic"}
    colors = {"full_model": "#555555", "random": "#aaaaaa", "attention": "#819754",
              "image_only": "#3565a3", "conditioned": "#d97925", "label_informed_teacher": "#9262ad"}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    for method in METHODS:
        records = [next(row for row in main[str(k)]["table"] if row["method"] == method) for k in budgets]
        for ax, metric, scale in zip(axes, METRICS, (100, 1)):
            values = [row[metric]["mean"] * scale for row in records]
            ax.plot(budgets, values, marker="o", color=colors[method],
                    linestyle="--" if method in ("full_model", "label_informed_teacher") else "-", label=labels[method])
    for ax, ylabel in zip(axes, ("Task/class-balanced accuracy (%)", "Mean signed geometric margin")):
        ax.set(xticks=budgets, xlabel="Retained patch tokens", ylabel=ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Unseen class combinations: three-seed averages")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output / f"budget_comparison.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    for ax, metric, factor in zip(axes, METRICS, (100, 1)):
        for i, budget in enumerate(budgets):
            estimate = main[str(budget)]["conditioned_minus_image_only"][metric]
            mean = estimate["mean"] * factor
            intervals = [("ci95", "#d97925", 3, 1.8)]
            if "bonferroni_family_ci95" in estimate:
                intervals.insert(0, ("bonferroni_family_ci95", "#bbbbbb", 5, 3))
            for interval, color, cap, width in intervals:
                low, high = np.asarray(estimate[interval]) * factor
                ax.errorbar(i, mean, yerr=[[mean - low], [high - mean]],
                            fmt="o", color=color, capsize=cap, elinewidth=width,
                            label=("Family-adjusted" if interval.startswith("bonferroni") else "Pointwise 95%") if i == 0 else None)
        ax.axhline(0, color="#666666", linewidth=.8)
        ax.set(xticks=range(len(budgets)), xticklabels=budgets, xlabel="Retained patch tokens",
               ylabel="Accuracy difference (percentage points)" if metric == "accuracy" else "Geometric margin difference")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Conditioned minus image-only: paired image-bootstrap intervals")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output / f"conditioning_by_budget.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    for ax, metric, factor in zip(axes, METRICS, (100, 1)):
        for i, budget in enumerate(budgets):
            estimate = exchange["budgets"][str(budget)]["task_splits"]["unseen"]["metrics"][f"own_minus_swapped_{metric}"]
            mean = estimate["mean"] * factor
            low, high = np.asarray(estimate["ci95"]) * factor
            ax.errorbar(i, mean, yerr=[[mean-low], [high-mean]], fmt="o", color="#2e7d8f", capsize=4)
        ax.axhline(0, color="#666666", linewidth=.8)
        ax.set(xticks=range(len(budgets)), xticklabels=budgets, xlabel="Retained patch tokens",
               ylabel="Accuracy difference (percentage points)" if metric == "accuracy" else "Geometric margin difference")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Correct minus exchanged task descriptor: image-equal diagnostic")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output / f"exchange_by_budget.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_sources(values):
    sources = {}
    for value in values:
        budget, path = value.split("=", 1)
        budget = int(budget)
        assert budget not in sources
        sources[budget] = Path(path)
    return sources


def run(sources, random_sources, output, pruning_summary):
    output.mkdir(parents=True, exist_ok=True)
    rows_by_budget, metadata = {}, {}
    for budget, source in sorted(sources.items()):
        metadata[budget] = json.loads((source / "eval_summary.json").read_text())
        assert metadata[budget]["config"]["keep_patch_tokens"] == budget
        rows_by_budget[budget] = (read_rows(source / "references" / "predictions.csv")
                                  + read_rows(source / "full" / "predictions.csv")
                                  + read_rows(random_sources.get(budget, source) / "random" / "predictions.csv"))
    first = metadata[min(sources)]
    training_seeds = first["config"]["training_seeds"]
    for item in metadata.values():
        assert item["config"]["training_seeds"] == training_seeds
        for key in ("model", "prune_after", "prompt_template", "manifest_dir"):
            assert item["config"][key] == first["config"][key]
    random_seeds = sorted({int(r["seed"]) for r in rows_by_budget[min(sources)] if r["method"] == "random"})
    replicates = first["config"].get("bootstrap_replicates", 2000)
    seed = first["config"].get("bootstrap_seed", 2052)
    result = analyze_rows(rows_by_budget, training_seeds, random_seeds, replicates, seed)
    result["sources"] = {}
    for budget, source in sorted(sources.items()):
        random_source = random_sources.get(budget, source)
        random_metadata = (json.loads((random_source / "eval_summary.json").read_text())
                           if random_source != source else metadata[budget])
        result["sources"][str(budget)] = {
            "evaluation": str(source), "random_evaluation": str(random_source),
            "evaluation_code_commit": metadata[budget]["code_commit"],
            "random_evaluation_code_commit": random_metadata["code_commit"],
            "config": metadata[budget]["config"], "recorded_source_evaluation_cost": metadata[budget]["cost"],
            "separate_random_evaluation_cost": random_metadata["cost"] if random_source != source else None,
            "cost_scope": "These are recorded source-run costs. A reused source can contain additional earlier conditions; do not treat these totals as new per-budget compute or deployment latency."}
    (output / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    edges = json.loads(pruning_summary.read_text())["config"]["margin_quartile_edges"]
    exchange = analyze_exchange(sources, output, training_seeds, edges, replicates, seed)
    (output / "exchange_analysis.json").write_text(json.dumps(exchange, indent=2, allow_nan=False) + "\n")
    return result, exchange


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", default=[], help="BUDGET=EVALUATION_DIR; repeat for each budget.")
    parser.add_argument("--budgets", nargs="+", type=int, default=[49, 98, 147],
                        help="Predeclared budget family; all listed sources must be supplied.")
    parser.add_argument("--random-source", action="append", default=[], help="BUDGET=DIR for a separate random-only supplement.")
    parser.add_argument("--output", type=Path, default=Path("results/pets-selector-budgets-v1"))
    parser.add_argument("--pruning-summary", type=Path, default=Path("results/pets-pruning-v1/summary.json"))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="Render saved analysis files without loading raw GPU outputs.")
    return parser


if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()
    if args.plot_only:
        result = json.loads((args.output / "analysis.json").read_text())
        exchange = json.loads((args.output / "exchange_analysis.json").read_text())
    else:
        sources, random_sources = parse_sources(args.source), parse_sources(args.random_source)
        assert len(set(args.budgets)) == len(args.budgets)
        assert sorted(sources) == sorted(args.budgets)
        result, exchange = run(sources, random_sources, args.output, args.pruning_summary)
    if args.plot or args.plot_only:
        plot(result, exchange, args.output)
    print(json.dumps({k: v["conditioned_minus_image_only"]
                      for k, v in result["task_splits"]["unseen"]["budgets"].items()}, indent=2))
