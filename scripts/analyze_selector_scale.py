"""Paired, task-balanced analysis of fixed-step training-image coverage."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from token_compression.selection_analysis import cluster_summary
from token_compression.selector_exchange import build_exchange_cases, summarize_exchange


def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def panel_weights(rows, panel):
    """Equal tasks and equal class sides, omitting tasks missing a panel side."""
    eligible = np.array([panel == "all" or (int(r["old_panel"]) == 1) == (panel == "old") for r in rows])
    complete = []
    for task in sorted({r["task_id"] for r in rows}):
        selected = [i for i, r in enumerate(rows) if eligible[i] and r["task_id"] == task]
        if len({int(rows[i]["class_id"]) for i in selected}) == 2:
            complete.append((task, selected))
    weights = np.zeros(len(rows))
    for _, indices in complete:
        counts = {c: sum(int(rows[i]["class_id"]) == c for i in indices)
                  for c in {int(rows[i]["class_id"]) for i in indices}}
        for i in indices:
            weights[i] = 1.0 / (2 * len(complete) * counts[int(rows[i]["class_id"])])
    return weights


def weighted_cluster_summary(ids, weights, values, replicates=2000, seed=2052):
    """Bootstrap images while preserving the point estimate's task/class weights.

    Each image contributes its weighted row numerator and denominator. Bootstrap
    replicates take the ratio of their resampled totals. Repeated training seeds
    must already have been averaged within each image/task row.
    """
    ids, weights, values = np.asarray(ids), np.asarray(weights, float), np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    chosen = weights > 0
    ids, weights, values = ids[chosen], weights[chosen], values[chosen]
    unique, inverse = np.unique(ids, return_inverse=True)
    numerator = np.zeros((len(unique), values.shape[1]))
    denominator = np.bincount(inverse, weights=weights, minlength=len(unique))
    np.add.at(numerator, inverse, weights[:, None] * values)
    point = numerator.sum(axis=0) / denominator.sum()
    rng = np.random.default_rng(seed)
    boot = np.empty((replicates, values.shape[1]))
    for start in range(0, replicates, 100):
        draw = rng.integers(0, len(unique), (min(100, replicates - start), len(unique)))
        boot[start:start + len(draw)] = numerator[draw].sum(axis=1) / denominator[draw].sum(axis=1)[:, None]
    low, high = np.percentile(boot, [2.5, 97.5], axis=0)
    return {"image_count": len(unique), "row_count": len(ids), "mean": point,
            "ci95_low": low, "ci95_high": high}


def named_estimates(stats, names):
    return {name: {"mean": float(stats["mean"][i]),
                   "ci95": [float(stats["ci95_low"][i]), float(stats["ci95_high"][i])]}
            for i, name in enumerate(names)}


def analyze_rows(reference, learned, seeds, replicates=2000, seed=2052):
    result = {"primary": "Unseen combinations, all eligible development images: change in conditioning advantage (full minus small coverage).",
              "aggregation": "Each task has equal weight and each class side has half that task's weight; images within a class side are equal. Seeds are averaged before paired contrasts.",
              "uncertainty": "Image-cluster bootstrap ratio of weighted numerators and denominators, conditional on fixed tasks and three seeds. Image-equal estimates are separately labeled.",
              "bootstrap_replicates": replicates, "bootstrap_seed": seed, "task_splits": {}}
    metrics = ("correct", "margin")
    for split in ("seen", "unseen"):
        base = [r for r in reference if r["method"] == "full_model" and r["task_split"] == split]
        keys = [(r["image_id"], r["task_id"]) for r in base]
        assert len(keys) == len(set(keys))
        ids = np.asarray([r["image_id"] for r in base])

        def values(rows, method, training_seed):
            selected = [r for r in rows if r["method"] == method and int(r["seed"]) == training_seed
                        and r["task_split"] == split]
            mapping = {(r["image_id"], r["task_id"]): r for r in selected}
            assert len(mapping) == len(selected) == len(keys)
            assert set(mapping) == set(keys)
            return np.asarray([[float(mapping[k][m]) for m in metrics] for k in keys])

        arrays = {coverage: {arm: np.stack([values(rows, arm, s) for s in seeds])
                             for arm in ("conditioned", "image_only")}
                  for coverage, rows in learned.items()}
        reference_values = {m: values(reference, m, -1) for m in ("full_model", "attention", "label_informed_teacher")}
        differences = {
            "small_conditioned_minus_image_only": arrays["small"]["conditioned"] - arrays["small"]["image_only"],
            "full_conditioned_minus_image_only": arrays["full"]["conditioned"] - arrays["full"]["image_only"],
            "conditioned_full_minus_small": arrays["full"]["conditioned"] - arrays["small"]["conditioned"],
            "image_only_full_minus_small": arrays["full"]["image_only"] - arrays["small"]["image_only"],
        }
        differences["change_in_conditioning_advantage"] = (
            differences["full_conditioned_minus_image_only"] - differences["small_conditioned_minus_image_only"])
        names = [f"{contrast}.{metric}" for contrast in differences for metric in ("accuracy", "margin")]
        matrix = np.column_stack([delta.mean(axis=0)[:, j] for delta in differences.values() for j in range(2)])
        panels = {}
        for panel in ("all", "old", "new"):
            weights = panel_weights(base, panel)
            take = weights > 0
            weighted = weighted_cluster_summary(ids, weights, matrix, replicates, seed)
            equal = cluster_summary(ids[take], matrix[take], replicates, seed)
            table = []
            for method, data in reference_values.items():
                point = (weights[:, None] * data).sum(axis=0)
                table.append({"coverage": "references", "method": method,
                              "accuracy": float(point[0]), "mean_margin": float(point[1])})
            full_correct = reference_values["full_model"][:, 0]
            for coverage, arms in arrays.items():
                for arm, data in arms.items():
                    points = (weights[None, :, None] * data).sum(axis=1)
                    repairs = (data[:, :, 0] == 1) & (full_correct[None, :] == 0)
                    new_errors = (data[:, :, 0] == 0) & (full_correct[None, :] == 1)
                    table.append({"coverage": coverage, "method": arm,
                                  "accuracy": float(points[:, 0].mean()), "mean_margin": float(points[:, 1].mean()),
                                  "weighted_repairs_vs_full_model": float((repairs * weights[None]).sum(axis=1).mean()),
                                  "weighted_new_errors_vs_full_model": float((new_errors * weights[None]).sum(axis=1).mean()),
                                  "per_seed": [{"seed": s, "accuracy": float(point[0]), "mean_margin": float(point[1]),
                                                "weighted_repairs_vs_full_model": float(np.sum(repairs[si] * weights)),
                                                "weighted_new_errors_vs_full_model": float(np.sum(new_errors[si] * weights))}
                                               for si, (s, point) in enumerate(zip(seeds, points))]})
            per_seed = []
            for si, s in enumerate(seeds):
                points = np.asarray([(weights * delta[si, :, j]).sum() for delta in differences.values() for j in range(2)])
                per_seed.append({"seed": s, "effects": dict(zip(names, points.tolist()))})
            correctness_changes = {}
            for coverage, arms in arrays.items():
                conditioned, shared = arms["conditioned"][:, :, 0], arms["image_only"][:, :, 0]
                repairs = (conditioned == 1) & (shared == 0)
                new_errors = (conditioned == 0) & (shared == 1)
                correctness_changes[coverage] = [{"seed": s,
                    "repair_image_task_cases": int(np.sum(repairs[si] & take)),
                    "new_error_image_task_cases": int(np.sum(new_errors[si] & take)),
                    "weighted_repair_fraction": float(np.sum(repairs[si] * weights)),
                    "weighted_new_error_fraction": float(np.sum(new_errors[si] * weights))}
                    for si, s in enumerate(seeds)]
            panels[panel] = {"eligible_image_count": weighted["image_count"], "image_task_examples": int(take.sum()),
                             "task_count": len({r["task_id"] for i, r in enumerate(base) if take[i]}),
                             "table": table, "task_class_balanced_effects": named_estimates(weighted, names),
                             "image_equal_effects": named_estimates(equal, names), "per_seed": per_seed,
                             "conditioned_vs_image_only_correctness_changes": correctness_changes}
        result["task_splits"][split] = {"panels": panels}
    return result


EXCHANGE_KEYS = ("seed", "method", "task_split", "image_ids", "task_ids", "labels", "class_pairs",
                 "keep_patch_tokens", "patch_indices", "cosine_scores")


def read_unseen_predictions(path, full_only=False):
    with np.load(path) as saved:
        selected = saved["task_split"] == "unseen"
        if full_only:
            selected &= saved["method"] == "full_model"
        out = {k: saved[k][selected] for k in EXCHANGE_KEYS}
        out["text_features"] = saved["text_features"].copy()
    return out


def analyze_exchange(source, output, seeds, margin_edges, replicates, seed):
    full = read_unseen_predictions(source / "references" / "predictions.npz", full_only=True)
    cases_by_coverage, old_ids = {}, set()
    with np.load(source / "small" / "predictions.npz") as saved:
        old_ids = set(saved["image_ids"][saved["old_panel"].astype(bool)].tolist())
    summaries = {"scope": "Secondary diagnostic on exchange-eligible unseen task pairs, image-equal. These are not task-balanced classification effects.",
                 "coverages": {}}
    for coverage in ("small", "full"):
        learned = read_unseen_predictions(source / coverage / "predictions.npz")
        np.testing.assert_array_equal(full["text_features"], learned["text_features"])
        merged = {k: np.concatenate([full[k], learned[k]]) for k in EXCHANGE_KEYS}
        merged["text_features"] = full["text_features"]
        merged["row_id"] = np.arange(len(merged["seed"]))
        cases = build_exchange_cases(merged, seeds, ["unseen"])
        cases_by_coverage[coverage] = cases
        np.savez_compressed(output / f"{coverage}_unseen_exchange_cases.npz", **cases)
        summaries["coverages"][coverage] = {}
        old = np.isin(cases["image_ids"], list(old_ids))
        for panel, chosen in (("all", np.ones(len(old), bool)), ("old", old), ("new", ~old)):
            part = {k: v[chosen] for k, v in cases.items()}
            summaries["coverages"][coverage][panel] = summarize_exchange(part, margin_edges, replicates, seed)
    small, large = cases_by_coverage["small"], cases_by_coverage["full"]
    for k in ("image_ids", "task_a", "task_b", "seed"):
        np.testing.assert_array_equal(small[k], large[k])
    diffs = []
    for metric in ("correct", "margin"):
        effects = [(c[f"{metric}_aa"].astype(float) + c[f"{metric}_bb"]
                    - c[f"{metric}_ba"] - c[f"{metric}_ab"]) / 2 for c in (small, large)]
        diffs.append(effects[1] - effects[0])
    summaries["full_minus_small_exchange_effect"] = {}
    old = np.isin(small["image_ids"], list(old_ids))
    for panel, take in (("all", np.ones(len(old), bool)), ("old", old), ("new", ~old)):
        values = np.column_stack(diffs)[take]
        stats = cluster_summary(small["image_ids"][take], values, replicates, seed)
        summaries["full_minus_small_exchange_effect"][panel] = {
            "image_count": stats["image_count"], **named_estimates(stats, ["accuracy", "margin"])}
    return summaries


def plot(result, output, training_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    colors = {"old": "#a7a7a7", "new": "#2e7d8f", "all": "#d97925"}
    contrasts = ["small_conditioned_minus_image_only", "full_conditioned_minus_image_only", "change_in_conditioning_advantage"]
    panels = result["task_splits"]["unseen"]["panels"]
    for ax, metric, factor in zip(axes, ("accuracy", "margin"), (100, 1)):
        for pi, panel in enumerate(("old", "new", "all")):
            for ci, contrast in enumerate(contrasts):
                estimate = panels[panel]["task_class_balanced_effects"][f"{contrast}.{metric}"]
                mean = estimate["mean"] * factor
                low, high = np.asarray(estimate["ci95"]) * factor
                ax.errorbar(ci + (pi - 1) * .18, mean, yerr=[[mean - low], [high - mean]],
                            fmt="o", capsize=3, color=colors[panel], label=panel if ci == 0 else None)
        ax.axhline(0, color="#888888", linewidth=.8)
        ax.set(xticks=range(3), xticklabels=["Small coverage\nconditioning gain", "Full coverage\nconditioning gain", "Change in\nconditioning gain"],
               ylabel="Accuracy difference (percentage points)" if metric == "accuracy" else "Geometric margin difference",
               title="Unseen combinations: task/class balanced")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(title="Development image panel", frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output / f"coverage_comparison.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    if training_dir is None:
        return
    rows = read_rows(training_dir / "training_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    for coverage, style in (("small", "--"), ("full", "-")):
        for arm, color in (("conditioned", "#d97925"), ("image_only", "#3565a3")):
            selected = [r for r in rows if r["coverage"] == coverage and r["arm"] == arm]
            seeds = sorted({int(r["seed"]) for r in selected})
            for ax, metric in zip(axes, ("train_kl", "val_kl")):
                curves = []
                for s in seeds:
                    part = sorted([r for r in selected if int(r["seed"]) == s and int(r["step"]) > 0], key=lambda r: int(r["step"]))
                    steps = [int(r["step"]) for r in part]
                    curves.append([float(r[metric]) for r in part])
                ax.plot(steps, np.mean(curves, axis=0), style, color=color,
                        label=f"{coverage}: {arm.replace('_', ' ')}")
    for ax, title in zip(axes, ("Sampled training loss (preceding interval)", "Balanced development loss: unseen pairs")):
        ax.set(xlabel="Optimizer step", ylabel="Teacher-distribution KL", title=title)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output / f"training_curves.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runs/pets-selector-scale-eval-v1"))
    parser.add_argument("--output", type=Path, default=Path("results/pets-selector-scale-v1"))
    parser.add_argument("--training", type=Path)
    parser.add_argument("--pruning-summary", type=Path, default=Path("results/pets-pruning-v1/summary.json"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.source / "eval_summary.json").read_text())
    config = metadata["config"]
    seeds = config["training_seeds"]
    replicates, seed = config.get("bootstrap_replicates", 2000), config.get("bootstrap_seed", 2052)
    reference = read_rows(args.source / "references" / "predictions.csv")
    learned = {c: read_rows(args.source / c / "predictions.csv") for c in ("small", "full")}
    result = analyze_rows(reference, learned, seeds, replicates, seed)
    result["evaluation_source"] = str(args.source)
    result["evaluation_code_commit"] = metadata["code_commit"]
    result["config"] = config
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    edges = json.loads(args.pruning_summary.read_text())["config"]["margin_quartile_edges"]
    exchange = analyze_exchange(args.source, args.output, seeds, edges, replicates, seed)
    (args.output / "exchange_analysis.json").write_text(json.dumps(exchange, indent=2, allow_nan=False) + "\n")
    if args.plot:
        plot(result, args.output, args.training)
    print(json.dumps(result["task_splits"]["unseen"]["panels"]["all"]["task_class_balanced_effects"], indent=2))
