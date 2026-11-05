"""Compare external preservation prediction with matched CLIP selector results."""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_selector_budgets import weighted_estimates
from analyze_selector_scale import panel_weights, read_rows
from evaluate_selectors import write_rows


METRICS = ("accuracy", "margin")
METHODS = ("full_model", "random", "attention", "image_only", "image_only_wide",
           "conditioned", "preservation_teacher", "patch_ranking")
LABELS = {"full_model": "Full CLIP", "random": "Random", "attention": "Attention",
          "image_only": "Original shared", "image_only_wide": "Strong shared",
          "conditioned": "Conditioned", "preservation_teacher": "Preservation teacher",
          "patch_ranking": "Patch Ranking (adapted)"}


def aligned_values(rows, method, keys, metadata, seeds, budget):
    """Match image/task identities and labels before taking paired differences."""
    runs = []
    for seed in seeds:
        selected = [row for row in rows if row["method"] == method
                    and int(row["seed"]) == seed and row["task_split"] == "unseen"]
        lookup = {(row["image_id"], row["task_id"]): row for row in selected}
        assert len(lookup) == len(selected) == len(keys) and set(lookup) == set(keys)
        data = []
        for key in keys:
            row = lookup[key]
            assert tuple(int(row[n]) for n in ("class_id", "class_a", "class_b")) == metadata[key]
            assert int(row["keep_patch_tokens"]) == (196 if method == "full_model" else budget)
            data.append([float(row["correct"]), float(row["margin"])])
        runs.append(data)
    return np.asarray(runs)


def run(config, source, output):
    output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(source / "predictions.csv")
    base = sorted([row for row in rows if row["method"] == "full_model"],
                  key=lambda row: (row["image_id"], row["task_id"]))
    keys = [(row["image_id"], row["task_id"]) for row in base]
    metadata = {key: tuple(int(row[n]) for n in ("class_id", "class_a", "class_b"))
                for key, row in zip(keys, base)}
    ids = np.array([key[0] for key in keys])
    weights = panel_weights(base, "all")
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-12)
    old = read_rows(Path(config["baseline_eval_dir"]) / "full/predictions.csv")
    wide = read_rows(Path(config["ablation_eval_dir"]) / "full/predictions.csv")
    budget, seeds = config["keep_patch_tokens"], config["seeds"]
    random_seeds = config.get("random_seeds", [2055, 2056, 2057])
    arrays, run_seeds = {}, {}
    for method in METHODS:
        selected = old if method in ("conditioned", "image_only") else wide if method == "image_only_wide" else rows
        current_seeds = (seeds if method in ("patch_ranking", "conditioned", "image_only", "image_only_wide")
                         else random_seeds if method == "random" else [-1])
        arrays[method] = aligned_values(selected, method, keys, metadata, current_seeds, budget)
        run_seeds[method] = current_seeds
    reps = config.get("bootstrap_replicates", 2000)
    bootstrap_seed = config.get("bootstrap_seed", 2501)
    means = {method: value.mean(axis=0) for method, value in arrays.items()}
    full_by_key = dict(zip(keys, means["full_model"]))
    for row in old + wide:
        if row["task_split"] == "unseen":
            full = full_by_key[row["image_id"], row["task_id"]]
            assert int(row["base_correct"]) == full[0]
            np.testing.assert_allclose(float(row["base_margin"]), full[1], atol=1e-6, rtol=0)
    table = []
    for method, data in arrays.items():
        estimates = weighted_estimates(ids, weights, means[method], METRICS, reps, bootstrap_seed)
        points = np.einsum("n,snm->sm", weights, data)
        full_correct = means["full_model"][:, 0]
        repaired = (data[:, :, 0] == 1) & (full_correct == 0)
        new_errors = (data[:, :, 0] == 0) & (full_correct == 1)
        table.append({"method": method, **estimates,
                      "repairs_vs_full": float((repaired * weights).sum(axis=1).mean()),
                      "new_errors_vs_full": float((new_errors * weights).sum(axis=1).mean()),
                      "per_seed": [{"seed": seed, "accuracy": float(point[0]), "margin": float(point[1])}
                                   for seed, point in zip(run_seeds[method], points)]})
    comparisons = (("patch_ranking", "image_only_wide"), ("patch_ranking", "conditioned"),
                   ("patch_ranking", "image_only"), ("patch_ranking", "attention"),
                   ("patch_ranking", "preservation_teacher"), ("preservation_teacher", "full_model"))
    contrasts, contrast_rows = {}, []
    for index, (a, b) in enumerate(comparisons):
        name = f"{a}_minus_{b}"
        family = 4 if index < 2 else 1
        estimates = weighted_estimates(ids, weights, means[a] - means[b], METRICS,
                                        reps, bootstrap_seed, family)
        contrasts[name] = estimates
        for metric, stats in estimates.items():
            interval = stats.get("bonferroni_family_ci95", [None, None])
            contrast_rows.append({"contrast": name, "metric": metric, "mean": stats["mean"],
                                  "ci95_low": stats["ci95"][0], "ci95_high": stats["ci95"][1],
                                  "adjusted_low": interval[0], "adjusted_high": interval[1]})

    with np.load(source / "predictions.npz") as raw:
        full_ids, labels = raw["image_ids"], raw["labels"]
        multi, recovery = [], []
        for method in ("full_model", "random", "attention", "preservation_teacher", "patch_ranking"):
            take = raw["method"] == method
            scores = raw["cosine_scores"][take]
            correct = scores.argmax(axis=-1) == labels[None]
            multi.append({"method": method, **weighted_estimates(
                full_ids, np.ones(len(labels)) / len(labels), correct.mean(axis=0),
                ["accuracy"], reps, bootstrap_seed),
                "per_seed": [{"seed": int(seed), "accuracy": float(values.mean())}
                             for seed, values in zip(raw["seed"][take], correct)]})
            if method != "full_model":
                overlaps = raw["teacher_overlap"][take]
                recovery.append({"method": method, "at_k": weighted_estimates(
                    full_ids, np.ones(len(labels)) / len(labels), overlaps.mean(axis=0),
                    ["fraction"], reps, bootstrap_seed)["fraction"]})

    evaluation = json.loads((source / "evaluation_summary.json").read_text())
    train_path = Path(evaluation["models_dir"]) / "training_summary.json"
    training = json.loads(train_path.read_text())
    summary = {"dataset": config["dataset"], "budget": budget, "prune_after": config["prune_after"],
               "source": str(source), "reference_sources": {"original": config["baseline_eval_dir"],
                                                               "wide": config["ablation_eval_dir"]},
               "seeds": seeds, "image_count": len(set(ids)), "image_task_count": len(base),
               "task_count": len({key[1] for key in keys}), "table": table,
               "primary_contrasts": dict(list(contrasts.items())[:2]),
               "secondary_contrasts": dict(list(contrasts.items())[2:]),
               "multiclass": {"image_count": len(full_ids), "class_count": len(np.unique(labels)),
                              "aggregation": "Equal validation images, all class prompts.", "table": multi},
               "teacher_recovery": recovery,
               "evaluation_runs": evaluation["runs"], "training": training,
               "aggregation": "Equal tasks and equal class sides within tasks; seeds averaged before image bootstrap.",
               "primary_intervals": "Ordinary 95% and adjustment for two datasets x two primary contrasts per metric.",
               "bootstrap_replicates": reps, "bootstrap_seed": bootstrap_seed}
    (output / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_rows(output / "classification_table.csv", [
        {"method": row["method"], "accuracy_percent": 100 * row["accuracy"]["mean"],
         "accuracy_ci95_low_percent": 100 * row["accuracy"]["ci95"][0],
         "accuracy_ci95_high_percent": 100 * row["accuracy"]["ci95"][1],
         "geometric_margin": row["margin"]["mean"],
         "margin_ci95_low": row["margin"]["ci95"][0], "margin_ci95_high": row["margin"]["ci95"][1]}
        for row in table])
    write_rows(output / "contrast_table.csv", contrast_rows)
    write_rows(output / "per_seed.csv", [{"method": row["method"], **seed}
                                         for row in table for seed in row["per_seed"]])
    write_rows(output / "reference_predictions.csv", [row for row in old + wide
        if row["method"] in ("conditioned", "image_only", "image_only_wide")
        and row["task_split"] == "unseen"])
    np.savez_compressed(output / "paired_cases.npz", image_ids=ids,
                        task_ids=np.array([key[1] for key in keys]), weights=weights,
                        **{f"{method}_per_seed": value for method, value in arrays.items()})
    print(json.dumps({"dataset": config["dataset"], "primary_contrasts": summary["primary_contrasts"]}), flush=True)
    return summary


def plot(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, metric, scale, label in zip(axes[:2], METRICS, (100, 1),
                                      ("Task/class-balanced accuracy (%)", "Geometric margin")):
        for index, row in enumerate(summary["table"]):
            color = "#276f9e" if row["method"] == "patch_ranking" else "#ae733f" if row["method"] == "preservation_teacher" else "#7c8790"
            values = row[metric]
            low, high = np.asarray(values["ci95"]) * scale
            ax.plot([low, high], [index, index], color=color, lw=2)
            ax.plot(values["mean"] * scale, index, "o", color=color)
        ax.set(yticks=range(len(summary["table"])),
               yticklabels=[LABELS[row["method"]] for row in summary["table"]] if ax is axes[0] else [],
               xlabel=label)
        ax.invert_yaxis()
    recovery = summary["teacher_recovery"]
    axes[2].bar(range(len(recovery)), [100*row["at_k"]["mean"] for row in recovery],
                color=["#276f9e" if row["method"] == "patch_ranking" else "#7c8790" for row in recovery])
    axes[2].set(xticks=range(len(recovery)),
                xticklabels=[LABELS[row["method"]] for row in recovery],
                ylabel=f"Teacher set overlap at {summary['budget']} (%)", ylim=(0, 105))
    axes[2].tick_params(axis="x", labelrotation=30)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x" if ax is not axes[2] else "y", alpha=.15)
    fig.suptitle(f"{summary['dataset'].title()}: preservation teacher, predictor and classification")
    fig.tight_layout()
    fig.savefig(output / "external_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()
    result = run(json.loads(args.config.read_text()), args.source, args.output)
    if not args.skip_plot:
        plot(result, args.output)
