"""Paired development analysis for learned conditional and image-only selectors."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from token_compression.selection_analysis import cluster_summary, image_means


def read_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def estimate(ids, values):
    result = cluster_summary(ids, values, bootstrap_replicates=2000, seed=2050)
    return {"image_count": result["image_count"], "row_count": result["row_count"],
            "mean": float(result["mean"]),
            "ci95": [float(result["ci95_low"]), float(result["ci95_high"])]}


def analyze(rows):
    output = {"scope": "Development transfer to unseen combinations of known classes; official test untouched.",
              "aggregation": "Tables balance task examples; paired effects average seeds and task rows within image, then images equally.",
              "uncertainty": "Image bootstrap conditional on these training seeds and task collections; seed variation reported separately.",
              "task_splits": {}}
    domain_effects = {}
    for split in ("seen", "unseen"):
        selected = [r for r in rows if r["task_split"] == split]
        seeds = sorted({int(r["seed"]) for r in selected if r["method"] == "conditioned"})
        table = []
        for method in sorted({r["method"] for r in selected}):
            method_rows = [r for r in selected if r["method"] == method]
            method_seeds = sorted({int(r["seed"]) for r in method_rows})
            by_seed = []
            for seed in method_seeds:
                values = [r for r in method_rows if int(r["seed"]) == seed]
                assert len({(r["image_id"], r["task_id"]) for r in values}) == len(values)
                entry = {"seed": seed, "row_count": len(values),
                         "accuracy": float(np.mean([float(r["correct"]) for r in values])),
                         "mean_margin": float(np.mean([float(r["margin"]) for r in values]))}
                correlations = [float(r["teacher_utility_correlation"]) for r in values
                                if r.get("teacher_utility_correlation", "") != ""]
                if correlations:
                    entry["teacher_utility_correlation"] = float(np.mean(correlations))
                by_seed.append(entry)
            table.append({"method": method, "seeds": by_seed,
                          "mean_accuracy": float(np.mean([r["accuracy"] for r in by_seed])),
                          "seed_accuracy_std": float(np.std([r["accuracy"] for r in by_seed], ddof=1)) if len(by_seed)>1 else None,
                          "mean_margin": float(np.mean([r["mean_margin"] for r in by_seed]))})
        paired = {}
        for method in ("conditioned", "image_only"):
            records = [r for r in selected if r["method"] == method]
            paired[method] = {(int(r["seed"]), r["image_id"], r["task_id"]): r for r in records}
            assert len(records) == len(paired[method])
        keys = sorted(paired["conditioned"])
        assert keys == sorted(paired["image_only"])
        contrasts = {}
        for metric in ("correct", "margin"):
            delta = np.array([float(paired["conditioned"][k][metric]) - float(paired["image_only"][k][metric]) for k in keys])
            ids = [k[1] for k in keys]
            contrasts[metric] = {"across_seed_image_effect": estimate(ids, delta),
                                 "balanced_row_effect": float(delta.mean()), "per_seed": []}
            for seed in seeds:
                chosen = np.array([k[0] == seed for k in keys])
                contrasts[metric]["per_seed"].append({"seed": seed,
                    **estimate(np.asarray(ids)[chosen], delta[chosen])})
            domain_effects[(split, metric)] = image_means(ids, delta)
        output["task_splits"][split] = {"image_count": len({r["image_id"] for r in selected}),
            "task_count": len({r["task_id"] for r in selected}),
            "training_seeds": seeds, "table": table, "conditional_minus_image_only": contrasts}
    output["unseen_minus_seen_conditional_effect"] = {}
    for metric in ("correct", "margin"):
        ids, unseen = domain_effects[("unseen", metric)]
        seen_ids, seen = domain_effects[("seen", metric)]
        assert np.array_equal(ids, seen_ids)
        output["unseen_minus_seen_conditional_effect"][metric] = estimate(ids, unseen - seen)
    return output


def plot(result, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"conditioned": "#d97925", "image_only": "#3565a3", "attention": "#41946b"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    labels = ["Seen combinations", "Unseen combinations"]
    for method, offset in (("conditioned", -.2), ("image_only", 0), ("attention", .2)):
        for x, split in enumerate(("seen", "unseen")):
            row = next(r for r in result["task_splits"][split]["table"] if r["method"] == method)
            axes[0].bar(x+offset, (1-row["mean_accuracy"])*100, width=.18, color=colors[method],
                        label=method.replace("_", " ") if x==0 else None, alpha=.75)
            axes[0].plot(np.full(len(row["seeds"]), x+offset), [(1-r["accuracy"])*100 for r in row["seeds"]],
                         ".", color="#222222", markersize=5)
    for x, split in enumerate(("seen", "unseen")):
        effect = result["task_splits"][split]["conditional_minus_image_only"]["margin"]
        summary = effect["across_seed_image_effect"]
        low, high = summary["ci95"]
        axes[1].errorbar(x, summary["mean"], yerr=[[summary["mean"]-low], [high-summary["mean"]]],
                         fmt="o", color="#d97925", capsize=4)
        axes[1].scatter(np.full(len(effect["per_seed"]), x)+np.linspace(-.07,.07,len(effect["per_seed"])),
                        [r["mean"] for r in effect["per_seed"]], marker="x", color="#3565a3", s=24)
    axes[0].set(ylabel="Balanced task error (%) — lower is better", ylim=(0, 3.2), title="Actual compressed CLIP predictions")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set(ylabel="Conditional minus image-only margin", title="Paired effect, image bootstrap 95% CI")
    axes[1].axhline(0, color="#888888", linewidth=.8)
    for ax in axes:
        ax.set(xticks=[0,1], xticklabels=labels)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(directory / f"selector_comparison.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_training(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = read_rows(directory / "training" / "training_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for arm, color in (("conditioned", "#d97925"), ("image_only", "#3565a3")):
        seeds = sorted({int(r["seed"]) for r in rows if r["arm"] == arm})
        for ax, metric in zip(axes, ("train_kl", "val_kl")):
            curves = []
            for seed in seeds:
                values = sorted((r for r in rows if r["arm"] == arm and int(r["seed"]) == seed),
                                key=lambda r: int(r["epoch"]))
                epochs = [int(r["epoch"]) for r in values]
                curve = [float(r[metric]) for r in values]
                curves.append(curve)
                ax.plot(epochs, curve, color=color, linewidth=.7, alpha=.3)
            ax.plot(epochs, np.mean(curves, axis=0), color=color, linewidth=2,
                    label=arm.replace("_", " "))
    for ax, title in zip(axes, ("Training fit", "New images and unseen pairs")):
        ax.set(xlabel="Epoch (fixed final checkpoint)", ylabel="Teacher-distribution KL", title=title)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(directory / f"training_curves.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    result = analyze(read_rows(args.results / "evaluation" / "predictions.csv"))
    (args.results / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    if args.plot:
        plot(result, args.results)
        plot_training(args.results)
    for split, value in result["task_splits"].items():
        print(split, [(row["method"], round(row["mean_accuracy"], 4)) for row in value["table"]])
