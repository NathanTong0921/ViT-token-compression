"""Analyze exchange_rows.csv and optional single_task_rows.csv from an offline pilot."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from token_compression.selection_analysis import analyze_rows, candidate_search_depth, fit_h2, image_means, row_metrics, summarize_single_tasks


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def plot_results(rows, analysis, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    budgets = [item["keep_patch_tokens"] for item in analysis["budgets"]]
    for key, label, marker in (("own_accuracy", "Task-specific deletion ranking", "o"),
                                ("shared_proxy_accuracy", "Shared deletion ranking", "s"),
                                ("shared_search_mean_accuracy", "Shared candidate search (mean)", "^"),
                                ("task_search_own_accuracy", "Task-specific candidate search", "D")):
        axes[0].plot(budgets, [100 * item["metrics"][key]["mean"] for item in analysis["budgets"]],
                     marker=marker, label=label)
    axes[0].set(xlabel="Retained patch tokens", ylabel="Image-equal panel accuracy (%)",
                title="Label-informed offline selections", xticks=budgets)
    axes[0].legend(fontsize=8, frameon=False)
    budget = min(budgets)
    selected = [row for row in rows if int(row["keep_patch_tokens"]) == budget]
    target = row_metrics(selected)["exchange_margin_loss"]
    _, angle, loss = fit_h2(selected, target, bootstrap_replicates=20)
    _, means = image_means([row["image_id"] for row in selected], np.column_stack([angle, loss]))
    axes[1].scatter(means[:, 0] * 90, means[:, 1], alpha=.35, s=18)
    edges = np.unique(np.quantile(means[:, 0], np.linspace(0, 1, 6)))
    group = np.searchsorted(edges[1:-1], means[:, 0], side="right")
    centers = np.asarray([means[group == index].mean(axis=0) for index in np.unique(group)])
    axes[1].plot(centers[:, 0] * 90, centers[:, 1], "o-", color="#b44726", label="Quantile-group means")
    axes[1].axhline(0, color="#888888", linewidth=.7)
    axes[1].set(xlabel="Angle residual after margin controls (degrees)",
                ylabel="Exchange margin-loss residual", title=f"Exploratory H2 association: {budget} patches")
    axes[1].legend(fontsize=8, frameon=False)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.text(.5, .01, "Left: anchor-image panel, not balanced binary accuracy. Right: each dot is one image; association is not causal.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"selection_diagnostics.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--h2-bootstrap-replicates", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2030)
    parser.add_argument("--plot", action="store_true", help="Also save descriptive PNG/PDF plots (requires Matplotlib).")
    parser.add_argument("--search-depth", action="store_true",
                        help="Analyze saved candidate-bank prefixes at 16, 32 and 64 candidates.")
    args = parser.parse_args()
    rows = read_csv(args.results / "exchange_rows.csv")
    analysis = analyze_rows(rows, args.bootstrap_replicates, args.h2_bootstrap_replicates, args.seed)
    single_path = args.results / "single_task_rows.csv"
    if single_path.exists():
        analysis["single_task_panel"] = summarize_single_tasks(read_csv(single_path))
    if args.search_depth:
        with np.load(args.results / "deletion_evidence.npz") as evidence:
            text = evidence["text_features"].copy()
        depth_rows = []
        for bank_path in sorted(args.results.glob("candidate_bank_k*.npz")):
            budget = int(bank_path.stem.split("_k")[-1])
            with np.load(bank_path) as bank:
                depth_rows.extend(candidate_search_depth(bank, text, budget))
        with (args.results / "search_depth.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(depth_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(depth_rows)
        analysis["candidate_search_depth"] = {
            "aggregation": "mean within image, then equal-weight mean across images",
            "scope": "Nested finite-bank prefixes; stability does not establish global optimality.",
            "rows": depth_rows,
        }
    (args.results / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    table = []
    for budget in analysis["budgets"]:
        for metric, estimate in budget["metrics"].items():
            table.append({"keep_patch_tokens": budget["keep_patch_tokens"],
                          "row_count": budget["row_count"], "image_count": budget["image_count"],
                          "metric": metric, "mean": estimate["mean"],
                          "ci95_low": estimate["ci95"][0], "ci95_high": estimate["ci95"][1]})
    with (args.results / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    if args.plot:
        plot_results(rows, analysis, args.results)
    print(f"Analyzed {len(rows)} rows across {len(analysis['budgets'])} budgets.")


if __name__ == "__main__":
    main()
