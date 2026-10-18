"""Re-score saved learned selections under another same-image candidate task."""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np

from token_compression.selector_exchange import build_exchange_cases, summarize_exchange


def plot_results(analysis, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = [s for s in ("unseen", "seen") if s in analysis["task_splits"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    for ax, metric, ylabel in zip(axes, ("margin", "accuracy"),
                                  ("Own minus exchanged margin", "Own minus exchanged accuracy (pp)")):
        factor = 100 if metric == "accuracy" else 1
        for x, split in enumerate(splits):
            summary = analysis["task_splits"][split]
            estimate = summary["metrics"][f"own_minus_swapped_{metric}"]
            mean, (low, high) = estimate["mean"], estimate["ci95"]
            ax.errorbar(x, factor * mean,
                        yerr=[[factor * (mean-low)], [factor * (high-mean)]],
                        fmt="o", capsize=4, color="#d97925")
            individual = [r["metrics"][f"own_minus_swapped_{metric}"]["mean"] for r in summary["per_seed"]]
            ax.scatter(x + np.linspace(-.06, .06, len(individual)), factor * np.asarray(individual),
                       marker="x", color="#3565a3", s=25)
        ax.axhline(0, color="#777777", linewidth=.8)
        ax.set(xticks=range(len(splits)), xticklabels=[s.title()+" combinations" for s in splits], ylabel=ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Same image, exchanged task descriptor; fixed trained weights")
    fig.text(.5, .01, "Image-equal effects; image-bootstrap 95% intervals; crosses denote training seeds.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    for extension in ("png", "pdf"):
        fig.savefig(output / f"selector_exchange.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(config, output, plot=False, unseen_csv=False):
    output.mkdir(parents=True, exist_ok=True)
    source = Path(config["source_dir"]) / "evaluation" / "predictions.npz"
    with np.load(source) as saved:
        predictions = {key: saved[key] for key in saved.files}
    cases = build_exchange_cases(predictions, config["training_seeds"], config["task_splits"])
    pruning = json.loads(Path(config["pruning_summary"]).read_text())
    edges = pruning["config"]["margin_quartile_edges"]
    analysis = summarize_exchange(cases, edges, config["bootstrap_replicates"], config["bootstrap_seed"])
    analysis["config"] = config
    analysis["code_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    analysis["source_predictions"] = str(source)
    analysis["artifact_schema"] = {
        "outcome_suffix": "First letter selects descriptor; second letter selects classification task. ba means descriptor B evaluated on task A.",
        "source_rows": "source_row_a/b and image_only_row_a/b refer to row_id in the parent predictions artifact.",
        "row_unit": "One training seed, one image and one unordered pair of distinct tasks sharing the true class.",
        "image_only_prefix": "Same exchange evaluation using the matched image-only selector; expected exact null.",
    }
    np.savez_compressed(output / "exchange_cases.npz", **cases)
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False)+"\n")
    if unseen_csv:
        indices = np.flatnonzero(cases["task_split"] == "unseen")
        with (output / "unseen_exchange_cases.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(cases)
            writer.writerows([cases[key][i].item() for key in cases] for i in indices)
    if plot:
        plot_results(analysis, output)
    for split, summary in analysis["task_splits"].items():
        print(split, summary["image_count"], "images;", summary["unique_image_task_pair_count"],
              "image/task pairs; exchange margin", summary["metrics"]["own_minus_swapped_margin"])
    return analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pets_selector_exchange.json"))
    parser.add_argument("--output", type=Path, default=Path("results/pets-selector-exchange-v1"))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--unseen-csv", action="store_true")
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output, args.plot, args.unseen_csv)
