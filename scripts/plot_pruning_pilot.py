"""Plot the saved pruning pilot tables; requires NumPy and Matplotlib."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/pets-pruning-v1"))
    args = parser.parse_args()
    with (args.results / "metrics.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["task_split"] == "val"]
    baseline = next(r for r in rows if r["method"] == "keep_all")
    budgets = sorted({int(r["keep_patch_tokens"]) for r in rows})
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    for ax, key, title in zip(axes, ("accuracy_37class", "binary_accuracy_macro"),
                              ("37-class classification", "66 validation pairs (macro)")):
        full = 100 * float(baseline[key])
        attention, means, low, high = [], [], [], []
        for k in budgets:
            if k == int(baseline["keep_patch_tokens"]):
                attention.append(full)
                means.append(full)
                low.append(full)
                high.append(full)
                continue
            attention.append(100 * float(next(r[key] for r in rows
                                              if r["method"] == "attention"
                                              and int(r["keep_patch_tokens"]) == k)))
            random = [100 * float(r[key]) for r in rows
                      if r["method"] == "random" and int(r["keep_patch_tokens"]) == k]
            means.append(np.mean(random))
            low.append(min(random))
            high.append(max(random))
        ax.axhline(full, color="#777777", linestyle=":", linewidth=1.3, label="Full tokens")
        ax.plot(budgets, attention, "o-", color="#1764a0", linewidth=2, label="Attention")
        ax.plot(budgets, means, "s-", color="#d47b26", linewidth=2, label="Random (mean of 3 seeds)")
        ax.fill_between(budgets, low, high, color="#d47b26", alpha=0.18)
        ax.set(title=title, xlabel="Retained patch tokens (CLS always retained)", ylabel="Accuracy (%)")
        ax.set_xticks(budgets)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Single-stage pruning after block 6: frozen CLIP ViT-B/16", y=0.99)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=3, frameon=False)
    fig.text(0.5, 0.008, "736 development images; shaded band = random-seed range, not a confidence interval.",
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    for suffix in ("png", "pdf"):
        fig.savefig(args.results / ("accuracy_by_budget." + suffix), dpi=180, bbox_inches="tight")
    print("Saved accuracy_by_budget.png and accuracy_by_budget.pdf")


if __name__ == "__main__":
    main()
