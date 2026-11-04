"""Plot all fixed class partitions and both prespecified aggregate summaries."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ENTRIES = [("2099", "2099 · historical", "#8C939C", "o"),
           ("2100", "2100 · replication", "#176A9D", "o"),
           ("2101", "2101 · replication", "#176A9D", "o"),
           ("new_partitions_only", "New partitions mean", "#103F66", "D"),
           ("historical_plus_new", "Historical + new mean", "#8C939C", "D")]


def effect(data, entry, metric, contrast):
    if entry.isdigit():
        return data["per_partition"][entry]["metrics"][metric][contrast]["seen_minus_held"]
    return data["aggregates"][entry]["metrics"][metric]["contrasts"][contrast]


def run(root):
    saved = json.loads((root / "summary.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    rows = []
    for contrast, name, comparison in (("real_minus_permuted_mean", "identity", "Real text versus fixed identity codes"),
                                        ("real_minus_wide", "shared", "Real text versus wide image-only selection")):
        fig, axes = plt.subplots(2, 2, figsize=(10.3, 6.7), sharex="row", sharey=True)
        for column, (dataset, title) in enumerate((("pets", "Pets · 49 patches"), ("flowers", "Flowers · 98 patches"))):
            data = saved["datasets"][dataset]
            for row, metric in enumerate(("correct", "margin")):
                ax = axes[row, column]
                scale = 1. if metric == "correct" else 1000.
                for index, (entry, label, color, marker) in enumerate(ENTRIES):
                    record = effect(data, entry, metric, contrast)
                    point, lower, upper = record["estimate"] * scale, *(np.asarray(record["ci95"]) * scale)
                    y = len(ENTRIES) - 1 - index
                    ax.errorbar(point, y, xerr=[[point - lower], [upper - point]], fmt=marker,
                                color=color, markerfacecolor="white" if entry == "historical_plus_new" else color,
                                markersize=4.5, capsize=2.5, elinewidth=1.4 if entry == "new_partitions_only" else .9)
                    rows.append(dict(dataset=dataset, contrast=contrast, metric=metric, summary=entry,
                                     estimate=record["estimate"], lower95=record["ci95"][0], upper95=record["ci95"][1]))
                ax.axvline(0, color="#414A54", lw=.8, linestyle="--")
                ax.axhline(1.5, color="#CCD0D5", lw=.6)
                ax.grid(axis="x", alpha=.15)
                ax.set_yticks(np.arange(len(ENTRIES))[::-1], [item[1] for item in ENTRIES])
                ax.set_ylim(-.45, 4.45)
                ax.set_xlabel("H0-choice frequency contrast (pp)" if metric == "correct"
                              else "Candidate-direction margin shift (×10³)")
                if row == 0:
                    ax.set_title(title, weight="bold", pad=10)
        fig.suptitle(comparison, weight="bold", fontsize=12, y=.98)
        fig.text(.55, .045, "Means of 3 paired training seeds and 2 fixed code maps; bars: paired image-bootstrap 95% intervals.",
                 ha="center", fontsize=8)
        fig.text(.55, .025, "Aggregates weight partitions equally and jointly resample shared image IDs. New partitions are 2100/2101.",
                 ha="center", fontsize=8)
        fig.text(.55, .005, "H0 changes from held to seen; the role-accuracy contrast equals a change in H0-choice frequency.",
                 ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .09, 1, .94), h_pad=2., w_pad=2.)
        for suffix in ("png", "pdf"):
            fig.savefig(root / f"familiarity_partition_replication_{name}.{suffix}", dpi=220, bbox_inches="tight")
        plt.close(fig)
    with (root / "plot_data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/familiarity-replications-v1"))
    run(parser.parse_args().results_root)
