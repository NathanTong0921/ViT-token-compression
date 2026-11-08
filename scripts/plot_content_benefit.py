"""Plot content-based benefit prediction with source-value tables."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SETTINGS = ("pets_b16_49", "pets_b16_98", "flowers_b16_49", "flowers_b16_98")
LABELS = ("Pets · 49", "Pets · 98", "Flowers · 49", "Flowers · 98")
BLUE, ORANGE, PURPLE = "#2166A5", "#C97927", "#8056A6"


def estimate(panel, key, scale=1):
    value = panel["estimates"][key]
    low, high = value["four_setting_adjusted_ci95"]
    return np.asarray([value["mean"], low, high]) * scale


def forest(ax, points, y, color, marker="o", label=None):
    points = np.asarray(points)
    ax.hlines(y, points[:, 1], points[:, 2], color=color, lw=1.5, zorder=3)
    ax.scatter(points[:, 0], y, color=color, marker=marker, s=36,
               label=label, zorder=4)


def setup_forest(ax, labels=LABELS):
    y = np.arange(4)[::-1]
    ax.set_yticks(y, labels)
    ax.set_ylim(-.5, 3.5)
    ax.axvline(0, color="#6F7780", lw=.9)
    ax.grid(axis="x", color="#E5E9EE", lw=.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    return y


def save(fig, output, stem):
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{extension}", dpi=220, facecolor="white")
    plt.close(fig)


def main(source, output):
    output.mkdir(parents=True, exist_ok=True)
    records = json.loads((source / "analysis/summary.json").read_text())
    panels = {(x["mode"], x["source"], x["setting"]): x for x in records}
    current = [panels[("double_holdout", "", setting)] for setting in SETTINGS]
    headroom = {x["setting"]: x for x in json.loads((source / "headroom/summary.json").read_text())["settings"]}
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#24323E", "text.color": "#24323E",
                         "axes.edgecolor": "#ADB5BF", "xtick.color": "#46515D",
                         "ytick.color": "#46515D"})
    source_rows = []

    fig, (a, b) = plt.subplots(1, 2, figsize=(12.6, 5.8), gridspec_kw={"width_ratios": [1.12, 1]})
    fig.subplots_adjust(left=.11, right=.98, bottom=.24, top=.73, wspace=.50)
    y = setup_forest(a)
    a.set_xlim(0, 1.75)
    for offset, key, name, color in ((.14, "conditioned_only", "C correct; U wrong", BLUE),
                                      (-.14, "shared_only", "U correct; C wrong", ORANGE)):
        values = [headroom[s]["correctness_percent"][key] for s in SETTINGS]
        a.barh(y+offset, values, height=.23, color=color, alpha=.9, label=name, zorder=2)
        for yi, value, setting in zip(y+offset, values, SETTINGS):
            a.text(value+.025, yi, f"{value:.2f}", va="center", fontsize=9)
            source_rows.append(dict(figure="headroom_prediction", panel="A", setting=setting,
                                    metric=key, mean=value, low="", high="", unit="percent"))
    for yi, setting in zip(y, SETTINGS):
        value = headroom[setting]["metrics"]["accuracy"]["mean_per_seed_oracle_gain"]
        a.text(1.58, yi, f"+{value:.2f}", ha="center", va="center", fontsize=10,
               fontweight="bold", color="#333C45")
        source_rows.append(dict(figure="headroom_prediction", panel="A", setting=setting,
                                metric="mean_per_seed_oracle_gain", mean=value, low="", high="", unit="pp"))
    a.axvline(1.43, color="#D7DDE4", lw=.8)
    a.text(1.58, 3.65, "Ideal gain\n(pp)", ha="center", va="bottom", fontsize=9)
    a.set_xticks([0, .4, .8, 1.2])
    a.set_xlabel("Image–task cases (%)")
    a.set_title("A   Where C and U correct different errors", loc="left", pad=34,
                fontsize=11, fontweight="bold")
    a.legend(loc="upper left", bbox_to_anchor=(0, 1.12), ncol=1, frameon=False, fontsize=9)

    y = setup_forest(b)
    for offset, key, name, color, marker in (
            (.11, "content_increment_r2_raw", "Ridge", BLUE, "o"),
            (-.11, "tree_content_increment_r2_raw", "Shallow tree", PURPLE, "s")):
        points = [estimate(p, key) for p in current]
        forest(b, points, y+offset, color, marker, name)
        for setting, values in zip(SETTINGS, points):
            source_rows.append(dict(figure="headroom_prediction", panel="B", setting=setting,
                                    metric=key, mean=values[0], low=values[1], high=values[2], unit="R2"))
    b.set_xlabel(r"Added predictive value, $\Delta R^2$")
    b.set_title("B   Content adds information beyond context", loc="left", pad=34,
                fontsize=11, fontweight="bold")
    b.legend(loc="upper left", bbox_to_anchor=(0, 1.12), ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Content-based prediction of conditioning benefits", x=.54, y=.96, fontsize=15,
                 fontweight="bold")
    fig.text(.54, .895, "Development data · CLIP B/16 · block 6 · three fixed selector seeds",
             ha="center", fontsize=10.5, color="#566271")
    fig.text(.06, .12,
             "A  Ideal gain: per-example oracle over the better fixed C/U selector, computed per seed and then averaged.\n"
             "B  Outcome: C−U raw classification gap; predictors hold out both images and class combinations.\n"
             "     Context = seven selection statistics + projected full-image gap. Bars: four-setting-adjusted 95% image-bootstrap intervals.",
             fontsize=9, linespacing=1.55, va="center")
    save(fig, output, "headroom_prediction")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.1), gridspec_kw={"width_ratios": [1.17, 1.12, 1]})
    fig.subplots_adjust(left=.12, right=.98, bottom=.25, top=.73, wspace=.63)
    a, b, c = axes
    group_labels = []
    for setting, label, panel in zip(SETTINGS, LABELS, current):
        e = panel["estimates"]
        lo = 100*e["group_ridge_predicted_raw_gain_low_fraction"]["mean"]
        hi = 100*e["group_ridge_predicted_raw_gain_high_fraction"]["mean"]
        group_labels.append(f"{label}\n{lo:.0f}% low / {hi:.0f}% high")
        for name, value in (("low", lo), ("high", hi)):
            source_rows.append(dict(figure="groups_and_transfer", panel="A", setting=setting,
                                    metric="predicted_group_"+name+"_fraction", mean=value,
                                    low="", high="", unit="percent"))
    y = setup_forest(a, group_labels)
    a.tick_params(axis="y", labelsize=9)
    for offset, group, name, color, marker in ((.12, "high", "High predicted benefit", BLUE, "o"),
                                              (-.12, "low", "Low predicted benefit", ORANGE, "s")):
        key = f"group_ridge_predicted_raw_gain_{group}_raw"
        points = [estimate(p, key, 1000) for p in current]
        forest(a, points, y+offset, color, marker, name)
        for setting, point in zip(SETTINGS, points):
            source_rows.append(dict(figure="groups_and_transfer", panel="A", setting=setting,
                                    metric=key, mean=point[0], low=point[1], high=point[2], unit="gap_x1000"))
    a.set_xlabel("Actual C−U raw gap × 1,000")
    a.set_title("A   Predicted groups separate\nactual benefits", loc="left", pad=34,
                fontsize=11, fontweight="bold")
    a.legend(loc="upper left", bbox_to_anchor=(0, 1.15), frameon=False, fontsize=9)

    transfer_keys = (("pets_b16_49", "flowers_b16_49"), ("flowers_b16_49", "pets_b16_49"),
                     ("pets_b16_98", "flowers_b16_98"), ("flowers_b16_98", "pets_b16_98"))
    trans = [panels[("cross_dataset", s, t)] for s, t in transfer_keys]
    y = setup_forest(b, ("Pets → Flowers · 49", "Flowers → Pets · 49",
                         "Pets → Flowers · 98", "Flowers → Pets · 98"))
    b.tick_params(axis="y", labelsize=9)
    for offset, key, name, color, marker in ((.11, "content_increment_r2_raw", "Ridge", BLUE, "o"),
                                             (-.11, "tree_content_increment_r2_raw", "Shallow tree", PURPLE, "s")):
        points = [estimate(p, key) for p in trans]
        forest(b, points, y+offset, color, marker, name)
        for (src, dst), point in zip(transfer_keys, points):
            source_rows.append(dict(figure="groups_and_transfer", panel="B", setting=src+"->"+dst,
                                    metric=key, mean=point[0], low=point[1], high=point[2], unit="R2"))
    b.set_xlabel(r"Transferred benefit rule: $\Delta R^2$")
    b.set_title("B   Transfer across datasets\nand token budgets", loc="left", pad=34,
                fontsize=11, fontweight="bold")
    b.legend(loc="upper left", bbox_to_anchor=(0, 1.15), frameon=False, fontsize=9)

    y = setup_forest(c)
    key = "group_u_omitted_contrast_energy_high_minus_low_raw"
    points = [estimate(p, key, 1000) for p in current]
    forest(c, points, y, BLUE)
    c.set_xlabel("High−low omission: C−U gap × 1,000")
    c.set_title("C   Omitted contrast:\nFlowers and Pets differ", loc="left", pad=34,
                fontsize=11, fontweight="bold")
    for setting, point in zip(SETTINGS, points):
        source_rows.append(dict(figure="groups_and_transfer", panel="C", setting=setting,
                                metric=key, mean=point[0], low=point[1], high=point[2], unit="gap_x1000"))
    fig.suptitle("Predicting conditioning benefits within and across datasets", x=.53, y=.96,
                 fontsize=15, fontweight="bold")
    fig.text(.53, .895, "Development data · CLIP B/16 · positive raw gaps favor C over U",
             ha="center", fontsize=10.5, color="#566271")
    fig.text(.04, .125,
             "A  Ridge-predicted lower/upper groups; cutoffs learned in training folds. Labels show weighted group sizes.\n"
             "B  Benefit rule fitted on source only; the common prefix readout uses training images from both datasets. ΔR² is relative to context, not absolute R².\n"
             "C  High/low groups use U's omitted task-contrast energy. Panels A and C measure score-gap differences.\n"
             "All bars: four-setting-adjusted 95% image-bootstrap intervals; the readout, selectors and fitted benefit predictors are held fixed.",
             fontsize=9, linespacing=1.55, va="center")
    save(fig, output, "groups_and_transfer")
    with (output / "source_values.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(source_rows)
    (output / "sources.json").write_text(json.dumps({
        "analysis": "../analysis/summary.json", "headroom": "../headroom/summary.json",
        "script": "scripts/plot_content_benefit.py",
        "confidence_intervals": "four_setting_adjusted_ci95; oracle column shows point estimates",
        "data": "development panels; no reserved final-test data",
    }, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/conditioning-content-v1"))
    parser.add_argument("--output", type=Path, default=Path("results/conditioning-content-v1/figures"))
    args = parser.parse_args()
    main(args.source, args.output)
