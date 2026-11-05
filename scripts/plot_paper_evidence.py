"""Build the study overview and a compact evidence panel from saved analyses."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = (("pets", 49), ("pets", 98), ("flowers", 49), ("flowers", 98))
INK = "#25354A"
BLUE = "#3174A7"
ORANGE = "#CB6A32"
TEAL = "#267E79"
GRAY = "#7B8795"


def save(fig, out, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=240, facecolor="white")
    svg = out / f"{name}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def framework(out):
    fig, ax = plt.subplots(figsize=(13.2, 6.3))
    fig.subplots_adjust(left=.01, right=.99, top=.95, bottom=.025)
    ax.set(xlim=(0, 13.2), ylim=(0, 6.3))
    ax.axis("off")

    def text(x, y, s, size=11, color=INK, weight="normal", ha="center", **kwargs):
        ax.text(x, y, s, fontsize=size, color=color, weight=weight,
                ha=ha, va="center", **kwargs)

    def box(x, y, w, h, label, color=INK, fill="white", size=11):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=.03,rounding_size=.07", facecolor=fill,
                     edgecolor=color, linewidth=1.15))
        text(x + w / 2, y + h / 2, label, size=size, color=color)

    def arrow(a, b, color=GRAY, dashed=False, style="-|>", rad=0):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style,
                     connectionstyle=f"arc3,rad={rad}", mutation_scale=12,
                     linewidth=1.25, color=color, linestyle="--" if dashed else "-"))

    def grid(x, y, selected, color, size=.58):
        cell = size / 4
        for i in range(16):
            r, c = divmod(i, 4)
            ax.add_patch(Rectangle((x+c*cell, y+(3-r)*cell), cell-.025, cell-.025,
                         facecolor=color if i in selected else "#F4F6F8",
                         edgecolor="#BAC5D0", linewidth=.55))

    text(6.6, 6.07, "Same image, different candidate classes", 17, weight="bold")
    for x, w, title in ((.12, 3.65, "(a) Controlled inputs"),
                         (4.0, 4.30, "(b) Paired token selection"),
                         (8.55, 4.50, "(c) Three value tests")):
        ax.add_patch(FancyBboxPatch((x, .65), w, 4.90,
                     boxstyle="round,pad=.03,rounding_size=.09",
                     facecolor="#FAFBFC", edgecolor="#D9E0E6", linewidth=.8))
        text(x+w/2, 5.27, title, 13, weight="bold")

    # The patch grids illustrate positions only; they are not measured masks.
    grid(1.48, 4.18, range(16), "#D6E1EC", .78)
    text(1.87, 4.02, "one image x", 10)
    arrow((1.87, 3.85), (1.87, 3.62))
    box(.74, 3.02, 2.26, .57, "Frozen ViT prefix", fill="#EAF0F6")
    arrow((1.87, 3.01), (1.87, 2.68))
    box(.72, 2.12, 2.30, .53, "patch + CLS features H", fill="#F2F5F8", size=10.5)
    text(1.87, 1.78, "Change the competitor", 10.5, weight="bold")
    box(.40, 1.01, 1.42, .51, r"$T_A=\{a,b\}$", color=BLUE, fill="#E9F2FA")
    box(1.99, 1.01, 1.42, .51, r"$T_B=\{a,c\}$", color=BLUE, fill="#E9F2FA")
    text(1.87, .80, r"frozen text $\rightarrow$ descriptors $d_A,d_B$", 9.6)
    ax.plot([3.05, 4.16], [2.39, 2.39], color=GRAY, lw=1.25)
    ax.plot([4.16, 4.16], [1.535, 4.305], color=GRAY, lw=1.25)
    for y in (4.305, 3.115, 1.535):
        arrow((4.16, y), (4.35, y))
    text(3.62, 2.60, "same H", 9.3)

    # A single trained conditional model is called with two task descriptions.
    box(4.35, 4.02, 1.80, .57, r"$C(H,d_A)$", color=BLUE, fill="#E9F2FA", size=13)
    box(4.35, 2.83, 1.80, .57, r"$C(H,d_B)$", color=BLUE, fill="#E9F2FA", size=13)
    text(5.25, 3.72, "same trained selector", 9.6, color=BLUE)
    text(5.25, 4.82, r"$d_A$", 11, color=BLUE)
    arrow((5.25, 4.70), (5.25, 4.60), BLUE)
    text(6.25, 3.64, r"$d_B$", 11, color=BLUE)
    arrow((6.15, 3.59), (5.85, 3.41), BLUE)
    box(4.35, 1.25, 1.80, .57, r"$U(H)$", color=ORANGE, fill="#FFF1E7", size=13)
    text(5.25, 1.00, "shared across tasks", 9.6, color=ORANGE)
    for y, pattern, color, label in ((4.02, {1, 2, 5, 9}, BLUE, r"$S_A$"),
                                    (2.83, {2, 6, 10, 11}, BLUE, r"$S_B$"),
                                    (1.25, {1, 5, 6, 10}, ORANGE, r"$S_U$")):
        arrow((6.18, y+.285), (6.60, y+.285), color)
        grid(6.65, y-.01, pattern, color)
        text(7.43, y+.28, label, 12, color)
    text(6.93, 2.14, "same k", 10, weight="bold")
    text(6.10, .79, "C: 348k parameters   U: 347k parameters", 9.1)
    arrow((8.04, 2.70), (8.73, 2.70), color=INK)

    # The three checks correspond to three distinct empirical questions.
    text(8.82, 4.76, "1  Use", 12, weight="bold", ha="left")
    text(10.86, 4.76, "Exchange selections", 11)
    x0, y0, cw, ch = 10.02, 3.65, 1.13, .40
    text(9.55, y0+.60, r"$T_A$", 11)
    text(9.55, y0+.20, r"$T_B$", 11)
    text(x0+cw/2, y0+.97, r"$S_A$", 11, BLUE)
    text(x0+1.5*cw, y0+.97, r"$S_B$", 11, BLUE)
    for r in range(2):
        for c in range(2):
            ax.add_patch(Rectangle((x0+c*cw, y0+(1-r)*ch), cw, ch,
                         facecolor="#DDECF7" if r == c else "white", edgecolor="#B8C9D8", lw=.75))
            text(x0+(c+.5)*cw, y0+(1.5-r)*ch, "own" if r == c else "swapped", 9.8)
    text(8.82, 3.15, "2  Cost", 12, weight="bold", ha="left")
    text(10.87, 3.15, "Reallocate selector capacity", 10.5)
    box(9.48, 2.31, 1.18, .49, "C", BLUE, "#E9F2FA", 12)
    text(10.97, 2.56, "vs.", 10)
    box(11.29, 2.31, 1.18, .49, "wider U", ORANGE, "#FFF1E7", 10.7)
    text(10.98, 2.06, "comparable parameter budgets", 9.8)
    text(8.82, 1.64, "3  Value", 12, weight="bold", ha="left")
    text(11.03, 1.64, "Mixing, budget, depth", 10.6)
    text(10.94, 1.13, r"$C_s + U_t$    vs.    $U_s + U_t$", 12)
    text(10.94, .81, "same donor and retained-token budget", 9.4)

    text(6.6, .33, "Evaluate retained tokens through the same frozen ViT suffix and CLIP classifier; test unseen class combinations.", 10.5)
    save(fig, out, "framework_overview")


def load_evidence():
    rows, absolute = [], []
    for dataset, budget in SETTINGS:
        src = f"results/{dataset}-selector-ablations-v1/analysis.json"
        summary = json.loads((ROOT / src).read_text())["results"][str(budget)]
        hybrid_src = f"results/hybrid-specificity-v1/{dataset}_b16_{budget}/summary.json"
        hybrid = json.loads((ROOT / hybrid_src).read_text())
        exchange = next(v for v in summary["exchange"]["table"] if v["method"] == "conditioned")
        effects = (
            ("C_own_minus_swapped", exchange["own_minus_swapped"], src, 1),
            ("C_minus_original_U", summary["secondary_contrasts"]["conditioned_minus_image_only"], src, 1),
            ("C_minus_wide_U", summary["primary_contrasts"]["conditioned_minus_image_only_wide"], src, 2),
            ("CU_minus_UU", hybrid["classification"]["contrasts"]["cu_minus_uu"], hybrid_src, 4),
        )
        for contrast, stats, source, family in effects:
            for metric in ("margin", "accuracy"):
                v = stats[metric]
                adjusted = v.get("bonferroni_family_ci95", [None, None])
                rows.append(dict(dataset=dataset, budget=budget, depth=6,
                    contrast=contrast, metric=metric, estimate=v["mean"],
                    ci95_low=v["ci95"][0], ci95_high=v["ci95"][1],
                    adjusted_low=adjusted[0], adjusted_high=adjusted[1],
                    comparison_family=family, source=source))
        for v in summary["table"]:
            if v["method"] in ("conditioned", "image_only", "image_only_wide"):
                absolute.append(dict(dataset=dataset, budget=budget, depth=6, method=v["method"],
                    accuracy_percent=100*v["accuracy"]["mean"], geometric_margin=v["margin"]["mean"], source=src))
        for v in hybrid["classification"]["table"]:
            if v["method"] in ("cu_half", "uu_half"):
                absolute.append(dict(dataset=dataset, budget=budget, depth=6, method=v["method"],
                    accuracy_percent=100*v["accuracy"]["mean"], geometric_margin=v["margin"]["mean"], source=hybrid_src))
    return rows, absolute


def evidence_figure(out, rows):
    lookup = {(v["dataset"], v["budget"], v["contrast"], v["metric"]): v for v in rows}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False})
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.4))
    fig.subplots_adjust(left=.11, right=.97, top=.86, bottom=.20, hspace=.68, wspace=.30)
    configs = (("margin", (("C_own_minus_swapped", BLUE, 0),), "(a) Task information is used"),
               ("margin", (("C_minus_original_U", GRAY, -.12), ("C_minus_wide_U", ORANGE, .12)), "(b) Image capacity changes the comparison"),
               ("margin", (("CU_minus_UU", TEAL, 0),), "(c) Additional mixing value depends on setting"),
               ("accuracy", (("C_minus_wide_U", ORANGE, -.12), ("CU_minus_UU", TEAL, .12)), "(d) Accuracy effects remain small"))
    for ax, (metric, contrasts, title) in zip(axes.flat, configs):
        scale = 1000 if metric == "margin" else 100
        for i, (dataset, budget) in enumerate(SETTINGS):
            for contrast, color, offset in contrasts:
                v = lookup[dataset, budget, contrast, metric]
                y = i + offset
                if v["adjusted_low"] is not None:
                    ax.hlines(y, scale*v["adjusted_low"], scale*v["adjusted_high"],
                              color=color, linewidth=1.3, alpha=.5)
                lo, hi = scale*v["ci95_low"], scale*v["ci95_high"]
                ax.hlines(y, lo, hi, color=color, linewidth=3)
                ax.plot(scale*v["estimate"], y, "o", color=color, ms=5)
        ax.axvline(0, color="#AAB2BB", linestyle="--", lw=1)
        ax.set_yticks(range(4), [f"{'Pets' if d == 'pets' else 'Flowers'} / {k}" for d, k in SETTINGS])
        ax.set_ylim(3.6, -.6)
        ax.grid(axis="x", color="#EBEEF2", lw=.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, loc="left", fontsize=11.2, weight="bold", pad=13)
        ax.set_xlabel("Geometric-margin difference (×10⁻³)" if metric == "margin" else "Accuracy difference (percentage points)")
    axes[0, 0].text(.02, 1.015, "C with its own task minus C with exchanged task", transform=axes[0, 0].transAxes, fontsize=8.5, color=GRAY)
    handles = [Line2D([0], [0], color=GRAY, marker="o", lw=2, label="C − original U"),
               Line2D([0], [0], color=ORANGE, marker="o", lw=2, label="C − wider U"),
               Line2D([0], [0], color=TEAL, marker="o", lw=2, label="CU − UU")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.55, .925), ncol=3, frameon=False)
    fig.suptitle("Task utilization and value over shared selection are different questions", fontsize=15, weight="bold", y=.98)
    fig.text(.11, .075, "Frozen CLIP ViT-B/16; prune after block 6; 49 or 98 retained patches. C = conditioned; U = shared image-only.\n"
             "Development images and unseen class combinations. Points average three seeds (C/U) or six ordered seed pairs (CU/UU).\n"
             "Thick bars: 95% image-bootstrap intervals. Thin bars: original adjustments (C−wider U: 2 budgets; CU−UU: 4 settings).",
             fontsize=9.2, va="center", linespacing=1.5, color=INK)
    save(fig, out, "core_evidence")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/paper-evidence-v1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.labelcolor": INK,
                         "text.color": INK, "xtick.color": INK, "ytick.color": INK})
    rows, absolute = load_evidence()
    write_csv(args.output / "core_contrasts.csv", rows)
    write_csv(args.output / "core_results.csv", absolute)
    framework(args.output)
    evidence_figure(args.output, rows)
    print(f"Wrote two figures and two tables to {args.output}")


if __name__ == "__main__":
    main()
