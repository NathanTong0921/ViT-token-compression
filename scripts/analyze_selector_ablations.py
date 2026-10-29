"""Paired image-capacity and class-descriptor ablations on development tasks."""

import argparse
from collections import defaultdict
import csv
from itertools import combinations
import json
from pathlib import Path
import subprocess

import numpy as np

from analyze_selector_budgets import weighted_estimates
from analyze_selector_scale import panel_weights, read_rows


SEEDS = (2041, 2042, 2043)
METHODS = ("conditioned", "image_only", "image_only_wide", "mean_only", "difference_only")
SOURCE_METHODS = {"baseline": METHODS[:2], "ablations": METHODS[2:]}
METRICS = ("accuracy", "margin")
PRIMARY = {"conditioned_minus_image_only_wide": 2,
           "conditioned_minus_mean_only": 4,
           "conditioned_minus_difference_only": 4}
LABELS = {"conditioned": "Full conditioning", "image_only": "Image-only (original)",
          "image_only_wide": "Image-only (wide)", "mean_only": "Mean only",
          "difference_only": "Difference only"}


def align_rows(rows, seeds, budget):
    """Identity-align all five methods and all seeds on unseen validation tasks."""
    base = sorted([r for r in rows["baseline"] if r["task_split"] == "unseen"
                   and r["method"] == "conditioned" and int(r["seed"]) == seeds[0]],
                  key=lambda r: (r["image_id"], r["task_id"]))
    keys = [(r["image_id"], r["task_id"]) for r in base]
    assert keys and len(keys) == len(set(keys))
    metadata = {key: tuple(int(r[n]) for n in ("class_id", "class_a", "class_b"))
                for key, r in zip(keys, base)}
    values = {}
    for source, methods in SOURCE_METHODS.items():
        for method in methods:
            selected = [r for r in rows[source] if r["task_split"] == "unseen" and r["method"] == method]
            assert {int(r["seed"]) for r in selected} == set(seeds)
            runs = []
            for seed in seeds:
                run = [r for r in selected if int(r["seed"]) == seed]
                lookup = {(r["image_id"], r["task_id"]): r for r in run}
                assert len(run) == len(lookup) == len(keys) and set(lookup) == set(keys)
                for key, r in lookup.items():
                    assert tuple(int(r[n]) for n in ("class_id", "class_a", "class_b")) == metadata[key]
                    assert int(r["keep_patch_tokens"]) == budget
                runs.append([[float(lookup[key][m]) for m in ("correct", "margin")] for key in keys])
            values[method] = np.asarray(runs, float)
            assert np.isfinite(values[method]).all() and np.isin(values[method][:, :, 0], [0., 1.]).all()
    weights = panel_weights(base, "all")
    assert np.all(weights > 0)
    np.testing.assert_allclose(weights.sum(), 1, rtol=0, atol=1e-12)
    return base, values, weights


def contrasts(values):
    output = {f"conditioned_minus_{method}": values["conditioned"] - values[method]
              for method in METHODS[1:]}
    output["image_only_wide_minus_image_only"] = values["image_only_wide"] - values["image_only"]
    output["mean_only_minus_difference_only"] = values["mean_only"] - values["difference_only"]
    return output


def analyze_classification(rows, seeds, budget, replicates=2000, seed=2091):
    base, values, weights = align_rows(rows, seeds, budget)
    ids = np.asarray([r["image_id"] for r in base])
    differences = contrasts(values)
    estimates = {name: weighted_estimates(ids, weights, value.mean(axis=0), METRICS,
                                          replicates, seed, PRIMARY.get(name, 1))
                 for name, value in differences.items()}
    table = []
    for method, value in values.items():
        points = np.einsum("n,snm->sm", weights, value)
        table.append({"method": method, **weighted_estimates(ids, weights, value.mean(axis=0),
                                                             METRICS, replicates, seed),
                      "per_seed": [{"seed": s, **dict(zip(METRICS, p.tolist()))}
                                   for s, p in zip(seeds, points)]})
    cases = {"image_id": ids, "task_id": np.asarray([r["task_id"] for r in base]),
             "class_id": np.asarray([int(r["class_id"]) for r in base]),
             "class_a": np.asarray([int(r["class_a"]) for r in base]),
             "class_b": np.asarray([int(r["class_b"]) for r in base]), "task_class_weight": weights}
    for method, value in {**values, **differences}.items():
        for index, metric in enumerate(METRICS):
            cases[f"{method}_{metric}"] = value.mean(axis=0)[:, index]
    packed = {**cases, "training_seeds": np.asarray(seeds),
              **{f"{method}_per_seed": value for method, value in values.items()}}
    summary = {"image_count": len(set(ids)), "image_task_examples": len(ids),
               "task_count": len({r["task_id"] for r in base}), "table": table,
               "primary_contrasts": {k: estimates[k] for k in PRIMARY},
               "secondary_contrasts": {k: v for k, v in estimates.items() if k not in PRIMARY},
               "per_seed": [{"seed": s, "contrasts": {name: dict(zip(METRICS,
                   np.einsum("n,nm->m", weights, value[si]).tolist()))
                   for name, value in differences.items()}} for si, s in enumerate(seeds)]}
    return summary, cases, packed, base, values


def binary_scores(scores, labels, competitors, text):
    """Seed x example score vectors; ties go to the lower class index."""
    rows = np.arange(len(labels))
    own, other = scores[:, rows, labels], scores[:, rows, competitors]
    margin = (own - other) / np.linalg.norm(text[labels] - text[competitors], axis=1)
    correct = (own > other) | ((own == other) & (labels < competitors))
    return np.stack((correct.astype(float), margin), axis=-1)


def aligned_raw(sources, base, seeds, budget, values):
    """Load each required compressed array once, then align raw score vectors."""
    keys = [(r["image_id"], r["task_id"]) for r in base]
    labels = np.asarray([int(r["class_id"]) for r in base])
    pairs = np.asarray([[int(r["class_a"]), int(r["class_b"])] for r in base])
    competitors = np.where(labels == pairs[:, 0], pairs[:, 1], pairs[:, 0])
    aligned, text = {}, None
    names = ("row_id", "image_ids", "task_ids", "task_split", "method", "seed", "labels",
             "class_pairs", "keep_patch_tokens", "cosine_scores", "patch_indices")
    for source, methods in SOURCE_METHODS.items():
        with np.load(sources[source] / "full" / "predictions.npz") as archive:
            take = (archive["task_split"] == "unseen") & np.isin(archive["method"], methods)
            raw = {name: archive[name][take] for name in names}
            current_text = archive["text_features"].astype(float)
        if text is None:
            text = current_text
        else:
            np.testing.assert_array_equal(text, current_text)
        for method in methods:
            score_runs, position_runs, source_rows = [], [], []
            for seed in seeds:
                rows = np.flatnonzero((raw["method"] == method) & (raw["seed"] == seed))
                lookup = {(str(raw["image_ids"][i]), str(raw["task_ids"][i])): i for i in rows}
                assert len(rows) == len(lookup) == len(keys) and set(lookup) == set(keys)
                order = np.asarray([lookup[key] for key in keys])
                np.testing.assert_array_equal(raw["labels"][order], labels)
                np.testing.assert_array_equal(raw["class_pairs"][order], pairs)
                assert np.all(raw["keep_patch_tokens"][order] == budget)
                positions = raw["patch_indices"][order]
                assert np.all(positions[:, :budget] >= 0) and np.all(positions[:, budget:] == -1)
                assert np.all(np.diff(positions[:, :budget], axis=1) > 0)
                score_runs.append(raw["cosine_scores"][order].astype(float))
                position_runs.append(positions)
                source_rows.append(raw["row_id"][order])
            scores = np.stack(score_runs)
            reconstructed = binary_scores(scores, labels, competitors, text)
            np.testing.assert_array_equal(reconstructed[:, :, 0], values[method][:, :, 0])
            np.testing.assert_allclose(reconstructed[:, :, 1], values[method][:, :, 1], rtol=0, atol=2e-7)
            aligned[method] = {"scores": scores, "positions": np.stack(position_runs),
                               "source_rows": np.stack(source_rows)}
    return aligned, text


def exchange_pairs(base):
    by_image = defaultdict(list)
    for index, row in enumerate(base):
        by_image[row["image_id"]].append(index)
    pairs = [(a, b) for image in sorted(by_image) for a, b in combinations(by_image[image], 2)]
    assert pairs
    a, b = np.asarray(pairs, dtype=int).T
    labels = np.asarray([int(r["class_id"]) for r in base])
    np.testing.assert_array_equal(labels[a], labels[b])
    return a, b


def analyze_exchange(base, raw, text, seeds, budget, replicates=2000, seed=2092):
    """Rescore own/exchanged tasks; average seeds and task pairs within image."""
    a, b = exchange_pairs(base)
    labels = np.asarray([int(r["class_id"]) for r in base])
    pairs = np.asarray([[int(r["class_a"]), int(r["class_b"])] for r in base])
    other = np.where(labels == pairs[:, 0], pairs[:, 1], pairs[:, 0])
    ids = np.asarray([base[i]["image_id"] for i in a])
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    weights = 1 / (len(counts) * counts[inverse])
    cases = {"image_id": ids, "task_a": np.asarray([base[i]["task_id"] for i in a]),
             "task_b": np.asarray([base[i]["task_id"] for i in b]),
             "class_id": labels[a], "competitor_a": other[a], "competitor_b": other[b],
             "image_pair_weight": weights}
    packed = {"training_seeds": np.asarray(seeds)}
    effects, table, invariant = {}, [], {}
    for method in METHODS:
        scores, positions = raw[method]["scores"], raw[method]["positions"]
        own_a = binary_scores(scores[:, a], labels[a], other[a], text)
        own_b = binary_scores(scores[:, b], labels[b], other[b], text)
        swap_a = binary_scores(scores[:, b], labels[a], other[a], text)
        swap_b = binary_scores(scores[:, a], labels[b], other[b], text)
        own, swapped = (own_a + own_b) / 2, (swap_a + swap_b) / 2
        effects[method] = own - swapped
        masks = np.zeros((*positions.shape[:2], positions.shape[-1]), dtype=bool)
        np.put_along_axis(masks, positions[:, :, :budget].astype(int), True, axis=2)
        overlap = (masks[:, a] & masks[:, b]).sum(axis=2) / budget
        table.append({"method": method,
                      "own": weighted_estimates(ids, weights, own.mean(axis=0), METRICS, replicates, seed),
                      "swapped": weighted_estimates(ids, weights, swapped.mean(axis=0), METRICS, replicates, seed),
                      "own_minus_swapped": weighted_estimates(ids, weights, effects[method].mean(axis=0),
                                                              METRICS, replicates, seed),
                      "token_overlap": weighted_estimates(ids, weights, overlap.mean(axis=0),
                                                           ["fraction"], replicates, seed)["fraction"],
                      "per_seed": [{"seed": s, **dict(zip(METRICS,
                          np.einsum("n,nm->m", weights, effects[method][si]).tolist()))}
                                   for si, s in enumerate(seeds)]})
        for name, value in (("own", own), ("swapped", swapped), ("own_minus_swapped", effects[method])):
            packed[f"{method}_{name}_per_seed"] = value
            for mi, metric in enumerate(METRICS):
                cases[f"{method}_{name}_{metric}"] = value.mean(axis=0)[:, mi]
        packed[f"{method}_overlap_per_seed"] = overlap
        packed[f"{method}_source_row_a"] = raw[method]["source_rows"][:, a]
        packed[f"{method}_source_row_b"] = raw[method]["source_rows"][:, b]
        if method in ("image_only", "image_only_wide"):
            score_error = float(np.max(np.abs(scores[:, a] - scores[:, b])))
            same_positions = bool(np.array_equal(positions[:, a], positions[:, b]))
            assert score_error == 0 and same_positions
            np.testing.assert_array_equal(effects[method], 0)
            invariant[method] = {"max_abs_cross_task_score_difference": score_error,
                                 "identical_positions": same_positions}
    contrasts = {f"conditioned_minus_{method}": weighted_estimates(
        ids, weights, (effects["conditioned"] - effects[method]).mean(axis=0), METRICS, replicates, seed)
                 for method in ("mean_only", "difference_only")}
    summary = {"image_count": len(counts), "image_task_pairs": len(a),
               "aggregation": "Equal images, then equal same-image task pairs; fixed seeds averaged before image bootstrap.",
               "bootstrap_seed": seed, "table": table, "secondary_effect_contrasts": contrasts,
               "image_only_invariance": invariant}
    return summary, cases, {**cases, **packed}


def write_cases(path, cases):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(cases)
        writer.writerows(zip(*(cases[name] for name in cases)))


def run(dataset, sources, output, replicates=2000, seed=2091):
    assert set(sources) == set(SOURCE_METHODS)
    assert set(sources["baseline"]) == set(sources["ablations"]) == {49, 98}
    output.mkdir(parents=True, exist_ok=True)
    result = {"dataset": dataset, "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "budgets": [49, 98], "training_seeds": list(SEEDS), "methods": list(METHODS),
              "scope": "Existing development images and unseen validation class combinations only.",
              "aggregation": "Task-equal and class-within-task-equal classification; average per-example metrics over fixed seeds before image resampling.",
              "uncertainty": "Paired image bootstrap conditional on these tasks and seeds. Capacity: 2-budget family per dataset/metric. Components: 4-comparison family per dataset/metric. Exchange contrasts are secondary ordinary 95% intervals.",
              "metric_units": {"accuracy": "fraction; multiply effects and intervals by 100 for percentage points", "margin": "signed normalized geometric margin"},
              "bootstrap_replicates": replicates, "bootstrap_seed": seed, "exchange_bootstrap_seed": 2092,
              "sources": {}, "results": {}}
    previous_panel = None
    for budget in (49, 98):
        paths = {name: value[budget] for name, value in sources.items()}
        rows = {name: read_rows(path / "full" / "predictions.csv") for name, path in paths.items()}
        summary, cases, packed, base, values = analyze_classification(rows, SEEDS, budget, replicates, seed)
        panel = [(r["image_id"], r["task_id"], int(r["class_id"])) for r in base]
        if previous_panel is not None:
            assert panel == previous_panel
        previous_panel = panel
        raw, text = aligned_raw(paths, base, SEEDS, budget, values)
        exchange, ecases, epacked = analyze_exchange(base, raw, text, SEEDS, budget, replicates, 2092)
        folder = output / f"budget_{budget}"
        folder.mkdir(exist_ok=True)
        write_cases(folder / "paired_cases.csv", cases)
        np.savez_compressed(folder / "paired_cases.npz", **packed)
        write_cases(folder / "exchange_cases.csv", ecases)
        np.savez_compressed(folder / "exchange_cases.npz", **epacked)
        summary["exchange"] = exchange
        result["results"][str(budget)] = summary
        result["sources"][str(budget)] = {name: str(path) for name, path in paths.items()}
        print(json.dumps({"dataset": dataset, "budget": budget,
                          "primary_contrasts": summary["primary_contrasts"]}), flush=True)
    (output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def plot(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"image_only_wide": "#426c9e", "mean_only": "#ca8033", "difference_only": "#2b8d80",
              "conditioned": "#80529b", "image_only": "#999999"}
    budgets = result["budgets"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    comparisons = [(k, method) for k in budgets for method in ("image_only_wide", "mean_only", "difference_only")]
    for ax, metric, scale in zip(axes, METRICS, (100, 1)):
        for y, (budget, method) in enumerate(comparisons):
            estimate = result["results"][str(budget)]["primary_contrasts"][f"conditioned_minus_{method}"][metric]
            mean = estimate["mean"] * scale
            for interval, color, width in (("bonferroni_family_ci95", "#c6c6c6", 4), ("ci95", colors[method], 1.5)):
                lo, hi = np.asarray(estimate[interval]) * scale
                ax.plot([lo, hi], [y, y], color=color, linewidth=width, solid_capstyle="round")
            ax.plot(mean, y, "o", color=colors[method], markersize=5)
        ax.axvline(0, color="#555555", linewidth=.8)
        ax.set(yticks=range(len(comparisons)), yticklabels=[f"{k}: full − {LABELS[m]}" for k, m in comparisons],
               xlabel="Accuracy difference (percentage points)" if metric == "accuracy" else "Geometric margin difference")
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=.2)
    fig.suptitle(f"{result['dataset'].title()}: capacity and descriptor ablations")
    fig.text(.5, .01, "Colored: pointwise 95%; grey: capacity 97.5% / component 98.75% intervals. Positive favors full conditioning.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .045, 1, .95])
    for ext in ("png", "pdf"):
        fig.savefig(output / f"ablation_classification.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))
    for ax, metric, scale in zip(axes, METRICS, (100, 1)):
        for mi, method in enumerate(METHODS):
            offset = (mi - 2) * .12
            for ki, budget in enumerate(budgets):
                record = next(row for row in result["results"][str(budget)]["exchange"]["table"] if row["method"] == method)
                estimate = record["own_minus_swapped"][metric]
                point = estimate["mean"] * scale
                lo, hi = np.asarray(estimate["ci95"]) * scale
                ax.plot([ki + offset] * 2, [lo, hi], color=colors[method], linewidth=1.4)
                ax.plot(ki + offset, point, "o", color=colors[method], markersize=5,
                        label=LABELS[method] if ki == 0 else None)
        ax.axhline(0, color="#555555", linewidth=.8)
        ax.set(xticks=range(len(budgets)), xticklabels=budgets, xlabel="Retained patch tokens",
               ylabel="Own − exchanged accuracy (percentage points)" if metric == "accuracy" else "Own − exchanged geometric margin")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{result['dataset'].title()}: task exchange after descriptor ablation")
    fig.text(.5, .01, "Image-equal task-pair panel; ordinary 95% paired image-bootstrap intervals.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, .95])
    for ext in ("png", "pdf"):
        fig.savefig(output / f"ablation_exchange.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_sources(items):
    output = {}
    for value in items:
        budget, path = value.split("=", 1)
        assert int(budget) not in output
        output[int(budget)] = Path(path)
    return output


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("pets", "flowers"))
    parser.add_argument("--baseline", action="append", help="budget=existing KL evaluation directory")
    parser.add_argument("--ablations", action="append", help="budget=new three-variant evaluation directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2091)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    return parser


if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()
    if args.plot_only:
        result = json.loads((args.output / "analysis.json").read_text())
    else:
        if not args.dataset or not args.baseline or not args.ablations:
            parser.error("--dataset, --baseline and --ablations are required unless --plot-only is used")
        result = run(args.dataset, {"baseline": parse_sources(args.baseline), "ablations": parse_sources(args.ablations)},
                     args.output, args.replicates, args.seed)
    if not args.skip_plots:
        plot(result, args.output)
