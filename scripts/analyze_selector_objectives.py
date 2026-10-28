"""Paired 2x2 comparison of task conditioning and selector training objective.

Example:
  PYTHONPATH=src python scripts/analyze_selector_objectives.py --dataset pets \
    --kl 49=runs/pets-selector-scale-eval-v1 \
    --kl 98=runs/pets-selector-budget-98-eval-v1 \
    --rank 49=runs/pets-selector-rank-49-eval-v1 \
    --rank 98=runs/pets-selector-rank-98-eval-v1 \
    --output results/pets-selector-objectives-v1 --skip-plots
"""

import argparse
import csv
import json
from pathlib import Path
import subprocess

import numpy as np

from analyze_selector_budgets import weighted_estimates
from analyze_selector_scale import panel_weights, read_rows


OBJECTIVES = ("kl", "rank")
ARMS = ("conditioned", "image_only")
METRICS = ("accuracy", "margin")
DIAGNOSTICS = ("teacher_kl", "teacher_topk_overlap", "teacher_additive_utility_regret",
               "teacher_boundary_violation_fraction", "teacher_topk_rank_loss",
               "teacher_topk_tie_aware_overlap", "teacher_boundary_tie_fraction")


def align_objectives(rows, seeds, budget):
    """Align by identities, rejecting duplicated/missing rows or changed labels."""
    first = [r for r in rows["kl"] if r["task_split"] == "unseen"
             and r["method"] == "conditioned" and int(r["seed"]) == seeds[0]]
    first.sort(key=lambda r: (r["image_id"], r["task_id"]))
    keys = [(r["image_id"], r["task_id"]) for r in first]
    assert keys and len(keys) == len(set(keys))
    metadata = {key: tuple(int(r[k]) for k in ("class_id", "class_a", "class_b"))
                for key, r in zip(keys, first)}
    values = {}
    for objective in OBJECTIVES:
        selected = [r for r in rows[objective] if r["task_split"] == "unseen"]
        assert {int(r["seed"]) for r in selected if r["method"] in ARMS} == set(seeds)
        values[objective] = {}
        for arm in ARMS:
            runs = []
            for seed in seeds:
                run = [r for r in selected if r["method"] == arm and int(r["seed"]) == seed]
                mapping = {(r["image_id"], r["task_id"]): r for r in run}
                assert len(run) == len(mapping) == len(keys)
                assert set(mapping) == set(keys)
                for key, r in mapping.items():
                    assert tuple(int(r[k]) for k in ("class_id", "class_a", "class_b")) == metadata[key]
                    assert int(r["keep_patch_tokens"]) == budget
                array = np.asarray([[float(mapping[key][m]) for m in ("correct", "margin")]
                                    for key in keys])
                assert np.isfinite(array).all() and np.isin(array[:, 0], [0., 1.]).all()
                runs.append(array)
            values[objective][arm] = np.stack(runs)
    weights = panel_weights(first, "all")
    assert np.all(weights > 0)
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-12, rtol=0)
    return first, values, weights


def objective_contrasts(values):
    """Each array is paired seed x image/task x metric, before seed averaging."""
    result = {f"{objective}_conditioned_minus_image_only":
              values[objective]["conditioned"] - values[objective]["image_only"]
              for objective in OBJECTIVES}
    result.update({f"{arm}_rank_minus_kl": values["rank"][arm] - values["kl"][arm]
                   for arm in ARMS})
    result["interaction"] = (result["rank_conditioned_minus_image_only"]
                             - result["kl_conditioned_minus_image_only"])
    return result


def analyze_budget(rows, seeds, budget, replicates=2000, seed=2085, family_size=2):
    base, values, weights = align_objectives(rows, seeds, budget)
    ids = np.asarray([r["image_id"] for r in base])
    differences = objective_contrasts(values)
    contrasts = {}
    for name, data in differences.items():
        contrasts[name] = weighted_estimates(ids, weights, data.mean(axis=0), METRICS,
                                             replicates, seed, family_size if name == "interaction" else 1)
    table = []
    for objective in OBJECTIVES:
        for arm in ARMS:
            data = values[objective][arm]
            estimates = weighted_estimates(ids, weights, data.mean(axis=0), METRICS, replicates, seed)
            points = np.einsum("n,snm->sm", weights, data)
            table.append({"objective": objective, "arm": arm, **estimates,
                          "per_seed": [{"seed": s, **dict(zip(METRICS, point.tolist()))}
                                       for s, point in zip(seeds, points)]})
    per_seed = []
    for si, training_seed in enumerate(seeds):
        per_seed.append({"seed": training_seed, "contrasts": {
            name: dict(zip(METRICS, np.einsum("n,nm->m", weights, data[si]).tolist()))
            for name, data in differences.items()}})
    cases = {"image_id": ids, "task_id": np.asarray([r["task_id"] for r in base]),
             "class_id": np.asarray([int(r["class_id"]) for r in base]),
             "class_a": np.asarray([int(r["class_a"]) for r in base]),
             "class_b": np.asarray([int(r["class_b"]) for r in base]),
             "task_class_weight": weights}
    for objective in OBJECTIVES:
        for arm in ARMS:
            for mi, metric in enumerate(METRICS):
                cases[f"{objective}_{arm}_{metric}"] = values[objective][arm].mean(axis=0)[:, mi]
    for name, data in differences.items():
        for mi, metric in enumerate(METRICS):
            cases[f"{name}_{metric}"] = data.mean(axis=0)[:, mi]
    packed = {**cases, "training_seeds": np.asarray(seeds),
              **{f"{objective}_{arm}_per_seed": values[objective][arm]
                 for objective in OBJECTIVES for arm in ARMS}}
    summary = {"image_count": len(set(ids)), "image_task_examples": len(ids),
               "task_count": len({r["task_id"] for r in base}), "table": table,
               "primary_interaction": contrasts.pop("interaction"),
               "secondary_contrasts": contrasts, "per_seed": per_seed}
    return summary, cases, packed, base, weights


def teacher_metrics(logits, utility, k, temperature=1.):
    """Common diagnostics for either loss; teacher utility is not real CLIP payoff.

    Boundary ties are expanded symmetrically, matching the training loss. A flat
    teacher has zero ranking loss/violations and perfect tie-aware agreement;
    its canonical overlap is index-based and not a preferred target.
    """
    logits, utility = np.asarray(logits, float), np.asarray(utility, float)
    assert logits.shape == utility.shape and logits.ndim == 2
    teacher_order = np.argsort(-utility, axis=1, kind="stable")
    selected = np.argsort(-logits, axis=1, kind="stable")[:, :k]
    rows = np.arange(len(logits))[:, None]
    top, outside = teacher_order[:, :k], teacher_order[:, k:]
    teacher_sum = utility[rows, top].sum(axis=1)
    selected_sum = utility[rows, selected].sum(axis=1)
    regret = teacher_sum - selected_sum
    assert np.all(regret >= -1e-10)
    teacher_selected = np.zeros_like(logits, bool)
    teacher_selected[rows, top] = True
    overlap = teacher_selected[rows, selected].mean(axis=1)
    standardized = utility - utility.mean(axis=1, keepdims=True)
    standardized /= np.maximum(utility.std(axis=1, keepdims=True), 1e-6)
    target_logit = standardized / temperature
    target_logit -= target_logit.max(axis=1, keepdims=True)
    target_log_probability = target_logit - np.log(np.exp(target_logit).sum(axis=1, keepdims=True))
    predicted = logits / temperature
    predicted -= predicted.max(axis=1, keepdims=True)
    predicted_log_probability = predicted - np.log(np.exp(predicted).sum(axis=1, keepdims=True))
    kl = (np.exp(target_log_probability) * (target_log_probability - predicted_log_probability)).sum(axis=1)
    cutoff = utility[rows, top[:, -1:]]
    tied_boundary = cutoff[:, 0] == utility[rows, outside[:, :1]][:, 0]
    possible_in = utility >= cutoff
    possible_out = (utility < cutoff) | (tied_boundary[:, None] & (utility == cutoff))
    score_difference = logits[:, None, :] - logits[:, :, None]
    valid = (possible_in[:, :, None] & possible_out[:, None, :]
             & (utility[:, :, None] > utility[:, None, :]))
    count = valid.sum(axis=(1, 2))
    denominator = np.maximum(count, 1)
    rank = (np.logaddexp(0, score_difference) * valid).sum(axis=(1, 2)) / denominator
    violation = ((score_difference >= 0) & valid).sum(axis=(1, 2)) / denominator
    selected_utility = utility[rows, selected]
    strict_hits = (selected_utility > cutoff).sum(axis=1)
    tied_hits = (selected_utility == cutoff).sum(axis=1)
    remaining = k - (utility > cutoff).sum(axis=1)
    tie_aware_overlap = (strict_hits + np.minimum(tied_hits, remaining)) / k
    return np.column_stack((kl, overlap, regret, violation, rank,
                            tie_aware_overlap, tied_boundary.astype(float))), count


def common_teacher_diagnostics(source, cache, base, seeds, budget, temperature):
    """Read only unseen validation rows; no new model inference or teacher fit."""
    with np.load(cache / "val" / "teacher.npz") as data:
        teacher_ids = data["image_ids"].copy()
        coefficients = data["coefficients"].copy()
    with np.load(cache / "val" / "evidence.npz") as data:
        np.testing.assert_array_equal(data["image_ids"], teacher_ids)
        text = data["text_features"].copy()
    lookup = {str(value): i for i, value in enumerate(teacher_ids)}
    ix = np.asarray([lookup[r["image_id"]] for r in base])
    labels = np.asarray([int(r["class_id"]) for r in base])
    competitors = np.asarray([int(r["class_b"]) if int(r["class_id"]) == int(r["class_a"])
                              else int(r["class_a"]) for r in base])
    norms = np.linalg.norm(text[labels] - text[competitors], axis=1)
    utilities = np.stack([(coefficients[i, :, a] - coefficients[i, :, b]) / n
                          for i, a, b, n in zip(ix, labels, competitors, norms)])
    keys = [(r["image_id"], r["task_id"]) for r in base]
    output = {}
    with np.load(source / "full" / "predictions.npz") as data:
        chosen = data["task_split"] == "unseen"
        selected_rows = np.flatnonzero(chosen)
        index = {(str(data["image_ids"][i]), str(data["task_ids"][i]),
                  str(data["method"][i]), int(data["seed"][i])): int(i) for i in selected_rows}
        assert len(index) == len(selected_rows)
        score_array = data["selection_logits"]
        for arm in ARMS:
            runs = []
            for training_seed in seeds:
                aligned = np.asarray([index[im, task, arm, training_seed] for im, task in keys])
                assert np.all(data["keep_patch_tokens"][aligned] == budget)
                np.testing.assert_array_equal(data["labels"][aligned], labels)
                np.testing.assert_array_equal(data["class_pairs"][aligned],
                                              [[int(r["class_a"]), int(r["class_b"])] for r in base])
                chunks = []
                for start in range(0, len(aligned), 64):
                    end = min(start + 64, len(aligned))
                    metrics, _ = teacher_metrics(score_array[aligned[start:end]], utilities[start:end],
                                                  budget, temperature)
                    chunks.append(metrics)
                runs.append(np.concatenate(chunks))
            output[arm] = np.stack(runs)
    return output, {"flat_teacher_examples": int(np.sum(np.ptp(utilities, axis=1) == 0)),
                    "teacher_boundary_tied_examples": int(np.sum(
                        np.sort(utilities, axis=1)[:, -budget] == np.sort(utilities, axis=1)[:, -budget - 1]))}


def write_csv(path, columns):
    names = list(columns)
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(names)
        writer.writerows(zip(*(columns[name] for name in names)))


def parse_sources(values):
    result = {}
    for value in values or []:
        budget, path = value.split("=", 1)
        assert int(budget) not in result
        result[int(budget)] = Path(path)
    return result


def run(dataset, sources, caches, output, replicates=2000, seed=2085):
    budgets = sorted(sources["kl"])
    assert budgets == [49, 98] and sorted(sources["rank"]) == budgets
    assert not caches or sorted(caches) == budgets
    output.mkdir(parents=True, exist_ok=True)
    result = {"dataset": dataset, "code_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(), "budgets": budgets,
        "primary": "[(conditioned - image_only)_rank - (conditioned - image_only)_KL] on unseen validation combinations.",
        "aggregation": "Equal tasks, equal class sides within each task, then equal images within side. Fixed training seeds are averaged before image-cluster resampling.",
        "uncertainty": "Paired image-cluster weighted ratio bootstrap conditional on tasks and three fixed seeds. Primary 97.5% intervals adjust across two budgets separately for accuracy and margin within this dataset; secondary intervals are pointwise 95%.",
        "bootstrap_replicates": replicates, "bootstrap_seed": seed, "scope": "Validation only; test images and reserved test combinations are not read.",
        "training_objective_caution": "Raw KL and ranking training losses have different targets/scales and are not compared. Common teacher diagnostics and actual pruned-CLIP classification are reported separately.",
        "sources": {objective: {str(k): str(v) for k, v in items.items()} for objective, items in sources.items()},
        "results": {}}
    original_keys = None
    for budget in budgets:
        metadata = {objective: json.loads((sources[objective][budget] / "eval_summary.json").read_text())
                    for objective in OBJECTIVES}
        configs = {objective: item["config"] for objective, item in metadata.items()}
        seeds = configs["kl"]["training_seeds"]
        assert seeds == [2041, 2042, 2043]
        for key in ("model", "prune_after", "keep_patch_tokens", "prompt_template", "manifest_dir",
                    "training_seeds", "hidden_dim", "training_steps", "batch_size", "temperature",
                    "learning_rate", "weight_decay", "cache_dir"):
            assert configs["kl"][key] == configs["rank"][key], key
        assert configs["kl"]["keep_patch_tokens"] == budget
        rows = {objective: read_rows(sources[objective][budget] / "full" / "predictions.csv")
                for objective in OBJECTIVES}
        summary, cases, packed, base, weights = analyze_budget(rows, seeds, budget, replicates, seed, len(budgets))
        keys = list(zip(cases["image_id"], cases["task_id"]))
        if original_keys is None:
            original_keys = keys
        else:
            assert keys == original_keys
        result["training_seeds"] = seeds
        result["results"][str(budget)] = summary
        if caches:
            diagnostics, checks = {}, {}
            for objective in OBJECTIVES:
                diagnostics[objective], checks[objective] = common_teacher_diagnostics(
                    sources[objective][budget], caches[budget], base, seeds, budget, configs[objective]["temperature"])
            assert checks["kl"] == checks["rank"]
            diagnostic_differences = objective_contrasts(diagnostics)
            diagnostic_summary = {"definition": "Teacher-derived diagnostics, not actual classifier margin or accuracy. Utility regret compares additive teacher sums at equal k, not true CLIP rewards. Stable-index overlap may be arbitrary for tied teachers.",
                                  "checks": checks["kl"], "table": [], "secondary_contrasts": {}}
            for objective in OBJECTIVES:
                for arm in ARMS:
                    data = diagnostics[objective][arm]
                    estimates = weighted_estimates(cases["image_id"], weights, data.mean(axis=0),
                                                    DIAGNOSTICS, replicates, seed)
                    diagnostic_summary["table"].append({"objective": objective, "arm": arm, **estimates})
                    packed[f"{objective}_{arm}_teacher_diagnostics_per_seed"] = data
                    for mi, name in enumerate(DIAGNOSTICS):
                        cases[f"{objective}_{arm}_{name}"] = data.mean(axis=0)[:, mi]
            for name, data in diagnostic_differences.items():
                diagnostic_summary["secondary_contrasts"][name] = weighted_estimates(
                    cases["image_id"], weights, data.mean(axis=0), DIAGNOSTICS, replicates, seed)
            summary["common_teacher_diagnostics"] = diagnostic_summary
            packed["teacher_diagnostic_names"] = np.asarray(DIAGNOSTICS)
        folder = output / f"budget_{budget}"
        folder.mkdir(exist_ok=True)
        write_csv(folder / "paired_cases.csv", cases)
        np.savez_compressed(folder / "paired_cases.npz", **packed)
    (output / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def plot(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    budgets = result["budgets"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, metric, scale in zip(axes, METRICS, (100, 1)):
        for index, budget in enumerate(budgets):
            estimate = result["results"][str(budget)]["primary_interaction"][metric]
            value = scale * estimate["mean"]
            for key, color, width in (("bonferroni_family_ci95", "#b7c6d8", 5), ("ci95", "#326a9d", 2)):
                low, high = scale * np.asarray(estimate[key])
                ax.vlines(index, low, high, color=color, linewidth=width,
                          label="Two-budget adjusted" if key.startswith("bonferroni") else "Pointwise 95%" if index == 0 else None)
            ax.plot(index, value, "o", color="#326a9d")
        ax.axhline(0, color=".5", linewidth=1)
        ax.set(xticks=range(len(budgets)), xticklabels=budgets, xlabel="Retained patch tokens",
               ylabel="Change in conditioning advantage (pp)" if metric == "accuracy" else "Change in conditioning advantage (margin)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), frameon=False, fontsize=8)
    fig.suptitle(f"{result['dataset']}: (conditioned - image-only), ranking minus KL")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"objective_interaction.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for objective, style in (("kl", "--"), ("rank", "-")):
        for arm, color in (("conditioned", "#d27730"), ("image_only", "#326a9d")):
            records = [next(r for r in result["results"][str(k)]["table"]
                            if r["objective"] == objective and r["arm"] == arm) for k in budgets]
            for ax, metric, scale in zip(axes, METRICS, (100, 1)):
                ax.plot(budgets, [r[metric]["mean"] * scale for r in records], marker="o", color=color,
                        linestyle=style, label=f"{objective.upper()} / {arm.replace('_', ' ')}")
    for ax, ylabel in zip(axes, ("Task/class-balanced accuracy (%)", "Mean signed geometric margin")):
        ax.set(xticks=budgets, xlabel="Retained patch tokens", ylabel=ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{result['dataset']}: matched objective and conditioning comparison")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"objective_performance.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("pets", "flowers"))
    parser.add_argument("--kl", action="append", help="budget=existing KL evaluation directory")
    parser.add_argument("--rank", action="append", help="budget=new ranking evaluation directory")
    parser.add_argument("--cache", action="append", help="budget=shared teacher cache; enables common teacher diagnostics")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2085)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    return parser


if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()
    if args.plot_only:
        result = json.loads((args.output / "analysis.json").read_text())
    else:
        if not args.dataset or not args.kl or not args.rank:
            parser.error("--dataset, --kl and --rank are required unless --plot-only is used")
        result = run(args.dataset, {"kl": parse_sources(args.kl), "rank": parse_sources(args.rank)},
                     parse_sources(args.cache), args.output, args.replicates, args.seed)
    if not args.skip_plots:
        plot(result, args.output)
