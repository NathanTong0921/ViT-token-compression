"""Analyze the fixed reserved-image and reserved-class-pair confirmation."""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_selector_budgets import weighted_estimates
from analyze_selector_geometry import average_seeds, construct_geometry, scalar_cases, write_csv
from analyze_selector_scale import panel_weights, read_rows
from evaluate_selectors import binary_outcomes, write_rows
from token_compression.selector_geometry import adjusted_angle_association, image_weights


METRICS = ("accuracy", "margin")


def prediction_path(setting, method, seed):
    return Path(setting["eval_dir"]) / method / f"seed_{seed}" / "predictions.npz"


def method_seeds(setting, config):
    result = {"full_model": [-1], "attention": [-1],
              "random": config.get("random_seeds", [2055, 2056, 2057])}
    for method in setting["checkpoints"]:
        result[method] = setting.get("seeds", config.get("seeds", [2041, 2042, 2043]))
    return result


def load_outcomes(setting, method, seeds, examples):
    """Check saved outcomes against the common row ordering and class scores."""
    labels = np.array([int(r["class_id"]) for r in examples])
    pairs = np.array([[int(r["class_a"]), int(r["class_b"])] for r in examples])
    arrays = []
    for seed in seeds:
        with np.load(prediction_path(setting, method, seed)) as saved:
            assert saved["cosine_scores"].shape[0] == len(examples)
            margin, correct = binary_outcomes(saved["cosine_scores"], labels, pairs,
                                              saved["text_features"])
            np.testing.assert_array_equal(saved["correct"], correct)
            np.testing.assert_allclose(saved["margin"], margin, rtol=0, atol=1e-7)
            arrays.append(np.column_stack([correct, margin]))
    return np.asarray(arrays, dtype=float)


def geometry_predictions(setting, seeds, examples):
    """Adapt the final evaluator records to the existing exchange analysis."""
    ids = np.array([r["image_id"] for r in examples])
    tasks = np.array([r["task_id"] for r in examples])
    labels = np.array([int(r["class_id"]) for r in examples])
    pairs = np.array([[int(r["class_a"]), int(r["class_b"])] for r in examples])
    parts, text = [], None
    jobs = [("full_model", -1)] + [(method, seed) for seed in seeds
                                    for method in ("conditioned", "image_only")]
    for method, seed in jobs:
        with np.load(prediction_path(setting, method, seed)) as saved:
            if text is not None:
                np.testing.assert_array_equal(text, saved["text_features"])
            text = saved["text_features"].copy()
            n = len(examples)
            positions = saved["patch_indices"].copy()
            parts.append({"seed": np.full(n, seed), "method": np.full(n, method),
                          "task_split": np.full(n, "unseen"), "image_ids": ids,
                          "task_ids": tasks, "labels": labels, "class_pairs": pairs,
                          "keep_patch_tokens": np.full(n, positions.shape[1] if method == "full_model"
                                                        else setting["keep_patch_tokens"]),
                          "patch_indices": positions,
                          "cosine_scores": saved["cosine_scores"].copy(),
                          "image_features": saved["image_features"].copy()})
    result = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    result["text_features"] = text
    result["row_id"] = np.arange(len(result["seed"]))
    result["original_row_id"] = result["row_id"].copy()
    return result


def contrast_rows(setting_id, contrast, estimates, family_size):
    rows = []
    for metric, stats in estimates.items():
        adjusted = stats.get("bonferroni_family_ci95", stats["ci95"])
        rows.append({"setting": setting_id, "contrast": contrast, "metric": metric,
                     "mean": stats["mean"], "ci95_low": stats["ci95"][0],
                     "ci95_high": stats["ci95"][1], "family_size": family_size,
                     "family_ci95_low": adjusted[0], "family_ci95_high": adjusted[1]})
    return rows


def analyze_setting(setting, config, output, families):
    output.mkdir(parents=True, exist_ok=True)
    examples = read_rows(Path(setting["eval_dir"]) / "examples.csv")
    keys = [(r["image_id"], r["task_id"]) for r in examples]
    assert len(keys) == len(set(keys)) and {r["task_split"] for r in examples} == {"unseen"}
    ids = np.array([r["image_id"] for r in examples])
    weights = panel_weights(examples, "all")
    np.testing.assert_allclose(weights, [float(r["task_class_weight"]) for r in examples],
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(weights.sum(), 1., atol=1e-12)
    reps, rng_seed = config.get("bootstrap_replicates", 2000), config.get("bootstrap_seed", 2421)
    runs = method_seeds(setting, config)
    arrays = {method: load_outcomes(setting, method, seeds, examples) for method, seeds in runs.items()}
    means = {method: values.mean(axis=0) for method, values in arrays.items()}
    table, seed_rows, flat_table = [], [], []
    full_correct = means["full_model"][:, 0]
    for method, values in arrays.items():
        stats = weighted_estimates(ids, weights, means[method], METRICS, reps, rng_seed)
        repairs = (values[:, :, 0] == 1) & (full_correct[None] == 0)
        errors = (values[:, :, 0] == 0) & (full_correct[None] == 1)
        row = {"method": method, **stats,
               "repairs_vs_full": float(np.mean(repairs @ weights)),
               "new_errors_vs_full": float(np.mean(errors @ weights))}
        table.append(row)
        flat_table.append({"setting": setting["id"], "method": method,
                           "accuracy_percent": 100 * stats["accuracy"]["mean"],
                           "accuracy_ci95_low_percent": 100 * stats["accuracy"]["ci95"][0],
                           "accuracy_ci95_high_percent": 100 * stats["accuracy"]["ci95"][1],
                           "margin": stats["margin"]["mean"],
                           "margin_ci95_low": stats["margin"]["ci95"][0],
                           "margin_ci95_high": stats["margin"]["ci95"][1],
                           "repairs_vs_full": row["repairs_vs_full"],
                           "new_errors_vs_full": row["new_errors_vs_full"]})
        for seed, value in zip(runs[method], np.einsum("n,snm->sm", weights, values)):
            seed_rows.append({"setting": setting["id"], "method": method, "seed": seed,
                              "accuracy": float(value[0]), "margin": float(value[1])})
    specifications = [("conditioned", "image_only", families["original"])]
    if "image_only_wide" in arrays:
        specifications.append(("conditioned", "image_only_wide", families["strong"]))
    if "patch_ranking" in arrays:
        specifications += [("image_only_wide", "patch_ranking", families["external"]),
                           ("conditioned", "patch_ranking", families["external"])]
    contrasts, flat_contrasts = {}, []
    for a, b, family in specifications:
        name = f"{a}_minus_{b}"
        stats = weighted_estimates(ids, weights, means[a] - means[b], METRICS, reps, rng_seed, family)
        contrasts[name] = stats
        flat_contrasts.extend(contrast_rows(setting["id"], name, stats, family))

    seeds = runs["conditioned"]
    predictions = geometry_predictions(setting, seeds, examples)
    cases, audit = construct_geometry(predictions, seeds)
    del predictions
    averaged = average_seeds(cases, seeds)
    np.savez_compressed(output / "exchange_per_seed.npz", **scalar_cases(cases))
    del cases
    np.savez_compressed(output / "exchange_seed_mean.npz", **scalar_cases(averaged))
    write_csv(output / "exchange_seed_mean.csv", scalar_cases(averaged))
    exchange = weighted_estimates(
        averaged["image_ids"], image_weights(averaged["image_ids"]),
        np.column_stack([averaged["correctness_exchange_loss"], averaged["exchange_loss"],
                         averaged["overlap_fraction"]]),
        ["accuracy", "margin", "selection_overlap"], reps, rng_seed, families["original"])
    geometry = {
        "difficulty_adjusted": adjusted_angle_association(averaged, replicates=reps, seed=rng_seed,
                                                          family_size=families["original"]),
        "within_image": adjusted_angle_association(averaged, replicates=reps, seed=rng_seed,
                                                    family_size=families["original"],
                                                    image_fixed_effects=True)}
    summary = {"setting": setting, "image_count": len(set(ids)), "image_task_count": len(ids),
               "task_count": len({r["task_id"] for r in examples}), "table": table,
               "contrasts": contrasts, "exchange_own_minus_swapped": exchange,
               "exchange_image_count": len(set(averaged["image_ids"])),
               "exchange_pair_count": len(averaged["image_ids"]), "h2": geometry, "audit": audit}
    (output / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_rows(output / "classification_table.csv", flat_table)
    write_rows(output / "contrasts.csv", flat_contrasts)
    write_rows(output / "per_seed.csv", seed_rows)
    np.savez_compressed(output / "paired_outcomes.npz", image_ids=ids,
                        task_ids=np.array([r["task_id"] for r in examples]), weights=weights,
                        **{f"{method}_per_seed": values for method, values in arrays.items()})
    retained = {"setting": setting, "keys": keys, "ids": ids, "weights": weights,
                "metadata": [(int(r["class_id"]), int(r["class_a"]), int(r["class_b"])) for r in examples],
                "means": means}
    return summary, retained, flat_table, flat_contrasts, seed_rows


def budget_contrasts(lower, higher, replicates, seed, family_size):
    """Compare the conditioning increment, aligning image/task identities first."""
    lookup = {key: i for i, key in enumerate(higher["keys"])}
    assert set(lookup) == set(lower["keys"])
    ix = np.array([lookup[key] for key in lower["keys"]])
    assert [higher["metadata"][i] for i in ix] == lower["metadata"]
    np.testing.assert_allclose(higher["weights"][ix], lower["weights"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(higher["means"]["full_model"][ix], lower["means"]["full_model"],
                               rtol=0, atol=1e-7)
    result = {}
    for shared in ("image_only", "image_only_wide"):
        if shared not in lower["means"] or shared not in higher["means"]:
            continue
        low = lower["means"]["conditioned"] - lower["means"][shared]
        high = (higher["means"]["conditioned"] - higher["means"][shared])[ix]
        result[f"conditioned_minus_{shared}"] = weighted_estimates(
            lower["ids"], lower["weights"], high - low, METRICS, replicates, seed, family_size)
    return result


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    settings = config["settings"]
    families = {"original": len(settings),
                "strong": sum("image_only_wide" in s["checkpoints"] for s in settings),
                "external": sum("patch_ranking" in s["checkpoints"] for s in settings)}
    summary = {"final_test_used": True,
               "scope": "Fixed trained weights on reserved official test images and reserved class combinations.",
               "classification_weighting": "Equal tasks and class sides; equal images within each class side; average seeds before image bootstrap.",
               "exchange_weighting": "Average paired seeds and task pairs within each image; equal image weight.",
               "intervals": "Ordinary 95% intervals and Bonferroni family intervals per metric and named contrast; fixed task collections and training seeds.",
               "families": families, "bootstrap_replicates": config.get("bootstrap_replicates", 2000),
               "bootstrap_seed": config.get("bootstrap_seed", 2421), "settings": {}, "budget_changes": {}}
    retained, tables, contrasts, seeds = {}, [], [], []
    for setting in settings:
        result, keep, table, contrast, seed_rows = analyze_setting(
            setting, config, output / setting["id"], families)
        summary["settings"][setting["id"]] = result
        retained[setting["id"]] = keep
        tables.extend(table)
        contrasts.extend(contrast)
        seeds.extend(seed_rows)
        print(json.dumps({"completed_setting": setting["id"], "contrasts": result["contrasts"]}), flush=True)
    budget_rows = []
    for dataset in sorted({s["dataset"] for s in settings}):
        selected = {s["keep_patch_tokens"]: retained[s["id"]] for s in settings
                    if s["dataset"] == dataset and s["model"] == "ViT-B/16"}
        if 49 not in selected or 98 not in selected:
            continue
        name = f"{dataset}_98_minus_49"
        result = budget_contrasts(selected[49], selected[98], summary["bootstrap_replicates"],
                                 summary["bootstrap_seed"], 2)
        summary["budget_changes"][name] = result
        for contrast, estimates in result.items():
            budget_rows.extend(contrast_rows(name, contrast, estimates, 2))
    (output / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_rows(output / "classification_table.csv", tables)
    write_rows(output / "contrasts.csv", contrasts)
    write_rows(output / "per_seed.csv", seeds)
    if budget_rows:
        write_rows(output / "budget_changes.csv", budget_rows)
    h2_rows, exchange_rows = [], []
    for setting_id, result in summary["settings"].items():
        for name, value in result["h2"].items():
            h2_rows.append({"setting": setting_id, "analysis": name, "angle_slope_per_10deg": value["angle_slope"],
                            "ci95_low": value["ci95"][0], "ci95_high": value["ci95"][1],
                            "family_ci95_low": value["ci_familywise_budgets"][0],
                            "family_ci95_high": value["ci_familywise_budgets"][1],
                            "angle_estimable": value["angle_estimable_beyond_controls"],
                            "image_count": value["image_count"], "pair_count": value["pair_count"]})
        exchange_rows.extend(contrast_rows(setting_id, "correct_minus_exchanged_task",
                                            result["exchange_own_minus_swapped"], families["original"]))
    write_rows(output / "h2.csv", h2_rows)
    write_rows(output / "exchange.csv", exchange_rows)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)
