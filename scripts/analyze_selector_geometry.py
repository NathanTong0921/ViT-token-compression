"""Predeclared H2 analysis of full-feature learned-selector exchanges.

Example:
  PYTHONPATH=src python scripts/analyze_selector_geometry.py \\
    --source 49=runs/pets-selector-scale-eval-v1 \\
    --source 98=runs/pets-selector-budget-98-eval-v1 \\
    --source 147=runs/pets-selector-budget-147-eval-v1 \\
    --output results/pets-selector-budgets-v1/geometry
"""

import argparse
import csv
import json
from pathlib import Path
import subprocess

import numpy as np

from token_compression.selector_exchange import build_exchange_cases
from token_compression.selector_geometry import (
    adjusted_angle_association, cross_validated_angle_prediction, exchange_geometry,
)


def load_predictions(source, seeds):
    parts = []
    for group in ("references", "full"):
        with np.load(source / group / "predictions.npz") as saved:
            chosen = (saved["task_split"] == "unseen")
            if group == "references":
                chosen &= saved["method"] == "full_model"
            else:
                chosen &= np.isin(saved["seed"], seeds) & np.isin(saved["method"], ["conditioned", "image_only"])
            part = {k: saved[k][chosen].copy() for k in saved.files if k != "text_features"}
            part["original_row_id"] = part["row_id"].copy()
            part["source_group"] = np.full(chosen.sum(), group)
            text = saved["text_features"].copy()
            if parts:
                np.testing.assert_array_equal(text, text_features)
            text_features = text
            parts.append(part)
    assert set(parts[0]) == set(parts[1])
    output = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    output["row_id"] = np.arange(len(output["row_id"]))
    output["text_features"] = text_features
    return output


def construct_geometry(predictions, seeds):
    cases = build_exchange_cases(predictions, seeds, ["unseen"])
    ra, rb = cases["source_row_a"], cases["source_row_b"]
    full_map = {str(image_id): i for i, (image_id, method) in enumerate(zip(
        predictions["image_ids"], predictions["method"])) if method == "full_model"}
    full_rows = np.asarray([full_map[str(i)] for i in cases["image_ids"]])
    z_a, z_b, z_full = [predictions["image_features"][r] for r in (ra, rb, full_rows)]
    geometry = exchange_geometry(z_a, z_b, z_full, predictions["text_features"],
                                 cases["labels"], cases["competitor_a"], cases["competitor_b"])
    score_loss = .5 * (cases["margin_aa"] - cases["margin_ba"] + cases["margin_bb"] - cases["margin_ab"])
    score_error = float(np.max(np.abs(score_loss - geometry["exchange_loss"])))
    np.testing.assert_allclose(score_loss, geometry["exchange_loss"], rtol=0, atol=1e-6)
    cases.update(geometry)
    cases["source_original_row_a"] = predictions["original_row_id"][ra]
    cases["source_original_row_b"] = predictions["original_row_id"][rb]
    cases["exchange_loss_from_scores"] = score_loss
    cases["correctness_exchange_loss"] = .5 * (
        cases["correct_aa"].astype(float) - cases["correct_ba"] +
        cases["correct_bb"].astype(float) - cases["correct_ab"])
    image_only_delta = predictions["image_features"][cases["image_only_row_a"]] - predictions["image_features"][cases["image_only_row_b"]]
    image_only_null = float(np.max(np.abs(image_only_delta)))
    assert image_only_null == 0.
    return cases, {"score_vs_full_feature_exchange_loss_max_abs_error": score_error,
                   "image_only_feature_exchange_max_abs_difference": image_only_null,
                   "geometry_identity_max_abs_error": float(np.max(geometry["identity_error"]))}


def average_seeds(cases, seeds):
    """Align the same pairs, average losses and recompute mean-displacement geometry."""
    seed_parts = []
    for seed in seeds:
        indices = np.flatnonzero(cases["seed"] == seed)
        order = np.lexsort((cases["task_b"][indices], cases["task_a"][indices], cases["image_ids"][indices]))
        seed_parts.append(indices[order])
    first = seed_parts[0]
    for indices in seed_parts[1:]:
        for key in ("image_ids", "task_a", "task_b", "labels", "competitor_a", "competitor_b", "keep_patch_tokens"):
            np.testing.assert_array_equal(cases[key][indices], cases[key][first])
        for key in ("base_margin_a", "base_margin_b", "angle_degrees", "sin_half_angle"):
            np.testing.assert_allclose(cases[key][indices], cases[key][first], rtol=0, atol=1e-12)
    metadata = ("image_ids", "task_a", "task_b", "labels", "competitor_a", "competitor_b", "keep_patch_tokens",
                "base_margin_a", "base_margin_b", "angle_degrees", "sin_half_angle", "direction_norm_a", "direction_norm_b")
    result = {k: cases[k][first].copy() for k in metadata}
    for key in ("exchange_loss", "loss_a", "loss_b", "overlap_fraction", "correctness_exchange_loss"):
        result[key] = np.mean([cases[key][i] for i in seed_parts], axis=0)
    displacement = np.mean([cases["feature_displacement"][i] for i in seed_parts], axis=0)
    direction = cases["direction_displacement"][first]
    norm, direction_norm = np.linalg.norm(displacement, axis=1), np.linalg.norm(direction, axis=1)
    dot = np.einsum("ij,ij->i", displacement, direction)
    alignment = np.full(len(first), np.nan)
    valid = (norm > 1e-12) & (direction_norm > 1e-12)
    alignment[valid] = dot[valid] / (norm[valid] * direction_norm[valid])
    result.update(displacement_norm=norm, alignment=alignment, alignment_valid=valid,
                  identity_error=np.abs(result["exchange_loss"] - norm * result["sin_half_angle"] * np.nan_to_num(alignment)))
    np.testing.assert_allclose(result["exchange_loss"], dot / 2, rtol=0, atol=1e-12)
    result["seed_count"] = np.full(len(first), len(seeds))
    return result


def subset(cases, selected):
    return {k: v[selected] for k, v in cases.items()}


def scalar_cases(cases):
    return {k: v for k, v in cases.items() if np.asarray(v).ndim == 1}


def write_csv(path, columns):
    keys = list(columns)
    with Path(path).open("w") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(keys)
        for row in zip(*[columns[k] for k in keys]):
            writer.writerow(["" if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v for v in row])


def describe(cases):
    angle, loss, norm = [np.asarray(cases[k], float) for k in ("angle_degrees", "exchange_loss", "displacement_norm")]
    return {"image_count": len(np.unique(cases["image_ids"])), "pair_count": len(angle),
            "angle_degrees_range": [float(angle.min()), float(angle.max())],
            "exchange_loss_range": [float(loss.min()), float(loss.max())],
            "displacement_norm_range": [float(norm.min()), float(norm.max())],
            "zero_or_near_zero_displacement_pairs_at_1e12": int(np.sum(norm <= 1e-12)),
            "small_displacement_pairs_at_1e6": int(np.sum(norm <= 1e-6)),
            "geometry_identity_max_abs_error": float(np.max(cases["identity_error"]))}


def plot_summaries(summary, output):
    import matplotlib.pyplot as plt
    budgets = sorted(summary["budgets"], key=int)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    specifications = [
        ("primary_difficulty_adjusted", "angle_slope", "ci_familywise_three_budgets", "Adjusted exchange-loss slope\nper 10 degrees (98.33% intervals)"),
        ("image_fixed_effects_sensitivity", "angle_slope", "ci95", "Within-image slope\nper 10 degrees (95% intervals)"),
        ("held_image_prediction", "incremental_out_of_fold_r2", "incremental_out_of_fold_r2_ci95", "Held-image incremental R²\nfrom adding angle (95% intervals)"),
    ]
    for ax, (section, point, interval, title) in zip(axes, specifications):
        for i, budget in enumerate(budgets):
            result = summary["budgets"][budget][section]
            value, bounds = result[point], result[interval]
            if value is not None:
                ax.plot(i, value, "o", color="C0")
                ax.vlines(i, bounds[0], bounds[1], color="C0")
                ax.hlines(bounds, i - .07, i + .07, color="C0")
        ax.axhline(0, color="0.5", linewidth=1)
        ax.set_xticks(range(len(budgets)), budgets)
        ax.set_xlabel("Retained patch tokens")
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"h2_geometry.{suffix}", dpi=180)
    plt.close(fig)


def run(sources, output, seeds=(2041, 2042, 2043), replicates=2000,
        bootstrap_seed=2061, fold_seed=2060, plot=False):
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "primary": "Unseen class combinations on development images, full training coverage; conditional selector's symmetric exchange loss.",
        "seed_aggregation": "Average the same image/task-pair loss across the three fixed seeds before fitting or resampling. Alignment is recomputed from seed-mean feature displacement; per-seed diagnostics are reported separately.",
        "weighting": "Images have equal total weight; all eligible unordered task pairs are equally weighted within each image.",
        "difficulty_controls": "Intercept and symmetric quadratic basis of min/max full-model signed geometric margins; primary adds a linear angle term per 10 degrees.",
        "identity": "L = .5*(z_A-z_B) dot (u_A-u_B) = ||z_A-z_B|| * sin(theta/2) * alignment. This identity is not a mechanism discovery.",
        "uncertainty": "Paired image-cluster bootstrap, conditional on task collection and fixed training seeds. Shared sorted image IDs and bootstrap seed align resamples across budgets when coverage matches.",
        "multiplicity": "Primary three budget slopes have 98.333% Bonferroni intervals; other 95% intervals are exploratory diagnostics.",
        "interpretation": "A positive adjusted slope is predictive association, not a causal angle intervention. Image-fixed-effects sensitivity and held-image prediction assess robustness. Alignment diagnostics remove the algebraic displacement/angle scale but do not prove mechanism.",
        "training_seeds": list(seeds), "bootstrap_replicates": replicates,
        "bootstrap_seed": bootstrap_seed, "fold_seed": fold_seed,
        "final_test_used": False, "budgets": {},
    }
    common_keys = None
    for budget, source in sorted(sources.items()):
        predictions = load_predictions(source, seeds)
        cases, audit = construct_geometry(predictions, seeds)
        assert np.all(cases["keep_patch_tokens"] == budget)
        averaged = average_seeds(cases, seeds)
        keys = np.column_stack([averaged["image_ids"], averaged["task_a"], averaged["task_b"]])
        if common_keys is not None:
            np.testing.assert_array_equal(keys, common_keys)
        common_keys = keys
        folder = output / f"budget_{budget}"
        folder.mkdir(exist_ok=True)
        np.savez_compressed(folder / "seed_mean_cases.npz", **scalar_cases(averaged))
        np.savez_compressed(folder / "per_seed_cases.npz", **scalar_cases(cases))
        write_csv(folder / "seed_mean_cases.csv", scalar_cases(averaged))
        primary = adjusted_angle_association(averaged, replicates=replicates, seed=bootstrap_seed)
        fixed = adjusted_angle_association(averaged, replicates=replicates, seed=bootstrap_seed, image_fixed_effects=True)
        cv = cross_validated_angle_prediction(averaged, folds=5, seed=fold_seed,
                                               replicates=replicates, bootstrap_seed=bootstrap_seed)
        np.savez_compressed(folder / "cross_fitted_predictions.npz", **cv.pop("out_of_fold_predictions"),
                            image_ids=averaged["image_ids"], task_a=averaged["task_a"], task_b=averaged["task_b"],
                            exchange_loss=averaged["exchange_loss"])
        per_seed = []
        for seed in seeds:
            part = subset(cases, cases["seed"] == seed)
            per_seed.append({"seed": int(seed), "scope": describe(part),
                             "difficulty_adjusted": adjusted_angle_association(part, replicates=replicates, seed=bootstrap_seed),
                             "image_fixed_effects": adjusted_angle_association(part, replicates=replicates, seed=bootstrap_seed, image_fixed_effects=True)})
        alignment = []
        for threshold in (1e-12, 1e-6):
            eligible = averaged["alignment_valid"] & (averaged["displacement_norm"] > threshold)
            part = subset(averaged, eligible)
            if not np.any(eligible):
                alignment.append({"minimum_displacement_norm": threshold, "eligible_pair_count": 0,
                                  "status": "No nonzero feature displacements; normalized alignment is undefined."})
                continue
            alignment.append({"minimum_displacement_norm": threshold, "scope": describe(part),
                              "excluded_pair_count": int((~eligible).sum()),
                              "difficulty_adjusted": adjusted_angle_association(part, outcome="alignment", replicates=replicates, seed=bootstrap_seed),
                              "image_fixed_effects": adjusted_angle_association(part, outcome="alignment", replicates=replicates, seed=bootstrap_seed, image_fixed_effects=True)})
        summary["budgets"][str(budget)] = {"source": str(source), "scope": describe(averaged), "audit": audit,
                                            "primary_difficulty_adjusted": primary,
                                            "image_fixed_effects_sensitivity": fixed,
                                            "held_image_prediction": cv, "per_seed": per_seed,
                                            "normalized_alignment_diagnostics": alignment}
        (output / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"budget": budget, "slope_per_10_degrees": primary["angle_slope"],
                          "interval_familywise": primary["ci_familywise_three_budgets"],
                          "image_fixed_effect_slope": fixed["angle_slope"],
                          "incremental_cv_r2": cv["incremental_out_of_fold_r2"]}), flush=True)
        del predictions, cases, averaged
    if plot:
        plot_summaries(summary, output)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, help="BUDGET=EVALUATION_DIRECTORY; repeat for 49,98,147")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2041, 2042, 2043])
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2061)
    parser.add_argument("--fold-seed", type=int, default=2060)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    sources = {int(value.split("=", 1)[0]): Path(value.split("=", 1)[1]) for value in args.source}
    run(sources, args.output, args.seeds, args.bootstrap_replicates, args.bootstrap_seed, args.fold_seed, args.plot)
