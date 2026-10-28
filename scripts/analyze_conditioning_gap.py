"""Diagnose conditioning gains using augmented and independent probe libraries.

Example (run from the repository, after the source version is committed):
  PYTHONPATH=src python scripts/analyze_conditioning_gap.py \
    --config 49=configs/pets_selector_scale.json \
    --config 98=configs/pets_selector_budget_98.json \
    --source 49=runs/pets-selector-scale-eval-v1 \
    --source 98=runs/pets-selector-budget-98-eval-v1 \
    --shared 49=runs/pets-selector-budget-49-shared-v1 \
    --shared 98=results/pets-selector-budgets-v1/shared/budget_98 \
    --output results/pets-conditioning-gap-v1

All inputs are existing development predictions and measured random probes.
This script performs no new training, model inference, or final-test loading.
"""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from token_compression.conditioning_gap import (
    average_seed_cases, cross_validated_gap_prediction, library_potential,
    probe_gap_association, regret_decomposition, summarize_columns,
)
from token_compression.selector_exchange import build_exchange_cases
from token_compression.selector_geometry import image_weights


def load_predictions(source, seeds):
    keys = ["image_ids", "task_ids", "task_split", "labels", "class_pairs", "method",
            "seed", "cosine_scores", "patch_indices", "keep_patch_tokens"]
    pieces, text = [], None
    for group in ("references", "full"):
        with np.load(source / group / "predictions.npz") as saved:
            assert set(saved["task_split"]) <= {"seen", "unseen"}
            selected = saved["task_split"] == "unseen"
            if group == "references":
                selected &= saved["method"] == "full_model"
            else:
                selected &= np.isin(saved["seed"], seeds) & np.isin(saved["method"], ["conditioned", "image_only"])
            pieces.append({key: saved[key][selected] for key in keys})
            if text is not None:
                np.testing.assert_array_equal(text, saved["text_features"])
            text = saved["text_features"].copy()
    result = {key: np.concatenate([p[key] for p in pieces]) for key in keys}
    result["row_id"] = np.arange(len(result["method"]))
    result["text_features"] = text
    return result


def load_probes(cache, shared_folder):
    with np.load(cache / "val" / "evidence.npz") as saved:
        evidence = {key: saved[key].copy() for key in ("image_ids", "labels", "text_features")}
    with np.load(cache / "val" / "teacher.npz") as saved:
        np.testing.assert_array_equal(saved["image_ids"], evidence["image_ids"])
        responses, available, masks = [saved[key].copy() for key in ("responses", "response_available", "masks")]
    missing = np.flatnonzero(~available)
    if len(missing):
        with np.load(shared_folder / "missing_probe_responses.npz") as saved:
            np.testing.assert_array_equal(saved["source_indices"], missing)
            np.testing.assert_array_equal(saved["image_ids"], evidence["image_ids"][missing])
            np.testing.assert_array_equal(saved["masks"], masks)
            responses[missing] = saved["responses"]
    assert responses.shape == (len(evidence["image_ids"]), 512, len(evidence["text_features"]))
    assert np.isfinite(responses).all()
    return evidence, responses, masks, len(missing)


def attach_library_cases(cases, shared, evidence, responses, split_seed=2083):
    """Align augmented records and separately score only 512 fixed random masks."""
    lookup = {(str(i), str(a), str(b)): j for j, (i, a, b) in enumerate(zip(
        shared["image_ids"], shared["task_a"], shared["task_b"]))}
    keys = list(zip(cases["image_ids"], cases["task_a"], cases["task_b"]))
    assert len(lookup) == len(keys) and set(lookup) == set(keys)
    order = np.asarray([lookup[key] for key in keys])
    for key in ("labels", "competitor_a", "competitor_b"):
        np.testing.assert_array_equal(cases[key], shared[key][order])
    cases["augmented_separate_margin"] = shared["separate_margin"][order]
    cases["augmented_shared_margin"] = shared["shared_margin"][order]
    cases.update(regret_decomposition(cases["augmented_separate_margin"], cases["augmented_shared_margin"],
                                      cases["conditioned_margin"], cases["image_only_margin"]))
    images = {str(name): j for j, name in enumerate(evidence["image_ids"])}
    image_index = np.asarray([images[str(i)] for i in cases["image_ids"]])
    np.testing.assert_array_equal(cases["labels"], evidence["labels"][image_index])
    text = evidence["text_features"].astype(float)
    labels, ca, cb = [cases[k].astype(int) for k in ("labels", "competitor_a", "competitor_b")]
    na, nb = [np.linalg.norm(text[labels]-text[c], axis=1) for c in (ca, cb)]
    # Gather only the three needed class responses, avoiding an examples x masks x classes copy.
    true_scores = responses[image_index, :, labels].astype(float)
    a = (true_scores-responses[image_index, :, ca]) / na[:, None]
    b = (true_scores-responses[image_index, :, cb]) / nb[:, None]
    permutation = np.random.default_rng(split_seed).permutation(512)
    libraries = {"512": np.arange(512), "half_a": permutation[:256], "half_b": permutation[256:]}
    for name, indices in libraries.items():
        result = library_potential(a[:, indices], b[:, indices])
        for metric, values in result.items():
            cases[f"probe_{metric}_{name}"] = values
    cases["probe_half_gap_difference"] = cases["probe_gap_half_a"]-cases["probe_gap_half_b"]
    return {name: value.tolist() for name, value in libraries.items()}


def write_cases(output, cases):
    np.savez_compressed(output / "cases.npz", **cases)
    columns = list(cases)
    assert all(np.asarray(cases[key]).ndim == 1 for key in columns)
    with (output / "cases.csv").open("w") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(zip(*(cases[key] for key in columns)))


def analyze_budget(config, source, shared_folder, output, replicates, bootstrap_seed, split_seed, family_size):
    output.mkdir(parents=True, exist_ok=True)
    seeds, budget = config["training_seeds"], config["keep_patch_tokens"]
    predictions = load_predictions(source, seeds)
    exchange = build_exchange_cases(predictions, seeds, ["unseen"])
    cases = average_seed_cases(exchange, seeds)
    assert np.all(cases["keep_patch_tokens"] == budget)
    with np.load(shared_folder / "shared_search_cases.npz") as saved:
        shared = {key: saved[key].copy() for key in saved.files}
    evidence, responses, masks, supplied = load_probes(Path(config["cache_dir"]), shared_folder)
    np.testing.assert_array_equal(evidence["text_features"], predictions["text_features"])
    np.testing.assert_array_equal(masks, shared["probe_masks"])
    assert np.all(masks.sum(axis=1) == budget)
    library_indices = attach_library_cases(cases, shared, evidence, responses, split_seed)
    # Check containment for each individual trained seed, not only its average.
    per_seed = {}
    for seed in seeds:
        values = regret_decomposition(cases["augmented_separate_margin"], cases["augmented_shared_margin"],
                                      cases[f"conditioned_margin_seed_{seed}"], cases[f"image_only_margin_seed_{seed}"])
        per_seed[str(seed)] = summarize_columns({"image_ids": cases["image_ids"], **values}, list(values),
                                                replicates, bootstrap_seed, family_size)
    columns = ["conditioned_margin", "image_only_margin", "learned_gain", "learned_accuracy_gain",
               "augmented_separate_margin", "augmented_shared_margin", "augmented_specialization_potential",
               "conditioned_regret", "image_only_regret", "regret_difference", "probe_gap_512",
               "probe_gap_half_a", "probe_gap_half_b", "probe_half_gap_difference"]
    summary = summarize_columns(cases, columns, replicates, bootstrap_seed, family_size)
    weights = image_weights(cases["image_ids"])
    a, b = cases["probe_gap_half_a"], cases["probe_gap_half_b"]
    ac, bc = a-np.sum(weights*a), b-np.sum(weights*b)
    denominator = np.sqrt(np.sum(weights*ac**2)*np.sum(weights*bc**2))
    stability = {"split_seed": split_seed, "library_mask_indices": library_indices,
                 "half_gap_weighted_correlation": float(np.sum(weights*ac*bc)/denominator) if denominator>0 else None,
                 "scope": "Descriptive two-half finite-library stability; halves are not independent image samples.",
                 "half_gap_difference": summary["probe_half_gap_difference"]}
    association = probe_gap_association(cases, replicates, bootstrap_seed, family_size)
    prediction = cross_validated_gap_prediction(cases, replicates=replicates, bootstrap_seed=bootstrap_seed,
                                                family_size=family_size)
    np.savez_compressed(output / "cross_fitted_predictions.npz", image_ids=cases["image_ids"],
                        task_a=cases["task_a"], task_b=cases["task_b"], observed=cases["learned_gain"],
                        **prediction.pop("out_of_fold_predictions"))
    cases["fold"] = np.asarray([prediction["fold_assignment"][str(i)] for i in cases["image_ids"]], np.int8)
    write_cases(output, cases)
    result = {"budget": budget, "image_count": len(np.unique(cases["image_ids"])),
              "task_pair_count": len(cases["image_ids"]), "training_seeds": seeds,
              "aggregation": "Average paired training seeds first; equal images, then equal task pairs within image.",
              "source": str(source), "teacher_cache": config["cache_dir"], "shared_source": str(shared_folder),
              "missing_probes_restored_from_existing_supplement": supplied,
              "metrics": summary, "per_seed_decomposition": per_seed,
              "decomposition_max_abs_residual": float(np.max(np.abs(cases["decomposition_residual"]))),
              "pure_probe_stability": stability, "primary_probe_gap_association": association,
              "held_image_prediction": prediction}
    (output / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


def plot_summary(summary, output):
    import matplotlib.pyplot as plt
    budgets = sorted(summary["budgets"], key=int)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for color, (offset, key, label) in enumerate([(-.18, "augmented_specialization_potential", "Library potential"),
                                                 (0, "regret_difference", "Regret difference"),
                                                 (.18, "learned_gain", "Learned C−U")]):
        for j, budget in enumerate(budgets):
            metric = summary["budgets"][budget]["metrics"][key]
            y, bounds = metric["mean"], metric["ci95"]
            axes[0].plot(j+offset, y, "o", color=f"C{color}", label=label if j == 0 else None)
            axes[0].vlines(j+offset, bounds[0], bounds[1], color=f"C{color}")
            axes[0].hlines(bounds, j+offset-.04, j+offset+.04, color=f"C{color}")
    for j, budget in enumerate(budgets):
        result = summary["budgets"][budget]
        slope = result["primary_probe_gap_association"]["primary_gap_coefficient"]
        values = [(slope["estimate"], slope["ci_familywise_budgets"]),
                  (result["held_image_prediction"]["incremental_out_of_fold_r2"],
                   result["held_image_prediction"]["incremental_out_of_fold_r2_ci_familywise_budgets"])]
        for ax, (value, bounds) in zip(axes[1:], values):
            if value is not None:
                ax.plot(j, value, "o", color="C0")
                ax.vlines(j, bounds[0], bounds[1], color="C0")
                ax.hlines(bounds, j-.06, j+.06, color="C0")
    axes[0].legend(fontsize=8)
    titles = ["Finite-library decomposition (95% intervals)", "Probe-gap partial slope\nper 0.01 margin", "Held-image incremental R²\nfrom pure-probe gap"]
    for ax, title in zip(axes, titles):
        ax.set_xticks(range(len(budgets)), budgets)
        ax.set_xlabel("Retained patch tokens")
        ax.set_title(title)
        ax.axhline(0, color=".5", linewidth=1)
        ax.grid(axis="y", alpha=.2)
    level = 100*(1-.05/summary["family_size"])
    fig.suptitle(f"Shared-baseline advantage: label-informed diagnostics\nSlope and held-image intervals: {level:g}% per budget")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output/f"conditioning_gap.{suffix}", dpi=180)
    plt.close(fig)


def keyed_paths(items):
    result = {}
    for item in items:
        key, value = item.split("=", 1)
        budget = int(key)
        assert budget not in result
        result[budget] = Path(value)
    return result


def main(args):
    if args.plot_only:
        plot_summary(json.loads((args.output/"analysis.json").read_text()), args.output)
        return
    configs, sources, shared = [keyed_paths(items) for items in (args.config, args.source, args.shared)]
    assert set(configs) == set(sources) == set(shared) == {49, 98}
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "scope": "Development-only common-library diagnostic; no new training, inference or test data.",
               "interpretation": "Exact decomposition is accounting, not causal evidence. Augmented oracles know labels and both tasks. Independent pure-probe predictors are also label-informed, not deployable.",
               "bootstrap_replicates": args.replicates, "bootstrap_seed": args.bootstrap_seed,
               "probe_split_seed": args.split_seed, "fold_seed": 2060, "family_size": 2,
               "multiplicity": "Two-budget intervals separately for the probe-gap slope and held-image prediction gain; no claim of global coverage across auxiliary metrics.",
               "budgets": {}}
    for budget in sorted(configs):
        config = json.loads(configs[budget].read_text())
        assert config["keep_patch_tokens"] == budget
        summary["budgets"][str(budget)] = analyze_budget(config, sources[budget], shared[budget],
            args.output/f"budget_{budget}", args.replicates, args.bootstrap_seed, args.split_seed, 2)
        print(json.dumps({"budget": budget, "status": "complete"}), flush=True)
    summary["wall_seconds"] = time.perf_counter()-started
    (args.output/"analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    if args.plot:
        plot_summary(summary, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--shared", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2084)
    parser.add_argument("--split-seed", type=int, default=2083)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    main(parser.parse_args())
