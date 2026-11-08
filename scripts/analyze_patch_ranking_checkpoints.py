"""Compare validation-KL-selected PR checkpoints with final-epoch predictors."""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_patch_ranking import METRICS, aligned_values
from analyze_selector_budgets import weighted_estimates
from analyze_selector_scale import panel_weights, read_rows
from evaluate_selectors import write_rows


def auxiliary_values(folder, seeds, method="patch_ranking"):
    with np.load(folder / "predictions.npz") as raw:
        indices = []
        for seed in seeds:
            found = np.flatnonzero((raw["method"] == method) & (raw["seed"] == seed))
            assert len(found) == 1
            indices.append(int(found[0]))
        return {"image_ids": raw["image_ids"].copy(), "labels": raw["labels"].copy(),
                "scores": raw["cosine_scores"][indices],
                "overlap": raw["teacher_overlap"][indices],
                "patch_indices": raw["patch_indices"][indices]}


def run(config, run_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    seeds, budget = config["seeds"], config["keep_patch_tokens"]
    previous = Path(config.get("previous_run_dir", f"runs/patch-ranking-v1/{config['dataset']}"))
    folders = {"pr_best_val_kl": run_dir / "eval_best",
               "pr_final": run_dir / "eval_final", "pr_previous_final": previous / "evaluation"}
    sources = {name: read_rows(path / "predictions.csv") for name, path in folders.items()}
    base = sorted([r for r in sources["pr_final"] if r["method"] == "full_model"],
                  key=lambda r: (r["image_id"], r["task_id"]))
    keys = [(r["image_id"], r["task_id"]) for r in base]
    metadata = {key: tuple(int(r[n]) for n in ("class_id", "class_a", "class_b"))
                for key, r in zip(keys, base)}
    ids = np.array([key[0] for key in keys])
    weights = panel_weights(base, "all")
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-12)
    arrays = {name: aligned_values(rows, "patch_ranking", keys, metadata, seeds, budget)
              for name, rows in sources.items()}
    arrays["full_model"] = aligned_values(sources["pr_final"], "full_model", keys,
                                          metadata, [-1], budget)
    for rows in sources.values():
        np.testing.assert_allclose(aligned_values(rows, "full_model", keys, metadata, [-1], budget),
                                   arrays["full_model"], atol=1e-6, rtol=0)
    reference_rows = []
    for method, key in (("image_only_wide", "ablation_eval_dir"), ("conditioned", "baseline_eval_dir")):
        rows = read_rows(Path(config[key]) / "full/predictions.csv")
        selected = [r for r in rows if r["method"] == method and r["task_split"] == "unseen"]
        reference_rows.extend(selected)
        arrays[method] = aligned_values(selected, method, keys, metadata, seeds, budget)
        full = dict(zip(keys, arrays["full_model"][0]))
        for row in selected:
            expected = full[row["image_id"], row["task_id"]]
            np.testing.assert_allclose([float(row["base_correct"]), float(row["base_margin"])],
                                       expected, atol=1e-6, rtol=0)
    reps, rng_seed = config.get("bootstrap_replicates", 2000), config.get("bootstrap_seed", 2411)
    means = {name: values.mean(axis=0) for name, values in arrays.items()}
    table, seed_rows = [], []
    for name, values in arrays.items():
        estimate = weighted_estimates(ids, weights, means[name], METRICS, reps, rng_seed)
        per_seed = np.einsum("n,snm->sm", weights, values)
        method_seeds = [-1] if name == "full_model" else seeds
        records = [{"method": name, "seed": seed, "accuracy": float(v[0]), "margin": float(v[1])}
                   for seed, v in zip(method_seeds, per_seed)]
        seed_rows.extend(records)
        table.append({"method": name, **estimate, "per_seed": records})

    comparisons = (("pr_best_val_kl", "pr_final", 2),
                   ("pr_best_val_kl", "pr_previous_final", 1),
                   ("pr_best_val_kl", "image_only_wide", 1),
                   ("pr_best_val_kl", "conditioned", 1))
    contrasts, contrast_rows = {}, []
    for a, b, family in comparisons:
        name = f"{a}_minus_{b}"
        estimates = weighted_estimates(ids, weights, means[a] - means[b], METRICS,
                                        reps, rng_seed, family)
        contrasts[name] = estimates
        for metric, value in estimates.items():
            adjusted = value.get("bonferroni_family_ci95", [None, None])
            contrast_rows.append({"contrast": name, "metric": metric, "mean": value["mean"],
                                  "ci95_low": value["ci95"][0], "ci95_high": value["ci95"][1],
                                  "adjusted_low": adjusted[0], "adjusted_high": adjusted[1]})

    aux = {name: auxiliary_values(path, seeds) for name, path in folders.items()}
    aux["full_model"] = auxiliary_values(folders["pr_final"], [-1], "full_model")
    all_ids, labels = aux["pr_final"]["image_ids"], aux["pr_final"]["labels"]
    uniform = np.ones(len(all_ids)) / len(all_ids)
    auxiliary, aux_means = {}, {}
    for name, values in aux.items():
        np.testing.assert_array_equal(values["image_ids"], all_ids)
        np.testing.assert_array_equal(values["labels"], labels)
        correct = values["scores"].argmax(axis=-1) == labels[None]
        mean_correct = correct.mean(axis=0)
        entry = {"multiclass_accuracy": weighted_estimates(
            all_ids, uniform, mean_correct, ["accuracy"], reps, rng_seed)["accuracy"]}
        if name != "full_model":
            entry["teacher_overlap_at_98"] = weighted_estimates(
                all_ids, uniform, values["overlap"].mean(axis=0), ["fraction"], reps, rng_seed)["fraction"]
            aux_means[name] = np.stack((mean_correct, values["overlap"].mean(axis=0)), axis=-1)
        auxiliary[name] = entry
        for row in (r for r in seed_rows if r["method"] == name):
            index = 0 if name == "full_model" else seeds.index(row["seed"])
            row["multiclass_accuracy"] = float(correct[index].mean())
            row["teacher_overlap_at_98"] = None if name == "full_model" else float(values["overlap"][index].mean())
    for row in seed_rows:
        row.setdefault("multiclass_accuracy", None)
        row.setdefault("teacher_overlap_at_98", None)
    auxiliary_contrast = weighted_estimates(all_ids, uniform,
        aux_means["pr_best_val_kl"] - aux_means["pr_final"],
        ["multiclass_accuracy", "teacher_overlap_at_98"], reps, rng_seed)

    training = json.loads((run_dir / "training/training_summary.json").read_text())
    selected_rows, replay = [], []
    for index, seed in enumerate(seeds):
        item = training["seeds"][str(seed)]
        selected = item["best_val_kl"]
        history = read_rows(run_dir / f"training/seed_{seed}/training_history.csv")
        best = min(history, key=lambda r: float(r["val_kl"]))
        assert int(best["epoch"]) == selected["epoch"]
        np.testing.assert_allclose(float(best["val_kl"]), selected["validation"]["kl"], atol=1e-10)
        selected_rows.append({"dataset": config["dataset"], "seed": seed,
                              "selected_epoch": selected["epoch"], "final_epoch": int(history[-1]["epoch"]),
                              "selected_val_kl": selected["validation"]["kl"],
                              "final_val_kl": item["final_val"]["kl"],
                              "selected_overlap_at_98": selected["validation"]["overlap_at_98"],
                              "final_overlap_at_98": item["final_val"]["overlap_at_98"]})
        delta = arrays["pr_final"][index] - arrays["pr_previous_final"][index]
        points = weights @ delta
        with np.load(run_dir / f"training/seed_{seed}/predictions_val.npz") as new, \
                np.load(previous / f"training/seed_{seed}/predictions_val.npz") as old:
            np.testing.assert_array_equal(new["image_ids"], old["image_ids"])
            score_difference = float(np.max(np.abs(new["predicted_scores"] - old["predicted_scores"])))
        replay.append({"seed": seed, "max_abs_predictor_score_difference": score_difference,
                       "max_abs_class_score_difference": float(np.max(np.abs(
                           aux["pr_final"]["scores"][index] - aux["pr_previous_final"]["scores"][index]))),
                       "changed_selected_sets": int(np.any(
                           aux["pr_final"]["patch_indices"][index] != aux["pr_previous_final"]["patch_indices"][index], axis=1).sum()),
                       "changed_binary_predictions": int(np.count_nonzero(delta[:, 0])),
                       "max_abs_binary_margin_difference": float(np.max(np.abs(delta[:, 1]))),
                       "accuracy_difference": float(points[0]), "margin_difference": float(points[1])})

    summary = {"dataset": config["dataset"], "budget": budget, "prune_after": config["prune_after"],
               "seeds": seeds, "sources": {name: str(path) for name, path in folders.items()},
               "image_count": len(set(ids)), "task_count": len({key[1] for key in keys}),
               "image_task_count": len(keys), "table": table,
               "primary_contrast": contrasts["pr_best_val_kl_minus_pr_final"],
               "contrasts": contrasts, "checkpoint_selection": selected_rows, "replay_comparison": replay,
               "auxiliary": {"image_count": len(all_ids), "aggregation": "Equal validation images.",
                             "methods": auxiliary, "best_minus_final": auxiliary_contrast},
               "training": training,
               "aggregation": "Equal tasks and equal class sides; seeds averaged before image bootstrap.",
               "intervals": "Best minus rerun-final: two-dataset adjustment per binary metric. Reference and auxiliary comparisons: ordinary 95% intervals.",
               "bootstrap_replicates": reps, "bootstrap_seed": rng_seed}
    (output / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_rows(output / "classification_table.csv", [
        {"method": row["method"], "accuracy_percent": 100 * row["accuracy"]["mean"],
         "accuracy_ci95_low_percent": 100 * row["accuracy"]["ci95"][0],
         "accuracy_ci95_high_percent": 100 * row["accuracy"]["ci95"][1],
         "geometric_margin": row["margin"]["mean"],
         "margin_ci95_low": row["margin"]["ci95"][0], "margin_ci95_high": row["margin"]["ci95"][1]}
        for row in table])
    write_rows(output / "contrast_table.csv", contrast_rows)
    write_rows(output / "per_seed.csv", seed_rows)
    write_rows(output / "checkpoint_selection.csv", selected_rows)
    write_rows(output / "reference_predictions.csv", reference_rows)
    np.savez_compressed(output / "paired_cases.npz", image_ids=ids,
                        task_ids=np.array([key[1] for key in keys]), weights=weights,
                        **{f"{name}_per_seed": values for name, values in arrays.items()})
    print(json.dumps({"dataset": config["dataset"], "selected_epochs": [r["selected_epoch"] for r in selected_rows],
                      "primary_contrast": summary["primary_contrast"], "replay_comparison": replay}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.run_dir, args.output)
