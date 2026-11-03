"""Compare additive-teacher and actual margin changes under KL -> ranking.

Uses the saved development-set paired cases; no model inference or training.
For a fixed image/task, the teacher top-k utility and intercept cancel, so
regret_KL - regret_rank is the teacher-predicted selected-mask margin change.
"""

import argparse
import csv
import json
from pathlib import Path
import subprocess

import numpy as np


SETTINGS = (("pets", 49), ("pets", 98), ("flowers", 49), ("flowers", 98))
ARMS = ("conditioned", "image_only")


def weighted_correlation(x, y, weights):
    x_centered = x - np.dot(weights, x)
    y_centered = y - np.dot(weights, y)
    denominator = np.sqrt(np.dot(weights, x_centered**2)
                          * np.dot(weights, y_centered**2))
    if denominator == 0:
        return None
    return float(np.dot(weights, x_centered * y_centered) / denominator)


def summarize_change(teacher_change, actual_change, weights):
    discrepancy = actual_change - teacher_change
    return {
        "teacher_predicted_margin_change": float(np.dot(weights, teacher_change)),
        "actual_margin_change": float(np.dot(weights, actual_change)),
        "actual_minus_teacher_change": float(np.dot(weights, discrepancy)),
        "mean_absolute_change_discrepancy": float(np.dot(weights, np.abs(discrepancy))),
        "weighted_change_correlation": weighted_correlation(
            teacher_change, actual_change, weights),
    }


def analyze_setting(results_root, dataset, budget):
    source_dir = results_root / f"{dataset}-selector-objectives-v1"
    cases_path = source_dir / f"budget_{budget}" / "paired_cases.npz"
    gradient_path = source_dir / "execution" / f"budget_{budget}" / "initial_gradient_diagnostics.json"
    with np.load(cases_path, allow_pickle=False) as source:
        data = {name: source[name] for name in source.files}

    weights = np.asarray(data["task_class_weight"], dtype=np.float64)
    np.testing.assert_allclose(weights.sum(), 1., rtol=0, atol=1e-12)
    assert np.all(weights > 0)
    seeds = data["training_seeds"].astype(int).tolist()
    regret_index = list(data["teacher_diagnostic_names"]).index(
        "teacher_additive_utility_regret")
    teacher_changes, actual_changes = {}, {}
    for arm in ARMS:
        teacher_changes[arm] = (
            data[f"kl_{arm}_teacher_diagnostics_per_seed"][:, :, regret_index]
            - data[f"rank_{arm}_teacher_diagnostics_per_seed"][:, :, regret_index])
        # The originating analysis records metrics in (accuracy, margin) order.
        actual_changes[arm] = (data[f"rank_{arm}_per_seed"][:, :, 1]
                               - data[f"kl_{arm}_per_seed"][:, :, 1])
        assert teacher_changes[arm].shape == actual_changes[arm].shape == (len(seeds), len(weights))
        np.testing.assert_allclose(actual_changes[arm].mean(axis=0),
                                   data[f"{arm}_rank_minus_kl_margin"], rtol=0, atol=1e-12)

    teacher_changes["conditioning_interaction"] = (
        teacher_changes["conditioned"] - teacher_changes["image_only"])
    actual_changes["conditioning_interaction"] = (
        actual_changes["conditioned"] - actual_changes["image_only"])
    np.testing.assert_allclose(actual_changes["conditioning_interaction"].mean(axis=0),
                               data["interaction_margin"], rtol=0, atol=1e-12)

    contrasts, rows = {}, []
    for contrast in teacher_changes:
        predicted, measured = teacher_changes[contrast], actual_changes[contrast]
        assert np.isfinite(predicted).all() and np.isfinite(measured).all()
        mean_summary = summarize_change(predicted.mean(axis=0), measured.mean(axis=0), weights)
        per_seed = [{"training_seed": seed,
                     **summarize_change(predicted[i], measured[i], weights)}
                    for i, seed in enumerate(seeds)]
        contrasts[contrast] = {"seed_mean": mean_summary, "per_seed": per_seed}
        for item in [{"training_seed": "mean", **mean_summary}, *per_seed]:
            rows.append({"dataset": dataset, "retained_patch_tokens": budget,
                         "contrast": contrast, **item})

    gradient_records = json.loads(gradient_path.read_text())
    gradient_summary = [{
        "training_seed": record["seed"], "arm": record["arm"],
        "batch_count": record["batch_count"],
        "mean_initial_kl_gradient_l2": record["mean_kl_gradient_l2"],
        "mean_initial_rank_gradient_l2": record["mean_topk_rank_gradient_l2"],
        "ratio_of_mean_initial_l2": (record["mean_topk_rank_gradient_l2"]
                                     / record["mean_kl_gradient_l2"]),
    } for record in gradient_records]
    summary = {
        "dataset": dataset, "retained_patch_tokens": budget,
        "source_cases": str(cases_path), "source_gradient_records": str(gradient_path),
        "training_seeds": seeds, "image_task_count": len(weights),
        "image_count": len(np.unique(data["image_id"])),
        "task_count": len(np.unique(data["task_id"])),
        "contrasts": contrasts, "recorded_initial_gradient_norms": gradient_summary,
    }
    return summary, rows


def run(results_root, output):
    summaries, rows = {}, []
    for dataset, budget in SETTINGS:
        summary, setting_rows = analyze_setting(results_root, dataset, budget)
        summaries[f"{dataset}_{budget}"] = summary
        rows.extend(setting_rows)
    repo_root = Path(__file__).resolve().parents[1]
    report = {
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip(),
        "analysis_role": "Exploratory descriptive point estimates from existing objective comparisons; no new resampling or significance tests.",
        "scope": "Existing development images and unseen development class combinations; frozen CLIP ViT-B/16, block 6. The image-only arm is the original fixed-context selector, not the later wide variant.",
        "reserved_test_used": False, "new_training": False, "new_inference": False,
        "units": "Signed geometric classification margin, cosine-score difference divided by the class text-direction norm.",
        "weighting": "Use saved task_class_weight: each task equally weighted, then its two classes equally weighted, then images within each task/class equally weighted. Average the three fixed seeds before the aggregate correlation; also report each seed separately.",
        "definitions": {
            "teacher_predicted_margin_change": "regret_KL - regret_rank; the common teacher top-k utility and intercept cancel for each image/task.",
            "actual_margin_change": "Actual pruned-CLIP margin_rank - margin_KL, from saved predictions.",
            "conditioning_interaction": "The conditioned arm's KL-to-ranking change minus the original image-only arm's change.",
            "actual_minus_teacher_change": "Change in the signed actual-minus-surrogate residual; this is not an absolute teacher-error estimate.",
            "mean_absolute_change_discrepancy": "Weighted absolute difference between actual and teacher-predicted changes; after seed averaging in the seed_mean row.",
            "weighted_change_correlation": "Weighted Pearson correlation across image/task rows under the same task/class-balanced weights; descriptive only.",
        },
        "gradient_record_interpretation": "These historical records measure initial raw gradient L2 norms, not parameter-update norms. With AdamW, a constant positive loss scaling approximately cancels between first and second moments (apart from epsilon/numerical effects); the norm ratio alone does not establish smaller effective updates. KL and ranking can still differ in gradient direction and relative coordinate magnitudes.",
        "settings": summaries,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / "teacher_transfer.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(output), "settings": len(summaries),
                      "summary_rows": len(rows), "reserved_test_used": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.results_root, arguments.output)
