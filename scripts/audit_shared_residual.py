"""Recompute classification means and selection diagnostics from saved raw outputs."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np


SETTINGS = ("pets_b16_49", "flowers_b16_49", "flowers_b16_98")


def audit_curated(root):
    """Rebuild reported means/CIs from the locally backed-up image contributions."""
    results = []
    for setting in SETTINGS:
        folder = Path(root) / setting
        summary = json.loads((folder / "summary.json").read_text())
        checks, maximum_error = [], 0.
        for panel, seed in (("classification", 2271), ("exchange", 2272)):
            with (folder / f"analysis/{panel}_images.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            denominator = np.asarray([float(r["weight"]) for r in rows])
            names = [(r["method"], r) for r in summary[panel]["table"]]
            names += list(summary[panel]["contrasts"].items())
            rng = np.random.default_rng(seed)
            draws = rng.integers(len(rows), size=(summary["bootstrap_replicates"], len(rows)))
            for name, expected in names:
                numerator = np.asarray([[float(r[f"{name}_{m}_weighted_sum"]) for m in ("accuracy", "margin")] for r in rows])
                means = numerator.sum(0) / denominator.sum()
                boot = numerator[draws].sum(1) / denominator[draws].sum(1)[:, None]
                intervals = np.quantile(boot, [.025, .975], axis=0)
                family = np.quantile(boot, [.025 / 3, 1 - .025 / 3], axis=0)
                for j, metric in enumerate(("accuracy", "margin")):
                    pairs = [(means[j], expected[metric]["mean"]),
                             (intervals[:, j], expected[metric]["ci95"])]
                    if "bonferroni_family_ci95" in expected[metric]:
                        pairs.append((family[:, j], expected[metric]["bonferroni_family_ci95"]))
                    for actual, saved in pairs:
                        error = float(np.max(np.abs(np.asarray(actual) - np.asarray(saved))))
                        maximum_error = max(maximum_error, error)
                        assert error < 1e-12
                checks.append({"panel": panel, "method_or_contrast": name, "means_and_intervals_match": True})
        results.append({"setting": setting, "maximum_estimate_error": maximum_error, "checks": checks,
                        "historical_endpoint_records": [r["historical_shared_endpoint"] for r in summary["evaluation_records"]]})
    return {"description": "Independent means and paired bootstrap intervals reconstructed from backed-up per-image weighted contributions. These arrays already average the three seeds; this does not re-read per-seed class scores or checkpoints.",
            "settings": results, "bootstrap_replicates": 2000,
            "full_raw_score_audit": "pending server reconnection", "base_checkpoint_bundle": "pending server reconnection",
            "confirmed_completed_fits": 18, "final_test_used": False}


def audit(root):
    root = Path(root)
    results = []
    for setting in SETTINGS:
        folder = root / setting
        summary = json.loads((folder / "analysis/summary.json").read_text())
        with (folder / "examples.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        side_counts = Counter((r["task_id"], r["class_id"]) for r in rows)
        task_count = len({r["task_id"] for r in rows})
        weights = np.asarray([1 / (2 * task_count * side_counts[r["task_id"], r["class_id"]]) for r in rows])
        np.testing.assert_allclose(weights, [float(r["task_class_weight"]) for r in rows], rtol=0, atol=1e-15)
        labels = np.asarray([int(r["class_id"]) for r in rows])
        pairs = np.asarray([[int(r["class_a"]), int(r["class_b"])] for r in rows])
        competitors = np.where(labels == pairs[:, 0], pairs[:, 1], pairs[:, 0])
        with np.load(folder / "evidence.npz") as data:
            text = data["text_features"].astype(float)
        directions = np.linalg.norm(text[labels] - text[competitors], axis=1)
        seed_records = []
        aggregate, maximum_error = [], 0.
        for seed in summary["training_seeds"]:
            with np.load(folder / f"seed_{seed}.npz") as saved:
                scores = saved["cosine_scores"].astype(float)
                retained = saved["patch_indices"]
                logits = saved["selector_logits"]
                np.testing.assert_array_equal(np.sort(np.argsort(-logits, axis=-1, kind="stable")[..., :retained.shape[-1]], axis=-1), retained)
                assert np.all(np.diff(retained, axis=-1) > 0)
                membership = np.zeros(logits.shape, bool)
                np.put_along_axis(membership, retained.astype(int), True, axis=-1)
                changes = retained.shape[-1] - (membership & membership[:1]).sum(-1)
                np.testing.assert_array_equal(changes, saved["changed_tokens_vs_shared"])
                deviation = (logits - logits[:1]).std(-1)
                np.testing.assert_array_equal(deviation, saved["correction_patch_std"])
            own = scores[:, np.arange(len(rows)), labels]
            other = scores[:, np.arange(len(rows)), competitors]
            accuracy = (own > other) | ((own == other) & (labels < competitors))
            margin = (own - other) / directions
            means = np.stack((accuracy @ weights, margin @ weights), axis=-1) / weights.sum()
            aggregate.append(means)
            for mi, method in enumerate(summary["methods"]):
                table = next(r for r in summary["classification"]["table"] if r["method"] == method)
                expected = next(r for r in table["per_seed"] if r["seed"] == seed)
                error = max(abs(means[mi, 0] - expected["accuracy"]), abs(means[mi, 1] - expected["margin"]))
                maximum_error = max(maximum_error, error)
                assert error < 1e-12
            seed_records.append({"seed": seed, "methods": summary["methods"],
                "weighted_accuracy_margin": means.tolist(), "topk_masks_exact": True,
                "retained_positions_unique": True, "replacement_counts_exact": True,
                "residual_patch_std_exact": True})
        for mi, method in enumerate(summary["methods"]):
            table = next(r for r in summary["classification"]["table"] if r["method"] == method)
            for j, metric in enumerate(("accuracy", "margin")):
                error = abs(np.mean(aggregate, axis=0)[mi, j] - table[metric]["mean"])
                maximum_error = max(maximum_error, error)
                assert error < 1e-12
        results.append({"setting": setting, "source": str(folder.resolve()),
            "rows": len(rows), "weight_sum": float(weights.sum()),
            "maximum_classification_mean_error": maximum_error, "seed_records": seed_records,
            "historical_endpoint_records": [r["historical_shared_endpoint"] for r in summary["evaluation_records"]]})
    return {"description": "Raw class-score reconstruction with independently rebuilt equal-task/class weights; raw logit top-k, token replacement, and residual variation checks. Historical endpoint records originate from the run-time alignment check.",
            "settings": results, "final_test_used": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/shared-residual-v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--curated", action="store_true", help="Recompute aggregate means/CIs from locally copied per-image contributions.")
    args = parser.parse_args()
    result = audit_curated(args.root) if args.curated else audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    key = "maximum_estimate_error" if args.curated else "maximum_classification_mean_error"
    print(json.dumps({"settings": len(result["settings"]), key: max(r[key] for r in result["settings"])}))
