"""Compare near-full and budget-matched ranking supervision on paired images."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from token_compression.selection_analysis import cluster_summary, image_means


def read_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def paired_summary(ids, values):
    result = cluster_summary(ids, values, bootstrap_replicates=1000, seed=2030)
    return {"image_count": result["image_count"], "row_count": result["row_count"],
            "mean": float(result["mean"]),
            "ci95": [float(result["ci95_low"]), float(result["ci95_high"])]}


def compare(results, previous):
    sources = {"legacy_block": previous,
               "near_full": results / "near_full",
               "budget_matched": results / "budget_matched"}
    singles = {name: read_rows(path / "single_task_rows.csv") for name, path in sources.items()}
    exchanges = {name: read_rows(path / "exchange_rows.csv") for name, path in sources.items()}
    budgets = sorted({int(row["keep_patch_tokens"]) for row in singles["near_full"]})
    output = {"scope": "Paired development comparison; no learned selector or final test evaluation.",
              "aggregation": "Paired effects average tasks within image, then images; accuracy table uses balanced task examples.",
              "budgets": []}
    prediction_rows = read_rows(results / "holdout_prediction_rows.csv")
    for k in budgets:
        single_lookup = {}
        table = []
        for name, rows in singles.items():
            selected = [row for row in rows if int(row["keep_patch_tokens"]) == k and row["method"] == "task_rank"]
            single_lookup[name] = {(row["image_id"], row["task_id"]): row for row in selected}
            assert len(selected) == len(single_lookup[name])
            table.append({"method": name, "n": len(selected),
                          "accuracy": float(np.mean([int(row["correct"]) for row in selected])),
                          "new_errors": sum(int(row["base_correct"]) and not int(row["correct"]) for row in selected),
                          "recoveries": sum(not int(row["base_correct"]) and int(row["correct"]) for row in selected)})
        keys = sorted(single_lookup["near_full"])
        assert all(sorted(values) == keys for values in single_lookup.values())
        ids = [key[0] for key in keys]
        contrasts = {}
        for reference in ("near_full", "legacy_block"):
            for metric in ("correct", "margin"):
                delta = [float(single_lookup["budget_matched"][key][metric]) - float(single_lookup[reference][key][metric]) for key in keys]
                contrasts[f"budget_minus_{reference}_{metric}"] = paired_summary(ids, delta)
        attention = [row for row in singles["budget_matched"] if int(row["keep_patch_tokens"]) == k and row["method"] == "attention"]
        table.append({"method": "attention", "n": len(attention),
                      "accuracy": float(np.mean([int(row["correct"]) for row in attention])),
                      "new_errors": sum(int(row["base_correct"]) and not int(row["correct"]) for row in attention),
                      "recoveries": sum(not int(row["base_correct"]) and int(row["correct"]) for row in attention)})

        # A common union reference prevents a weaker bank from manufacturing an advantage.
        bank_values = []
        case_ids = None
        with np.load(results / "evidence.npz") as evidence:
            text = evidence["text_features"].copy()
        for name, path in sources.items():
            with np.load(path / f"candidate_bank_k{k}.npz") as bank:
                ids_in_bank = bank["image_ids"]
                cases = bank["cases"]
                keyed = [(str(ids_in_bank[i]), int(y), int(a), int(b)) for i, y, a, b in cases]
                if case_ids is None:
                    case_ids = keyed
                assert keyed == case_ids
                bank_values.append(bank["cosine_scores"].copy())
        scores = np.concatenate(bank_values, axis=1)
        margins = np.empty(scores.shape[:2] + (2,))
        correctness = np.empty_like(margins)
        for i, (_, y, a, b) in enumerate(case_ids):
            norm = np.linalg.norm(text[y] - text[[a, b]], axis=1)
            margins[i] = (scores[i, :, y, None] - scores[i][:, [a, b]]) / norm
            for side, other in enumerate((a, b)):
                difference = scores[i, :, y] - scores[i, :, other]
                correctness[i, :, side] = (difference > 0) | ((difference == 0) & (y < other))
        candidates = np.arange(len(case_ids))
        common = {}
        for objective, winner in (("mean", margins.mean(2).argmax(1)), ("maxmin", margins.min(2).argmax(1))):
            common[f"margin_{objective}"] = margins[candidates, winner].mean(1)
            common[f"accuracy_{objective}"] = correctness[candidates, winner].mean(1)
        task_winners = margins.argmax(1)
        common["task_search_margin"] = margins[candidates[:, None], task_winners, np.arange(2)[None, :]].mean(1)
        common["task_search_accuracy"] = correctness[candidates[:, None], task_winners, np.arange(2)[None, :]].mean(1)
        image_ids = [row[0] for row in case_ids]
        shared = {name: paired_summary(image_ids, values) for name, values in common.items()}
        for arm in sources:
            exchange_selected = [row for row in exchanges[arm] if int(row["keep_patch_tokens"]) == k]
            exchange_by_key = {(row["image_id"], int(row["class_id"]), int(row["competitor_a"]), int(row["competitor_b"])): row for row in exchange_selected}
            assert sorted(exchange_by_key) == sorted(case_ids)
            for metric, prefix in (("margin", "margin"), ("accuracy", "correct")):
                own = np.asarray([(float(exchange_by_key[key][f"{prefix}_task_a_a"]) + float(exchange_by_key[key][f"{prefix}_task_b_b"])) / 2 for key in case_ids])
                shared[f"{arm}_own_{metric}"] = paired_summary(image_ids, own)
                for objective in ("mean", "maxmin"):
                    shared[f"{arm}_own_minus_union_shared_{metric}_{objective}"] = paired_summary(image_ids, own - common[f"{metric}_{objective}"])

        prediction = {}
        for arm in ("near_full", "budget_matched"):
            rows = [row for row in prediction_rows if int(row["keep_patch_tokens"]) == k and row["arm"] == arm]
            prediction[arm] = {}
            for metric in ("raw_mse", "centered_mse", "pearson"):
                selected = [row for row in rows if metric != "pearson" or int(row.get("pearson_valid", "1"))]
                prediction[arm][metric] = paired_summary([row["image_id"] for row in selected], [float(row[metric]) for row in selected])
        output["budgets"].append({"keep_patch_tokens": k, "balanced_accuracy": table,
                                  "paired_effects": contrasts,
                                  "independent_mask_prediction": prediction,
                                  "union_candidate_count": scores.shape[1], "union_shared_reference": shared})
    return output


def plot(comparison, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    budgets = [b["keep_patch_tokens"] for b in comparison["budgets"]]
    labels = {"legacy_block": "Previous block-deletion ranking", "near_full": "Near-full regression ranking",
              "budget_matched": "Budget-matched regression ranking", "attention": "Attention"}
    for method, label in labels.items():
        values = [next(row["accuracy"] for row in b["balanced_accuracy"] if row["method"] == method) for b in comparison["budgets"]]
        axes[0].plot(budgets, np.asarray(values) * 100, "o-", label=label)
    for arm in ("near_full", "budget_matched"):
        values = [b["independent_mask_prediction"][arm]["pearson"]["mean"] for b in comparison["budgets"]]
        axes[1].plot(budgets, values, "o-", label=labels[arm])
    axes[0].set(ylabel="Balanced task accuracy (%)", title="Actual top-k selection performance")
    axes[1].set(ylabel="Mean held-out mask correlation", title="Prediction on independent fixed-budget masks")
    for ax in axes:
        ax.set(xlabel="Retained patches", xticks=budgets)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(destination / f"budget_comparison.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    comparison = compare(args.results, args.previous)
    (args.results / "comparison.json").write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n")
    if args.plot:
        plot(comparison, args.results)
    for budget in comparison["budgets"]:
        print(budget["keep_patch_tokens"], [(row["method"], round(row["accuracy"], 4)) for row in budget["balanced_accuracy"]])
