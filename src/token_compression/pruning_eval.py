"""Paired classification diagnostics for a pruned CLIP and its full-token baseline.

These metrics describe compression effects. They do not measure inference
speed or establish whether task-conditioned selection is beneficial.
"""

from collections import defaultdict

import numpy as np

from .metrics import binary_margin
from .pets import balanced_task_rows


def _summarize(rows):
    """Summarize (baseline correctness, new correctness, margin loss) rows."""
    n = len(rows)
    baseline_correct = sum(row[0] for row in rows)
    correct = sum(row[1] for row in rows)
    new_errors = sum(base and not new for base, new, _ in rows)
    recoveries = sum(not base and new for base, new, _ in rows)
    baseline_wrong = n - baseline_correct
    return {
        "n": n,
        "correct": correct,
        "baseline_correct": baseline_correct,
        "accuracy": correct / n if n else None,
        "baseline_accuracy": baseline_correct / n if n else None,
        "new_errors": new_errors,
        "recoveries": recoveries,
        "new_error_rate_among_baseline_correct": (
            new_errors / baseline_correct if baseline_correct else None
        ),
        "recovery_rate_among_baseline_wrong": (
            recoveries / baseline_wrong if baseline_wrong else None
        ),
        "mean_geometric_margin_loss": (
            float(np.mean([row[2] for row in rows])) if n else None
        ),
    }


def evaluate_variant(scores, baseline_scores, text_features, labels, image_ids,
                     images, tasks, image_split, task_splits, balance_seed,
                     quartile_edges):
    """Compare paired scores, reusing a fixed baseline-margin stratification.

    Accuracies and error rates are fractions; ``*_pp`` values are percentage
    points. Quartile edges must be fixed before inspecting variant results.
    A boundary value belongs to the higher stratum. Signed negative margins
    are retained, and binary ties follow the baseline's first-candidate rule.
    """
    baseline_correct = baseline_scores.argmax(axis=1) == labels
    correct = scores.argmax(axis=1) == labels
    summary = {
        "image_count": len(labels),
        "accuracy_37class": float(np.mean(correct)),
        "baseline_accuracy_37class": float(np.mean(baseline_correct)),
        "net_accuracy_change_pp": float(100 * (np.mean(correct) - np.mean(baseline_correct))),
        "baseline_correct_to_wrong_count": int(np.sum(baseline_correct & ~correct)),
        "baseline_wrong_to_correct_count": int(np.sum(~baseline_correct & correct)),
        "binary": {},
        "interpretation_scope": (
            "Compression diagnostics only; no inference speed claim or test of H1/H2."
        ),
    }
    lookup = {image_id: index for index, image_id in enumerate(image_ids)}
    per_task_rows, per_stratum_rows = [], []
    for split in task_splits:
        by_task = defaultdict(list)
        by_stratum = defaultdict(list)
        task_classes = {}
        for row in balanced_task_rows(images, tasks, image_split, split, seed=balance_seed):
            index = lookup[row["image_id"]]
            a, b, label = row["class_a"], row["class_b"], row["class_id"]
            competitor = b if label == a else a
            base_prediction = (a, b)[int(np.argmax(baseline_scores[index, [a, b]]))]
            prediction = (a, b)[int(np.argmax(scores[index, [a, b]]))]
            _, base_margin = binary_margin(baseline_scores[index], text_features, label, competitor)
            _, margin = binary_margin(scores[index], text_features, label, competitor)
            result = (base_prediction == label, prediction == label, base_margin - margin)
            by_task[row["task_id"]].append(result)
            task_classes[row["task_id"]] = (a, b)
            stratum = int(np.searchsorted(quartile_edges, base_margin, side="right")) + 1
            by_stratum[stratum].append(result)

        split_tasks = []
        for task_id, rows in by_task.items():
            a, b = task_classes[task_id]
            task_result = {"task_split": split, "task_id": task_id,
                           "class_a": a, "class_b": b, **_summarize(rows)}
            per_task_rows.append(task_result)
            split_tasks.append(task_result)
        aggregate = _summarize([row for rows in by_task.values() for row in rows])
        summary["binary"][split] = {
            "task_count": len(split_tasks),
            "example_count": aggregate["n"],
            "accuracy_macro": float(np.mean([row["accuracy"] for row in split_tasks])),
            "accuracy_micro": aggregate["accuracy"],
            "baseline_accuracy_macro": float(np.mean([row["baseline_accuracy"] for row in split_tasks])),
            "baseline_accuracy_micro": aggregate["baseline_accuracy"],
            "net_accuracy_change_pp": 100 * (aggregate["accuracy"] - aggregate["baseline_accuracy"]),
            "baseline_correct_count": aggregate["baseline_correct"],
            "new_errors": aggregate["new_errors"],
            "recoveries": aggregate["recoveries"],
            "new_error_rate_among_baseline_correct": aggregate["new_error_rate_among_baseline_correct"],
            "recovery_rate_among_baseline_wrong": aggregate["recovery_rate_among_baseline_wrong"],
            "mean_geometric_margin_loss": aggregate["mean_geometric_margin_loss"],
        }
        for stratum in range(1, 5):
            per_stratum_rows.append({"task_split": split, "stratum": f"Q{stratum}",
                                     **_summarize(by_stratum[stratum])})
    return summary, per_task_rows, per_stratum_rows
