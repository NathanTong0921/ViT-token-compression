"""Describe all paired selector error changes using pre-existing difficulty bins."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from token_compression.selection_analysis import cluster_summary


def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def row_key(row):
    return row["task_split"], row["image_id"], row["task_id"]


def quartile(margin, edges):
    """The existing protocol assigns an exact boundary to the higher bin."""
    return f"Q{np.searchsorted(edges, margin, side='right') + 1}"


def transitions(reference, model):
    reference, model = np.asarray(reference, dtype=int), np.asarray(model, dtype=int)
    repaired = int(np.sum((reference == 0) & (model == 1)))
    introduced = int(np.sum((reference == 1) & (model == 0)))
    return {"repaired_errors": repaired, "new_errors": introduced,
            "net_correct_gain": repaired - introduced,
            "unchanged_correct": int(np.sum((reference == 1) & (model == 1))),
            "unchanged_wrong": int(np.sum((reference == 0) & (model == 0)))}


def estimate(ids, values):
    if not len(ids):
        return {"row_count": 0, "image_count": 0, "mean": None, "ci95": None}
    result = cluster_summary(ids, values, bootstrap_replicates=2000, seed=2051)
    return {"row_count": result["row_count"], "image_count": result["image_count"],
            "mean": float(result["mean"]),
            "ci95": [float(result["ci95_low"]), float(result["ci95_high"])]}


def seed_pattern(differences):
    present = set(differences)
    if present == {0}:
        return "never_changed"
    if present == {1}:
        return "persistent_repair"
    if present == {-1}:
        return "persistent_new_error"
    if 1 in present and -1 in present:
        return "direction_reversal"
    return "intermittent_repair" if 1 in present else "intermittent_new_error"


def analyze(rows, edges):
    lookup = defaultdict(dict)
    for row in rows:
        if row["method"] not in {"conditioned", "image_only", "full_model", "attention"}:
            continue
        group = (int(row["seed"]), row["method"])
        key = row_key(row)
        assert key not in lookup[group], (group, key)
        lookup[group][key] = row
    full, attention = lookup[(-1, "full_model")], lookup[(-1, "attention")]
    keys = sorted(full)
    assert keys and sorted(attention) == keys
    seeds = sorted(seed for seed, method in lookup if method == "conditioned")
    assert seeds and seeds == sorted(seed for seed, method in lookup if method == "image_only")
    for group, values in lookup.items():
        assert sorted(values) == keys, group
        for key in keys:
            row, reference = values[key], full[key]
            assert all(int(row[field]) == int(reference[field])
                       for field in ("class_id", "class_a", "class_b")), (group, key)
            assert int(row["base_correct"]) == int(reference["correct"]), (group, key)
            assert abs(float(row["base_margin"]) - float(reference["margin"])) < 1e-8, (group, key)
    bins = {key: quartile(float(full[key]["margin"]), edges) for key in keys}
    differences = {key: [int(lookup[(seed, "conditioned")][key]["correct"])
                         - int(lookup[(seed, "image_only")][key]["correct"]) for seed in seeds]
                   for key in keys}
    patterns = {key: seed_pattern(values) for key, values in differences.items()}
    output = {
        "scope": "Exploratory error diagnostics on all existing paired development examples; no examples selected for aggregate claims based on their outcomes.",
        "training_seeds": seeds,
        "fixed_margin_quartile_edges": list(edges),
        "quartile_rule": "Q1: margin < first edge; an exact edge belongs to the higher quartile. Negative full-model margins remain in Q1.",
        "uncertainty": "2,000 image-bootstrap replicates, seed 2051. Tasks average within image before images receive equal weight. Seed-averaged effects average paired seeds per example first; intervals condition on these seeds and task collections.",
        "counting": "Error counts refer to image/task examples. One image may contribute multiple tasks; unique-image counts are separate.",
        "task_splits": {},
    }
    changed = []
    for split in ("seen", "unseen"):
        split_keys = [key for key in keys if key[0] == split]
        pattern_counts = dict(Counter(patterns[key] for key in split_keys))
        cases = [{"image_id": key[1], "task_id": key[2],
                  "category": patterns[key], "conditional_minus_image_only_correct_by_seed": differences[key]}
                 for key in split_keys if patterns[key] != "never_changed"]
        split_result = {"example_count": len(split_keys),
                        "image_count": len({key[1] for key in split_keys}),
                        "task_count": len({key[2] for key in split_keys}),
                        "correctness_seed_patterns": {"counts": pattern_counts, "changed_cases": cases},
                        "strata": {}}
        for stratum in ("all", "Q1", "Q2", "Q3", "Q4"):
            selected = [key for key in split_keys if stratum == "all" or bins[key] == stratum]
            ids = [key[1] for key in selected]
            full_correct = np.array([int(full[key]["correct"]) for key in selected])
            attention_correct = np.array([int(attention[key]["correct"]) for key in selected])
            info = {"example_count": len(selected), "image_count": len(set(ids)),
                    "task_count": len({key[2] for key in selected}),
                    "full_model_correct": int(full_correct.sum()),
                    "attention_correct": int(attention_correct.sum()),
                    "attention_vs_full_model": transitions(full_correct, attention_correct),
                    "per_seed": []}
            margin_deltas, correctness_deltas = [], []
            for seed in seeds:
                conditional, shared = lookup[(seed, "conditioned")], lookup[(seed, "image_only")]
                c = np.array([int(conditional[key]["correct"]) for key in selected])
                s = np.array([int(shared[key]["correct"]) for key in selected])
                delta = np.array([float(conditional[key]["margin"]) - float(shared[key]["margin"])
                                  for key in selected])
                margin_deltas.append(delta)
                correctness_deltas.append(c - s)
                info["per_seed"].append({
                    "seed": seed, "conditioned_correct": int(c.sum()), "image_only_correct": int(s.sum()),
                    "conditioned_vs_image_only": transitions(s, c),
                    "conditioned_vs_full_model": transitions(full_correct, c),
                    "image_only_vs_full_model": transitions(full_correct, s),
                    "conditioned_vs_attention": transitions(attention_correct, c),
                    "image_only_vs_attention": transitions(attention_correct, s),
                    "conditioned_minus_image_only_margin": estimate(ids, delta),
                    "conditioned_minus_image_only_accuracy": estimate(ids, c - s)})
            info["seed_averaged_conditional_minus_image_only"] = {
                "margin": estimate(ids, np.mean(margin_deltas, axis=0)),
                "accuracy": estimate(ids, np.mean(correctness_deltas, axis=0))}
            split_result["strata"][stratum] = info
        output["task_splits"][split] = split_result
        for key in split_keys:
            for seed, difference in zip(seeds, differences[key]):
                if difference == 0:
                    continue
                c, s = lookup[(seed, "conditioned")][key], lookup[(seed, "image_only")][key]
                reference, att = full[key], attention[key]
                changed.append({"seed": seed, "task_split": split, "image_id": key[1], "task_id": key[2],
                                "class_id": int(reference["class_id"]), "class_a": int(reference["class_a"]),
                                "class_b": int(reference["class_b"]), "margin_quartile": bins[key],
                                "change": "repaired_error" if difference == 1 else "new_error",
                                "seed_consistency": patterns[key],
                                "full_model_correct": int(reference["correct"]),
                                "full_model_margin": float(reference["margin"]),
                                "attention_correct": int(att["correct"]), "attention_margin": float(att["margin"]),
                                "image_only_correct": int(s["correct"]), "image_only_margin": float(s["margin"]),
                                "conditioned_correct": int(c["correct"]), "conditioned_margin": float(c["margin"]),
                                "conditional_minus_image_only_margin": float(c["margin"]) - float(s["margin"])})
    output["changed_case_row_count"] = len(changed)
    return output, changed


CHANGED_FIELDS = ["seed", "task_split", "image_id", "task_id", "class_id", "class_a", "class_b",
                  "margin_quartile", "change", "seed_consistency", "full_model_correct", "full_model_margin",
                  "attention_correct", "attention_margin", "image_only_correct", "image_only_margin",
                  "conditioned_correct", "conditioned_margin", "conditional_minus_image_only_margin"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/pets-selector-pilot-v1"))
    parser.add_argument("--pruning-summary", type=Path, default=Path("results/pets-pruning-v1/summary.json"))
    parser.add_argument("--output", type=Path, default=Path("results/pets-selector-exchange-v1/error_analysis"))
    args = parser.parse_args()
    edges = json.loads(args.pruning_summary.read_text())["config"]["margin_quartile_edges"]
    summary, changed = analyze(read_rows(args.source / "evaluation" / "predictions.csv"), edges)
    summary["source"] = str(args.source)
    summary["quartile_edge_source"] = str(args.pruning_summary)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (args.output / "changed_cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CHANGED_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(changed)
    for split, result in summary["task_splits"].items():
        print(split, [(row["seed"], row["conditioned_vs_image_only"])
                      for row in result["strata"]["all"]["per_seed"]])
