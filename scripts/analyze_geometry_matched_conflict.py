#!/usr/bin/env python3
"""Match same/different-object task pairs using native scores, then evaluate.

The design stage never reads compressed predictions. No model inference is used.
"""
import argparse
import csv
import json
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def candidates_for_panel(panel, task_rows, text, native_scores, task_pool="val"):
    """Enumerate all valid development tasks, then unordered task pairs."""
    present = {source["class_id"] for source in panel["sources"]}
    tasks = []
    for slot, source in enumerate(panel["sources"]):
        label = source["class_id"]
        for row in sorted(task_rows, key=lambda row: row["task_id"]):
            if row["split"] not in (("val",) if task_pool == "val" else ("train", "val")):
                continue
            pair = (int(row["class_a"]), int(row["class_b"]))
            if label not in pair:
                continue
            other = pair[1] if pair[0] == label else pair[0]
            if other in present:
                continue
            direction = text[label] - text[other]
            norm = float(np.linalg.norm(direction))
            tasks.append({"task_id": row["task_id"], "task_split": row["split"], "slot": slot, "label": label,
                          "distractor": other, "direction_norm": norm,
                          "native_margin": float((native_scores[label] - native_scores[other]) / norm),
                          "unit_direction": direction / norm})
    pairs = []
    for a, b in combinations(range(len(tasks)), 2):
        first, second = tasks[a], tasks[b]
        angle = float(np.degrees(np.arccos(np.clip(np.dot(first["unit_direction"], second["unit_direction"]), -1, 1))))
        margins = sorted([first["native_margin"], second["native_margin"]])
        pairs.append({"key": f'{first["slot"]}:{first["task_id"]}|{second["slot"]}:{second["task_id"]}',
                      "different_objects": first["slot"] != second["slot"],
                      "angle_degrees": angle, "margin_min": margins[0], "margin_max": margins[1],
                      "tasks": [{k: v for k, v in task.items() if k != "unit_direction"}
                                for task in (first, second)]})
    return pairs, len(tasks)


def choose_match(pairs, angle_caliper=5., margin_caliper=.015, native_correct=False):
    """One deterministic within-composite match, using geometry only."""
    eligible = [p for p in pairs if not native_correct or p["margin_min"] > 0]
    same = sorted([p for p in eligible if not p["different_objects"]], key=lambda p: p["key"])
    different = sorted([p for p in eligible if p["different_objects"]], key=lambda p: p["key"])
    diagnostic = {"same_candidates": len(same), "different_candidates": len(different),
                  "cartesian_comparisons": len(same) * len(different),
                  "angle_pass": 0, "margin_pass": 0, "joint_pass": 0}
    if not same or not different:
        diagnostic["reason"] = "no_same_candidates" if not same else "no_different_candidates"
        return None, diagnostic
    coordinates = lambda values: np.asarray([[p["angle_degrees"], p["margin_min"], p["margin_max"]]
                                            for p in values], dtype=np.float64)
    gaps = np.abs(coordinates(same)[:, None, :] - coordinates(different)[None, :, :])
    distances = np.sum((gaps / [angle_caliper, margin_caliper, margin_caliper]) ** 2, axis=2)
    angle_ok, margin_ok = gaps[:, :, 0] <= angle_caliper, np.all(gaps[:, :, 1:] <= margin_caliper, axis=2)
    valid = angle_ok & margin_ok
    diagnostic.update(angle_pass=int(angle_ok.sum()), margin_pass=int(margin_ok.sum()), joint_pass=int(valid.sum()))
    nearest = np.unravel_index(np.argmin(distances), distances.shape)
    diagnostic["nearest_standardized_squared_distance"] = float(distances[nearest])
    gap_keys = ("angle_degrees", "margin_min", "margin_max")
    diagnostic["nearest_gaps"] = dict(zip(gap_keys, gaps[nearest].tolist()))
    if not valid.any():
        diagnostic["reason"] = ("no_angle_overlap" if not diagnostic["angle_pass"] else
                                "no_margin_overlap" if not diagnostic["margin_pass"] else "no_joint_caliper_match")
        return None, diagnostic
    # Sorted task keys and row-major argmin implement the prespecified tie break.
    best = np.unravel_index(np.argmin(np.where(valid, distances, np.inf)), distances.shape)
    diagnostic["reason"] = "matched"
    return {"same": same[best[0]], "different": different[best[1]],
            "squared_distance": float(distances[best]),
            "absolute_gaps": dict(zip(gap_keys, gaps[best].tolist()))}, diagnostic


def build_design(input_root, angle_caliper, margin_caliper, task_pool="val"):
    datasets = {}
    for dataset in ("pets", "flowers"):
        folder = input_root / dataset
        manifest = json.loads((folder / "panel_manifest.json").read_text())
        task_path = Path(manifest["source_manifest"]) / "tasks.csv"
        with task_path.open() as handle:
            task_rows = list(csv.DictReader(handle))
        with np.load(folder / "native_scores.npz") as native:
            text = native["text_features"].astype(np.float64)
            scores = native["panel_scores"].astype(np.float64)
        entries = []
        for index, panel in enumerate(manifest["panels"]):
            assert all(source["split"] == "val" for source in panel["sources"])
            pairs, task_count = candidates_for_panel(panel, task_rows, text, scores[index], task_pool)
            panels = {}
            for name, correct_only in (("all_native_difficulties", False), ("native_both_correct", True)):
                match, diagnostic = choose_match(pairs, angle_caliper, margin_caliper, correct_only)
                panels[name] = {"match": match, "diagnostic": diagnostic}
            entries.append({"panel_index": index, "panel_id": panel["panel_id"],
                            "sources": panel["sources"], "layout": panel["layout"],
                            "valid_binary_tasks": task_count, "panels": panels})
        coverage = {}
        for name in ("all_native_difficulties", "native_both_correct"):
            matched = [entry for entry in entries if entry["panels"][name]["match"] is not None]
            reasons = {}
            for entry in entries:
                reason = entry["panels"][name]["diagnostic"]["reason"]
                reasons[reason] = reasons.get(reason, 0) + 1
            coverage[name] = {"matched_composites": len(matched), "total_composites": len(entries),
                              "coverage_fraction": len(matched) / len(entries), "reasons": reasons,
                              "meets_20_composite_screen": len(matched) >= 20,
                              "unique_source_images": len({s["image_id"] for e in matched for s in e["sources"]})}
        datasets[dataset] = {"task_manifest": str(task_path), "coverage": coverage, "entries": entries}
    return datasets


def design_balance(design):
    """Describe residual matching differences and reused sources before evaluation."""
    result = {}
    for dataset, data in design["datasets"].items():
        result[dataset] = {}
        for panel_name in ("all_native_difficulties", "native_both_correct"):
            selected = [e for e in data["entries"] if e["panels"][panel_name]["match"] is not None]
            sources = [source["image_id"] for entry in selected for source in entry["sources"]]
            _, counts = np.unique(sources, return_counts=True)
            out = {"matched_composites": len(selected), "unique_source_images": len(counts),
                   "source_images_reused": int((counts > 1).sum()),
                   "maximum_source_image_reuse": int(counts.max()) if len(counts) else 0}
            for coordinate in ("angle_degrees", "margin_min", "margin_max"):
                same = np.asarray([e["panels"][panel_name]["match"]["same"][coordinate] for e in selected])
                different = np.asarray([e["panels"][panel_name]["match"]["different"][coordinate] for e in selected])
                out[coordinate] = ({"same_mean": float(same.mean()), "different_mean": float(different.mean()),
                    "paired_signed_difference_mean": float((different - same).mean()),
                    "paired_absolute_difference_mean": float(np.abs(different - same).mean()),
                    "paired_absolute_difference_max": float(np.abs(different - same).max())} if len(selected) else None)
            result[dataset][panel_name] = out
    return result


def bank_outcomes(scores, pair):
    margins = np.stack([(scores[:, t["label"]] - scores[:, t["distractor"]]) / t["direction_norm"]
                        for t in pair["tasks"]], axis=1)
    separate = margins.max(0)
    return {"separately_feasible": int(np.all(separate > 0)),
            "common_feasible": int(np.any(np.all(margins > 0, axis=1))),
            "conflict": int(np.all(separate > 0) and not np.any(np.all(margins > 0, axis=1))),
            "mean_margin_gap": float(separate.mean() - margins.mean(1).max())}


def paired_summary(rows, replicates, seed):
    summary = {"composites": len(rows), "outcomes": {}}
    if not rows:
        return summary
    rng = np.random.default_rng(seed)
    ix = rng.integers(len(rows), size=(replicates, len(rows)))
    for metric in ("conflict", "common_feasible", "separately_feasible", "mean_margin_gap"):
        same = np.asarray([r[f"same_{metric}"] for r in rows])
        different = np.asarray([r[f"different_{metric}"] for r in rows])
        delta = different - same
        scale = 1. if metric == "mean_margin_gap" else 100.
        summary["outcomes"][metric] = {"same_mean": float(same.mean() * scale),
            "different_mean": float(different.mean() * scale), "different_minus_same": float(delta.mean() * scale),
            "paired_image_bootstrap_ci95": (np.quantile(delta[ix].mean(1), [.025, .975]) * scale).tolist(),
            "units": "geometric margin" if metric == "mean_margin_gap" else "percentage points"}
        if metric != "mean_margin_gap":
            summary["outcomes"][metric].update({"same_count": int(same.sum()), "different_count": int(different.sum()),
                                               "different_only": int(((different == 1) & (same == 0)).sum()),
                                               "same_only": int(((different == 0) & (same == 1)).sum())})
    return summary


def evaluate_design(design, input_root, replicates, seed):
    all_rows, summary = [], {}
    for dataset, data in design["datasets"].items():
        summary[dataset] = {}
        fixed_masks = {}
        for layer in (3, 6):
            for keep in (16, 49):
                source = input_root / dataset / f"composite_layer{layer}_k{keep}.npz"
                with np.load(source) as saved:
                    fixed = saved["methods"] != "attention"
                    scores, masks = saved["cosine_scores"][:, fixed].astype(np.float64), saved["masks"][:, fixed]
                    assert fixed.sum() == 35
                if keep in fixed_masks:
                    np.testing.assert_array_equal(masks, fixed_masks[keep])
                else:
                    fixed_masks[keep] = masks
                for panel_name in ("all_native_difficulties", "native_both_correct"):
                    condition_rows = []
                    for entry in data["entries"]:
                        match = entry["panels"][panel_name]["match"]
                        if match is None:
                            continue
                        row = {"dataset": dataset, "panel": panel_name, "layer": layer, "keep": keep,
                               "panel_id": entry["panel_id"], "source_image_0": entry["sources"][0]["image_id"],
                               "source_image_1": entry["sources"][1]["image_id"]}
                        for arm in ("same", "different"):
                            values = bank_outcomes(scores[entry["panel_index"]], match[arm])
                            row.update({f"{arm}_{key}": value for key, value in values.items()})
                        condition_rows.append(row)
                    all_rows.extend(condition_rows)
                    summary[dataset][f"{panel_name}_layer{layer}_k{keep}"] = paired_summary(condition_rows, replicates, seed)
    return all_rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/evidence-conflict-v1"))
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--design-only", action="store_true")
    mode.add_argument("--evaluate-only", action="store_true", help="Read the saved design without rebuilding or overwriting it")
    parser.add_argument("--task-pool", choices=["val", "train_val"], default="val")
    parser.add_argument("--angle-caliper", type=float, default=5.)
    parser.add_argument("--margin-caliper", type=float, default=.015)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2111)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.evaluate_only:
        design = json.loads((args.output / "matched_design.json").read_text())
    else:
        design = {"source_root": str(args.input), "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "protocol": {"angle_caliper_degrees": args.angle_caliper, "each_ordered_margin_caliper": args.margin_caliper,
                               "matching_metric": "squared distance after dividing each coordinate gap by its caliper",
                               "selection": "one match per composite per panel; minimum distance, then lexicographic task keys",
                               "task_pool": args.task_pool,
                               "task_split": "preassigned val" if args.task_pool == "val" else "preassigned train+val, excluding all test pairs",
                               "interpretation": "mechanism diagnostic, not unseen-combination generalization" if args.task_pool == "train_val" else "development val-pair diagnostic",
                               "screen_minimum_matched_composites": 20,
                               "screen_interpretation": "20 is a prespecified coverage screen, not a statistical power guarantee",
                               "compressed_scores_used_for_matching": False},
                  "datasets": build_design(args.input, args.angle_caliper, args.margin_caliper, args.task_pool)}
        write_json(args.output / "matched_design.json", design)
        coverage = {dataset: values["coverage"] for dataset, values in design["datasets"].items()}
        write_json(args.output / "coverage.json", coverage)
        write_json(args.output / "matching_balance.json", design_balance(design))
        print(json.dumps(coverage, indent=2), flush=True)
    if args.design_only:
        return
    rows, summary = evaluate_design(design, Path(design["source_root"]), args.bootstrap_replicates, args.bootstrap_seed)
    if rows:
        with (args.output / "paired_outcomes.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(args.output / "summary.json", {"primary_outcome": "different-object minus same-object joint-conflict rate",
        "evaluation_code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "task_pool": design["protocol"].get("task_pool", "val"),
        "margin_units": "geometric signed margin", "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_replicates": args.bootstrap_replicates,
        "uncertainty_scope": "paired composite bootstrap conditional on selected tasks/classes/source panel; reused source images are not independent across composites",
        "zero_event_interpretation": "zero-width empirical bootstrap intervals with no conflicts do not establish equivalence",
        "datasets": summary})


if __name__ == "__main__":
    main()
