"""Compare learned task and image score corrections with the same frozen base."""

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from analyze_core_protection import exchange_outcomes, save_cases
from analyze_selector_ablations import binary_scores
from analyze_selector_budgets import weighted_estimates
from analyze_selector_scale import panel_weights, read_rows
from evaluate_selectors import write_rows


METRICS = ("accuracy", "margin")
COMPARISONS = (("task_residual", "image_residual"), ("task_residual", "shared"),
               ("image_residual", "shared"))


def summarize(ids, weights, values, seeds, replicates, bootstrap_seed, primary=False):
    table, contrasts, differences = [], {}, {}
    for name, value in values.items():
        points = np.einsum("n,snm->sm", weights, value) / weights.sum()
        table.append({"method": name, **weighted_estimates(ids, weights, value.mean(0), METRICS, replicates, bootstrap_seed),
                      "per_seed": [{"seed": seed, **dict(zip(METRICS, point.tolist()))} for seed, point in zip(seeds, points)]})
    for a, b in COMPARISONS:
        name = f"{a}_minus_{b}"
        delta = values[a] - values[b]
        family = 3 if primary and b == "image_residual" else 1
        contrasts[name] = weighted_estimates(ids, weights, delta.mean(0), METRICS, replicates, bootstrap_seed, family)
        points = np.einsum("n,snm->sm", weights, delta) / weights.sum()
        contrasts[name]["per_seed"] = [{"seed": seed, **dict(zip(METRICS, point.tolist()))} for seed, point in zip(seeds, points)]
        differences[name] = delta
    return {"table": table, "contrasts": contrasts}, differences


def analyze(source, output, replicates=2000):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((source / "protocol.json").read_text())
    seeds, methods = protocol["training_seeds"], protocol["methods"]
    base = read_rows(source / "examples.csv")
    ix = np.asarray([int(r["image_index"]) for r in base])
    ids, tasks = np.asarray([r["image_id"] for r in base]), np.asarray([r["task_id"] for r in base])
    labels = np.asarray([int(r["class_id"]) for r in base])
    pairs = np.asarray([[int(r["class_a"]), int(r["class_b"])] for r in base])
    others = np.where(labels == pairs[:, 0], pairs[:, 1], pairs[:, 0])
    weights = panel_weights(base, "all")
    np.testing.assert_allclose(weights, [float(r["task_class_weight"]) for r in base], rtol=0, atol=1e-12)
    with np.load(source / "evidence.npz") as saved:
        text, full_scores = saved["text_features"].astype(float), saved["full_scores"][ix].astype(float)
        np.testing.assert_array_equal(saved["image_ids"][ix], ids)
    scores, masks, diagnostics = [], [], []
    names = ("changed_tokens_vs_shared", "correction_patch_std", "correction_patch_rms", "shared_score_patch_std")
    for seed in seeds:
        with np.load(source / f"seed_{seed}.npz") as saved:
            np.testing.assert_array_equal(saved["methods"], methods)
            scores.append(saved["cosine_scores"].astype(float))
            masks.append(saved["patch_indices"].copy())
            fields = [saved[name].astype(float) for name in names[:3]]
            fields.append(np.broadcast_to(saved["shared_score_patch_std"], fields[0].shape))
            diagnostics.append(np.stack(fields, axis=-1))
    scores, masks, diagnostics = np.stack(scores), np.stack(masks), np.stack(diagnostics)
    raw = binary_scores(scores.reshape(-1, len(base), len(text)), labels, others, text)
    raw = raw.reshape(len(seeds), len(methods), len(base), 2)
    values = {method: raw[:, mi] for mi, method in enumerate(methods)}
    classification, differences = summarize(ids, weights, values, seeds, replicates, 2271, primary=True)
    classification["full_model_reference"] = weighted_estimates(ids, weights,
        binary_scores(full_scores[None], labels, others, text)[0], METRICS, replicates, 2271)
    metadata = {"image_id": ids, "task_id": tasks, "class_id": labels, "class_a": pairs[:, 0], "class_b": pairs[:, 1]}
    save_cases(output, "classification", metadata, {**values, **differences}, seeds, weights)
    np.savez_compressed(output / "raw_classification_outcomes.npz", training_seeds=seeds,
                        methods=methods, outcomes=raw, metric_names=METRICS)

    groups = defaultdict(list)
    for row, image_id in enumerate(ids):
        groups[image_id].append(row)
    left, right = np.asarray([p for image_id in sorted(groups) for p in combinations(groups[image_id], 2)]).T
    np.testing.assert_array_equal(labels[left], labels[right])
    exchange_ids = ids[left]
    unique, counts = np.unique(exchange_ids, return_counts=True)
    lookup = dict(zip(unique, counts))
    exchange_weights = np.asarray([1 / (len(unique) * lookup[i]) for i in exchange_ids])
    own_raw, swapped_raw = exchange_outcomes(scores, labels, others, text, left, right)
    own = {method: own_raw[:, mi] for mi, method in enumerate(methods)}
    swapped = {method: swapped_raw[:, mi] for mi, method in enumerate(methods)}
    effects = {method: own[method] - swapped[method] for method in methods}
    for method in ("shared", "image_residual"):
        np.testing.assert_array_equal(effects[method], np.zeros_like(effects[method]))
        np.testing.assert_array_equal(masks[:, methods.index(method), left], masks[:, methods.index(method), right])
    exchange, exchange_differences = summarize(exchange_ids, exchange_weights, effects, seeds, replicates, 2272)
    exchange["correct_task_table"] = summarize(exchange_ids, exchange_weights, own, seeds, replicates, 2272)[0]["table"]
    exchange["swapped_task_table"] = summarize(exchange_ids, exchange_weights, swapped, seeds, replicates, 2272)[0]["table"]
    exchange.update(image_count=len(unique), task_pair_count=len(left))
    exchange_meta = {"image_id": exchange_ids, "task_a": tasks[left], "task_b": tasks[right], "class_id": labels[left]}
    save_cases(output, "exchange", exchange_meta,
               {**effects, **exchange_differences, **{f"{m}_own": v for m, v in own.items()},
                **{f"{m}_swapped": v for m, v in swapped.items()}}, seeds, exchange_weights)
    np.savez_compressed(output / "raw_exchange_outcomes.npz", training_seeds=seeds, methods=methods,
                        own=own_raw, swapped=swapped_raw, left_rows=left, right_rows=right, metric_names=METRICS)
    diagnostic_rows, diagnostic_table = [], []
    for mi, method in enumerate(methods):
        value = diagnostics[:, mi]
        diagnostic_table.append({"method": method, **weighted_estimates(ids, weights, value.mean(0), names, replicates, 2271),
                                "per_seed": [{"seed": seed, **dict(zip(names, (weights @ run).tolist()))}
                                             for seed, run in zip(seeds, value)]})
        diagnostic_rows.extend({"image_id": image_id, "task_id": task, "method": method,
                                 **dict(zip(names, row.tolist()))}
                                for image_id, task, row in zip(ids, tasks, value.mean(0)))
    write_rows(output / "correction_diagnostics.csv", diagnostic_rows)
    np.savez_compressed(output / "correction_diagnostics.npz", training_seeds=seeds, methods=methods,
                        image_ids=ids, task_ids=tasks, values=diagnostics, metric_names=names)
    training_records = json.loads((source / "training_records.json").read_text())
    evaluation_records = json.loads((source / "evaluation_records.json").read_text())
    result = {"setting": protocol["setting"], "source": str(source.resolve()), "git_commit": protocol["git_commit"],
              "training_seeds": seeds, "methods": methods, "classification_image_count": len(set(ids)),
              "classification_task_count": len(set(tasks)), "classification_image_task_count": len(base),
              "primary_family_size": 3, "bootstrap_replicates": replicates,
              "primary": "Task-residual minus image-residual classification at three fixed dataset/budget settings; margin primary, accuracy secondary. Both use the same frozen wide base and nearly equal total/learned parameter budgets.",
              "uncertainty": "Paired image bootstrap after averaging fixed training seeds. Three-comparison adjusted intervals separately for each primary metric. Endpoint and task-exchange analyses have ordinary intervals.",
              "classification": classification, "exchange": exchange,
              "correction_diagnostics": diagnostic_table, "training_records": training_records,
              "evaluation_records": evaluation_records,
              "new_training_seconds": sum(r["training_seconds"] for r in training_records)}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()
    result = analyze(args.source, args.output, args.replicates)
    print(json.dumps({"setting": result["setting"], "contrasts": result["classification"]["contrasts"]}, indent=2))
