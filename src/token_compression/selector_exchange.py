"""Exchange candidate-task descriptors using saved learned-selector predictions."""

from collections import defaultdict
from itertools import combinations

import numpy as np

from .selection_analysis import cluster_summary


def _outcomes(scores, labels, competitors, text):
    rows = np.arange(len(labels))
    norms = np.linalg.norm(text[labels] - text[competitors], axis=1)
    margins = (scores[rows, labels] - scores[rows, competitors]) / norms
    a, b = np.minimum(labels, competitors), np.maximum(labels, competitors)
    predicted = np.where(scores[rows, a] >= scores[rows, b], a, b)
    return margins, (predicted == labels).astype(np.int8)


def build_exchange_cases(predictions, training_seeds, task_splits):
    """Enumerate same-image task pairs and re-score each saved token selection.

    In outcome names, the first letter identifies the descriptor used to select
    tokens and the second identifies the task used to evaluate classification.
    Example: ``margin_ba`` uses B's selection but classifies task A.
    """
    a = predictions
    assert np.array_equal(a["row_id"], np.arange(len(a["row_id"])))
    groups = defaultdict(list)
    image_only, full = {}, {}
    for i in range(len(a["row_id"])):
        split, image_id, task = a["task_split"][i], a["image_ids"][i], a["task_ids"][i]
        seed, method = int(a["seed"][i]), a["method"][i]
        if split not in task_splits:
            continue
        if method == "full_model":
            full[split, image_id, task] = i
        elif seed in training_seeds:
            key = seed, split, image_id
            if method == "conditioned":
                groups[key].append(i)
            elif method == "image_only":
                image_only[seed, split, image_id, task] = i
    rows_a, rows_b, image_rows_a, image_rows_b, full_a, full_b = [], [], [], [], [], []
    for (seed, split, image_id), indices in sorted(groups.items()):
        indices.sort(key=lambda i: a["task_ids"][i])
        assert len({a["task_ids"][i] for i in indices}) == len(indices)
        for ra, rb in combinations(indices, 2):
            task_a, task_b = a["task_ids"][[ra, rb]]
            assert a["labels"][ra] == a["labels"][rb]
            rows_a.append(ra)
            rows_b.append(rb)
            image_rows_a.append(image_only[seed, split, image_id, task_a])
            image_rows_b.append(image_only[seed, split, image_id, task_b])
            full_a.append(full[split, image_id, task_a])
            full_b.append(full[split, image_id, task_b])
    ra, rb, ia, ib, fa, fb = [np.asarray(x, dtype=np.int64) for x in
                              (rows_a, rows_b, image_rows_a, image_rows_b, full_a, full_b)]
    labels = a["labels"][ra]
    pa, pb = a["class_pairs"][ra], a["class_pairs"][rb]
    ca = np.where(pa[:, 0] == labels, pa[:, 1], pa[:, 0])
    cb = np.where(pb[:, 0] == labels, pb[:, 1], pb[:, 0])
    assert np.all(ca != cb)
    budgets = a["keep_patch_tokens"][ra]
    assert np.array_equal(budgets, a["keep_patch_tokens"][rb])
    positions = a["patch_indices"]
    masks = np.zeros((len(positions), positions.shape[1]), dtype=bool)
    row_indices, slots = np.nonzero(positions >= 0)
    masks[row_indices, positions[row_indices, slots]] = True
    overlaps = np.sum(masks[ra] & masks[rb], axis=1) / budgets
    triples, inverse = np.unique(np.column_stack([labels, ca, cb]), axis=0, return_inverse=True)
    text = a["text_features"]
    da = text[triples[:, 0]].astype(float) - text[triples[:, 1]]
    db = text[triples[:, 0]].astype(float) - text[triples[:, 2]]
    cosines = np.sum(da * db, axis=1) / (np.linalg.norm(da, axis=1) * np.linalg.norm(db, axis=1))
    angles = np.degrees(np.arccos(np.clip(cosines, -1, 1)))[inverse]
    cases = {"source_row_a": a["row_id"][ra], "source_row_b": a["row_id"][rb],
             "image_only_row_a": a["row_id"][ia], "image_only_row_b": a["row_id"][ib],
             "image_ids": a["image_ids"][ra], "task_a": a["task_ids"][ra],
             "task_b": a["task_ids"][rb], "seed": a["seed"][ra],
             "task_split": a["task_split"][ra], "labels": labels,
             "competitor_a": ca, "competitor_b": cb,
             "keep_patch_tokens": budgets, "angle_degrees": angles,
             "overlap_fraction": overlaps}
    for prefix, row_a, row_b in (("", ra, rb), ("image_only_", ia, ib)):
        for name, source, competitor in (("aa", row_a, ca), ("ba", row_b, ca),
                                          ("bb", row_b, cb), ("ab", row_a, cb)):
            margin, correct = _outcomes(a["cosine_scores"][source], labels, competitor, text)
            cases[f"{prefix}margin_{name}"] = margin
            cases[f"{prefix}correct_{name}"] = correct
    for name, source, competitor in (("a", fa, ca), ("b", fb, cb)):
        margin, correct = _outcomes(a["cosine_scores"][source], labels, competitor, text)
        cases[f"base_margin_{name}"] = margin
        cases[f"base_correct_{name}"] = correct
    return cases


def _paired_metrics(cases, directed=False):
    def values(prefix, metric, own):
        endings = ("aa", "bb") if own else ("ba", "ab")
        parts = [cases[f"{prefix}{metric}_{ending}"].astype(float) for ending in endings]
        return np.concatenate(parts) if directed else (parts[0] + parts[1]) / 2

    result = {}
    for source, name in (("margin", "margin"), ("correct", "accuracy")):
        own, swapped = values("", source, True), values("", source, False)
        shared, shared_swapped = values("image_only_", source, True), values("image_only_", source, False)
        result[f"own_{name}"] = own
        result[f"swapped_{name}"] = swapped
        result[f"image_only_{name}"] = shared
        result[f"own_minus_swapped_{name}"] = own - swapped
        result[f"own_minus_image_only_{name}"] = own - shared
        result[f"image_only_exchange_{name}_loss"] = shared - shared_swapped
    result["overlap_fraction"] = np.tile(cases["overlap_fraction"], 2) if directed else cases["overlap_fraction"]
    def pair_values(a, b):
        return np.concatenate([a, b]) if directed else (a.astype(float) + b.astype(float)) / 2

    for name, own_value, swapped_value in (("correct_descriptor_repairs", 1, 0),
                                           ("correct_descriptor_new_errors", 0, 1)):
        a = (cases["correct_aa"] == own_value) & (cases["correct_ba"] == swapped_value)
        b = (cases["correct_bb"] == own_value) & (cases["correct_ab"] == swapped_value)
        result[name] = pair_values(a, b)
    result["prediction_change_fraction"] = pair_values(
        cases["correct_aa"] != cases["correct_ba"], cases["correct_bb"] != cases["correct_ab"])
    result["full_model_margin"] = pair_values(cases["base_margin_a"], cases["base_margin_b"])
    result["full_model_accuracy"] = pair_values(cases["base_correct_a"], cases["base_correct_b"])
    return result


def _summarize(ids, seeds, values, bootstrap_replicates, seed):
    names = list(values)
    if not len(ids):
        return {"observation_count": 0, "image_count": 0, "metrics": {}, "per_seed": []}
    matrix = np.column_stack([values[name] for name in names])

    def estimate(selected):
        stats = cluster_summary(ids[selected], matrix[selected], bootstrap_replicates, seed)
        return {"observation_count": int(stats["row_count"]), "image_count": int(stats["image_count"]),
                "metrics": {name: {"mean": float(stats["mean"][j]),
                                    "ci95": [float(stats["ci95_low"][j]), float(stats["ci95_high"][j])]}
                            for j, name in enumerate(names)}}

    summary = estimate(np.ones(len(ids), dtype=bool))
    summary["per_seed"] = [{"seed": int(s), **estimate(seeds == s)} for s in np.unique(seeds)]
    return summary


def summarize_exchange(cases, margin_edges, bootstrap_replicates=2000, seed=2051):
    """Summarize all pairs, fixed difficulty strata, and same-stratum sensitivity."""
    edges = np.asarray(margin_edges, dtype=float)
    qa = np.searchsorted(edges, cases["base_margin_a"], side="right") + 1
    qb = np.searchsorted(edges, cases["base_margin_b"], side="right") + 1
    output = {"scope": "Descriptor exchange on the same image using fixed trained selectors and cached score vectors.",
              "aggregation": "Average repeated seeds and task-pair rows within image, then weight images equally. Bootstrap resamples images.",
              "panel_scope": "Exchange-eligible anchor-image panel; these accuracies are not balanced binary-task benchmark accuracies.",
              "task_split_scope": "Unseen combinations are primary and seen combinations secondary; their task collections differ, so their contrast alone is not a pure transfer effect.",
              "margin_quartile_edges": edges.tolist(), "quartile_boundary_rule": "An exact boundary belongs to the higher-numbered stratum.",
              "bootstrap_replicates": bootstrap_replicates, "bootstrap_seed": seed, "task_splits": {}}
    for split in sorted(np.unique(cases["task_split"])):
        selected = cases["task_split"] == split
        part = {key: values[selected] for key, values in cases.items()}
        metrics = _paired_metrics(part)
        summary = _summarize(part["image_ids"], part["seed"], metrics, bootstrap_replicates, seed)
        summary["unique_image_task_pair_count"] = len(set(zip(part["image_ids"], part["task_a"], part["task_b"])))
        same = qa[selected] == qb[selected]
        summary["same_quartile_sensitivity"] = _summarize(
            part["image_ids"][same], part["seed"][same],
            {key: value[same] for key, value in metrics.items()}, bootstrap_replicates, seed)
        summary["same_quartile_sensitivity"]["unique_image_task_pair_count"] = len(set(zip(
            part["image_ids"][same], part["task_a"][same], part["task_b"][same])))
        directed = _paired_metrics(part, directed=True)
        ids, seeds = np.tile(part["image_ids"], 2), np.tile(part["seed"], 2)
        strata = np.concatenate([qa[selected], qb[selected]])
        summary["directed_difficulty_strata"] = []
        for quartile in range(1, len(edges) + 2):
            chosen = strata == quartile
            summary["directed_difficulty_strata"].append({"quartile": quartile, **_summarize(
                ids[chosen], seeds[chosen], {key: value[chosen] for key, value in directed.items()},
                bootstrap_replicates, seed)})
        summary["image_only_null_max_abs_margin_difference"] = float(max(
            np.max(np.abs(part["image_only_margin_aa"] - part["image_only_margin_ba"])),
            np.max(np.abs(part["image_only_margin_bb"] - part["image_only_margin_ab"]))))
        summary["image_only_null_prediction_disagreements"] = int(np.count_nonzero(
            part["image_only_correct_aa"] != part["image_only_correct_ba"]) + np.count_nonzero(
            part["image_only_correct_bb"] != part["image_only_correct_ab"]))
        output["task_splits"][str(split)] = summary
    return output
