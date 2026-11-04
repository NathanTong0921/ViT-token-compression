"""Compare direct-set and KL training against the same fixed hybrid masks."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_selector_budgets import weighted_estimates
from analyze_selector_scale import panel_weights, read_rows


METRICS = ("accuracy", "margin", "raw_gap")
METHODS = ("shared", "task_residual", "image_residual")
BOOTSTRAP_SEED = 2372


def align_rows(reference, rows):
    """Match image/task identity, then check labels and both candidate classes."""
    keys = [(r["image_id"], r["task_id"]) for r in reference]
    lookup = {(r["image_id"], r["task_id"]): i for i, r in enumerate(rows)}
    assert len(keys) == len(set(keys)) == len(lookup) == len(rows)
    assert set(keys) == set(lookup)
    order = np.asarray([lookup[key] for key in keys])
    for r, i in zip(reference, order):
        assert tuple(int(r[k]) for k in ("class_id", "class_a", "class_b")) == tuple(
            int(rows[i][k]) for k in ("class_id", "class_a", "class_b"))
    return order


def binary_outcomes(scores, labels, competitors, text):
    rows = np.arange(len(labels))
    own, other = scores[..., rows, labels], scores[..., rows, competitors]
    raw = own - other
    correct = (own > other) | ((own == other) & (labels < competitors))
    return np.stack((correct.astype(float), raw / np.linalg.norm(
        text[labels] - text[competitors], axis=1), raw), axis=-1)


def membership_overlap(chosen, target, patches=196):
    """Intersection/k with broadcastable leading dimensions, without k-by-k pairs."""
    assert chosen.shape[-1] == target.shape[-1]
    c = np.zeros((*chosen.shape[:-1], patches), bool)
    t = np.zeros((*target.shape[:-1], patches), bool)
    np.put_along_axis(c, chosen.astype(int), True, axis=-1)
    np.put_along_axis(t, target.astype(int), True, axis=-1)
    assert np.all(c.sum(-1) == chosen.shape[-1]) and np.all(t.sum(-1) == target.shape[-1])
    return (c & t).sum(-1) / chosen.shape[-1]


def error_transitions(candidate, reference):
    """Per-row changes preserve the identity net accuracy = corrected - broken."""
    c, r = candidate[..., 0], reference[..., 0]
    fixed = (c == 1) & (r == 0)
    broken = (c == 0) & (r == 1)
    return np.stack((fixed, broken, fixed.astype(float) - broken, fixed | broken), axis=-1)


def summarize(ids, weights, values, seeds, metrics, replicates, family=1):
    result = weighted_estimates(ids, weights, values.mean(0), metrics,
                                replicates, BOOTSTRAP_SEED, family)
    points = np.einsum("n,snm->sm", weights, values) / weights.sum()
    result["per_seed"] = [{"seed": s, **dict(zip(metrics, row.tolist()))}
                          for s, row in zip(seeds, points)]
    return result


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def target_rows(path, base):
    with np.load(path) as saved:
        ids = saved["image_ids"][saved["image_indices"]]
        pairs = saved["class_pairs"][saved["pair_indices"]]
        keys = [(str(i), int(a), int(b)) for i, (a, b) in zip(ids, pairs)]
        lookup = {key: i for i, key in enumerate(keys)}
        wanted = [(r["image_id"], int(r["class_a"]), int(r["class_b"])) for r in base]
        assert len(lookup) == len(keys) == len(wanted) and set(lookup) == set(wanted)
        return saved["patch_indices"][[lookup[key] for key in wanted]].copy()


def analyze(source, output, replicates=2000, kl_source=None):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((source / "protocol.json").read_text())
    old = Path(kl_source or protocol["kl_source"])
    old_protocol = json.loads((old / "protocol.json").read_text())
    seeds = protocol["training_seeds"]
    assert seeds == old_protocol["training_seeds"]
    assert list(protocol["methods"]) == list(old_protocol["methods"]) == list(METHODS)
    base = read_rows(source / "examples.csv")
    old_rows = read_rows(old / "examples.csv")
    order = align_rows(base, old_rows)
    ids = np.asarray([r["image_id"] for r in base])
    tasks = np.asarray([r["task_id"] for r in base])
    labels = np.asarray([int(r["class_id"]) for r in base])
    pairs = np.asarray([[int(r["class_a"]), int(r["class_b"])] for r in base])
    rivals = np.where(labels == pairs[:, 0], pairs[:, 1], pairs[:, 0])
    weights = panel_weights(base, "all")
    np.testing.assert_allclose(weights, panel_weights(old_rows, "all")[order], rtol=0, atol=1e-12)
    np.testing.assert_allclose(weights, [float(r["task_class_weight"]) for r in base], rtol=0, atol=1e-12)
    ix = np.asarray([int(r["image_index"]) for r in base])
    with np.load(source / "evidence.npz") as saved:
        text = saved["text_features"].astype(float)
        full = saved["full_scores"][ix].astype(float)
        np.testing.assert_array_equal(saved["image_ids"][ix], ids)
    with np.load(old / "evidence.npz") as saved:
        np.testing.assert_array_equal(saved["text_features"], text)
        old_ix = np.asarray([int(old_rows[i]["image_index"]) for i in order])
        np.testing.assert_array_equal(saved["full_scores"][old_ix], full)
    new_values, old_values, teachers, new_masks, old_masks, targets = [], [], [], [], [], []
    for seed in seeds:
        with np.load(source / f"seed_{seed}.npz") as new, np.load(old / f"seed_{seed}.npz") as previous:
            for saved in (new, previous):
                np.testing.assert_array_equal(saved["methods"], METHODS)
            np.testing.assert_array_equal(new["patch_indices"][0], previous["patch_indices"][0, order])
            np.testing.assert_array_equal(new["cosine_scores"][0], previous["cosine_scores"][0, order])
            new_values.append(binary_outcomes(new["cosine_scores"].astype(float), labels, rivals, text))
            old_values.append(binary_outcomes(previous["cosine_scores"][:, order].astype(float), labels, rivals, text))
            new_masks.append(new["patch_indices"].copy())
            old_masks.append(previous["patch_indices"][:, order].copy())
            # Shared/image selectors cannot change their mask with candidate classes.
            for mi in (0, 2):
                for image in np.unique(ids):
                    where = np.flatnonzero(ids == image)
                    assert np.all(new["patch_indices"][mi, where] == new["patch_indices"][mi, where[0]])
        with np.load(old / f"seed_{seed}/teacher_reference.npz") as saved:
            np.testing.assert_array_equal(saved["methods"], ["conditioned", "wide", "top_50"])
            teachers.append(binary_outcomes(saved["cosine_scores"][:, order].astype(float), labels, rivals, text))
        targets.append(target_rows(old / f"seed_{seed}/val_targets.npz", base))
    new_values, old_values, teachers, new_masks, old_masks, targets = map(
        np.stack, (new_values, old_values, teachers, new_masks, old_masks, targets))
    np.testing.assert_array_equal(teachers[:, 1], new_values[:, 0])
    values = {"shared": new_values[:, 0], "set_task": new_values[:, 1], "set_image": new_values[:, 2],
              "kl_task": old_values[:, 1], "kl_image": old_values[:, 2],
              "conditioned_teacher": teachers[:, 0], "hybrid_teacher": teachers[:, 2]}
    definitions = {
        "set_task_minus_set_image": ("set_task", "set_image"),
        "kl_task_minus_kl_image": ("kl_task", "kl_image"),
        "set_task_minus_shared": ("set_task", "shared"),
        "set_image_minus_shared": ("set_image", "shared"),
        "set_task_minus_kl_task": ("set_task", "kl_task"),
        "set_image_minus_kl_image": ("set_image", "kl_image"),
        "set_task_minus_hybrid_teacher": ("set_task", "hybrid_teacher"),
        "set_image_minus_hybrid_teacher": ("set_image", "hybrid_teacher"),
        "hybrid_teacher_minus_shared": ("hybrid_teacher", "shared")}
    differences = {name: values[a] - values[b] for name, (a, b) in definitions.items()}
    differences["objective_by_conditioning_interaction"] = (
        differences["set_task_minus_set_image"] - differences["kl_task_minus_kl_image"])
    classification = {
        "absolute": {name: summarize(ids, weights, value, seeds, METRICS, replicates)
                     for name, value in values.items()},
        "contrasts": {name: summarize(ids, weights, value, seeds, METRICS, replicates,
                                      3 if name == "set_task_minus_set_image" else 1)
                      for name, value in differences.items()},
        "full_reference": weighted_estimates(ids, weights, binary_outcomes(full, labels, rivals, text),
                                              METRICS, replicates, BOOTSTRAP_SEED)}
    transitions = {name: error_transitions(values[a], values[b]) for name, (a, b) in definitions.items()}
    error_names = ("corrected", "broken", "net_accuracy", "prediction_disagreement")
    error_summary = {name: summarize(ids, weights, value, seeds, error_names, replicates)
                     for name, value in transitions.items()}
    for name, value in transitions.items():
        np.testing.assert_array_equal(value[..., 2], differences[name][..., 0])
    overlap_new = membership_overlap(new_masks, targets[:, None])
    overlap_old = membership_overlap(old_masks, targets[:, None])
    recovery_values = {"shared": overlap_new[:, 0, :, None],
                       "set_task": overlap_new[:, 1, :, None], "set_image": overlap_new[:, 2, :, None],
                       "kl_task": overlap_old[:, 1, :, None], "kl_image": overlap_old[:, 2, :, None]}
    recovery_diffs = {f"set_{arm}_minus_kl_{arm}": recovery_values[f"set_{arm}"] - recovery_values[f"kl_{arm}"]
                      for arm in ("task", "image")}
    recovery = {"absolute": {name: summarize(ids, weights, value, seeds, ("overlap",), replicates)
                             for name, value in recovery_values.items()},
                "contrasts": {name: summarize(ids, weights, value, seeds, ("overlap",), replicates, 6)
                              for name, value in recovery_diffs.items()},
                "source": "Actual online evaluation masks; same frozen hybrid target for both objectives."}
    # Keep all row-level values so a separate implementation can reconstruct every table.
    arrays = dict(image_id=ids, task_id=tasks, class_id=labels, class_a=pairs[:, 0], class_b=pairs[:, 1],
                  weights=weights, training_seeds=np.asarray(seeds), metric_names=np.asarray(METRICS))
    arrays.update({name: value for name, value in values.items()})
    arrays.update({name: value for name, value in differences.items()})
    arrays.update({f"recovery_{name}": value for name, value in {**recovery_values, **recovery_diffs}.items()})
    arrays.update({f"errors_{name}": value for name, value in transitions.items()})
    arrays["error_metric_names"] = np.asarray(error_names)
    np.savez_compressed(output / "paired_cases.npz", **arrays)
    unique, inverse = np.unique(ids, return_inverse=True)
    image_columns = {"image_id": unique, "weight": np.bincount(inverse, weights=weights)}
    for name, value in {**values, **differences}.items():
        for j, metric in enumerate(METRICS):
            image_columns[f"{name}_{metric}_weighted_sum"] = np.bincount(inverse, weights=weights * value.mean(0)[:, j])
    write_csv(output / "image_contributions.csv", [dict(zip(image_columns, row)) for row in zip(*image_columns.values())])
    fit = summarize_fit_rows(source, base, seeds, weights, recovery_values)
    result = dict(setting=protocol["setting"], source=str(source.resolve()), kl_source=str(old.resolve()),
                  git_commit=protocol["git_commit"], training_seeds=seeds,
                  image_count=len(unique), task_count=len(set(tasks)), image_task_rows=len(base),
                  primary_classification_family=3, primary_recovery_family=6,
                  bootstrap_replicates=replicates, bootstrap_seed=BOOTSTRAP_SEED,
                  aggregation="Average fixed seeds first, then resample images with all their task rows and original equal-task/equal-class weights.",
                  classification=classification, recovery=recovery, error_transitions=error_summary,
                  training_records=json.loads((source / "training_records.json").read_text()),
                  evaluation_records=json.loads((source / "evaluation_records.json").read_text()),
                  fit_diagnostics=fit,
                  interpretation="Same fixed hybrid targets and residual architecture; objective changes from KL to direct set loss. Recovery and classification are separate outcomes. Secondary contrasts use ordinary intervals.")
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def summarize_fit_rows(source, base, seeds, val_weights, recovery_values):
    """Training/cached validation diagnostics, separate from online recovery primary."""
    values, metadata = {}, {}
    for split in ("train", "val"):
        runs = []
        for seed in seeds:
            with np.load(source / f"seed_{seed}/{split}_fit_rows.npz") as saved:
                names = list(saved["model_names"])
                metrics = list(saved["metric_names"])
                current = {key: saved[key].copy() for key in
                           ("image_id", "task_id", "class_id", "class_a", "class_b", "weights")}
                if split not in metadata:
                    metadata[split] = current
                    metadata[split].update(model_names=names, metric_names=metrics)
                else:
                    for key, value in current.items():
                        np.testing.assert_array_equal(metadata[split][key], value)
                    assert names == metadata[split]["model_names"] and metrics == metadata[split]["metric_names"]
                runs.append(saved["values"].copy())
        values[split] = np.stack(runs)
    val_rows = [{k: str(v) for k, v in zip(("image_id", "task_id", "class_id", "class_a", "class_b"), row)}
                for row in zip(*(metadata["val"][k] for k in ("image_id", "task_id", "class_id", "class_a", "class_b")))]
    order = align_rows(base, val_rows)
    np.testing.assert_allclose(metadata["val"]["weights"][order], val_weights, rtol=0, atol=1e-12)
    result = {}
    for split in ("train", "val"):
        m = metadata[split]
        weights = m["weights"] / m["weights"].sum()
        result[split] = {
            name: {metric: {"mean": float(np.mean(values[split][:, mi, :, j] @ weights)),
                            "per_seed": [{"seed": s, "value": float(run @ weights)}
                                         for s, run in zip(seeds, values[split][:, mi, :, j])]}
                   for j, metric in enumerate(m["metric_names"])}
            for mi, name in enumerate(m["model_names"])}
    m = metadata["val"]
    j = m["metric_names"].index("overlap")
    result["online_cached_recovery_check"] = {
        name: {"max_row_absolute_difference": float(abs(values["val"][:, mi, order, j] - recovery_values[name][..., 0]).max()),
               "weighted_mean_difference": float(np.mean((values["val"][:, mi, order, j] - recovery_values[name][..., 0]) @ val_weights))}
        for mi, name in enumerate(m["model_names"])}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--kl-source", type=Path)
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()
    analyze(args.source, args.output or args.source / "analysis", args.replicates, args.kl_source)
