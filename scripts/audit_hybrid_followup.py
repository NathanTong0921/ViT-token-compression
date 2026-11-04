"""Independently check frozen-mixture and hybrid-supervision raw artifacts."""

import argparse
from collections import Counter
import csv
from itertools import permutations
import json
from pathlib import Path

import numpy as np


METHODS = ("conditioned", "shared_recipient", "shared_donor", "cu_half", "uu_half",
           "cu_d2", "uu_d2", "cu_d4", "uu_d4")
METRICS = ("accuracy", "margin")


def read_json(path):
    return json.loads(Path(path).read_text())


def panel(folder):
    with (folder / "examples.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    ids = np.asarray([r["image_id"] for r in rows])
    tasks = np.asarray([r["task_id"] for r in rows])
    labels = np.asarray([int(r["class_id"]) for r in rows])
    ix = np.asarray([int(r["image_index"]) for r in rows])
    pairs = np.asarray([[int(r["class_a"]), int(r["class_b"])] for r in rows])
    assert np.all((labels[:, None] == pairs).any(1))
    counts = Counter(zip(tasks, labels))
    assert all(len({label for task, label in counts if task == current}) == 2 for current in set(tasks))
    weights = np.asarray([1 / (2 * len(set(tasks)) * counts[(task, label)])
                          for task, label in zip(tasks, labels)])
    np.testing.assert_allclose(weights, [float(r["task_class_weight"]) for r in rows], rtol=0, atol=1e-12)
    with np.load(folder / "evidence.npz") as saved:
        text = saved["text_features"].astype(float)
        np.testing.assert_array_equal(saved["image_ids"][ix], ids)
        np.testing.assert_array_equal(saved["labels"][ix], labels)
    return dict(ids=ids, tasks=tasks, labels=labels, image_indices=ix,
                pairs=pairs, text=text, weights=weights)


def outcomes(scores, info):
    """Class decisions and margins directly from the two raw cosine scores."""
    labels, pairs = info["labels"], info["pairs"]
    row = np.arange(len(labels))
    alternative = np.where(pairs[:, 0] == labels, pairs[:, 1], pairs[:, 0])
    own, other = scores[..., row, labels], scores[..., row, alternative]
    correct = (own > other) | ((own == other) & (labels < alternative))
    length = np.sqrt(np.sum((info["text"][labels] - info["text"][alternative]) ** 2, axis=1))
    return np.stack((correct, (own - other) / length), axis=-1).astype(float)


def ranked_positions(scores, k):
    # Explicit index tie-breaking is independent of the producer's stable argsort.
    positions = np.arange(len(scores))
    return np.lexsort((positions, -scores))[:k]


def independent_mixture(a, b, d):
    """Set-based reconstruction from already ordered top-k positions."""
    shared = set(a) & set(b)
    remaining_a = [int(i) for i in a if i not in shared]
    remaining_b = [int(i) for i in b if i not in shared]
    assert len(remaining_a) == len(remaining_b)
    count = len(remaining_a)
    feasible = d is None or count >= d
    actual = count // 2 if d is None else d if feasible else 0
    retained = set(a) - set(remaining_a[len(remaining_a) - actual:] if actual else [])
    retained.update(remaining_b[:actual])
    return np.asarray(sorted(retained)), count, actual, feasible


def check_pair(saved, c_logits, u_logits, v_logits, info, k):
    np.testing.assert_array_equal(saved["methods"], METHODS)
    masks = saved["patch_indices"]
    assert masks.shape == (len(METHODS), len(info["ids"]), k)
    assert np.all(np.diff(masks, axis=-1) > 0)
    assert np.all((masks >= 0) & (masks < c_logits.shape[-1]))
    expected = np.empty_like(masks)
    disagreement = np.zeros((2, len(info["ids"])), dtype=int)
    replacements = np.zeros((len(METHODS), len(info["ids"])), dtype=int)
    eligible = np.empty((2, len(info["ids"])), dtype=bool)
    u_ranks = [ranked_positions(row, k) for row in u_logits]
    v_ranks = [ranked_positions(row, k) for row in v_logits]
    for row, image in enumerate(info["image_indices"]):
        c, u, v = ranked_positions(c_logits[row], k), u_ranks[image], v_ranks[image]
        expected[:3, row] = np.stack([np.sort(x) for x in (c, u, v)])
        for offset, recipient in enumerate((c, u)):
            m, count, d, _ = independent_mixture(recipient, v, None)
            expected[3 + offset, row] = m
            disagreement[offset, row] = count
            replacements[3 + offset, row] = d
        for di, d in enumerate((2, 4)):
            valid = []
            for offset, recipient in enumerate((c, u)):
                m, _, count, ok = independent_mixture(recipient, v, d)
                expected[5 + 2 * di + offset, row] = m
                replacements[5 + 2 * di + offset, row] = count
                valid.append(ok)
            eligible[di, row] = all(valid)
    np.testing.assert_array_equal(masks, expected)
    np.testing.assert_array_equal(saved["disagreement_sizes"], disagreement)
    np.testing.assert_array_equal(saved["changed_tokens_vs_recipient"], replacements)
    np.testing.assert_array_equal(saved["fixed_d_eligible"], eligible)
    np.testing.assert_array_equal(saved["fixed_d"], [2, 4])
    _, first, inverse = np.unique(info["image_indices"], return_index=True, return_inverse=True)
    for index in (1, 2, 4, 6, 8):
        np.testing.assert_array_equal(masks[index], masks[index, first[inverse]])
        np.testing.assert_array_equal(saved["cosine_scores"][index], saved["cosine_scores"][index, first[inverse]])
    return eligible


def compare_metrics(actual, expected):
    error = 0.
    for metric, value in zip(METRICS, actual):
        difference = abs(float(value) - expected[metric]["mean"])
        error = max(error, difference)
        np.testing.assert_allclose(value, expected[metric]["mean"], rtol=0, atol=1e-12)
    return error


def audit_specificity(folder):
    from analyze_selector_budgets import weighted_estimates

    folder = Path(folder)
    protocol = read_json(folder / "protocol.json")
    summary = read_json(folder / "analysis/summary.json")
    seeds = protocol["training_seeds"]
    primary = list(permutations(seeds, 2))
    assert [tuple(x) for x in protocol["primary_seed_pairs"]] == primary
    assert [tuple(x) for x in protocol["reference_seed_pairs"]] == [(s, s) for s in seeds]
    assert not protocol["final_test_used"]
    info = panel(folder)
    source = Path(protocol["source"])
    originals = {}
    for seed in seeds:
        with np.load(source / f"seed_{seed}.npz") as saved:
            originals[seed] = {key: saved[key].copy() for key in
                ("conditioned_logits", "wide_logits", "variants", "patch_indices", "cosine_scores")}
    k = protocol["config"]["keep_patch_tokens"]
    all_values, all_eligible, references, maximum_endpoint_error = [], [], {}, 0.
    for s, t in primary + [(s, s) for s in seeds]:
        with np.load(folder / f"pair_{s}_{t}.npz") as saved:
            assert int(saved["recipient_seed"]) == s and int(saved["donor_seed"]) == t
            eligible = check_pair(saved, originals[s]["conditioned_logits"], originals[s]["wide_logits"],
                                  originals[t]["wide_logits"], info, k)
            for index, (seed, method) in enumerate(((s, "conditioned"), (s, "wide"), (t, "wide"))):
                old = originals[seed]
                position = list(old["variants"]).index(method)
                np.testing.assert_array_equal(saved["patch_indices"][index], old["patch_indices"][position])
                delta = abs(saved["cosine_scores"][index] - old["cosine_scores"][position])
                maximum_endpoint_error = max(maximum_endpoint_error, float(delta.max()))
                assert delta.max() <= 1e-6
            values = outcomes(saved["cosine_scores"].astype(float), info)
            if s != t:
                all_values.append(values)
                all_eligible.append(eligible)
            else:
                references[s] = values
    raw, eligible = np.stack(all_values), np.stack(all_eligible)
    weights = info["weights"]
    contrasts = {"cu_minus_uu": raw[:, 3] - raw[:, 4],
        "cu_minus_conditioned": raw[:, 3] - raw[:, 0],
        "cu_minus_shared_recipient": raw[:, 3] - raw[:, 1],
        "cu_minus_shared_donor": raw[:, 3] - raw[:, 2],
        "uu_minus_shared_recipient": raw[:, 4] - raw[:, 1],
        "uu_minus_shared_donor": raw[:, 4] - raw[:, 2],
        "hybrid_increment_interaction": raw[:, 3] - raw[:, 0] - raw[:, 4] + raw[:, 1]}
    maximum_point_error = 0.
    reconstructed = {}
    for name, values in contrasts.items():
        point = np.sum(values * weights[None, :, None], axis=(0, 1)) / (len(primary) * weights.sum())
        target = summary["classification"]["contrasts"][name]
        maximum_point_error = max(maximum_point_error, compare_metrics(point, target))
        reconstructed[name] = dict(zip(METRICS, point.tolist()))
        for (s, t), row_values, expected in zip(primary, values, target["per_seed_pair"]):
            assert (expected["recipient_seed"], expected["donor_seed"]) == (s, t)
            np.testing.assert_allclose(weights @ row_values / weights.sum(),
                                       [expected[m] for m in METRICS], rtol=0, atol=1e-12)
    ci = weighted_estimates(info["ids"], weights, contrasts["cu_minus_uu"].mean(0), METRICS,
                            summary["bootstrap_replicates"], summary["bootstrap_seed"], 4)
    for metric in METRICS:
        for key in ("ci95", "bonferroni_family_ci95"):
            np.testing.assert_allclose(ci[metric][key], summary["classification"]["contrasts"]["cu_minus_uu"][metric][key],
                                       rtol=0, atol=1e-12)
    for i in range(5):
        expected = next(x for x in summary["classification"]["table"] if x["method"] == METHODS[i])
        point = np.sum(raw[:, i] * weights[None, :, None], axis=(0, 1)) / (len(primary) * weights.sum())
        maximum_point_error = max(maximum_point_error, compare_metrics(point, expected))
    for di, d in enumerate((2, 4)):
        keep = eligible[:, di]
        w = keep * weights[None, :] / len(primary)
        target = summary["fixed_displacement"][str(d)]
        np.testing.assert_allclose(w.sum() / weights.sum(), target["weighted_pair_row_coverage"], atol=1e-12, rtol=0)
        if w.sum():
            delta = raw[:, 5 + 2 * di] - raw[:, 6 + 2 * di]
            point = np.sum(w[..., None] * delta, axis=(0, 1)) / w.sum()
            maximum_point_error = max(maximum_point_error, compare_metrics(point, target))
        else:
            assert all(target[m] is None for m in METRICS)
    with np.load(folder / "analysis/classification_cases.npz") as stored:
        np.testing.assert_allclose(stored["outcomes"], raw, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(stored["eligible"], eligible)
    return dict(setting=protocol["setting"], status="passed", seed_pairs_checked=len(primary) + len(seeds),
        rows=len(weights), independent_set_based_masks=True, exact_fixed_d_feasibility=True,
        shared_mixtures_task_independent=True, original_endpoint_max_error=maximum_endpoint_error,
        independent_mean_max_error=maximum_point_error, primary_image_bootstrap_reproduced=True,
        primary_bootstrap_scope="Independent raw outcomes and weights; existing generic image-bootstrap helper reused.",
        reconstructed_contrasts=reconstructed, final_test_used=False)


def row_alignment(first, second):
    """Align raw rows by their actual image and task identities."""
    with (first / "examples.csv").open() as handle:
        a = list(csv.DictReader(handle))
    with (second / "examples.csv").open() as handle:
        b = list(csv.DictReader(handle))
    lookup = {(r["image_id"], r["task_id"]): i for i, r in enumerate(b)}
    keys = [(r["image_id"], r["task_id"]) for r in a]
    assert len(keys) == len(set(keys)) == len(lookup) == len(b)
    assert set(keys) == set(lookup)
    order = np.asarray([lookup[key] for key in keys])
    for r, j in zip(a, order):
        assert all(r[field] == b[j][field] for field in ("class_id", "class_a", "class_b"))
    return order


def mask_overlap(mask, teacher):
    return np.asarray([len(set(a) & set(b)) / len(b) for a, b in zip(mask, teacher)])


def uniform_mask_kl(logits, target, temperature):
    """Float64 numpy evaluation, independent of Torch's KL implementation."""
    values = logits.astype(float) / temperature
    maximum = values.max(axis=1)
    log_z = maximum + np.log(np.exp(values - maximum[:, None]).sum(axis=1))
    selected_logits = np.take_along_axis(values, target, axis=1).mean(axis=1)
    return log_z - selected_logits - np.log(target.shape[1])


def audit_distillation(folder):
    from analyze_selector_budgets import weighted_estimates

    folder = Path(folder)
    protocol, summary = read_json(folder / "protocol.json"), read_json(folder / "analysis/summary.json")
    original = Path(protocol["original_source"])
    old_protocol = read_json(original / "protocol.json")
    fields = ("training_seeds", "training_steps", "batch_size", "temperature", "learning_rate",
              "weight_decay", "keep_patch_tokens", "prune_after", "model", "cache_dir", "image_batch_size")
    assert all(protocol["config"][key] == old_protocol["config"][key] for key in fields)
    assert not protocol["final_test_used"]
    info = panel(folder)
    weights, config, seeds = info["weights"], protocol["config"], protocol["training_seeds"]
    order = row_alignment(folder, original)
    old_info = panel(original)
    np.testing.assert_array_equal(info["text"], old_info["text"])
    np.testing.assert_allclose(weights, old_info["weights"][order], atol=1e-12, rtol=0)
    fits = {(row["seed"], row["arm"]): row for row in read_json(folder / "target_fit.json")}
    assert len(fits) == 2 * len(seeds)
    np.testing.assert_equal(read_json(folder / "sampling_audit.json"), read_json(original / "sampling_audit.json"))
    evaluated = {row["seed"]: row for row in read_json(folder / "evaluation_records.json")}
    records = read_json(folder / "training_records.json")
    assert len(records) == 2 * len(seeds)
    with (folder / "training_curve.csv").open() as handle:
        curves = list(csv.DictReader(handle))
    for seed in seeds:
        for arm in ("task_residual", "image_residual"):
            assert max(int(row["step"]) for row in curves if int(row["seed"]) == seed and row["arm"] == arm) == config["training_steps"]
    new_values, old_values, teacher_values = [], [], []
    target_rows, maximum_fit_error = 0, 0.
    for seed in seeds:
        split_sets = {}
        for split in ("train", "val"):
            with np.load(folder / f"seed_{seed}/{split}_targets.npz") as target:
                h, c, u = [target[key] for key in ("patch_indices", "conditioned_patch_indices", "shared_patch_indices")]
                k = config["keep_patch_tokens"]
                assert h.shape == c.shape == u.shape and h.shape[1] == k
                assert all(np.all(np.diff(x, axis=1) > 0) for x in (h, c, u))
                assert all(np.all((x >= 0) & (x < 196)) for x in (h, c, u))
                np.testing.assert_allclose(float(target["target_probability"]), 1 / k, atol=1e-12, rtol=0)
                for hh, cc, uu in zip(h, c, u):
                    hs, cs, us = set(hh), set(cc), set(uu)
                    assert cs & us <= hs and hs <= cs | us
                    assert len(hs - cs) == len(cs - hs) == len(us - cs) // 2
                images, pairs = target["image_ids"], target["class_pairs"]
                split_sets[split] = (set(images), set(map(tuple, np.sort(pairs, axis=1))))
                target_rows += len(h)
                if split == "val":
                    np.testing.assert_array_equal(images[target["image_indices"]], info["ids"])
                    np.testing.assert_array_equal(pairs[target["pair_indices"]], info["pairs"])
                    target_mask = h.copy()
                    teacher_masks = np.stack((c, u, h))
        assert not split_sets["train"][0] & split_sets["val"][0]
        assert not split_sets["train"][1] & split_sets["val"][1]
        matched = Path(evaluated[seed]["teacher_reference"]["source"])
        reference_order = row_alignment(folder, matched)
        with np.load(folder / f"seed_{seed}/teacher_reference.npz") as teacher, np.load(matched / f"seed_{seed}.npz") as historical:
            np.testing.assert_array_equal(teacher["methods"], ["conditioned", "wide", "top_50"])
            for i, name in enumerate(teacher["methods"]):
                mi = list(historical["variants"]).index(name)
                np.testing.assert_array_equal(teacher_masks[i], historical["patch_indices"][mi, reference_order])
                np.testing.assert_array_equal(teacher["cosine_scores"][i], historical["cosine_scores"][mi, reference_order])
            teacher_values.append(outcomes(teacher["cosine_scores"].astype(float), info))
        with np.load(folder / f"seed_{seed}.npz") as new, np.load(original / f"seed_{seed}.npz") as old:
            methods = list(new["methods"])
            assert methods == list(old["methods"]) == ["shared", "task_residual", "image_residual"]
            for method in range(len(methods)):
                np.testing.assert_array_equal(np.sort(np.argsort(-new["selector_logits"][method], axis=1, kind="stable")[:, :k], axis=1),
                                              new["patch_indices"][method])
            np.testing.assert_array_equal(new["patch_indices"][0], old["patch_indices"][0, order])
            np.testing.assert_array_equal(new["cosine_scores"][0], old["cosine_scores"][0, order])
            _, first, inverse = np.unique(info["image_indices"], return_index=True, return_inverse=True)
            for method in (0, 2):
                np.testing.assert_array_equal(new["patch_indices"][method], new["patch_indices"][method, first[inverse]])
                np.testing.assert_array_equal(new["cosine_scores"][method], new["cosine_scores"][method, first[inverse]])
            for mi, arm in ((1, "task_residual"), (2, "image_residual")):
                measurements = (("hybrid_trained", new["patch_indices"][mi], new["selector_logits"][mi]),
                                ("original_trained", old["patch_indices"][mi, order], old["selector_logits"][mi, order]),
                                ("initial_val", new["patch_indices"][0], new["selector_logits"][0]))
                for key, mask, logits in measurements:
                    expected = fits[(seed, arm)][key] if key == "initial_val" else fits[(seed, arm)][key]["val"]
                    overlap = float(weights @ mask_overlap(mask, target_mask))
                    kl = float(weights @ uniform_mask_kl(logits, target_mask, config["temperature"]))
                    np.testing.assert_allclose(overlap, expected["hybrid_mask_overlap"], rtol=0, atol=1e-12)
                    np.testing.assert_allclose(kl, expected["hybrid_target_kl"], rtol=0, atol=2e-6)
                    maximum_fit_error = max(maximum_fit_error, abs(kl - expected["hybrid_target_kl"]))
            new_values.append(outcomes(new["cosine_scores"].astype(float), info))
            old_values.append(outcomes(old["cosine_scores"][:, order].astype(float), info))
    new, old, teacher = np.stack(new_values), np.stack(old_values), np.stack(teacher_values)
    contrasts = {"task_residual_minus_image_residual": new[:, 1] - new[:, 2],
                 "task_residual_minus_shared": new[:, 1] - new[:, 0],
                 "image_residual_minus_shared": new[:, 2] - new[:, 0]}
    target_contrasts = {
        "target_by_conditioning_interaction": new[:, 1] - new[:, 2] - old[:, 1] + old[:, 2],
        "task_target_change": new[:, 1] - old[:, 1],
        "image_target_change": new[:, 2] - old[:, 2],
        "hybrid_task_minus_hybrid_teacher": new[:, 1] - teacher[:, 2],
        "hybrid_image_minus_hybrid_teacher": new[:, 2] - teacher[:, 2],
        "hybrid_teacher_minus_shared": teacher[:, 2] - new[:, 0],
        "hybrid_teacher_minus_conditioned": teacher[:, 2] - teacher[:, 0]}
    maximum_point_error, reconstructed = 0., {}
    for group, values in (("classification", contrasts), ("target", target_contrasts)):
        for name, delta in values.items():
            expected = summary["classification"]["contrasts"][name] if group == "classification" else summary["target_comparisons"][name]
            point = np.sum(delta * weights[None, :, None], axis=(0, 1)) / (len(seeds) * weights.sum())
            maximum_point_error = max(maximum_point_error, compare_metrics(point, expected))
            reconstructed[name] = dict(zip(METRICS, point.tolist()))
            for seed, rows, per_seed in zip(seeds, delta, expected["per_seed"]):
                assert per_seed["seed"] == seed
                np.testing.assert_allclose(weights @ rows / weights.sum(), [per_seed[m] for m in METRICS], rtol=0, atol=1e-12)
    for values, expected, boot_seed, family in (
        (contrasts["task_residual_minus_image_residual"], summary["classification"]["contrasts"]["task_residual_minus_image_residual"], 2271, 3),
        (target_contrasts["target_by_conditioning_interaction"], summary["target_comparisons"]["target_by_conditioning_interaction"], 2321, 1)):
        bootstrap = weighted_estimates(info["ids"], weights, values.mean(0), METRICS,
                                        summary["bootstrap_replicates"], boot_seed, family)
        for metric in METRICS:
            for key in (("ci95", "bonferroni_family_ci95") if family > 1 else ("ci95",)):
                np.testing.assert_allclose(bootstrap[metric][key], expected[metric][key], rtol=0, atol=1e-12)
    with np.load(folder / "analysis/target_classification_outcomes.npz") as saved:
        for key, actual in (("hybrid_target", new), ("original_target", old), ("teacher", teacher)):
            np.testing.assert_allclose(saved[key], actual, rtol=0, atol=1e-12)
    return dict(setting=protocol["setting"], status="passed", new_branches=len(records), target_rows_checked=target_rows,
        train_val_images_and_class_pairs_disjoint=True, all_target_masks_fixed_k_and_half_exclusive_replacement=True,
        validation_hybrid_masks_match_previous_experiment=True, historical_shared_endpoint_exact=True,
        shared_and_image_correction_task_independent=True, same_training_recipe_and_sampling_audit=True,
        independent_mean_max_error=maximum_point_error, validation_hybrid_kl_max_error=maximum_fit_error,
        validation_mask_overlap_independently_recomputed=True, primary_and_interaction_bootstrap_reproduced=True,
        scope="Training target set structure checked; exact rank-priority matches checked on validation via historical masks. Raw validation logits independently verify hybrid-target KL and mask fit. Training target-fit losses are not independently rerun.",
        reconstructed_contrasts=reconstructed, final_test_used=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specificity", type=Path, help="One completed setting folder containing protocol.json.")
    parser.add_argument("--distillation", type=Path, help="One completed hybrid-supervision setting folder.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.specificity is None) == (args.distillation is None):
        parser.error("Provide exactly one of --specificity or --distillation.")
    result = audit_specificity(args.specificity) if args.specificity else audit_distillation(args.distillation)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
