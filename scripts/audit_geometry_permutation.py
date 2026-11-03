"""Independent raw-output and random-reference audit; no new geometry-core imports."""

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np


NAMES = ("difficulty_adjusted_slope", "image_fixed_effects_slope", "mean_alignment")


def close(a, b, atol=2e-10):
    np.testing.assert_allclose(a, b, rtol=0, atol=atol)


def saved_arrays(path):
    with np.load(path) as file:
        return {key: file[key].copy() for key in file.files}


def rows(path):
    with Path(path).open(newline="") as file:
        return list(csv.DictReader(file))


def raw_predictions(path, method, seeds=None):
    with np.load(path) as file:
        select = (file["method"] == method) & (file["task_split"] == "unseen")
        if seeds is not None:
            select &= np.isin(file["seed"], seeds)
        data = {key: file[key][select] for key in ("image_ids", "task_ids", "seed", "row_id",
                "labels", "class_pairs", "image_features", "keep_patch_tokens")}
        data["text_features"] = file["text_features"].copy()
    return data


def check_raw_panel(setting, panel, seeds):
    """Identity-index raw prediction rows independently of the extraction script."""
    source = Path(setting["source"])
    raw = raw_predictions(source / "full/predictions.npz", "conditioned", seeds)
    full = raw_predictions(source / "references/predictions.npz", "full_model")
    images = {r["image_id"]: int(r["class_id"]) for r in rows(Path(setting["manifest_dir"]) / "images.csv")
              if r["split"] == "val"}
    tasks = {r["task_id"]: (int(r["class_a"]), int(r["class_b"]))
             for r in rows(Path(setting["manifest_dir"]) / "tasks.csv") if r["split"] == "val"}
    for values in (raw, full):
        for image, task, label, pair in zip(values["image_ids"], values["task_ids"], values["labels"], values["class_pairs"]):
            assert images[str(image)] == label and tasks[str(task)] == tuple(pair) and label in pair
    keys = [(str(i), str(t), int(s)) for i, t, s in zip(raw["image_ids"], raw["task_ids"], raw["seed"])]
    assert len(set(keys)) == len(keys)
    index = {key: i for i, key in enumerate(keys)}
    assert set(raw["seed"]) == set(seeds)
    seed_keys = [{(i, t) for i, t, s in keys if s == seed} for seed in seeds]
    assert all(k == seed_keys[0] for k in seed_keys)
    image_tasks = {}
    for image, task in seed_keys[0]:
        image_tasks.setdefault(image, set()).add(task)
    expected = sorted((i, t) for i, t in seed_keys[0] if len(image_tasks[i]) >= 2)
    assert list(zip(panel["image_ids"], panel["task_ids"])) == expected
    np.testing.assert_array_equal(panel["training_seeds"], seeds)
    full_keys = list(zip(full["image_ids"], full["task_ids"]))
    assert len(set(full_keys)) == len(full_keys)
    fi = {key: i for i, key in enumerate(full_keys)}
    full_indices = [fi[key] for key in expected]
    np.testing.assert_array_equal(raw["text_features"], full["text_features"])
    np.testing.assert_array_equal(panel["text_features"], full["text_features"])
    np.testing.assert_array_equal(panel["reference_rows"], full["row_id"][full_indices])
    for s, seed in enumerate(seeds):
        selected = [index[i, t, seed] for i, t in expected]
        assert np.all(raw["keep_patch_tokens"][selected] == setting["keep_patch_tokens"])
        assert panel["seed_features"].dtype == raw["image_features"].dtype
        np.testing.assert_array_equal(panel["seed_features"][s], raw["image_features"][selected])
        np.testing.assert_array_equal(panel["source_rows_per_seed"][s], raw["row_id"][selected])
        for field in ("labels", "class_pairs"):
            np.testing.assert_array_equal(panel[field], raw[field][selected])
            np.testing.assert_array_equal(panel[field], full[field][full_indices])
    unique = sorted({i for i, _ in expected})
    np.testing.assert_array_equal(panel["unique_image_ids"], unique)
    positions = {key: i for i, key in enumerate(unique)}
    np.testing.assert_array_equal(panel["image_index"], [positions[i] for i, _ in expected])
    np.testing.assert_array_equal(panel["full_image_features"][panel["image_index"]], full["image_features"][full_indices])
    competitor = np.asarray([p[1] if p[0] == y else p[0] for y, p in zip(panel["labels"], panel["class_pairs"])])
    np.testing.assert_array_equal(panel["competitors"], competitor)
    text = full["text_features"].astype(np.float64)
    direction = text[panel["labels"]] - text[competitor]
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    close(direction, panel["task_directions"], 1e-12)
    margins = np.sum(full["image_features"][full_indices].astype(float) * direction, axis=1)
    close(margins, panel["full_margins"], 1e-12)
    return {"raw_feature_dtype": str(raw["image_features"].dtype), "raw_features_exactly_preserved": True,
            "development_manifests_verified": True, "image_count": len(unique), "image_task_count": len(expected)}


def permutations(dataset, image, tasks, draws, seed):
    # Documented byte encoding, independently reconstructed; no process hash.
    encoded = json.dumps([str(dataset), str(image), [str(t) for t in tasks]],
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed)] + list(encoded)))
    return np.stack([rng.permutation(len(tasks)) for _ in range(draws)])


def residual_coefficients(ids, margin_a, margin_b, angle):
    """Frisch-Waugh-Lovell slope via direct weighted nuisance regression."""
    ids = np.asarray(ids)
    unique, inverse, count = np.unique(ids, return_inverse=True, return_counts=True)
    weight = 1 / (len(unique) * count[inverse])
    lo, hi = np.minimum(margin_a, margin_b), np.maximum(margin_a, margin_b)
    nuisance = np.column_stack([lo, hi, lo * lo, lo * hi, hi * hi])
    angle = np.asarray(angle) / 10
    ordinary = np.column_stack([np.ones(len(ids)), nuisance])
    coefficients = []
    for fixed in (False, True):
        x, z = angle.copy(), (nuisance.copy() if fixed else ordinary.copy())
        if fixed:
            for image in unique:
                take = ids == image
                x[take] -= x[take].mean()
                z[take] -= z[take].mean(axis=0)
        root = np.sqrt(weight)
        beta = np.linalg.lstsq(z * root[:, None], x * root, rcond=None)[0]
        residual = x - z @ beta
        denominator = np.sum(weight * residual * residual)
        assert denominator > 1e-15
        linear = weight * residual / denominator
        if fixed:
            for image in unique:
                take = ids == image
                linear[take] -= linear[take].mean()
        coefficients.append(linear)
    return np.vstack([*coefficients, weight])


def image_geometry(features, directions):
    """Direct ordered feature-to-direction dot products, not an edge lookup."""
    z, u = np.asarray(features, float), np.asarray(directions, float)
    edges = np.asarray(list(itertools.combinations(range(len(z)), 2)))
    a, b = edges.T
    feature_dist = np.linalg.norm(z[:, None] - z[None, :], axis=2)
    direction_dist = np.linalg.norm(u[a] - u[b], axis=1)
    assert np.all(feature_dist[a, b] > 1e-12) and np.all(direction_dist > 1e-12)
    angles = np.rad2deg(np.arccos(np.clip(np.sum(u[a] * u[b], axis=1), -1, 1)))
    return edges, feature_dist, direction_dist, angles, z @ u.T


def assigned_alignment(scores, distances, direction_dist, edges, assignment):
    a, b = edges.T
    pa, pb = assignment[..., a], assignment[..., b]
    numerator = scores[pa, a] - scores[pb, a] - scores[pa, b] + scores[pb, b]
    return numerator / (distances[pa, pb] * direction_dist)


def auxiliary_seed_statistics(identity, alignment):
    """Recompute auxiliary-only weights and regressions on defined alignments."""
    alignment = np.asarray(alignment, dtype=float)
    eligible = np.isfinite(alignment)
    assert eligible.any()
    ids = np.asarray(identity["image_ids"])
    linear = residual_coefficients(ids[eligible], identity["base_margin_a"][eligible],
                                   identity["base_margin_b"][eligible], identity["angle_degrees"][eligible])
    values = linear @ alignment[eligible]
    excluded = [{"image_id": str(ids[i]), "task_a": str(identity["task_a"][i]),
                 "task_b": str(identity["task_b"][i])} for i in np.flatnonzero(~eligible)]
    return {**dict(zip(NAMES, values.tolist())), "total_pair_count": len(alignment),
            "eligible_pair_count": int(eligible.sum()), "excluded_pair_count": int((~eligible).sum()),
            "eligible_image_count": len(np.unique(ids[eligible])), "excluded_pairs": excluded,
            "minimum_displacement_norm": 1e-12, "minimum_direction_norm": 1e-12}


def check_setting(setting, panel, folder, summary, draws, seed, family_size=7):
    cases = saved_arrays(folder / "observed_pair_cases.npz")
    old = saved_arrays(Path(setting["geometry_source"]) / f'budget_{setting["keep_patch_tokens"]}/seed_mean_cases.npz')
    ids, a_tasks, b_tasks, all_alignment, per_seed = [], [], [], [], [[] for _ in panel["training_seeds"]]
    margin_a, margin_b, angles, norms, groups = [], [], [], [], []
    for i, image in enumerate(panel["unique_image_ids"]):
        take = np.flatnonzero(panel["image_index"] == i)
        task_ids = panel["task_ids"][take]
        assert list(task_ids) == sorted(set(task_ids)) and len(task_ids) >= 2
        z = panel["seed_features"][:, take].astype(float).mean(axis=0)
        u = panel["task_directions"][take]
        edges, dist, du, theta, scores = image_geometry(z, u)
        a, b = edges.T
        align = assigned_alignment(scores, dist, du, edges, np.arange(len(take)))
        ids.extend([str(image)] * len(edges)); a_tasks.extend(task_ids[a]); b_tasks.extend(task_ids[b])
        margin_a.extend(panel["full_margins"][take[a]]); margin_b.extend(panel["full_margins"][take[b]])
        angles.extend(theta); norms.extend(dist[a, b]); all_alignment.extend(align)
        for s, features in enumerate(panel["seed_features"][:, take]):
            dz = features[a].astype(float) - features[b].astype(float)
            norm = np.linalg.norm(dz, axis=1)
            valid = (norm > 1e-12) & (du > 1e-12)
            values = np.full(len(edges), np.nan)
            values[valid] = np.sum(dz[valid] * (u[a] - u[b])[valid], axis=1) / (norm[valid] * du[valid])
            per_seed[s].extend(values)
        groups.append((str(image), task_ids, z, u, edges, dist, du, scores))
    identity = {"image_ids": np.asarray(ids), "task_a": np.asarray(a_tasks), "task_b": np.asarray(b_tasks),
                "base_margin_a": np.asarray(margin_a), "base_margin_b": np.asarray(margin_b),
                "angle_degrees": np.asarray(angles), "displacement_norm": np.asarray(norms),
                "alignment": np.asarray(all_alignment)}
    old_lookup = {key: i for i, key in enumerate(zip(old["image_ids"], old["task_a"], old["task_b"]))}
    expected_keys = list(zip(ids, a_tasks, b_tasks))
    assert len(set(expected_keys)) == len(expected_keys) == len(old_lookup)
    order = [old_lookup[key] for key in expected_keys]
    for key, value in identity.items():
        if value.dtype.kind in 'US':
            np.testing.assert_array_equal(value, cases[key]); np.testing.assert_array_equal(value, old[key][order])
        else:
            close(value, cases[key]); close(value, old[key][order])
    exported = rows(folder / "observed_pair_cases.csv")
    assert len(exported) == len(ids)
    for csv_row, index in zip(exported, range(len(ids))):
        assert (csv_row["image_ids"], csv_row["task_a"], csv_row["task_b"]) == expected_keys[index]
        close(float(csv_row["alignment"]), identity["alignment"][index])
    linear = residual_coefficients(ids, np.asarray(margin_a), np.asarray(margin_b), np.asarray(angles))
    observed = linear @ np.asarray(all_alignment)
    old_summary = json.loads((Path(setting["geometry_source"]) / "analysis.json").read_text())["budgets"][str(setting["keep_patch_tokens"])]
    old_alignment = next(x for x in old_summary["normalized_alignment_diagnostics"] if x["minimum_displacement_norm"] == 1e-12)
    close(observed[:2], [old_alignment["difficulty_adjusted"]["angle_slope"], old_alignment["image_fixed_effects"]["angle_slope"]])
    close(observed, [summary["observed_statistics"][k] for k in NAMES])
    for s, entry in enumerate(summary["observed_statistics_per_seed"]):
        assert entry["training_seed"] == panel["training_seeds"][s]
        auxiliary = auxiliary_seed_statistics(identity, per_seed[s])
        close([auxiliary[k] for k in NAMES], [entry[k] for k in NAMES])
        for key in ("total_pair_count", "eligible_pair_count", "excluded_pair_count", "eligible_image_count",
                    "excluded_pairs", "minimum_displacement_norm", "minimum_direction_norm"):
            assert auxiliary[key] == entry[key], f"Auxiliary seed support differs: {key}"
    reference = np.zeros((draws, 3)); start = 0; direct_error = 0.
    for image, tasks, z, u, edges, dist, du, scores in groups:
        mapping = permutations(setting["dataset"], image, tasks, draws, seed)
        np.testing.assert_array_equal(np.sort(mapping, axis=1), np.tile(np.arange(len(tasks)), (draws, 1)))
        a, b = edges.T
        assigned_lengths = dist[mapping[:, a], mapping[:, b]]
        np.testing.assert_array_equal(np.sort(assigned_lengths, axis=1), np.tile(np.sort(dist[a, b]), (draws, 1)))
        alignment = assigned_alignment(scores, dist, du, edges, mapping)
        # Independently check the signed vector formula at multiple draw positions.
        for draw in sorted({0, draws // 2, draws - 1}):
            dz = z[mapping[draw, a]] - z[mapping[draw, b]]
            direct = np.sum(dz * (u[a] - u[b]), axis=1) / (np.linalg.norm(dz, axis=1) * du)
            close(direct, alignment[draw]); direct_error = max(direct_error, float(np.max(np.abs(direct-alignment[draw]))))
        reference += alignment @ linear[:, start:start + len(edges)].T
        start += len(edges)
    stored = saved_arrays(folder / "permutation_statistics.npz")
    np.testing.assert_array_equal(stored["statistic_names"], NAMES)
    np.testing.assert_array_equal(stored["draw_index"], np.arange(draws))
    assert int(stored["permutation_seed"]) == seed
    close(observed, stored["observed"]); close(reference, stored["null_statistics"])
    tails = (1 + np.sum(reference >= observed[None, :], axis=0)) / (draws + 1)
    close(tails, stored["upper_tail_p"], 0)
    for j, name in enumerate(NAMES):
        item = summary["statistics"][name]
        close(item["observed"], observed[j]); close(item["upper_tail_p"], tails[j], 0)
        assert item["exceedance_count"] == int(np.sum(reference[:, j] >= observed[j]))
        close(item["null_mean"], reference[:, j].mean())
        close(item["null_std"], reference[:, j].std(ddof=1))
        close(item["null_quantiles_025_50_975"], np.quantile(reference[:, j], [.025, .5, .975]))
    primary = summary["statistics"][NAMES[0]]
    close(primary["bonferroni_upper_tail_p"], min(1., family_size * tails[0]), 0)
    assert primary["above_assignment_reference_after_seven_settings"] == bool(observed[0] > 0 and family_size*tails[0] <= .05)
    return {"status": "passed", "image_count": len(groups), "pair_count": len(ids),
            "all_draws_recomputed": draws, "all_draws_preserve_complete_pair_norm_multisets": True,
            "raw_task_directions_angles_and_full_margins_fixed": True,
            "per_seed_observed_statistics_match": True,
            "observed_statistic_max_abs_error": float(np.max(np.abs(observed-stored["observed"]))),
            "all_reference_statistics_max_abs_error": float(np.max(np.abs(reference-stored["null_statistics"]))),
            "direct_signed_vector_max_abs_error": direct_error,
            "primary_exceedance_count": int(np.sum(reference[:, 0] >= observed[0])),
            "primary_upper_tail": float(tails[0]), "primary_bonferroni_upper_tail": float(min(1.,family_size*tails[0]))}


def audit(summary_path, output, panel_root=None):
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text())
    config = summary["config"]
    assert summary["status"] == "complete" and summary["final_test_used"] is False
    assert config["draws"] == 1999 and config["permutation_seed"] == 2094 and config["family_size"] == 7
    assert len(config["settings"]) == len(summary["settings"]) == 7
    record = {"status": "passed", "scope": "Independent raw prediction/manifests reconstruction, exact saved feature checks, old-H2 identity reproduction, direct weighted nuisance residualization, all 1999 references for all three statistics, all-draw complete-pair norm preservation, and seven-setting reference-tail correction. No geometry-permutation implementation is imported.",
              "source": str(summary_path), "final_test_used": False, "settings": {}}
    dataset_panels = {}
    for setting in config["settings"]:
        name = setting["id"]; result = summary["settings"][name]
        path = Path(panel_root) / f"{name}.npz" if panel_root else Path(result["panel_path"])
        panel = saved_arrays(path)
        metadata = json.loads(path.with_suffix('.json').read_text())
        assert metadata["setting"] == setting and metadata["final_test_used"] is False
        raw_audit = check_raw_panel(setting, panel, config["training_seeds"])
        keys = list(zip(panel["image_ids"], panel["task_ids"]))
        if setting["dataset"] in dataset_panels:
            assert keys == dataset_panels[setting["dataset"]]
        dataset_panels[setting["dataset"]] = keys
        result_audit = check_setting(setting, panel, summary_path.parent / name, result, config["draws"], config["permutation_seed"])
        record["settings"][name] = {**result_audit, "raw_panel": raw_audit}
        print(json.dumps({"setting": name, "audit": "passed"}), flush=True)
    record["shared_mapping_across_settings"] = "Identical dataset/image/sorted-task keys verified; the same documented deterministic RNG key is independently regenerated for every setting and all seeds share each assignment."
    Path(output).write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--panel-root', type=Path)
    args = parser.parse_args()
    audit(args.summary, args.output, args.panel_root)
