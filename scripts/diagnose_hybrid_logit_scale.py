"""Rank-preserving scale diagnosis from existing hybrid/original raw archives."""

import argparse
import csv
import io
import json
from pathlib import Path
import platform
import subprocess
import time
import zipfile

import numpy as np
from scipy.stats import rankdata

from analyze_selector_budgets import weighted_estimates
from token_compression.logit_scale import fit_global_scale, row_kl


SETTINGS = ("pets_b16_49", "flowers_b16_49", "flowers_b16_98")
SEEDS = (2041, 2042, 2043)
MODELS = ("shared", "original_task", "original_image", "hybrid_task", "hybrid_image")
CONDITIONS = ("raw", "global_scale", "row_standardized_scale")
ARMS = ("task", "image")


class Source:
    def __init__(self, path, prefix):
        self.path, self.prefix = Path(path), prefix
        self.archive = zipfile.ZipFile(path)

    def text(self, name):
        return self.archive.read(self.prefix + "/" + name).decode()

    def json(self, name):
        return json.loads(self.text(name))

    def rows(self, name):
        return list(csv.DictReader(io.StringIO(self.text(name))))

    def arrays(self, name, keys):
        with np.load(io.BytesIO(self.archive.read(self.prefix + "/" + name)), allow_pickle=False) as a:
            return {key: a[key].copy() for key in keys}


def write_csv(path, rows):
    with Path(path).open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def image_folds(ids):
    unique = np.unique(ids)
    shuffled = np.random.default_rng(2341).permutation(unique)
    lookup = {image: i % 5 for i, image in enumerate(shuffled)}
    return np.asarray([lookup[image] for image in ids])


def topk(logits, k):
    return np.sort(np.argsort(-logits, axis=1, kind="stable")[:, :k], axis=1)


def overlap(a, b):
    return (a[:, :, None] == b[:, None, :]).any(axis=-1).mean(axis=-1)


def calibrated(logits, masks, weights, folds, standardize, expected, context):
    centered = logits - logits.mean(axis=1, keepdims=True)
    if standardize:
        std = centered.std(axis=1, keepdims=True)
        assert np.all(std > 0), "Observed constant logits: investigate before standardizing"
        centered = centered / std
    losses, scales = np.empty(len(logits)), np.empty(len(logits))
    records = []
    for fold in range(5):
        train, held = folds != fold, folds == fold
        fit = fit_global_scale(centered[train], masks[train], weights[train])
        losses[held] = row_kl(centered[held], masks[held], fit["alpha"])
        scales[held] = fit["alpha"]
        np.testing.assert_array_equal(topk(centered[held] * fit["alpha"], masks.shape[1]), expected[held])
        records.append({**context, "condition": CONDITIONS[2 if standardize else 1],
                        "held_fold": fold, "calibration_rows": int(train.sum()),
                        "held_rows": int(held.sum()), **fit, "topk_exact": True})
    return losses, scales, records


def analyze_setting(new, old, setting, output, replicates):
    before = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    p = new.json(setting + "/protocol.json")
    rows, old_rows = new.rows(setting + "/examples.csv"), old.rows(setting + "/examples.csv")
    keys = [(r["image_id"], r["task_id"]) for r in rows]
    lookup = {(r["image_id"], r["task_id"]): i for i, r in enumerate(old_rows)}
    assert len(keys) == len(set(keys)) == len(lookup) and set(keys) == set(lookup)
    order = np.asarray([lookup[key] for key in keys])
    ids = np.asarray([r["image_id"] for r in rows])
    weights = np.asarray([float(r["task_class_weight"]) for r in rows])
    np.testing.assert_allclose(weights, [float(old_rows[i]["task_class_weight"]) for i in order], atol=1e-12, rtol=0)
    folds = image_folds(ids)
    write_csv(output / "image_folds.csv", [{"image_id": image, "fold": int(folds[np.flatnonzero(ids == image)[0]])}
                                          for image in np.unique(ids)])
    assert p["training_seeds"] == list(SEEDS)
    k, temperature = p["config"]["keep_patch_tokens"], p["config"]["temperature"]
    fits = {(x["seed"], x["arm"]): x for x in new.json(setting + "/target_fit.json")}
    losses = np.empty((len(SEEDS), len(MODELS), len(CONDITIONS), len(rows)))
    scales = np.empty((len(SEEDS), len(MODELS), 2, len(rows)))
    stds, overlaps = np.empty((len(SEEDS), len(MODELS), len(rows))), np.empty((len(SEEDS), len(MODELS), len(rows)))
    pair_overlap, rank_correlations = np.empty((len(SEEDS), 2, len(rows))), np.empty((len(SEEDS), 2, len(rows)))
    fold_records, fit_audits = [], []
    for si, seed in enumerate(SEEDS):
        arrays = [source.arrays(setting + f"/seed_{seed}.npz", ("methods", "selector_logits", "patch_indices"))
                  for source in (new, old)]
        n, o = arrays
        np.testing.assert_array_equal(n["methods"], ("shared", "task_residual", "image_residual"))
        np.testing.assert_array_equal(n["methods"], o["methods"])
        np.testing.assert_array_equal(n["selector_logits"][0], o["selector_logits"][0, order])
        target = new.arrays(setting + f"/seed_{seed}/val_targets.npz",
                            ("patch_indices", "image_indices", "image_ids", "pair_indices", "class_pairs"))
        np.testing.assert_array_equal(target["image_ids"][target["image_indices"]].astype(str), ids)
        np.testing.assert_array_equal(target["class_pairs"][target["pair_indices"]],
                                      [[int(r["class_a"]), int(r["class_b"])] for r in rows])
        masks = target["patch_indices"]
        logits = np.stack((n["selector_logits"][0], o["selector_logits"][1, order], o["selector_logits"][2, order],
                           n["selector_logits"][1], n["selector_logits"][2])).astype(float) / temperature
        saved = np.stack((n["patch_indices"][0], o["patch_indices"][1, order], o["patch_indices"][2, order],
                          n["patch_indices"][1], n["patch_indices"][2]))
        for mi, model in enumerate(MODELS):
            z = logits[mi]
            np.testing.assert_array_equal(topk(z, k), saved[mi])
            losses[si, mi, 0] = row_kl(z, masks)
            stds[si, mi] = z.std(axis=1)
            overlaps[si, mi] = overlap(saved[mi], masks)
            for ci, normalize in enumerate((False, True), start=1):
                l, a, records = calibrated(z, masks, weights, folds, normalize, saved[mi],
                                           {"setting": setting, "seed": seed, "model": model})
                losses[si, mi, ci], scales[si, mi, ci-1] = l, a
                fold_records.extend(records)
        for ai, arm in enumerate(ARMS):
            oi, ni = 1 + ai, 3 + ai
            pair_overlap[si, ai] = overlap(saved[oi], saved[ni])
            ro, rn = [rankdata(logits[mi], axis=1) for mi in (oi, ni)]
            ro -= ro.mean(axis=1, keepdims=True)
            rn -= rn.mean(axis=1, keepdims=True)
            rank_correlations[si, ai] = (ro * rn).sum(axis=1) / np.sqrt((ro * ro).sum(axis=1) * (rn * rn).sum(axis=1))
            for name, mi in (("original_trained", oi), ("hybrid_trained", ni)):
                ref = fits[(seed, arm + "_residual")][name]["val"]
                actual_kl = float(weights @ losses[si, mi, 0] / weights.sum())
                actual_overlap = float(weights @ overlaps[si, mi] / weights.sum())
                np.testing.assert_allclose(actual_kl, ref["hybrid_target_kl"], atol=1e-6, rtol=0)
                np.testing.assert_allclose(actual_overlap, ref["hybrid_mask_overlap"], atol=1e-10, rtol=0)
                fit_audits.append({"seed": seed, "model": MODELS[mi],
                                   "prior_kl_absolute_error": abs(actual_kl - ref["hybrid_target_kl"]),
                                   "prior_overlap_absolute_error": abs(actual_overlap - ref["hybrid_mask_overlap"])})
        print(json.dumps({"setting": setting, "seed": seed, "calibration_complete": True}), flush=True)

    contrasts, tables = {}, []
    max_decomposition_error = 0.0
    for ai, arm in enumerate(ARMS):
        oi, ni = 1 + ai, 3 + ai
        effects = {"raw_reduction": losses[:, oi, 0] - losses[:, ni, 0],
                   "old_target_overlap": overlaps[:, oi], "new_target_overlap": overlaps[:, ni],
                   "target_overlap_change": overlaps[:, ni] - overlaps[:, oi],
                   "old_new_mask_overlap": pair_overlap[:, ai], "old_new_rank_correlation": rank_correlations[:, ai]}
        for ci, name in enumerate(("global", "row"), start=1):
            effects[name + "_remaining"] = losses[:, oi, ci] - losses[:, ni, ci]
            effects[name + "_removable"] = (losses[:, oi, 0] - losses[:, oi, ci]) - (losses[:, ni, 0] - losses[:, ni, ci])
            effects[name + "_old_cal_minus_new_raw"] = losses[:, oi, ci] - losses[:, ni, 0]
            error = np.max(np.abs(effects["raw_reduction"] - effects[name + "_remaining"] - effects[name + "_removable"]))
            max_decomposition_error = max(max_decomposition_error, float(error))
            assert error < 1e-12
        estimates = weighted_estimates(ids, weights, np.column_stack([x.mean(axis=0) for x in effects.values()]),
                                       list(effects), replicates, 2342, 6)
        per_seed = [{"seed": seed, **{name: float(weights @ x[si] / weights.sum()) for name, x in effects.items()}}
                    for si, seed in enumerate(SEEDS)]
        ratios = {name + "_removed_fraction": (estimates[name + "_removable"]["mean"] / estimates["raw_reduction"]["mean"]
                  if estimates["raw_reduction"]["mean"] > 0 else None) for name in ("global", "row")}
        contrasts[arm] = {"effects": estimates, "ratios_of_aggregates": ratios, "per_seed": per_seed}
        table = {"setting": setting, "arm": arm}
        for tag, mi in (("original", oi), ("hybrid", ni)):
            for ci, cond in enumerate(("raw", "global", "row")):
                table[tag + "_" + cond] = float(weights @ losses[:, mi, ci].mean(axis=0) / weights.sum())
            table[tag + "_logit_std"] = float(weights @ stds[:, mi].mean(axis=0) / weights.sum())
        table.update({name: value["mean"] for name, value in estimates.items()})
        table.update(ratios)
        tables.append(table)

    model_table = [{"model": model, "condition": condition,
                    "kl": float(weights @ losses[:, mi, ci].mean(axis=0) / weights.sum()),
                    "target_overlap": float(weights @ overlaps[:, mi].mean(axis=0) / weights.sum())}
                   for mi, model in enumerate(MODELS) for ci, condition in enumerate(CONDITIONS)]
    result = {"setting": setting, "seeds": list(SEEDS), "images": len(np.unique(ids)), "rows": len(rows),
              "k": k, "uniform_limit_kl": float(np.log(logits.shape[-1] / k)), "contrasts": contrasts,
              "model_table": model_table, "boundary_fits": [r for r in fold_records if r["boundary"]],
              "calibration_fit_count": len(fold_records), "prior_fit_audits": fit_audits,
              "decomposition_max_error": max_decomposition_error, "all_transformed_masks_exact": True,
              "elapsed_seconds": time.perf_counter() - before}
    write_csv(output / "calibration_fits.csv", fold_records)
    write_csv(output / "summary.csv", tables)
    np.savez_compressed(output / "row_metrics.npz", seeds=SEEDS, models=MODELS, conditions=CONDITIONS,
                        image_ids=ids, task_ids=[r["task_id"] for r in rows], weights=weights, folds=folds,
                        kl=losses, fitted_scale=scales, logit_std=stds, target_overlap=overlaps,
                        old_new_mask_overlap=pair_overlap, old_new_rank_correlation=rank_correlations)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result, tables


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    protocol = {"git_commit": commit, "new_archive": str(args.new_archive.resolve()),
                "old_archive": str(args.old_archive.resolve()), "settings": SETTINGS, "seeds": SEEDS,
                "models": MODELS, "conditions": CONDITIONS, "image_fold_seed": 2341, "folds": 5,
                "alpha_bounds": [1e-4, 1e4], "bootstrap_replicates": args.replicates, "bootstrap_seed": 2342,
                "training_or_clip_inference": False, "final_test_used": False, "python": platform.python_version(),
                "numpy": np.__version__, "cpu_threads": 1,
                "scope": "Cross-fitted development-set probability diagnosis. Adjusted intervals span six setting/arm comparisons per metric, conditional on fitted scales. Residual KL differences are not pure ranking effects."}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    new = Source(args.new_archive, "runs/hybrid-distillation-v1")
    old = Source(args.old_archive, "runs/shared-residual-v1")
    results, tables = [], []
    for setting in SETTINGS:
        result, table = analyze_setting(new, old, setting, args.output / setting, args.replicates)
        results.append(result)
        tables.extend(table)
    new.archive.close()
    old.archive.close()
    write_csv(args.output / "summary.csv", tables)
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-archive", type=Path, required=True)
    parser.add_argument("--old-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    main(parser.parse_args())
