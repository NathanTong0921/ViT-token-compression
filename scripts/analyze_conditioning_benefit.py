"""Held-image/held-task prediction of gains over strong shared token selection."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from extract_benefit_features import load_setting
from token_compression.benefit_prediction import (fit_weighted_ridge, predict_ridge,
    two_way_folds, iter_two_way_splits, weighted_center)

SETTINGS = ("pets_b16_49", "pets_b16_98", "flowers_b16_49", "flowers_b16_98")
MODELS = ("mean", "base", "selection", "difficulty_base", "difficulty_selection")
OUTCOMES = ("geom", "raw")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def write_csv(path, rows):
    with Path(path).open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def model_inputs(data, panel, model):
    X = panel["X"]
    names = list(panel["feature_names"])
    if model in ("base", "difficulty_base"):
        X, names = X[:, :, :3], names[:3]
    if model.startswith("difficulty"):
        margin = data["full_signed_margin"]
        extra = np.stack((margin, margin * margin), axis=-1)
        X = np.concatenate((X, np.broadcast_to(extra, (len(X), *extra.shape))), axis=-1)
        names += ["full_signed_margin", "full_signed_margin_squared"]
    return X, names


def fit_probe(X, y, weights):
    s, n, p = X.shape
    return fit_weighted_ridge(X.reshape(s*n, p), y.reshape(s*n, 2), np.tile(weights/s, s))


def predict_probe(fit, X):
    s, n, p = X.shape
    return predict_ridge(fit, X.reshape(s*n, p)).reshape(s, n, 2)


def serial_fit(fit):
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in fit.items()}


def out_of_fold(data, panel):
    image_fold, task_fold = two_way_folds(data["ids"], data["tasks"])
    y, w = panel["outcomes"], data["weights"]
    predictions = np.empty((len(MODELS), *y.shape))
    cells, fits = [], []
    coverage = np.zeros(len(w), int)
    inputs = {model: model_inputs(data, panel, model) for model in MODELS[1:]}
    for i, j, train, test in iter_two_way_splits(image_fold, task_fold):
        assert train.any() and test.any()
        assert not set(data["ids"][train]) & set(data["ids"][test])
        assert not set(data["tasks"][train]) & set(data["tasks"][test])
        coverage[test] += 1
        mean = w[train] @ y[:, train].mean(axis=0) / w[train].sum()
        predictions[0, :, test, :] = mean
        cells.append({"image_fold": i, "task_fold": j, "training_rows": int(train.sum()),
                      "held_rows": int(test.sum()), "training_images": len(set(data["ids"][train])),
                      "held_images": len(set(data["ids"][test])),
                      "training_tasks": len(set(data["tasks"][train])),
                      "held_tasks": len(set(data["tasks"][test])), "training_gain": mean.tolist()})
        for mi, model in enumerate(MODELS[1:], start=1):
            X, names = inputs[model]
            fit = fit_probe(X[:, train], y[:, train], w[train])
            value = predict_probe(fit, X[:, test])
            # Index the model view first to preserve [run,row,outcome] axis order.
            predictions[mi][:, test] = value
            fits.append({"image_fold": i, "task_fold": j, "model": model,
                         "features": names, **serial_fit(fit)})
    np.testing.assert_array_equal(coverage, np.ones(len(w), int))
    return predictions, cells, fits, image_fold, task_fold


def transfer(source, target, source_panel, target_panel):
    sy, w = source_panel["outcomes"], source["weights"]
    predictions = np.empty((len(MODELS), *target_panel["outcomes"].shape))
    predictions[0] = w @ sy.mean(axis=0) / w.sum()
    fits = []
    for mi, model in enumerate(MODELS[1:], start=1):
        X, names = model_inputs(source, source_panel, model)
        target_X, target_names = model_inputs(target, target_panel, model)
        assert names == target_names
        fit = fit_probe(X, sy, w)
        predictions[mi] = predict_probe(fit, target_X)
        fits.append({"model": model, "features": names, **serial_fit(fit)})
    return predictions, fits


def derived_metrics(values, index, routing):
    """Ratios use pooled sums, so a common bootstrap weight denominator cancels."""
    result = {}
    get = lambda name: values[..., index[name]]
    for metric in OUTCOMES:
        null = get("mse_mean_" + metric)
        signal = get("within_signal_" + metric)
        for model in MODELS[1:]:
            result[model + "_r2_" + metric] = 1 - get("mse_" + model + "_" + metric) / null
            result[model + "_within_r2_" + metric] = 1 - get("within_mse_" + model + "_" + metric) / signal
        result["selection_increment_r2_" + metric] = (get("mse_base_" + metric) - get("mse_selection_" + metric)) / null
        result["selection_increment_within_r2_" + metric] = (get("within_mse_base_" + metric) - get("within_mse_selection_" + metric)) / signal
        result["difficulty_adjusted_increment_r2_" + metric] = (get("mse_difficulty_base_" + metric) - get("mse_difficulty_selection_" + metric)) / null
    if routing:
        for key in index:
            if key.startswith("route_"):
                result[key] = get(key) / get("weight")
    return result


def summarize(data, panel, predictions, replicates):
    y, ids, w = panel["outcomes"], data["ids"], data["weights"]
    centered_y = np.stack([weighted_center(run, ids, w) for run in y])
    values, names = [], []
    for mi, model in enumerate(MODELS):
        centered_p = np.stack([weighted_center(run, ids, w) for run in predictions[mi]])
        for qi, metric in enumerate(OUTCOMES):
            names += ["mse_" + model + "_" + metric, "within_mse_" + model + "_" + metric]
            values += [(y[:, :, qi] - predictions[mi, :, :, qi])**2,
                       (centered_y[:, :, qi] - centered_p[:, :, qi])**2]
    for qi, metric in enumerate(OUTCOMES):
        names.append("within_signal_" + metric)
        values.append(centered_y[:, :, qi]**2)
    route_choices = None
    if panel["routing"]:
        alt, base = panel["alternative_outcomes"], panel["base_outcomes"]
        choose = predictions[2, :, :, 0] > 0
        fixed = predictions[0, :, :, 0] > 0
        routed = np.where(choose[:, :, None], alt, base)
        chosen_fixed = np.where(fixed[:, :, None], alt, base)
        route_choices = choose
        for contrast, reference in (("base", base), ("alternative", alt), ("training_fixed", chosen_fixed)):
            for qi, metric in enumerate(("accuracy", "geom", "raw")):
                names.append("route_minus_" + contrast + "_" + metric)
                values.append(routed[:, :, qi] - reference[:, :, qi])
        names.append("route_alternative_fraction")
        values.append(choose.astype(float))
    names.append("weight")
    values.append(np.ones(y.shape[:2]))
    metrics = np.stack(values, axis=-1)  # run,row,metric
    point_sums = np.einsum("n,snk->sk", w, metrics)
    index = {name: i for i, name in enumerate(names)}
    mean_sums = point_sums.mean(axis=0)
    point = derived_metrics(mean_sums, index, panel["routing"])
    per_run = derived_metrics(point_sums, index, panel["routing"])
    unique, inverse = np.unique(ids, return_inverse=True)
    image_sums = np.zeros((len(unique), len(names)))
    np.add.at(image_sums, inverse, w[:, None] * metrics.mean(axis=0))
    rng = np.random.default_rng(2353)
    draws = np.empty((replicates, len(names)))
    for start in range(0, replicates, 100):
        ix = rng.integers(len(unique), size=(min(100, replicates-start), len(unique)))
        draws[start:start+len(ix)] = image_sums[ix].sum(axis=1)
    boot = derived_metrics(draws, index, panel["routing"])
    estimates = {name: {"mean": float(value), "ci95": np.quantile(boot[name], [.025, .975]).tolist(),
                       "four_setting_adjusted_ci95": np.quantile(boot[name], [.025/4, 1-.025/4]).tolist()}
                 for name, value in point.items()}
    mse = {name: float(mean_sums[i]/mean_sums[-1]) for i, name in enumerate(names)
           if name.startswith("mse_") or name.startswith("within_")}
    run_records = [{"run_id": np.asarray(run).tolist(), **{key: float(value[i]) for key,value in per_run.items()}}
                   for i, run in enumerate(panel["run_ids"])]
    return dict(estimates=estimates, mse=mse, per_run=run_records), names, metrics, route_choices


def save_analysis(output, data, panel, predictions, fits, replicates, context):
    output.mkdir(parents=True, exist_ok=True)
    result, metric_names, metrics, choices = summarize(data, panel, predictions, replicates)
    result.update(context, rows=len(data["ids"]), images=len(set(data["ids"])),
                  tasks=len(set(data["tasks"])), run_ids=panel["run_ids"].tolist(),
                  feature_names=panel["feature_names"], routing=panel["routing"])
    arrays = dict(predictions=predictions, models=MODELS, outcomes=panel["outcomes"],
                  image_ids=data["ids"], task_ids=data["tasks"], weights=data["weights"],
                  run_ids=panel["run_ids"], metric_names=metric_names, metric_values=metrics)
    if choices is not None:
        arrays.update(route_choices=choices, base_outcomes=panel["base_outcomes"],
                      alternative_outcomes=panel["alternative_outcomes"])
    np.savez_compressed(output / "predictions.npz", **arrays)
    write_json(output / "summary.json", result)
    write_json(output / "fits.json", fits)
    return result


def summary_rows(result):
    output = []
    for outcome in OUTCOMES:
        row = {key:result[key] for key in ("mode", "setting", "panel")}
        row.update(source=result.get("source", ""), outcome=outcome)
        for name in ("base_r2", "selection_r2", "selection_increment_r2", "selection_within_r2",
                     "selection_increment_within_r2", "difficulty_adjusted_increment_r2"):
            value = result["estimates"][name + "_" + outcome]
            row.update({name: value["mean"], name+"_ci_low":value["ci95"][0], name+"_ci_high":value["ci95"][1]})
        output.append(row)
    return output


def save_inputs(output, data):
    output.mkdir(parents=True, exist_ok=True)
    arrays = dict(image_ids=data["ids"], task_ids=data["tasks"], weights=data["weights"],
                  full_signed_margin=data["full_signed_margin"], full_raw_gap=data["full_raw_gap"],
                  full_outcomes=data["full_outcomes"])
    for name, panel in data["panels"].items():
        arrays[name+"_y"] = panel["outcomes"]
        arrays[name+"_run_ids"] = panel["run_ids"]
        if name != "increment":
            arrays[name+"_X"] = panel["X"]
            arrays[name+"_feature_names"] = panel["feature_names"]
            arrays[name+"_base"] = panel["base_outcomes"]
            arrays[name+"_alternative"] = panel["alternative_outcomes"]
    np.savez_compressed(output / "inputs.npz", **arrays)
    write_csv(output / "examples.csv", data["rows"])


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    protocol = dict(git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        core_archive=str(args.core_archive.resolve()), hybrid_archive=str(args.hybrid_archive.resolve()),
        settings=SETTINGS, models=MODELS, outcomes=OUTCOMES, ridge=.001, image_fold_seed=2351,
        task_fold_seed=2352, bootstrap_seed=2353, bootstrap_replicates=args.replicates,
        final_test_used=False, new_selector_training=False, new_clip_inference=False,
        uncertainty="Image bootstrap conditional on fixed tasks/runs and fitted probes; four-setting adjusted intervals per outcome/metric.")
    write_json(args.output / "protocol.json", protocol)
    data = {}
    all_summaries, table, routing_rows = [], [], []
    for setting in SETTINGS:
        data[setting] = load_setting(args.core_archive, args.hybrid_archive, setting)
        current = data[setting]
        folder = args.output / setting
        save_inputs(folder, current)
        for name, panel in current["panels"].items():
            predictions, cells, fits, ifold, tfold = out_of_fold(current, panel)
            result = save_analysis(folder / name, current, panel, predictions, fits, args.replicates,
                                   dict(mode="double_holdout", setting=setting, panel=name))
            write_json(folder / name / "split_cells.json", cells)
            if name == "primary":
                write_csv(folder / "row_folds.csv", [{"image_id":image,"task_id":task,"image_fold":int(i),"task_fold":int(j)}
                    for image,task,i,j in zip(current["ids"],current["tasks"],ifold,tfold)])
            all_summaries.append(result)
            table.extend(summary_rows(result))
            print(json.dumps({"setting":setting,"panel":name,"double_holdout_complete":True}),flush=True)
    for budget in (49,98):
        for src, dst in (("pets","flowers"),("flowers","pets")):
            source, target = f"{src}_b16_{budget}", f"{dst}_b16_{budget}"
            for name in data[source]["panels"]:
                sp, tp = data[source]["panels"][name], data[target]["panels"][name]
                predictions, fits = transfer(data[source], data[target], sp, tp)
                folder = args.output / "transfer" / (source + "_to_" + target) / name
                result = save_analysis(folder, data[target], tp, predictions, fits, args.replicates,
                                       dict(mode="cross_dataset", setting=target, source=source, panel=name))
                all_summaries.append(result)
                table.extend(summary_rows(result))
            print(json.dumps({"source":source,"target":target,"transfer_complete":True}),flush=True)
    for result in all_summaries:
        if not result["routing"]:
            continue
        for contrast in ("base", "alternative", "training_fixed"):
            for metric in ("accuracy", "geom", "raw"):
                value = result["estimates"]["route_minus_"+contrast+"_"+metric]
                routing_rows.append({"mode":result["mode"],"setting":result["setting"],"source":result.get("source", ""),
                    "panel":result["panel"],"reference":contrast,"metric":metric,**value})
    write_csv(args.output / "predictive_summary.csv", table)
    write_csv(args.output / "routing_summary.csv", routing_rows)
    write_json(args.output / "summary.json", all_summaries)
    write_json(args.output / "execution.json", dict(elapsed_seconds=time.perf_counter()-start,
        data_settings=len(data),double_holdout_panels=12,transfer_panels=12,cpu_threads=1,new_neural_fits=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-archive", type=Path, required=True)
    parser.add_argument("--hybrid-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    main(parser.parse_args())
