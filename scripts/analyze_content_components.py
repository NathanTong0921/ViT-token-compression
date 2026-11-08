"""Separate projected token-energy and selected-set gap contributions."""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from token_compression.benefit_prediction import (
    fit_weighted_ridge, predict_ridge, two_way_folds, iter_two_way_splits, weighted_center,
)


SETTINGS = ("pets_b16_49", "pets_b16_98", "flowers_b16_49", "flowers_b16_98")
MODELS = ("context", "energy", "gap", "full")
METRICS = ("raw", "geom")
ENERGY = ("u_omitted_contrast_energy", "c_minus_u_retained_energy")
GAP = ("u_surrogate_gap_loss", "c_minus_u_surrogate_gap", "cu_surrogate_prediction_disagreement")
CONTEXT = "full_surrogate_abs_gap"
CONTRASTS = (("energy_added", "context", "energy"),
             ("gap_added", "context", "gap"),
             ("gap_added_after_energy", "energy", "full"),
             ("energy_added_after_gap", "gap", "full"),
             ("all_content_added", "context", "full"))


def load_data(inputs, features):
    with np.load(inputs, allow_pickle=False) as source:
        data = {k: source[k].copy() for k in ("image_ids", "task_ids", "weights", "primary_X",
                    "primary_feature_names", "primary_run_ids")}
        data["y"] = (source["primary_alternative"] - source["primary_base"])[..., [2, 1]]
    with np.load(features, allow_pickle=False) as source:
        for a, b in (("image_ids", "image_ids"), ("task_ids", "task_ids"), ("run_ids", "primary_run_ids")):
            np.testing.assert_array_equal(source[a], data[b])
        data["content"] = source["X"].astype(float)
        data["content_names"] = source["feature_names"].tolist()
    return data


def design(data, model):
    requested = {CONTEXT}
    if model in ("energy", "full"):
        requested.update(ENERGY)
    if model in ("gap", "full"):
        requested.update(GAP)
    selected = [i for i, name in enumerate(data["content_names"]) if name in requested]
    return (np.concatenate((data["primary_X"], data["content"][..., selected]), axis=-1),
            data["primary_feature_names"].tolist() + [data["content_names"][i] for i in selected])


def fit_predict(data):
    y, w = data["y"], data["weights"]
    image_fold, task_fold = two_way_folds(data["image_ids"], data["task_ids"])
    predictions = np.empty((len(MODELS), *y.shape))
    null = np.empty_like(y)
    inputs = {model: design(data, model) for model in MODELS}
    records, coverage = [], np.zeros(len(w), dtype=int)
    for i, j, train, held in iter_two_way_splits(image_fold, task_fold):
        coverage[held] += 1
        null[:, held] = w[train] @ y[:, train].mean(axis=0) / w[train].sum()
        for mi, model in enumerate(MODELS):
            X, names = inputs[model]
            fit = fit_weighted_ridge(X[:, train].reshape(-1, X.shape[-1]),
                y[:, train].reshape(-1, len(METRICS)), np.tile(w[train] / len(y), len(y)), ridge=.001)
            predictions[mi][:, held] = predict_ridge(fit, X[:, held].reshape(-1, X.shape[-1])).reshape(len(y), -1, len(METRICS))
            records.append(dict(image_fold=i, task_fold=j, model=model, features=names,
                **{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in fit.items()}))
    np.testing.assert_array_equal(coverage, 1)
    return dict(predictions=predictions, null_predictions=null, image_fold=image_fold,
                task_fold=task_fold, fits=records)


def contributions(data, result):
    y, ids, w = data["y"], data["image_ids"], data["weights"]
    cy = np.array([weighted_center(run, ids, w) for run in y])
    names, columns = [], []
    for qi, metric in enumerate(METRICS):
        names += ["null_" + metric, "within_signal_" + metric]
        columns += [(y[..., qi] - result["null_predictions"][..., qi])**2, cy[..., qi]**2]
        for mi, model in enumerate(MODELS):
            pred = result["predictions"][mi]
            cp = np.array([weighted_center(run, ids, w) for run in pred])
            names += [model + "_mse_" + metric, model + "_within_mse_" + metric]
            columns += [(y[..., qi]-pred[..., qi])**2, (cy[..., qi]-cp[..., qi])**2]
    return names, np.stack(columns, axis=-1)


def derive(sums, names):
    ix = {name: i for i, name in enumerate(names)}
    get = lambda name: sums[..., ix[name]]
    out = {}
    for metric in METRICS:
        for model in MODELS:
            out[model + "_r2_" + metric] = 1-get(model+"_mse_"+metric)/get("null_"+metric)
        for contrast, base, alt in CONTRASTS:
            out[contrast+"_delta_r2_"+metric] = (get(base+"_mse_"+metric)-get(alt+"_mse_"+metric))/get("null_"+metric)
            out[contrast+"_within_delta_r2_"+metric] = (get(base+"_within_mse_"+metric)-get(alt+"_within_mse_"+metric))/get("within_signal_"+metric)
    return out


def summarize(data, result, replicates):
    names, values = contributions(data, result)
    per_seed_sums = np.einsum("n,snk->sk", data["weights"], values)
    estimates = derive(per_seed_sums.mean(axis=0), names)
    per_seed = derive(per_seed_sums, names)
    images, inverse = np.unique(data["image_ids"], return_inverse=True)
    image_sums = np.zeros((len(images), len(names)))
    np.add.at(image_sums, inverse, data["weights"][:, None] * values.mean(axis=0))
    rng = np.random.default_rng(2721)
    draws = np.empty((replicates, len(names)))
    for start in range(0, replicates, 100):
        chosen = rng.integers(len(images), size=(min(100,replicates-start),len(images)))
        draws[start:start+len(chosen)] = image_sums[chosen].sum(axis=1)
    boot = derive(draws, names)
    return {"estimates": {name: dict(mean=float(value), ci95=np.quantile(boot[name],[.025,.975]).tolist(),
                 four_setting_adjusted_ci95=np.quantile(boot[name],[.025/4,1-.025/4]).tolist())
                 for name,value in estimates.items()},
            "per_seed": [dict(seed=int(seed), **{name:float(values[i]) for name,values in per_seed.items()})
                         for i,seed in enumerate(data["primary_run_ids"])]}


def write_csv(path, rows):
    with path.open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main(args):
    started=time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol=dict(purpose="Interpretive follow-up to positive content-prediction results; fixed feature groups, no retuning.",
        models=MODELS, energy_features=ENERGY, gap_features=GAP, baseline="old seven statistics plus full_surrogate_abs_gap",
        contrasts=CONTRASTS, outcomes=METRICS, ridge=.001, fold_seeds=[2351,2352],
        bootstrap_replicates=args.replicates, bootstrap_seed=2721,
        interval_family="Four settings separately for each named contrast and outcome")
    (args.output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    summaries, table, seeds = [], [], []
    for setting in SETTINGS:
        data=load_data(args.inputs/setting/"inputs.npz",args.features/(setting+".npz"))
        result=fit_predict(data)
        summary=summarize(data,result,args.replicates)
        summary.update(setting=setting,rows=len(data["weights"]),images=len(set(data["image_ids"])),tasks=len(set(data["task_ids"])))
        summaries.append(summary)
        out=args.output/setting;out.mkdir(exist_ok=True)
        (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        (out/"fits.json").write_text(json.dumps(result["fits"],indent=2)+"\n")
        np.savez_compressed(out/"predictions.npz",predictions=result["predictions"],null_predictions=result["null_predictions"],
            outcomes=data["y"],image_ids=data["image_ids"],task_ids=data["task_ids"],weights=data["weights"],
            image_fold=result["image_fold"],task_fold=result["task_fold"],models=MODELS,outcome_names=METRICS,
            run_ids=data["primary_run_ids"])
        for name,value in summary["estimates"].items():
            table.append(dict(setting=setting,metric=name,mean=value["mean"],low=value["ci95"][0],high=value["ci95"][1],
                adjusted_low=value["four_setting_adjusted_ci95"][0],adjusted_high=value["four_setting_adjusted_ci95"][1]))
        seeds.extend([dict(setting=setting,**row) for row in summary["per_seed"]])
        print(json.dumps({"setting":setting, **{key:summary["estimates"][key]["mean"] for key in
             ["energy_added_delta_r2_raw","gap_added_delta_r2_raw","energy_added_after_gap_delta_r2_raw","gap_added_after_energy_delta_r2_raw"]}}),flush=True)
    write_csv(args.output/"summary.csv",table);write_csv(args.output/"per_seed.csv",seeds)
    (args.output/"summary.json").write_text(json.dumps(summaries,indent=2)+"\n")
    (args.output/"execution.json").write_text(json.dumps(dict(elapsed_seconds=time.perf_counter()-started,
           total_ridge_fits=4*25*4,bootstrap_replicates=args.replicates),indent=2)+"\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",type=Path,default=Path("results/conditioning-benefit-v1"))
    parser.add_argument("--features",type=Path,default=Path("results/conditioning-content-v1/features"))
    parser.add_argument("--output",type=Path,default=Path("results/conditioning-content-v1/components"))
    parser.add_argument("--replicates",type=int,default=2000)
    main(parser.parse_args())
