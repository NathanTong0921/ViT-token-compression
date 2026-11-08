"""Can pre-pruning image content explain benefits beyond selector statistics?"""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from token_compression.benefit_prediction import (
    fit_weighted_ridge, predict_ridge, two_way_folds, iter_two_way_splits,
    weighted_center,
)

SETTINGS = ("pets_b16_49", "pets_b16_98", "flowers_b16_49", "flowers_b16_98")
MODELS = ("mean", "statistics", "context", "content", "difficulty_context", "difficulty_content",
          "tree_context", "tree_content")
TREE_PARAMS = dict(max_iter=100, max_leaf_nodes=7, learning_rate=.05,
                   l2_regularization=1., min_samples_leaf=30, random_state=2361,
                   early_stopping=False)
OUTCOMES = ("raw", "geom", "accuracy")


def json_write(path, data):
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        return None if isinstance(value, float) and not np.isfinite(value) else value
    Path(path).write_text(json.dumps(clean(data), indent=2, allow_nan=False) + "\n")


def csv_write(path, rows):
    with Path(path).open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_data(inputs, features):
    with np.load(inputs, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in (
            "image_ids", "task_ids", "weights", "full_raw_gap", "full_signed_margin",
            "primary_X", "primary_feature_names", "primary_run_ids", "primary_base",
            "primary_alternative")}
    with np.load(features, allow_pickle=False) as archive:
        for key, old in (("image_ids", "image_ids"), ("task_ids", "task_ids"),
                         ("run_ids", "primary_run_ids")):
            np.testing.assert_array_equal(archive[key], data[old])
        data["content_X"] = archive["X"].astype(float)
        data["content_names"] = archive["feature_names"].tolist()
    assert data["content_X"].shape[:2] == data["primary_X"].shape[:2]
    assert np.isfinite(data["content_X"]).all()
    # Original absolute outcome columns are accuracy, geometric margin, raw gap.
    data["y"] = (data["primary_alternative"] - data["primary_base"])[..., [2, 1, 0]]
    return data


def design_inputs(data, model):
    X = data["primary_X"]
    names = data["primary_feature_names"].tolist()
    if model.endswith("context"):
        ci = data["content_names"].index("full_surrogate_abs_gap")
        X = np.concatenate((X, data["content_X"][..., ci:ci+1]), axis=-1)
        names += ["full_surrogate_abs_gap"]
    if model.endswith("content"):
        X = np.concatenate((X, data["content_X"]), axis=-1)
        names += data["content_names"]
    if model.startswith("difficulty"):
        # These outcome-informed controls are only for explanation, never routing.
        full = data["full_raw_gap"]
        diagnostic = np.column_stack((full, full ** 2))
        X = np.concatenate((X, np.broadcast_to(diagnostic, (*X.shape[:2], 2))), axis=-1)
        names += ["full_raw_gap", "full_raw_gap_squared"]
    return X, names


def fit_panel(X, y, w):
    runs, rows, features = X.shape
    return fit_weighted_ridge(X.reshape(runs * rows, features), y.reshape(runs * rows, -1),
                              np.tile(w / runs, runs))


def fit_model(model, X, y, w):
    if not model.startswith("tree"):
        return fit_panel(X, y, w)
    flat_X, flat_y = X.reshape(-1, X.shape[-1]), y.reshape(-1, y.shape[-1])
    weights = np.tile(w / w.mean(), len(X))
    return [HistGradientBoostingRegressor(**TREE_PARAMS).fit(flat_X, flat_y[:, qi], sample_weight=weights)
            for qi in range(y.shape[-1])]


def predict_panel(fit, X):
    flat = X.reshape(-1, X.shape[-1])
    if isinstance(fit, list):
        prediction = np.column_stack([model.predict(flat) for model in fit])
    else:
        prediction = predict_ridge(fit, flat)
    return prediction.reshape(*X.shape[:2], -1)


def weighted_quantile(values, weights, quantiles):
    values, weights = np.asarray(values).ravel(), np.asarray(weights).ravel()
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) / weights.sum()
    return np.interp(quantiles, cumulative, values)


def assign_tertiles(training, held, weights):
    """Use training-only weighted thresholds; ties can leave a group empty."""
    cuts = weighted_quantile(training, np.broadcast_to(weights, training.shape), [1/3, 2/3])
    group = np.zeros(held.shape, dtype=np.int8)
    group[held < cuts[0]] = -1
    group[held > cuts[1]] = 1
    return group, cuts


def serial_fit(fit):
    if isinstance(fit, list):
        return dict(parameters=TREE_PARAMS, iterations=[int(model.n_iter_) for model in fit])
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in fit.items()}


def out_of_fold(data, risk_names):
    y, w = data["y"], data["weights"]
    predictions = np.empty((len(MODELS), *y.shape))
    group_names = ["ridge_predicted_raw_gain", "tree_predicted_raw_gain"] + list(risk_names)
    groups = np.zeros((len(group_names), *y.shape[:2]), dtype=np.int8)
    image_fold, task_fold = two_way_folds(data["image_ids"], data["task_ids"])
    inputs = {model: design_inputs(data, model) for model in MODELS[1:]}
    cells, fits = [], []
    coverage = np.zeros(len(w), int)
    for i, j, train, held in iter_two_way_splits(image_fold, task_fold):
        assert train.any() and held.any()
        coverage[held] += 1
        mean = w[train] @ y[:, train].mean(axis=0) / w[train].sum()
        predictions[0][:, held] = mean
        cell = dict(image_fold=i, task_fold=j, train_rows=int(train.sum()), held_rows=int(held.sum()))
        for mi, model in enumerate(MODELS[1:], start=1):
            X, names = inputs[model]
            fit = fit_model(model, X[:, train], y[:, train], w[train])
            predictions[mi][:, held] = predict_panel(fit, X[:, held])
            fits.append(dict(image_fold=i, task_fold=j, model=model, features=names, **serial_fit(fit)))
            if model in ("content", "tree_content"):
                gi = 0 if model == "content" else 1
                training_scores = predict_panel(fit, X[:, train])[..., 0]
                groups[gi][:, held], cuts = assign_tertiles(training_scores, predictions[mi][:, held, 0], w[train])
                cell[group_names[gi] + "_thresholds"] = cuts.tolist()
        for gi, name in enumerate(risk_names, start=2):
            score = data["content_X"][..., data["content_names"].index(name)]
            groups[gi][:, held], cuts = assign_tertiles(score[:, train], score[:, held], w[train])
            cell[name + "_thresholds"] = cuts.tolist()
        cells.append(cell)
    np.testing.assert_array_equal(coverage, 1)
    return dict(predictions=predictions, groups=groups, group_names=group_names, fits=fits,
                cells=cells, image_fold=image_fold, task_fold=task_fold)


def transfer(source, target, risk_names):
    y, w = source["y"], source["weights"]
    predictions = np.empty((len(MODELS), *target["y"].shape))
    predictions[0] = w @ y.mean(axis=0) / w.sum()
    group_names = ["ridge_predicted_raw_gain", "tree_predicted_raw_gain"] + list(risk_names)
    groups = np.zeros((len(group_names), *target["y"].shape[:2]), dtype=np.int8)
    fits, thresholds = [], {}
    for mi, model in enumerate(MODELS[1:], start=1):
        X, names = design_inputs(source, model)
        target_X, target_names = design_inputs(target, model)
        assert names == target_names
        fit = fit_model(model, X, y, w)
        predictions[mi] = predict_panel(fit, target_X)
        fits.append(dict(model=model, features=names, **serial_fit(fit)))
        if model in ("content", "tree_content"):
            gi = 0 if model == "content" else 1
            groups[gi], cuts = assign_tertiles(predict_panel(fit, X)[..., 0], predictions[mi, ..., 0], w)
            thresholds[group_names[gi]] = cuts.tolist()
    for gi, name in enumerate(risk_names, start=2):
        x = source["content_X"][..., source["content_names"].index(name)]
        held = target["content_X"][..., target["content_names"].index(name)]
        groups[gi], cuts = assign_tertiles(x, held, w)
        thresholds[name] = cuts.tolist()
    return dict(predictions=predictions, groups=groups, group_names=group_names,
                fits=fits, cells=[thresholds])


def contributions(data, result):
    """Return additive per-run/per-row summaries for image-cluster resampling."""
    y, ids, w = data["y"], data["image_ids"], data["weights"]
    centered_y = np.stack([weighted_center(run, ids, w) for run in y])
    names, values = ["weight"], [np.ones(y.shape[:2])]
    for qi, metric in enumerate(OUTCOMES):
        names.append("within_signal_" + metric)
        values.append(centered_y[..., qi] ** 2)
    for mi, model in enumerate(MODELS):
        prediction = result["predictions"][mi]
        centered_prediction = np.stack([weighted_center(run, ids, w) for run in prediction])
        for qi, metric in enumerate(OUTCOMES):
            names += ["mse_" + model + "_" + metric, "within_mse_" + model + "_" + metric]
            values += [(y[..., qi] - prediction[..., qi]) ** 2,
                       (centered_y[..., qi] - centered_prediction[..., qi]) ** 2]
    for gi, group_name in enumerate(result["group_names"]):
        for side, code in (("low", -1), ("high", 1)):
            selected = (result["groups"][gi] == code).astype(float)
            prefix = "group_" + group_name + "_" + side
            names.append(prefix + "_weight")
            values.append(selected)
            for qi, metric in enumerate(OUTCOMES):
                names += [prefix + "_" + metric, prefix + "_centered_" + metric]
                values += [selected * y[..., qi], selected * centered_y[..., qi]]
    return names, np.stack(values, axis=-1)


def derive_metrics(sums, names, group_names):
    index = {key: i for i, key in enumerate(names)}
    get = lambda name: sums[..., index[name]]
    metrics = {}
    for outcome in OUTCOMES:
        null = get("mse_mean_" + outcome)
        within = get("within_signal_" + outcome)
        for model in MODELS[1:]:
            metrics[model + "_r2_" + outcome] = 1 - get("mse_" + model + "_" + outcome) / null
        pairs = (("", "context", "content"), ("difficulty_", "difficulty_context", "difficulty_content"),
                 ("tree_", "tree_context", "tree_content"), ("over_statistics_", "statistics", "content"))
        for prefix, base, content in pairs:
            metrics[prefix + "content_increment_r2_" + outcome] = (
                get("mse_" + base + "_" + outcome) - get("mse_" + content + "_" + outcome)) / null
            metrics[prefix + "content_increment_within_r2_" + outcome] = (
                get("within_mse_" + base + "_" + outcome) - get("within_mse_" + content + "_" + outcome)) / within
    for group in group_names:
        prefix = "group_" + group
        for side in ("low", "high"):
            denominator = get(prefix + "_" + side + "_weight")
            metrics[prefix + "_" + side + "_fraction"] = denominator / get("weight")
            for outcome in OUTCOMES:
                for centered in ("", "centered_"):
                    name = prefix + "_" + side + "_" + centered + outcome
                    metrics[name] = np.divide(get(name), denominator,
                        out=np.full(np.broadcast_shapes(get(name).shape, denominator.shape), np.nan), where=denominator > 0)
        for outcome in OUTCOMES:
            for centered in ("", "centered_"):
                suffix = centered + outcome
                metrics[prefix + "_high_minus_low_" + suffix] = (
                    metrics[prefix + "_high_" + suffix] - metrics[prefix + "_low_" + suffix])
    return metrics


def summarize(data, result, replicates=2000):
    names, values = contributions(data, result)
    w = data["weights"]
    per_run_sums = np.einsum("n,snk->sk", w, values)
    point = derive_metrics(per_run_sums.mean(axis=0), names, result["group_names"])
    per_run = derive_metrics(per_run_sums, names, result["group_names"])
    unique, inverse = np.unique(data["image_ids"], return_inverse=True)
    image_sums = np.zeros((len(unique), len(names)))
    np.add.at(image_sums, inverse, w[:, None] * values.mean(axis=0))
    rng = np.random.default_rng(2621)
    draws = np.empty((replicates, len(names)))
    for start in range(0, replicates, 100):
        ix = rng.integers(len(unique), size=(min(100, replicates-start), len(unique)))
        draws[start:start+len(ix)] = image_sums[ix].sum(axis=1)
    boot = derive_metrics(draws, names, result["group_names"])
    estimates = {}
    for name, value in point.items():
        estimate = dict(mean=float(value), ci95=np.quantile(boot[name], [.025, .975]).tolist(),
                        four_setting_adjusted_ci95=np.quantile(boot[name], [.025/4, 1-.025/4]).tolist())
        # A transfer threshold can select no examples on one side; record that fact.
        estimates[name] = {key: (None if isinstance(val, float) and not np.isfinite(val) else val)
                           for key, val in estimate.items()}
    run_records = [dict(seed=int(seed), **{key: float(value[i]) for key, value in per_run.items()})
                   for i, seed in enumerate(data["primary_run_ids"])]
    return dict(estimates=estimates, per_run=run_records), image_sums, names


def save_panel(output, data, result, replicates, context):
    output.mkdir(parents=True, exist_ok=True)
    summary, image_sums, contribution_names = summarize(data, result, replicates)
    summary.update(context, rows=len(data["weights"]), images=len(set(data["image_ids"])),
                   tasks=len(set(data["task_ids"])), group_names=result["group_names"],
                   content_features=data["content_names"], outcomes=OUTCOMES)
    json_write(output / "summary.json", summary)
    json_write(output / "fits.json", result["fits"])
    json_write(output / "folds.json", result["cells"])
    np.savez_compressed(output / "predictions.npz", predictions=result["predictions"],
        groups=result["groups"], group_names=result["group_names"], image_ids=data["image_ids"],
        task_ids=data["task_ids"], weights=data["weights"], outcomes=data["y"],
        models=MODELS, outcome_names=OUTCOMES, image_sums=image_sums,
        contribution_names=contribution_names, run_ids=data["primary_run_ids"])
    return summary


def routing_summary(data, predictions, replicates=2000):
    """Choose C iff the held-out accuracy-gain prediction is positive."""
    alternative = data["primary_alternative"]
    baseline = data["primary_base"]
    train_choice = predictions[0, ..., OUTCOMES.index("accuracy")] > 0
    fixed = np.where(train_choice[..., None], alternative, baseline)
    values, names, choices = [np.ones(baseline.shape[:2])], ["weight"], []
    for learner, name in (("ridge", "content"), ("tree", "tree_content")):
        choice = predictions[MODELS.index(name), ..., OUTCOMES.index("accuracy")] > 0
        choices.append(choice)
        routed = np.where(choice[..., None], alternative, baseline)
        for comparison, reference in (("C", alternative), ("U", baseline), ("training_fixed", fixed)):
            for qi, metric in enumerate(("accuracy", "geom", "raw")):
                names.append(learner + "_route_minus_" + comparison + "_" + metric)
                values.append(routed[..., qi] - reference[..., qi])
        names.append(learner + "_choose_C_fraction")
        values.append(choice.astype(float))
    values = np.stack(values, axis=-1)
    per_run = np.einsum("n,snk->sk", data["weights"], values)
    mean = per_run.mean(axis=0)
    unique, inverse = np.unique(data["image_ids"], return_inverse=True)
    image_sums = np.zeros((len(unique), len(names)))
    np.add.at(image_sums, inverse, data["weights"][:, None] * values.mean(axis=0))
    rng = np.random.default_rng(2621)
    draws = np.empty((replicates, len(names)))
    for start in range(0, replicates, 100):
        ix = rng.integers(len(unique), size=(min(100, replicates-start), len(unique)))
        draws[start:start+len(ix)] = image_sums[ix].sum(axis=1)
    draws /= draws[:, :1]
    estimates = {name: dict(mean=float(mean[i]/mean[0]),
        ci95=np.quantile(draws[:, i], [.025, .975]).tolist(),
        four_setting_adjusted_ci95=np.quantile(draws[:, i], [.025/4, 1-.025/4]).tolist())
        for i, name in enumerate(names) if name != "weight"}
    runs = [dict(seed=int(seed), **{name: float(per_run[si, i]/per_run[si, 0])
                for i, name in enumerate(names) if name != "weight"})
            for si, seed in enumerate(data["primary_run_ids"])]
    return dict(estimates=estimates, per_run=runs), np.stack(choices), image_sums, names


def run_routing(args):
    data = {setting: load_data(args.inputs / setting / "inputs.npz", args.features / (setting + ".npz"))
            for setting in SETTINGS}
    paths = [args.output / setting for setting in SETTINGS]
    paths += sorted((args.output / "transfer").glob("*_to_*"))
    summaries, rows = [], []
    for path in paths:
        context = json.loads((path / "summary.json").read_text())
        current = data[context["setting"]]
        with np.load(path / "predictions.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["models"], MODELS)
            predictions = archive["predictions"]
        summary, choices, image_sums, names = routing_summary(current, predictions, args.replicates)
        summary.update({key: context[key] for key in ("mode", "setting", "source")})
        summary["decision"] = "predicted accuracy gain > 0; use C, otherwise U"
        json_write(path / "routing_summary.json", summary)
        np.savez_compressed(path / "routing.npz", choices=choices, learners=["ridge", "tree"],
            image_sums=image_sums, metric_names=names, image_ids=current["image_ids"],
            task_ids=current["task_ids"], run_ids=current["primary_run_ids"])
        summaries.append(summary)
        for metric, value in summary["estimates"].items():
            rows.append(dict(mode=summary["mode"], setting=summary["setting"], source=summary["source"],
                metric=metric, mean=value["mean"], ci_low=value["ci95"][0], ci_high=value["ci95"][1],
                adjusted_low=value["four_setting_adjusted_ci95"][0], adjusted_high=value["four_setting_adjusted_ci95"][1]))
    json_write(args.output / "routing_summary.json", summaries)
    csv_write(args.output / "routing_summary.csv", rows)


def main(args):
    if args.routing_only:
        run_routing(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    data = {setting: load_data(args.inputs / setting / "inputs.npz", args.features / (setting + ".npz"))
            for setting in SETTINGS}
    results, table = [], []
    for setting in SETTINGS:
        result = out_of_fold(data[setting], args.risk_features)
        results.append(save_panel(args.output / setting, data[setting], result, args.replicates,
                                  dict(mode="double_holdout", setting=setting, source="")))
        print(setting + " double holdout complete", flush=True)
    for budget in (49, 98):
        for left, right in (("pets", "flowers"), ("flowers", "pets")):
            source, target = f"{left}_b16_{budget}", f"{right}_b16_{budget}"
            result = transfer(data[source], data[target], args.risk_features)
            results.append(save_panel(args.output / "transfer" / (source + "_to_" + target),
                data[target], result, args.replicates, dict(mode="cross_dataset", setting=target, source=source)))
            print(source + " to " + target + " complete", flush=True)
    for summary in results:
        for metric, value in summary["estimates"].items():
            table.append(dict(mode=summary["mode"], setting=summary["setting"], source=summary["source"],
                              metric=metric, mean=value["mean"], ci_low=value["ci95"][0], ci_high=value["ci95"][1],
                              adjusted_low=value["four_setting_adjusted_ci95"][0],
                              adjusted_high=value["four_setting_adjusted_ci95"][1]))
    csv_write(args.output / "summary.csv", table)
    json_write(args.output / "summary.json", results)
    json_write(args.output / "execution.json", dict(elapsed_seconds=time.perf_counter()-started,
        ridge=.001, tree_params=TREE_PARAMS, bootstrap_replicates=args.replicates, bootstrap_seed=2621,
        fold_seeds=[2351,2352], models=MODELS, risk_features=args.risk_features,
        double_holdout_panels=4, transfer_panels=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("results/conditioning-benefit-v1"))
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--risk-features", nargs="*", default=[])
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--routing-only", action="store_true")
    main(parser.parse_args())
