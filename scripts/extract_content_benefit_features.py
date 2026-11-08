"""Fit a shared prefix readout and extract content features for four dev panels."""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from token_compression.content_benefit import (
    FEATURE_NAMES, content_features, fit_content_readout, normalize_rows,
)


DATASETS = ("pets", "flowers")
SEEDS = (2041, 2042, 2043)


def evidence(root, dataset, split):
    folder = root / f"runs/{dataset}-selector-budget-98-supervision-v1" / split
    prefix = np.load(folder / "prefix_tokens.npy", mmap_mode="r")
    with np.load(folder / "evidence.npz") as archive:
        data = {key: archive[key].copy() for key in
                ("image_ids", "labels", "text_features", "full_features", "full_scores")}
    assert prefix.shape == (len(data["image_ids"]), 197, 768)
    return prefix, data


def patch_means(prefix, batch=32):
    return np.concatenate([prefix[start:start + batch, 1:].astype(np.float64).mean(axis=1)
                           for start in range(0, len(prefix), batch)])


def binary_quality(predicted, actual, labels, pairs, text, weights):
    direction = normalize_rows(text[pairs[:, 0]] - text[pairs[:, 1]])
    pred_gap = np.sum(predicted * direction, axis=1)
    true_gap = np.sum(actual * direction, axis=1)
    side = np.where(labels == pairs[:, 0], 1., -1.)
    weights = weights / weights.sum()
    pm, tm = weights @ pred_gap, weights @ true_gap
    cov = weights @ ((pred_gap - pm) * (true_gap - tm))
    corr = cov / np.sqrt((weights @ (pred_gap-pm)**2) * (weights @ (true_gap-tm)**2))
    return dict(surrogate_accuracy=float(weights @ ((pred_gap * side) > 0)),
                full_accuracy=float(weights @ ((true_gap * side) > 0)),
                prediction_agreement=float(weights @ ((pred_gap >= 0) == (true_gap >= 0))),
                signed_gap_correlation=float(corr))


def main(args):
    start = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    training, means, targets, weights = {}, [], [], []
    for dataset in DATASETS:
        prefix, data = evidence(args.root, dataset, "train")
        mean = patch_means(prefix)
        means.append(mean)
        targets.append(data["full_features"].astype(np.float64))
        weights.append(np.full(len(mean), .5 / len(mean)))
        training[dataset] = dict(images=len(mean), dataset_weight=.5,
                                 ids=data["image_ids"].tolist())
    fit = fit_content_readout(np.concatenate(means), np.concatenate(targets),
                             np.concatenate(weights), ridge=.01)
    np.savez_compressed(args.output / "shared_readout.npz", **fit)
    constant = normalize_rows(np.asarray(fit["intercept"])[None])[0]
    quality = {"training": training, "ridge": .01,
               "readout": "dataset-balanced training-image mean patch prefix to full CLIP embedding",
               "feature_names": list(FEATURE_NAMES), "datasets": {}}
    for dataset in DATASETS:
        prefix, data = evidence(args.root, dataset, "val")
        training_ids = set(training[dataset]["ids"])
        assert not training_ids.intersection(data["image_ids"].tolist())
        predicted = normalize_rows(patch_means(prefix) @ fit["matrix"] + fit["bias"])
        actual = normalize_rows(data["full_features"])
        dq = dict(images=len(prefix), embedding_cosine_mean=float(np.mean(np.sum(predicted*actual, axis=1))),
                  constant_embedding_cosine_mean=float(np.mean(actual @ constant)),
                  ordinary_class_accuracy=float(np.mean((predicted @ data["text_features"].T).argmax(axis=1)==data["labels"])),
                  full_ordinary_class_accuracy=float(np.mean(data["full_scores"].argmax(axis=1)==data["labels"])),
                  task_quality={})
        for keep in (49, 98):
            setting = f"{dataset}_b16_{keep}"
            source = args.root / "runs/core-protection-v1" / setting
            with (source / "examples.csv").open() as file:
                rows = list(csv.DictReader(file))
            indices = np.asarray([int(row["image_index"]) for row in rows])
            ids = np.asarray([row["image_id"] for row in rows])
            tasks = np.asarray([row["task_id"] for row in rows])
            pairs = np.asarray([[int(row["class_a"]), int(row["class_b"])] for row in rows])
            labels = np.asarray([int(row["class_id"]) for row in rows])
            weights = np.asarray([float(row["task_class_weight"]) for row in rows])
            np.testing.assert_array_equal(ids, data["image_ids"][indices])
            np.testing.assert_array_equal(labels, data["labels"][indices])
            with np.load(args.root / "runs/conditioning-benefit-v1" / setting / "inputs.npz") as old:
                np.testing.assert_array_equal(ids, old["image_ids"])
                np.testing.assert_array_equal(tasks, old["task_ids"])
            directions = normalize_rows(data["text_features"][pairs[:, 0]]-data["text_features"][pairs[:, 1]])
            dq["task_quality"][str(keep)] = binary_quality(predicted[indices], actual[indices],
                                                          labels, pairs, data["text_features"], weights)
            xs = []
            for seed in SEEDS:
                with np.load(source / f"seed_{seed}.npz") as archive:
                    variants = archive["variants"].tolist()
                    selected = archive["patch_indices"]
                    c = selected[variants.index("conditioned")]
                    u = selected[variants.index("wide")]
                x = np.empty((len(rows), len(FEATURE_NAMES)))
                for pos in range(0, len(rows), args.batch_size):
                    end = pos + args.batch_size
                    x[pos:end] = content_features(prefix[indices[pos:end], 1:], c[pos:end], u[pos:end],
                                                  directions[pos:end], fit["matrix"], fit["bias"])
                assert np.isfinite(x).all()
                xs.append(x)
            np.savez_compressed(args.output / f"{setting}.npz", X=np.stack(xs),
                                feature_names=FEATURE_NAMES, image_ids=ids, task_ids=tasks,
                                run_ids=SEEDS, weights=weights)
            print(json.dumps({"setting": setting, "rows": len(rows), "feature_shape": np.stack(xs).shape,
                              "quality": dq["task_quality"][str(keep)]}), flush=True)
        quality["datasets"][dataset] = dq
    quality["elapsed_seconds"] = time.perf_counter() - start
    (args.output / "readout_quality.json").write_text(json.dumps(quality, indent=2)+"\n")
    print(json.dumps({"elapsed_seconds": quality["elapsed_seconds"],
                      "cosine_quality": {d: quality["datasets"][d]["embedding_cosine_mean"] for d in DATASETS}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    main(parser.parse_args())
