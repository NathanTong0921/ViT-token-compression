"""One exploratory task/difficulty adjustment of mixed-class semantic effects.

Reads only saved validation predictions. No fitting of a new selector, GPU use,
additional mappings, model selection or final-test evaluation.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def weighted_quantiles(values, weights, probabilities):
    order = np.argsort(values, kind="stable")
    x, w = values[order], weights[order]
    return np.interp(probabilities, (np.cumsum(w) - .5 * w) / w.sum(), x)


def group_mean(values, weights, group, count):
    mass = np.bincount(group, weights=weights, minlength=count)
    total = np.bincount(group, weights=weights * values, minlength=count)
    return np.divide(total, mass, out=np.zeros(count), where=mass > 0), mass


def adjusted_effect(target, is_new, margin, weights, task_ids, image_ids):
    """WLS task fixed effects plus a fixed quadratic, image-cluster sandwich CI."""
    _, tasks = np.unique(task_ids, return_inverse=True)
    _, images = np.unique(image_ids, return_inverse=True)
    task_count, image_count = tasks.max() + 1, images.max() + 1
    center = np.average(margin, weights=weights)
    scale = np.sqrt(np.average((margin - center) ** 2, weights=weights))
    z = (margin - center) / scale
    x = np.column_stack((is_new.astype(float), z, z * z))
    within_x = np.column_stack([x[:, col] - group_mean(x[:, col], weights, tasks, task_count)[0][tasks]
                                for col in range(x.shape[1])])
    within_y = target - group_mean(target, weights, tasks, task_count)[0][tasks]
    gram = within_x.T @ (weights[:, None] * within_x)
    inverse = np.linalg.inv(gram)
    beta = inverse @ (within_x.T @ (weights * within_y))
    residual = within_y - within_x @ beta
    scores = np.zeros((image_count, 3))
    np.add.at(scores, images, weights[:, None] * within_x * residual[:, None])
    correction = image_count / (image_count - 1) * (len(target) - 1) / (len(target) - task_count - 3)
    covariance = correction * inverse @ (scores.T @ scores) @ inverse
    se = np.sqrt(np.diag(covariance))
    raw = np.average(target[is_new], weights=weights[is_new]) - np.average(target[~is_new], weights=weights[~is_new])
    return dict(raw_true_new_minus_known=float(raw), adjusted_true_new_minus_known=float(beta[0]),
                ci95_normal_cluster=[float(beta[0] - 1.96 * se[0]), float(beta[0] + 1.96 * se[0])],
                cluster_standard_error=float(se[0]), coefficients=beta.tolist(),
                column_order=["true_new", "standardized_full_margin", "standardized_full_margin_squared"],
                full_margin_center=float(center), full_margin_scale=float(scale),
                design_condition_number=float(np.linalg.cond(gram)), tasks=int(task_count),
                images=int(image_count), rows=len(target))


def overlap_standardization(targets, is_new, margin, weights, task_ids):
    """Ten pooled weighted-quantile bins; exact task/bin cells common to both sides."""
    edges = np.unique(weighted_quantiles(margin, weights, np.linspace(0, 1, 11)))
    bins = np.digitize(margin, edges[1:-1])
    bin_count = len(edges) - 1
    _, tasks = np.unique(task_ids, return_inverse=True)
    cell = tasks * bin_count + bins
    cell_count = (tasks.max() + 1) * bin_count
    group = 2 * cell + is_new
    masses = np.bincount(group, weights=weights, minlength=2 * cell_count).reshape(-1, 2)
    common = (masses > 0).all(1)
    common_weight = np.minimum(masses[:, 0], masses[:, 1])
    common_total = common_weight.sum()
    common_rows = common[cell]
    details = []
    for index in range(bin_count):
        chosen = bins == index
        side_details = {}
        for side, label in ((False, "true_known"), (True, "true_new")):
            mask = chosen & (is_new == side)
            side_details[label] = dict(rows=int(mask.sum()),
                fraction_of_side_weight=float(weights[mask].sum() / weights[is_new == side].sum()),
                mean_full_margin=float(np.average(margin[mask], weights=weights[mask])) if mask.any() else None)
        details.append(dict(lower=float(edges[index]), upper=float(edges[index + 1]), **side_details))
    effects = {}
    for name, target in {**targets, "full_margin_balance": margin}.items():
        mean, _ = group_mean(target, weights, group, 2 * cell_count)
        means = mean.reshape(-1, 2)
        known, new = (float(np.sum(common_weight * means[:, side]) / common_total) for side in (0, 1))
        effects[name] = dict(standardized_true_known=known, standardized_true_new=new,
                             standardized_true_new_minus_known=new - known)
    return dict(bin_policy="10 pooled weighted-quantile bins, fixed before analysis; task-by-bin cells retained only when both true-class sides occur; cell weight = minimum side mass.",
                uncertainty="Descriptive standardization point estimates; no confidence interval or rematching search.",
                bins=details, shared_task_bin_cells=int(common.sum()),
                tasks_with_shared_cells=int(len(np.unique(tasks[common_rows]))),
                total_tasks=int(tasks.max() + 1),
                common_rows=int(common_rows.sum()), total_rows=len(margin),
                fraction_rows_in_common_cells=float(common_rows.mean()),
                fraction_weight_in_common_cells=float(weights[common_rows].sum() / weights.sum()),
                overlap_mass_fraction=float(2 * common_total / weights.sum()), metrics=effects)


def analyze_dataset(root, dataset):
    folder = root / f"{dataset}-semantic-identity-holdout-expanded-v1"
    protocol = json.loads((folder / "protocol.json").read_text())
    with np.load(folder / "evaluation_panel.npz") as saved:
        panel = {key: saved[key].copy() for key in saved.files}
    mask = panel["strata"] == "known_new"
    effects = {metric: [] for metric in ("correct", "margin")}
    permutations = [arm for arm in protocol["arms"] if arm.startswith("permuted_")]
    for seed in protocol["training_seeds"]:
        with np.load(folder / f"seed_{seed}_real_text.npz") as saved:
            true = {metric: saved[metric][mask].astype(float) for metric in effects}
        permuted = {metric: [] for metric in effects}
        for arm in permutations:
            with np.load(folder / f"seed_{seed}_{arm}.npz") as saved:
                for metric in effects:
                    permuted[metric].append(saved[metric][mask].astype(float))
        for metric in effects:
            effects[metric].append(true[metric] - np.mean(permuted[metric], axis=0))
    targets = {metric: np.mean(values, axis=0) * (100 if metric == "correct" else 1)
               for metric, values in effects.items()}
    is_new = np.isin(panel["class_ids"][mask], protocol["held_classes"])
    margin, weights, task_ids, image_ids = (panel[key][mask] for key in ("full_margin", "weights", "task_ids", "image_ids"))
    regression = {metric: adjusted_effect(values, is_new, margin, weights, task_ids, image_ids)
                  for metric, values in targets.items()}
    return dict(source=str(folder), seeds=protocol["training_seeds"], permutations=permutations,
                final_test_used=False, units={"correct": "percentage points", "margin": "geometric margin"},
                regression=regression, common_support=overlap_standardization(targets, is_new, margin, weights, task_ids))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = dict(stage="Exploratory difficulty follow-up; fixed single specification, no causal interpretation.",
                  specification="Image-task real-minus-permutation effects averaged across all 3 seeds and 2 mappings. WLS: true-new indicator + quadratic full-model geometric margin + task fixed effects. Equal task/class weighting; CR1 image-cluster sandwich with normal 95% intervals.",
                  datasets={dataset: analyze_dataset(args.runs_root, dataset) for dataset in ("pets", "flowers")})
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
