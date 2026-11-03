"""Fixed development protocol for semantic versus identity conditioning."""

import numpy as np

from token_compression.selector_scale import balanced_training_stream


def class_partition(class_count, seed=2099, fraction=.25):
    held = np.sort(np.random.default_rng(seed).permutation(class_count)[:int(np.ceil(class_count * fraction))])
    seen = np.setdiff1d(np.arange(class_count), held)
    return seen, held


def class_permutation(class_count, seed, partitions=None):
    """A stable identity code; each partition retains exactly its vector bank."""
    mapping = np.arange(class_count)
    rng = np.random.default_rng(seed)
    for group in ([mapping.copy()] if partitions is None else partitions):
        group = np.asarray(group)
        mapping[group] = rng.permutation(group)
    return mapping


def training_membership(labels, pairs, seen_classes):
    return np.isin(labels, seen_classes), np.isin(pairs, seen_classes).all(axis=1)


def pair_strata(pairs, held_classes):
    counts = np.isin(pairs, held_classes).sum(axis=1)
    return np.asarray(["known_known", "known_new", "new_new"])[counts]


def fixed_step_stream(pairs, total_draws, seed):
    """Retain the original exact stream when divisible; otherwise balance to one."""
    group_count = 2 * len(pairs)
    if total_draws % group_count == 0:
        return balanced_training_stream(pairs, total_draws, seed)
    rng = np.random.default_rng(seed)
    complete, remainder = divmod(total_draws, group_count)
    groups = np.concatenate((np.tile(np.arange(group_count), complete),
                             rng.permutation(group_count)[:remainder]))
    rng.shuffle(groups)
    return {"pair_indices": groups // 2, "sides": groups % 2,
            "image_quantiles": rng.random(total_draws)}


def mapped_candidate_text(text, pairs, mapping):
    """Map only selector inputs, never the teacher or final classifier dictionary."""
    return text[mapping[pairs]]


def weighted_image_interval(image_ids, weights, values, replicates=2000, seed=2100):
    """Paired image bootstrap conditional on the fixed classes/tasks/models."""
    weights = np.asarray(weights, np.float64)
    values = np.asarray(values, np.float64)
    _, inverse = np.unique(image_ids, return_inverse=True)
    n = inverse.max() + 1
    numer = np.bincount(inverse, weights=weights * values, minlength=n)
    denom = np.bincount(inverse, weights=weights, minlength=n)
    rng = np.random.default_rng(seed)
    boot = np.empty(replicates)
    for start in range(0, replicates, 100):
        indices = rng.integers(n, size=(min(100, replicates - start), n))
        boot[start:start + len(indices)] = numer[indices].sum(1) / denom[indices].sum(1)
    return {"estimate": float(numer.sum() / denom.sum()),
            "ci95": np.percentile(boot, [2.5, 97.5]).tolist(),
            "images": int(n), "examples": len(values)}


def paired_stratum_interaction(image_ids, weights, values, strata, first, second,
                               replicates=2000, seed=2100):
    """Difference of paired effects with shared image resamples across strata."""
    selected = np.isin(strata, [first, second])
    ids, w, v, s = [np.asarray(x)[selected] for x in (image_ids, weights, values, strata)]
    _, inverse = np.unique(ids, return_inverse=True)
    n = inverse.max() + 1
    numerator, denominator = [], []
    for group in (first, second):
        mask = s == group
        numerator.append(np.bincount(inverse[mask], weights=w[mask] * v[mask], minlength=n))
        denominator.append(np.bincount(inverse[mask], weights=w[mask], minlength=n))
    rng = np.random.default_rng(seed)
    boot = []
    for start in range(0, replicates, 100):
        ix = rng.integers(n, size=(min(100, replicates - start), n))
        a = numerator[0][ix].sum(1) / denominator[0][ix].sum(1)
        b = numerator[1][ix].sum(1) / denominator[1][ix].sum(1)
        boot.extend(a - b)
    return {"contrast": f"{first} minus {second}",
            "estimate": float(numerator[0].sum() / denominator[0].sum()
                              - numerator[1].sum() / denominator[1].sum()),
            "ci95": np.percentile(boot, [2.5, 97.5]).tolist()}
