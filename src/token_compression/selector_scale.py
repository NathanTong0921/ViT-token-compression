"""Sampling and weighting for the fixed-update image-coverage experiment."""

import numpy as np


def validate_image_partition(train_ids, val_ids, small_ids):
    """Verify that the smaller training panel is nested and validation is held out."""
    train_ids, val_ids, small_ids = map(list, (train_ids, val_ids, small_ids))
    assert len(set(train_ids)) == len(train_ids)
    assert len(set(val_ids)) == len(val_ids)
    assert len(set(small_ids)) == len(small_ids)
    assert set(small_ids) <= set(train_ids)
    assert not set(train_ids) & set(val_ids)


def balanced_example_weights(labels, class_pairs, image_indices, pair_indices):
    """Give each task and each of its two classes equal total evaluation weight.

    Images within a task/class receive equal weight even when class counts differ.
    The supplied example list must contain both classes of every supplied task.
    """
    labels, pairs = np.asarray(labels), np.asarray(class_pairs)
    image_indices, pair_indices = np.asarray(image_indices), np.asarray(pair_indices)
    example_labels = labels[image_indices]
    assert np.all((pairs[pair_indices] == example_labels[:, None]).any(axis=1))
    sides = (example_labels == pairs[pair_indices, 1]).astype(np.int64)
    groups = 2 * pair_indices + sides
    counts = np.bincount(groups, minlength=2 * len(pairs))
    assert np.all(counts > 0)
    return 1.0 / (2 * len(pairs) * counts[groups])


def balanced_training_stream(class_pairs, total_draws, seed):
    """Shuffle equal task/class repetitions, then draw uniform within-class ranks.

    One CPU NumPy PCG64 generator determines the whole stream. Reusing its task,
    side, and image quantile arrays pairs coverage conditions before fitting.
    """
    pairs = np.asarray(class_pairs)
    group_count = 2 * len(pairs)
    assert total_draws > 0 and total_draws % group_count == 0
    generator = np.random.default_rng(seed)
    groups = np.tile(np.arange(group_count), total_draws // group_count)
    generator.shuffle(groups)
    return {"pair_indices": groups // 2,
            "sides": groups % 2,
            "image_quantiles": generator.random(total_draws)}


def map_stream_images(labels, image_ids, stream, class_pairs, allowed_image_ids=None):
    """Map shared image quantiles to each coverage's alphabetically ordered IDs."""
    labels, image_ids = np.asarray(labels), np.asarray(image_ids)
    pairs = np.asarray(class_pairs)
    allowed = (np.ones(len(labels), dtype=bool) if allowed_image_ids is None
               else np.isin(image_ids, list(allowed_image_ids)))
    draw_labels = pairs[stream["pair_indices"], stream["sides"]]
    quantiles = np.asarray(stream["image_quantiles"])
    assert np.all((quantiles >= 0) & (quantiles < 1))
    result = np.empty(len(draw_labels), dtype=np.int64)
    for label in np.unique(pairs):
        pool = np.flatnonzero(allowed & (labels == label))
        assert len(pool) > 0
        pool = pool[np.argsort(image_ids[pool], kind="stable")]
        selected = np.flatnonzero(draw_labels == label)
        ranks = (quantiles[selected] * len(pool)).astype(np.int64)
        result[selected] = pool[ranks]
    return result
