"""Offline patch-ranking utilities and equal-size candidate selection banks."""

import random
from collections import defaultdict

import numpy as np


def deletion_windows(grid_size=14, window_size=2):
    """Return stride-one square windows as row-major patch indices."""
    if not 1 <= window_size <= grid_size:
        raise ValueError("window_size must be between one and grid_size")
    grid = np.arange(grid_size * grid_size).reshape(grid_size, grid_size)
    return np.asarray([
        grid[row:row + window_size, col:col + window_size].reshape(-1)
        for row in range(grid_size - window_size + 1)
        for col in range(grid_size - window_size + 1)
    ])


def window_to_patch_scores(window_scores, windows, num_patches=196):
    """Average a window's utility over the windows covering each patch.

    Columns, when present, represent separate classification objectives.
    A window's utility is assigned to every patch it contains before averaging.
    """
    values = np.asarray(window_scores, dtype=np.float64)
    windows = np.asarray(windows)
    totals = np.zeros((num_patches,) + values.shape[1:], dtype=np.float64)
    counts = np.zeros(num_patches, dtype=np.int64)
    for indices, value in zip(windows, values):
        totals[indices] += value
        counts[indices] += 1
    if np.any(counts == 0):
        raise ValueError("Every patch must be covered by a deletion window")
    return totals / counts.reshape((num_patches,) + (1,) * (values.ndim - 1))


def select_panel(images, tasks, images_per_class, seed=2028,
                 image_split="val", task_split="val"):
    """Choose a balanced image panel using the existing manifest ordering.

    The RNG traverses every class in the image split, including classes outside
    the selected tasks, exactly as in ``balanced_task_rows``.
    """
    selected_tasks = sorted(
        (row for row in tasks if row["split"] == task_split),
        key=lambda row: row["task_id"],
    )
    task_classes = {row[key] for row in selected_tasks
                    for key in ("class_a", "class_b")}
    by_class = defaultdict(list)
    for row in images:
        if row["split"] == image_split:
            by_class[row["class_id"]].append(row)
    rng = random.Random(seed)
    selected_images = []
    for class_id in sorted(by_class):
        rows = sorted(by_class[class_id], key=lambda row: row["image_id"])
        rng.shuffle(rows)
        if class_id in task_classes:
            if len(rows) < images_per_class:
                raise ValueError("Not enough images for the requested balanced panel")
            selected_images.extend(rows[:images_per_class])
    if {row["class_id"] for row in selected_images} != task_classes:
        raise ValueError("Task classes must have images in the selected image split")
    return sorted(selected_images, key=lambda row: row["image_id"]), selected_tasks


def task_patch_utilities(full_scores, deleted_scores, text_features, true_class,
                         competitors, windows):
    """Map window-deletion geometric margin drops to per-patch utilities.

    Positive utility means deleting the window reduced the signed margin toward
    the true class. It does not establish the best selection at a target budget.
    """
    full_scores = np.asarray(full_scores)
    deleted_scores = np.asarray(deleted_scores)
    text_features = np.asarray(text_features)
    competitors = np.asarray(competitors, dtype=np.int64)
    norms = np.linalg.norm(text_features[true_class] - text_features[competitors], axis=1)
    full_margins = (full_scores[true_class] - full_scores[competitors]) / norms
    deleted_margins = (
        deleted_scores[:, true_class, None] - deleted_scores[:, competitors]
    ) / norms
    return window_to_patch_scores(
        full_margins - deleted_margins, windows, num_patches=int(np.max(windows)) + 1
    )


def _topk(scores, k):
    """Break utility ties by original patch position, then restore spatial order."""
    return np.sort(np.argsort(-np.asarray(scores), kind="stable")[:k])


def make_candidate_bank(utility_a, utility_b, attention, representation, k, seed,
                        candidate_count=64):
    """Build a shared finite bank of compressed selections for two tasks.

    The first ten entries initialize task, shared, attention, representation,
    mixture, and random selections. Remaining entries swap patches in those
    initial selections. Duplicates remain so every bank has the same size.
    """
    utility_a = np.asarray(utility_a)
    utility_b = np.asarray(utility_b)
    num_patches = len(utility_a)
    if not 0 < k < num_patches:
        raise ValueError("Candidate banks require a compressed budget: 0 < k < N")
    if candidate_count < 10:
        raise ValueError("candidate_count must accommodate the ten initial selections")
    rng = np.random.default_rng(seed)
    scores = [utility_a, utility_b, (utility_a + utility_b) / 2,
              attention, representation,
              0.25 * utility_a + 0.75 * utility_b,
              0.75 * utility_a + 0.25 * utility_b]
    selections = [_topk(values, k) for values in scores]
    names = ["task_a", "task_b", "shared_proxy", "attention", "representation",
             "mix_025", "mix_075"]
    for index in range(3):
        selections.append(np.sort(rng.choice(num_patches, k, replace=False)))
        names.append(f"random_{index}")
    swap_count = max(1, round(0.1 * min(k, num_patches - k)))
    all_indices = np.arange(num_patches)
    for index in range(candidate_count - 10):
        initial = selections[index % 10]
        removed = rng.choice(initial, swap_count, replace=False)
        available = np.setdiff1d(all_indices, initial, assume_unique=True)
        added = rng.choice(available, swap_count, replace=False)
        selections.append(np.sort(np.concatenate([
            np.setdiff1d(initial, removed, assume_unique=True), added
        ])))
        names.append(f"neighbor_{index}")
    return np.asarray(selections, dtype=np.int64), names
