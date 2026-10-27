"""Label-informed shared and task-specific search in one measured mask library."""

import numpy as np


def search_common_library(scores, label, competitors, text):
    """Compare separate and joint mean-margin optima in the exact same library."""
    scores, text = np.asarray(scores, float), np.asarray(text, float)
    competitors = np.asarray(competitors, int)
    norms = np.linalg.norm(text[label] - text[competitors], axis=1)
    margins = (scores[:, label, None] - scores[:, competitors]) / norms
    correct = (scores[:, label, None] > scores[:, competitors]) | (
        (scores[:, label, None] == scores[:, competitors]) & (label < competitors))
    separate = margins.argmax(axis=0)
    shared = int(margins.mean(axis=1).argmax())
    selected = np.r_[separate, shared]
    separate_margin = margins[separate, np.arange(2)].mean()
    shared_margin = margins[shared].mean()
    return {"selected_candidate_indices": selected,
            "separate_margin": float(separate_margin), "shared_margin": float(shared_margin),
            "separate_minus_shared_margin": float(separate_margin - shared_margin),
            "separate_accuracy": float(correct[separate, np.arange(2)].mean()),
            "shared_accuracy": float(correct[shared].mean()),
            "shared_both_correct": bool(correct[shared].all()),
            "any_shared_both_correct": bool(correct.all(axis=1).any()),
            "margins": margins, "correct": correct}
