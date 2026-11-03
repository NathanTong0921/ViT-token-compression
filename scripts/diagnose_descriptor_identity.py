"""Check whether pair means retain class identities in the training text dictionary.

For a row-full-rank class-text matrix T, the mean of rows a and b obeys
    ((T[a] + T[b]) / 2) @ pinv(T) = (e_a + e_b) / 2.

Enumerating text pairs here is an algebraic dictionary diagnostic, not a
classification evaluation. Only the ``text_features`` array from each training
cache is read; no reserved images, predictions or task-split files are accessed.
This does not establish that a trained selector implements this decoder.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np


def diagnose_dictionary(path):
    with np.load(path, allow_pickle=False) as evidence:
        saved_text = evidence["text_features"]
        source_dtype = str(saved_text.dtype)
        text = saved_text.astype(np.float64)

    class_count, dimension = text.shape
    singular_values = np.linalg.svd(text, compute_uv=False)
    rank_tolerance = max(text.shape) * np.finfo(text.dtype).eps * singular_values[0]
    rank = int(np.count_nonzero(singular_values > rank_tolerance))

    class_a, class_b = np.triu_indices(class_count, k=1)
    pair_means = (text[class_a] + text[class_b]) / 2
    # Set the tolerance explicitly so the numerical convention is recorded.
    pinv_rcond = 1e-15
    decoded = pair_means @ np.linalg.pinv(text, rcond=pinv_rcond)
    expected = np.zeros((len(class_a), class_count), dtype=np.float64)
    rows = np.arange(len(class_a))
    expected[rows, class_a] = .5
    expected[rows, class_b] = .5
    recovered_pairs = np.sort(np.argsort(decoded, axis=1)[:, -2:], axis=1)
    correct = np.all(recovered_pairs == np.column_stack((class_a, class_b)), axis=1)
    norms = np.linalg.norm(text, axis=1)

    return {
        "source_path": str(path),
        "array_read": "text_features",
        "source_dtype": source_dtype,
        "calculation_dtype": str(text.dtype),
        "text_shape": [class_count, dimension],
        "matrix_rank": rank,
        "row_full_rank": rank == class_count,
        "rank_tolerance": float(rank_tolerance),
        "pinv_rcond": pinv_rcond,
        "largest_singular_value": float(singular_values[0]),
        "smallest_singular_value": float(singular_values[-1]),
        "condition_number": (float(singular_values[0] / singular_values[-1])
                             if singular_values[-1] > 0 else None),
        "class_vector_norm_min": float(norms.min()),
        "class_vector_norm_max": float(norms.max()),
        "unordered_pair_count": len(class_a),
        "exact_identity_relation": "mean(T[a], T[b]) @ pinv(T) = 0.5 * (e_a + e_b), when T has full row rank",
        "identity_coefficient_max_abs_error": float(np.max(np.abs(decoded - expected))),
        "top_two_class_recovery_count": int(correct.sum()),
        "top_two_class_recovery_fraction": float(correct.mean()),
        "top_two_class_recovery_percent": float(100 * correct.mean()),
    }


def main(args):
    repo = Path(__file__).resolve().parents[1]
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "numpy_version": np.__version__,
        "final_test_used": False,
        "scope": "Training-cache text dictionaries only. All unordered class pairs are enumerated algebraically; this is not classification evaluation and does not access reserved images or outcomes.",
        "interpretation": "Checks information retained by the raw pair mean for the fixed known-class dictionary. The linear-algebra identity is not a new theorem, and the diagnostic does not show which computation the trained selector learned.",
        "datasets": {
            "pets": diagnose_dictionary(args.pets_source),
            "flowers": diagnose_dictionary(args.flowers_source),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "datasets": {key: {
            "row_full_rank": value["row_full_rank"],
            "pair_recovery_percent": value["top_two_class_recovery_percent"],
        } for key, value in result["datasets"].items()},
    }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pets-source", type=Path,
                        default=Path("runs/pets-selector-scale-supervision-v1/train/evidence.npz"))
    parser.add_argument("--flowers-source", type=Path,
                        default=Path("runs/flowers-selector-budget-49-supervision-v1/train/evidence.npz"))
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
