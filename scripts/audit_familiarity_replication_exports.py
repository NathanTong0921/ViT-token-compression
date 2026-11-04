"""Check local replication exports against independent prior and historical outputs."""

import json
from numbers import Real
from pathlib import Path

import numpy as np


def numeric_differences(actual, expected):
    if isinstance(actual, dict):
        return [difference for key in actual
                for difference in numeric_differences(actual[key], expected[key])]
    if isinstance(actual, list):
        assert len(actual) == len(expected)
        return [difference for a, b in zip(actual, expected)
                for difference in numeric_differences(a, b)]
    if isinstance(actual, Real):
        return [abs(float(actual) - float(expected))]
    assert actual == expected
    return []


if __name__ == "__main__":
    root = Path("results/familiarity-replications-v1")
    result = json.loads((root / "summary.json").read_text())
    prior_root = Path("results/familiarity-prior-v1")
    prior_audit = json.loads((prior_root / "verification.json").read_text())
    records = {}
    for dataset, data in result["datasets"].items():
        historical = json.loads((Path("results/familiarity-switch-v1") / dataset / "evaluation/summary.json").read_text())
        differences = []
        for metric, contrasts in data["per_partition"]["2099"]["metrics"].items():
            for contrast, actual in contrasts.items():
                differences.extend(numeric_differences(actual, historical["metrics"][metric]["contrasts"][contrast]))
        assert max(differences) < 1e-12
        records[dataset] = dict(historical_numeric_values_compared=len(differences),
                                historical_max_absolute_difference=max(differences), partitions={})
        for seed, part in data["per_partition"].items():
            key = f"{dataset}-p{seed}"
            audit = prior_audit[key]
            assert audit["all_source_predictions_match"]
            with np.load(prior_root / key / "analysis_arrays.npz") as saved:
                q, weights = saved["q"], saved["weights"]
                choice = (q > 0) | ((q == 0) & (saved["h0_class"] < saved["h1_class"]))
                arms = saved["arms"].tolist()
                real, maps = arms.index("real_text"), [i for i, arm in enumerate(arms) if arm.startswith("permuted_")]
                for metric, values, scale in (("correct", choice.astype(float), 100.), ("margin", q, 1.)):
                    phase_effects = values[:, :, real] - values[:, :, maps].mean(2)
                    point = np.average((phase_effects[1] - phase_effects[0]).mean(0), weights=weights) * scale
                    expected = part["metrics"][metric]["real_minus_permuted_mean"]["seen_minus_held"]["estimate"]
                    np.testing.assert_allclose(point, expected, atol=1e-12)
            records[dataset]["partitions"][seed] = dict(
                directional_choice_and_margin_reconstructed=True,
                source_predictions_verified=audit["source_feature_reconstruction_checks"])
    output = dict(status="passed", source="Existing local directional score arrays and independent historical summaries",
                  final_test_used=False, new_checkpoint_audit="pending server access; not covered by this export audit",
                  datasets=records)
    (root / "verification_inputs.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
