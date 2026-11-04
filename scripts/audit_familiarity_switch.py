"""Read-only CPU verification of completed class-familiarity switch artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def read(path):
    return json.loads(path.read_text())


def audit(root, dataset, partition_seed=None):
    stem = f"{dataset}-familiarity-switch"
    if partition_seed is not None:
        stem += f"-partition-{partition_seed}"
    training = root / f"{stem}-training-v1"
    evaluation = root / f"{stem}-evaluation-v1"
    plan, protocol = read(training / "frozen_plan.json"), read(evaluation / "protocol.json")
    assert plan["partition_seed"] == protocol["partition_seed"]
    if partition_seed is not None:
        assert plan["partition_seed"] == partition_seed
    summary = read(evaluation / "summary.json")
    with np.load(evaluation / "evaluation_panel.npz") as saved:
        panel = {key: saved[key].copy() for key in saved.files}
    h0, h1, remaining = [np.asarray(plan[key]) for key in ("H0", "H1", "always_seen")]
    assert not set(h0) & set(h1) and len(h0) == len(h1)
    assert np.all(panel["true_original_held"] ^ panel["true_switched_held"])
    assert np.all(np.isin(panel["class_pairs"], h0).sum(1) == 1)
    assert np.all(np.isin(panel["class_pairs"], h1).sum(1) == 1)
    assert np.all((panel["class_pairs"] == panel["class_ids"][:, None]).any(1))
    weights = panel["weights"]
    np.testing.assert_allclose(weights.sum(), 1., atol=1e-12)
    np.testing.assert_allclose(weights[panel["true_original_held"]].sum(), .5, atol=1e-12)
    tasks = np.unique(panel["task_ids"])
    for task in tasks:
        chosen = panel["task_ids"] == task
        for side in (True, False):
            np.testing.assert_allclose(weights[chosen & (panel["true_original_held"] == side)].sum(),
                                       1 / (2 * len(tasks)), atol=1e-12)
    reused, trained, predictions = [], [], {}
    for phase in ("original", "switched"):
        folder = training / phase / "class_holdout"
        saved_audit = read(folder / "mapping_audit.json")
        assert not set(saved_audit["training_image_ids"]) & set(panel["image_ids"])
        assert not set(saved_audit["training_task_ids"]) & set(panel["task_ids"])
        held = h0 if phase == "original" else h1
        assert not set(saved_audit["training_class_ids"]) & set(held)
        for arm, mapping in plan["mappings"].items():
            np.testing.assert_array_equal(saved_audit["mappings"][arm], mapping)
            mapping = np.asarray(mapping)
            for group in (h0, h1, remaining):
                np.testing.assert_array_equal(np.sort(mapping[group]), group)
        for record in read(folder / "sampling_audit.json"):
            assert record["task_side_max"] - record["task_side_min"] <= 1
            assert record["total_draws"] == plan["config"]["training_steps"] * plan["config"]["batch_size"]
        for record in read(folder / "training_records.json"):
            key = f"{phase}/seed_{record['seed']}_{record['arm']}"
            checkpoint = torch.load(folder / f"seed_{record['seed']}_{record['arm']}.pt", map_location="cpu", weights_only=True)
            np.testing.assert_array_equal(checkpoint["mapping"], saved_audit["mappings"][record["arm"]])
            if record["trained_in_this_run"]:
                assert checkpoint["source_checkpoint"] is None
                trained.append(key)
            else:
                source = torch.load(checkpoint["source_checkpoint"], map_location="cpu", weights_only=True)
                assert all(torch.equal(value, source["state_dict"][name]) for name, value in checkpoint["state_dict"].items())
                assert record["training_seconds"] == 0
                reused.append(key)
            with np.load(evaluation / phase / f"seed_{record['seed']}_{record['arm']}.npz") as saved:
                assert all(len(saved[field]) == len(weights) for field in saved.files)
                predictions[(phase, record["seed"], record["arm"])] = {
                    field: saved[field].astype(float) for field in ("correct", "margin")}
    total_models = 2 * 4 * len(plan["training_seeds"])
    assert len(trained) == plan["new_fits"] and len(reused) == total_models - plan["new_fits"]
    for metric in ("correct", "margin"):
        per_phase = {}
        for phase in ("original", "switched"):
            real = np.mean([predictions[(phase, seed, "real_text")][metric] for seed in plan["training_seeds"]], axis=0)
            permuted = np.mean([predictions[(phase, seed, arm)][metric] for seed in plan["training_seeds"]
                                for arm in plan["mappings"]], axis=0)
            per_phase[phase] = real - permuted
        oriented = (per_phase["switched"] - per_phase["original"]) * np.where(panel["true_original_held"], 1., -1.)
        point = np.average(oriented, weights=weights) * (100 if metric == "correct" else 1)
        contrast = summary["metrics"][metric]["contrasts"]["real_minus_permuted_mean"]
        np.testing.assert_allclose(point, contrast["seen_minus_held"]["estimate"], atol=1e-12)
        np.testing.assert_allclose(2 * point, contrast["difference_in_differences"]["estimate"], atol=1e-12)
    with np.load(Path(protocol["config"]["cache_dir"]) / "val" / "evidence.npz") as saved:
        lookup = {str(image): i for i, image in enumerate(saved["image_ids"])}
        ix = np.array([lookup[str(image)] for image in panel["image_ids"]])
        scores, text = saved["full_scores"][ix], saved["text_features"]
    labels, pairs = panel["class_ids"], panel["class_pairs"]
    competitor = np.where(labels == pairs[:, 0], pairs[:, 1], pairs[:, 0])
    rows = np.arange(len(labels))
    margin = (scores[rows, labels] - scores[rows, competitor]) / np.linalg.norm(text[labels] - text[competitor], axis=1)
    correct = np.where(scores[rows, pairs[:, 0]] >= scores[rows, pairs[:, 1]], pairs[:, 0], pairs[:, 1]) == labels
    np.testing.assert_allclose(panel["full_margin"], margin, atol=1e-7)
    np.testing.assert_array_equal(panel["full_correct"], correct)
    assert not plan["final_test_used"] and not protocol["final_test_used"] and not protocol["official_test_used"]
    return dict(status="passed", partition_seed=plan["partition_seed"],
                tasks=len(tasks), rows=len(weights), images=len(np.unique(panel["image_ids"])),
                holdout_group_sizes=[len(h0), len(h1)], identical_class_codes_across_fits=True,
                exact_reused_checkpoints=reused, new_training_runs=len(trained),
                shared_panel_lengths_verified=True, equal_task_and_true_class_weights_verified=True,
                training_image_and_task_overlap=False, full_model_outcomes_independently_verified=True,
                oriented_estimate_and_did_independently_recomputed=True,
                final_test_used=False, official_test_used=False,
                task_sources={str(source): len(np.unique(panel["task_ids"][panel["task_sources"] == source]))
                              for source in np.unique(panel["task_sources"])})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--partition-seed", type=int,
                        help="Use partition-specific run roots; omit for historical 2099 roots.")
    args = parser.parse_args()
    print(json.dumps({dataset: audit(args.runs_root, dataset, args.partition_seed)
                      for dataset in ("pets", "flowers")}, indent=2))
