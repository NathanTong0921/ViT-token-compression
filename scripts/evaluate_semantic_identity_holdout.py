"""Expand held-class task coverage without new training or final-test access."""

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from analyze_semantic_identity_holdout import summarize_holdout_panel
from evaluate_selectors import binary_outcomes
from run_semantic_identity_pilot import evaluate_model, make_model, summarize, synchronize, write_json
from semantic_identity_protocol import pair_strata
from token_compression.selector import build_examples
from token_compression.selector_scale import balanced_example_weights


def expanded_tasks(train_tasks, val_tasks, held_classes):
    """Original validation plus training-manifest pairs excluded by class holdout."""
    held = set(map(int, held_classes))
    val_keys = {tuple(sorted((int(t["class_a"]), int(t["class_b"])))) for t in val_tasks}
    train_keys = {tuple(sorted((int(t["class_a"]), int(t["class_b"])))) for t in train_tasks}
    assert not val_keys & train_keys
    rows = [{**t, "task_source": "original_val"} for t in val_tasks]
    rows += [{**t, "task_source": "removed_train"} for t in train_tasks
             if int(t["class_a"]) in held or int(t["class_b"]) in held]
    return rows


def load_panel(cache, tasks, device):
    with np.load(cache / "val" / "evidence.npz") as saved:
        ids, labels, text_array, full_scores = [saved[key].copy()
            for key in ("image_ids", "labels", "text_features", "full_scores")]
    pairs = np.array([[int(t["class_a"]), int(t["class_b"])] for t in tasks])
    pairs_tensor = torch.as_tensor(pairs, dtype=torch.long, device=device)
    examples = build_examples(torch.as_tensor(labels, dtype=torch.long, device=device), pairs_tensor)
    image_ix, pair_ix = [examples[key].cpu().numpy() for key in ("image_indices", "pair_indices")]
    weights = balanced_example_weights(labels, pairs, image_ix, pair_ix)
    return dict(image_ids=ids, labels=labels, text_array=text_array, full_scores=full_scores,
                text=torch.as_tensor(text_array, dtype=torch.float32, device=device),
                tokens=torch.as_tensor(np.load(cache / "val" / "prefix_tokens.npy"), device=device),
                pairs_array=pairs, pairs=pairs_tensor, examples=examples, image_ix=image_ix,
                pair_ix=pair_ix, weights=weights, task_ids=[t["task_id"] for t in tasks])


def run(models_dir, output, device_name, replicates):
    import clip
    from token_compression.pruning import SingleStagePruner

    protocol = json.loads((models_dir / "protocol.json").read_text())
    config = protocol["config"].copy()
    config["device"] = device_name
    cache, device = Path(config["cache_dir"]), torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    manifests = {}
    for split in ("train", "val"):
        with (cache / f"{split}_tasks.csv").open() as handle:
            manifests[split] = list(csv.DictReader(handle))
    tasks = expanded_tasks(manifests["train"], manifests["val"], protocol["held_classes"])
    audit = json.loads((models_dir / "class_holdout" / "mapping_audit.json").read_text())
    assert not set(t["task_id"] for t in tasks) & set(audit["training_task_ids"])
    data = load_panel(cache, tasks, device)
    assert not set(data["image_ids"]) & set(audit["training_image_ids"])
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    new_protocol = {**protocol, "evaluation_code_commit": commit, "models_dir": str(models_dir),
        "evaluation": "Original validation images; original validation pairs plus training-manifest pairs removed from class-holdout training.",
        "evaluation_scope": "Supplementary class-holdout coverage; no retraining, no official or reserved final-test pairs.",
        "decision_basis": "Original new/new validation strata contain only 2 Pets and 10 Flowers pairs; expansion chosen from pair counts before inspecting holdout outcomes.",
        "final_test_used": False, "official_test_used": False}
    write_json(output / "protocol.json", new_protocol)
    with (output / "evaluation_tasks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tasks[0]))
        writer.writeheader()
        writer.writerows(tasks)
    strata = pair_strata(data["pairs_array"][data["pair_ix"]], protocol["held_classes"])
    margin, correct = binary_outcomes(data["full_scores"][data["image_ix"]],
        data["labels"][data["image_ix"]], data["pairs_array"][data["pair_ix"]], data["text_array"])
    panel = dict(image_ids=data["image_ids"][data["image_ix"]],
        class_pairs=data["pairs_array"][data["pair_ix"]], class_ids=data["labels"][data["image_ix"]],
        task_ids=np.asarray(data["task_ids"])[data["pair_ix"]], weights=data["weights"], strata=strata,
        full_margin=margin, full_correct=correct,
        task_sources=np.asarray([t["task_source"] for t in tasks])[data["pair_ix"]])
    np.savez_compressed(output / "evaluation_panel.npz", **panel)
    backbone, _ = clip.load(config["model"], device=device, jit=False, download_root=config["model_cache"])
    backbone.eval().requires_grad_(False)
    pruner = SingleStagePruner(backbone.visual, config["prune_after"])
    predictions, records = {}, []
    for seed in protocol["training_seeds"]:
        for arm in protocol["arms"]:
            filename = f"seed_{seed}_{arm}"
            checkpoint_path = models_dir / "class_holdout" / f"{filename}.pt"
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            assert checkpoint["regime"] == "class_holdout"
            np.testing.assert_array_equal(checkpoint["mapping"], audit["mappings"][arm])
            prior = checkpoint["state_dict"].get("shared_descriptor", torch.zeros(2 * data["text"].shape[-1], device=device))
            model = make_model(prior, arm, checkpoint["config"], checkpoint["token_dim"], device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval().requires_grad_(False)
            mapping = torch.as_tensor(checkpoint["mapping"], dtype=torch.long, device=device)
            synchronize(device)
            started = time.perf_counter()
            predictions[(seed, arm)] = evaluate_model(model, data, mapping, pruner,
                config["keep_patch_tokens"], config["image_batch_size"], output / f"{filename}.npz")
            synchronize(device)
            record = dict(seed=seed, arm=arm, checkpoint=str(checkpoint_path),
                          evaluation_seconds=time.perf_counter() - started)
            records.append(record)
            write_json(output / "evaluation_records.json", records)
            print(json.dumps(record), flush=True)
            del model
    summary = summarize(data, predictions, protocol["arms"], protocol["training_seeds"], strata, replicates)
    write_json(output / "summary.json", summary)
    detail = summarize_holdout_panel(panel, predictions, protocol["arms"], protocol["training_seeds"],
                                    protocol["held_classes"], replicates)
    write_json(output / "class_novelty_details.json", detail)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    run(args.models, args.output, args.device, args.bootstrap_replicates)
