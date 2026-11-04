"""Evaluate original and role-switched fits on identical H0-by-H1 task rows."""

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from analyze_familiarity_switch import summarize_switch
from evaluate_selectors import binary_outcomes
from evaluate_semantic_identity_holdout import load_panel
from run_semantic_identity_pilot import evaluate_model, make_model, synchronize, write_json
from semantic_identity_protocol import class_partition


def switch_tasks(train_tasks, val_tasks, original_held, switched_held):
    """Select only development pairs excluded from both training regimes."""
    h0, h1 = set(original_held), set(switched_held)
    assert not h0 & h1
    rows = []
    seen_pairs = set()
    for source, tasks in (("original_train", train_tasks), ("original_val", val_tasks)):
        for task in tasks:
            a, b = int(task["class_a"]), int(task["class_b"])
            if (a in h0 and b in h1) or (a in h1 and b in h0):
                pair = tuple(sorted((a, b)))
                assert pair not in seen_pairs
                seen_pairs.add(pair)
                rows.append({**task, "task_source": source})
    return rows


def run(original_dir, switched_dir, output, device_name, replicates):
    import clip
    from token_compression.pruning import SingleStagePruner

    model_dirs = {"original": original_dir, "switched": switched_dir}
    protocols = {phase: json.loads((folder / "protocol.json").read_text()) for phase, folder in model_dirs.items()}
    old, new = protocols["original"], protocols["switched"]
    partition_seed = old["partition_seed"]
    assert new["partition_seed"] == partition_seed
    class_count = len(old["seen_classes"]) + len(old["held_classes"])
    for phase, block in (("original", 0), ("switched", 1)):
        np.testing.assert_array_equal(protocols[phase]["held_classes"],
            class_partition(class_count, partition_seed, holdout_block=block)[1])
    assert old["arms"] == new["arms"] and old["training_seeds"] == new["training_seeds"]
    for key in ("cache_dir", "model", "prune_after", "keep_patch_tokens", "training_steps", "batch_size"):
        assert old["config"][key] == new["config"][key]
    config, device = old["config"].copy(), torch.device(device_name)
    config["device"] = device_name
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    cache = Path(config["cache_dir"])
    manifests = {}
    for split in ("train", "val"):
        with (cache / f"{split}_tasks.csv").open() as handle:
            manifests[split] = list(csv.DictReader(handle))
    tasks = switch_tasks(manifests["train"], manifests["val"], old["held_classes"], new["held_classes"])
    data = load_panel(cache, tasks, device)
    audits = {phase: json.loads((folder / "class_holdout" / "mapping_audit.json").read_text())
              for phase, folder in model_dirs.items()}
    for arm in old["arms"]:
        np.testing.assert_array_equal(audits["original"]["mappings"][arm], audits["switched"]["mappings"][arm])
    for phase, audit in audits.items():
        assert not set(data["task_ids"]) & set(audit["training_task_ids"])
        assert not set(data["image_ids"]) & set(audit["training_image_ids"])
        assert not set(protocols[phase]["held_classes"]) & set(audit["training_class_ids"])
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    protocol = dict(code_commit=commit, model_roots={k: str(v) for k, v in model_dirs.items()},
                    original_held_classes=old["held_classes"], switched_held_classes=new["held_classes"],
                    arms=old["arms"], training_seeds=old["training_seeds"],
                    partition_seed=partition_seed, partition_blocks=[0, 1],
                    class_count=class_count, config=config, final_test_used=False, official_test_used=False,
                    training_audits={phase: {key: value[key] for key in ("training_images", "training_tasks", "training_class_ids")}
                                     for phase, value in audits.items()},
                    mappings={phase: value["mappings"] for phase, value in audits.items()},
                    evaluation="Same validation images and H0-by-H1 pairs drawn only from original train/val manifests, excluded from both training regimes.",
                    mapping_note="The same class-code map is used in both fits, preserving H0, H1 and always-seen R separately. Real-minus-wide is also reported.")
    write_json(output / "protocol.json", protocol)
    with (output / "evaluation_tasks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tasks[0]))
        writer.writeheader()
        writer.writerows(tasks)
    margin, correct = binary_outcomes(data["full_scores"][data["image_ix"]],
        data["labels"][data["image_ix"]], data["pairs_array"][data["pair_ix"]], data["text_array"])
    class_ids = data["labels"][data["image_ix"]]
    panel = dict(image_ids=data["image_ids"][data["image_ix"]], class_pairs=data["pairs_array"][data["pair_ix"]],
                 class_ids=class_ids, task_ids=np.asarray(data["task_ids"])[data["pair_ix"]], weights=data["weights"],
                 true_original_held=np.isin(class_ids, old["held_classes"]),
                 true_switched_held=np.isin(class_ids, new["held_classes"]),
                 full_margin=margin, full_correct=correct,
                 task_sources=np.asarray([t["task_source"] for t in tasks])[data["pair_ix"]])
    assert np.all(panel["true_original_held"] != panel["true_switched_held"])
    np.savez_compressed(output / "evaluation_panel.npz", **panel)
    backbone, _ = clip.load(config["model"], device=device, jit=False, download_root=config["model_cache"])
    backbone.eval().requires_grad_(False)
    pruner = SingleStagePruner(backbone.visual, config["prune_after"])
    predictions, records = {}, []
    for phase, models in model_dirs.items():
        folder = output / phase
        folder.mkdir(exist_ok=True)
        for seed in old["training_seeds"]:
            for arm in old["arms"]:
                filename = f"seed_{seed}_{arm}"
                checkpoint_path = models / "class_holdout" / f"{filename}.pt"
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
                assert checkpoint["regime"] == "class_holdout"
                np.testing.assert_array_equal(checkpoint["mapping"], audits[phase]["mappings"][arm])
                prior = checkpoint["state_dict"].get("shared_descriptor", torch.zeros(2 * data["text"].shape[-1], device=device))
                model = make_model(prior, arm, checkpoint["config"], checkpoint["token_dim"], device)
                model.load_state_dict(checkpoint["state_dict"])
                model.eval().requires_grad_(False)
                mapping = torch.as_tensor(checkpoint["mapping"], dtype=torch.long, device=device)
                synchronize(device)
                started = time.perf_counter()
                predictions[(phase, seed, arm)] = evaluate_model(model, data, mapping, pruner,
                    config["keep_patch_tokens"], config["image_batch_size"], folder / f"{filename}.npz")
                synchronize(device)
                record = dict(phase=phase, seed=seed, arm=arm, checkpoint=str(checkpoint_path),
                              evaluation_seconds=time.perf_counter() - started)
                records.append(record)
                write_json(output / "evaluation_records.json", records)
                print(json.dumps(record), flush=True)
                del model
    summary = summarize_switch(panel, predictions, old["arms"], old["training_seeds"], replicates)
    write_json(output / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-models", type=Path, required=True)
    parser.add_argument("--switched-models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    run(args.original_models, args.switched_models, args.output, args.device, args.bootstrap_replicates)
