"""Matched C/U capacity comparison using the original KL training implementation."""

import argparse
import csv
import gc
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from evaluate_direct_classification_panel import load_panel, read_pairs
from evaluate_selector_scale import selector_from_checkpoint
from run_direct_classification import evaluate, write_evaluation_manifest, write_json
from token_compression.pruning import SingleStagePruner
from token_compression.selector import make_selector, shared_descriptor
from train_selector_scale import train


ARMS = ("conditioned", "image_only_wide")
CAPACITIES = ("small", "medium", "large")


def training_config(study, dataset, capacity, device):
    config = json.loads(Path(study["datasets"][dataset]["base_config"]).read_text())
    widths = study["capacities"][capacity]
    config.update(device=device, hidden_dim=widths["conditioned"],
                  wide_hidden_dim=widths["image_only_wide"],
                  selector_arms=list(ARMS), training_seeds=study["training_seeds"],
                  objective="kl", coverages=["full"],
                  experiment_name=f"{study['experiment']}_{dataset}_{capacity}")
    assert config["model"] == "ViT-B/16" and config["prune_after"] == 6
    assert config["keep_patch_tokens"] == 49
    return config


def checkpoint_path(study, dataset, capacity, seed, arm):
    if capacity == "medium":
        base = Path(study["datasets"][dataset]["medium_models"][arm])
    else:
        base = Path(study["output"]) / dataset / "training" / capacity
    return base / "full" / f"seed_{seed}" / f"{arm}.pt"


def check_checkpoint(checkpoint, config, expected_width, arm):
    variant = checkpoint.get("selector_variant", "conditioned" if checkpoint["conditioned"] else "image_only")
    assert variant == arm
    assert checkpoint["hidden_dim"] == expected_width
    assert checkpoint["step"] == config["training_steps"]
    assert checkpoint.get("objective", "kl") == "kl"
    old = checkpoint["config"]
    for key in ("cache_dir", "batch_size", "learning_rate", "weight_decay", "temperature", "training_steps"):
        assert old[key] == config[key], (key, old[key], config[key])


def check_medium(study, dataset, device):
    config = training_config(study, dataset, "medium", device)
    records = []
    for seed in study["training_seeds"]:
        for arm in ARMS:
            path = checkpoint_path(study, dataset, "medium", seed, arm)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            width = study["capacities"]["medium"][arm]
            check_checkpoint(checkpoint, config, width, arm)
            count = sum(value.numel() for key, value in checkpoint["state_dict"].items()
                        if key != "shared_descriptor")
            records.append(dict(seed=seed, arm=arm, checkpoint=str(path), step=checkpoint["step"],
                                hidden_dim=width, parameter_count=count,
                                source_code_commit=checkpoint.get("code_commit")))
    return records


def retain_medium_training_records(study, dataset, output):
    """Keep the original middle-capacity learning curves without fitting again."""
    runs, history, sources = [], [], []
    for arm in ARMS:
        folder = Path(study["datasets"][dataset]["medium_models"][arm])
        saved = json.loads((folder / "training_summary.json").read_text())
        runs.extend(row for row in saved["runs"] if row["arm"] == arm and row["coverage"] == "full"
                    and row["seed"] in study["training_seeds"])
        with (folder / "training_history.csv").open() as handle:
            history.extend(row for row in csv.DictReader(handle)
                           if row["arm"] == arm and row["coverage"] == "full"
                           and int(row["seed"]) in study["training_seeds"])
        sources.append(str(folder))
    destination = output / "training" / "medium"
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "training_summary.json", dict(reused=True, sources=sources, runs=runs))
    with (destination / "training_history.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)


def fit_models(study, dataset, device, output):
    records = []
    for capacity in ("small", "large"):
        config = training_config(study, dataset, capacity, device)
        destination = output / "training" / capacity
        summary_path = destination / "training_summary.json"
        if summary_path.exists():
            saved = json.loads(summary_path.read_text())
            assert saved["config"] == config
        else:
            destination.mkdir(parents=True, exist_ok=True)
            write_json(destination / "config.json", config)
            train(config, destination)
            saved = json.loads(summary_path.read_text())
            saved.update(run_type="fixed_update_selector_capacity", capacity=capacity,
                         capacity_comparison="Conditioned and image-only networks use matched total parameter budgets at this capacity.")
            write_json(summary_path, saved)
        records.append(dict(capacity=capacity, summary=str(summary_path),
                            wall_seconds=saved["wall_seconds"], preparation_seconds=saved["preparation_seconds"],
                            runs=saved["runs"]))
        gc.collect()
        torch.cuda.empty_cache()
    return records


def evaluate_panels(study, dataset, device, output):
    import clip

    source = training_config(study, dataset, "medium", str(device))
    source["eval_batch_size"] = study["eval_batch_size"]
    backbone, _ = clip.load(source["model"], device=device, jit=False, download_root=source["model_cache"])
    backbone = backbone.float().eval().requires_grad_(False)
    pruner = SingleStagePruner(backbone.visual, source["prune_after"])
    scale = float(backbone.logit_scale.exp())
    with np.load(Path(source["cache_dir"]) / "train/evidence.npz") as saved:
        train_ids, train_text = saved["image_ids"].copy(), saved["text_features"].copy()
    _, train_pairs = read_pairs(Path(source["cache_dir"]) / "train_tasks.csv")
    descriptor = shared_descriptor(torch.as_tensor(train_text, device=device),
                                   torch.as_tensor(train_pairs, device=device))
    panels = []
    for panel, cache, split in (("dev", source["cache_dir"], "val"),
                               ("test", study["datasets"][dataset]["test_cache"], "test")):
        destination = output / "evaluation" / panel
        destination.mkdir(parents=True, exist_ok=True)
        data = load_panel(cache, split, split, device)
        assert not set(train_ids) & set(data["image_ids"])
        assert not set(map(tuple, train_pairs)) & set(map(tuple, data["pairs_array"]))
        np.testing.assert_array_equal(data["text_array"], train_text)
        write_evaluation_manifest(destination.parent / panel, data, scale)
        # Existing helper writes its manifest in an evaluation subdirectory.
        for name in ("examples.csv", "evidence.npz"):
            (destination / "evaluation" / name).replace(destination / name)
        (destination / "evaluation").rmdir()
        jobs = []
        for seed in study["training_seeds"]:
            for capacity in CAPACITIES:
                for arm in ARMS:
                    path = checkpoint_path(study, dataset, capacity, seed, arm)
                    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                    width = study["capacities"][capacity][arm]
                    check_checkpoint(checkpoint, source, width, arm)
                    model = selector_from_checkpoint(checkpoint, descriptor, arm, 768, device).float()
                    target = destination / f"seed_{seed}" / capacity / f"{arm}.npz"
                    summary, _ = evaluate(model, data, pruner, source, scale, target)
                    job = dict(seed=seed, capacity=capacity, arm=arm, width=width,
                               parameter_count=sum(p.numel() for p in model.parameters()),
                               checkpoint=str(path), reused_checkpoint=capacity == "medium", **summary)
                    write_json(target.with_suffix(".json"), job)
                    jobs.append(job)
                    print(json.dumps(dict(dataset=dataset, panel=panel, **job)), flush=True)
                    del model
        summary = dict(dataset=dataset, panel=panel, rows=len(data["weights"]),
                       images=len(set(data["image_ix"])), tasks=len(data["task_ids"]),
                       source_config=source, logit_scale=scale, jobs=jobs,
                       evaluation_seconds=sum(job["elapsed_seconds"] for job in jobs),
                       backbone_dtype="float32", prefix_dtype=str(data["tokens"].dtype),
                       inference="FP32 frozen CLIP suffix with native deletion; original prefix cache cast to float32.",
                       test_history="The existing panel was evaluated in earlier studies; no capacity endpoint is selected from this panel.")
        write_json(destination / "evaluation_summary.json", summary)
        panels.append(summary)
        del data
        gc.collect()
        torch.cuda.empty_cache()
    return panels


def main(args):
    study = json.loads(args.config.read_text())
    device_name = args.device or study["datasets"][args.dataset]["device"]
    device = torch.device(device_name)
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    output = Path(study["output"]) / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    medium = check_medium(study, args.dataset, device_name)
    write_json(output / "reused_medium.json", medium)
    write_json(output / "study_config.json", study)
    if args.mode == "check":
        print(json.dumps(medium), flush=True)
        return
    retain_medium_training_records(study, args.dataset, output)
    training = fit_models(study, args.dataset, device_name, output) if args.mode in ("all", "train") else []
    panels = evaluate_panels(study, args.dataset, device, output) if args.mode in ("all", "evaluate") else []
    write_json(output / "run_summary.json", dict(dataset=args.dataset, config=study, device=device_name,
        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        medium=medium, training=training, panels=panels, wall_seconds=time.perf_counter()-started,
        training_scope="Only small and large selectors are newly trained; original KL training implementation and final update reused.",
        pretrained_backbone_training=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/selector_capacity.json"))
    parser.add_argument("--dataset", choices=("pets", "flowers"), required=True)
    parser.add_argument("--device")
    parser.add_argument("--mode", choices=("all", "check", "train", "evaluate"), default="all")
    main(parser.parse_args())
