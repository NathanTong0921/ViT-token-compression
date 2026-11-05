"""Fit the released Patch Ranking predictor on cached preservation rankings."""

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import timm
import torch
from torch.nn import functional as F

from token_compression.patch_ranking import (PatchRankingPredictor, overlap_at,
                                            rank_scores, ranking_kl,
                                            ranking_probabilities)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_split(feature_cache, preservation, split, device, batch_size):
    folder = feature_cache / split
    with np.load(folder / "evidence.npz") as evidence:
        image_ids = evidence["image_ids"].copy()
    with np.load(preservation / split / "ranking.npz") as teacher:
        np.testing.assert_array_equal(teacher["image_ids"], image_ids)
        rankings = torch.as_tensor(teacher["rankings"], dtype=torch.long, device=device)
    prefix = np.load(folder / "prefix_tokens.npy", mmap_mode="r")
    # Pool the frozen channels once. Pooling 64 channels to 64 in the predictor
    # is then an identity; the learned Mixer still receives all 197 positions.
    pooled = []
    with torch.no_grad():
        for start in range(0, len(prefix), batch_size):
            values = torch.as_tensor(np.array(prefix[start:start + batch_size]), device=device).float()
            pooled.append(F.adaptive_avg_pool1d(values, 64))
    tokens = torch.cat(pooled)
    assert len(tokens) == len(rankings) == len(image_ids)
    return {"tokens": tokens, "rankings": rankings,
            "targets": ranking_probabilities(rankings), "image_ids": image_ids}


@torch.no_grad()
def evaluate(model, data, batch_size, keep_count, save_scores=False):
    model.eval()
    totals = dict(kl=0., overlap_at_98=0., overlap_at_100=0., deleted_50_overlap=0.)
    predictions = []
    n = len(data["tokens"])
    for start in range(0, n, batch_size):
        sl = slice(start, start + batch_size)
        scores = model(data["tokens"][sl])
        ranks = rank_scores(scores)
        teacher = data["rankings"][sl]
        totals["kl"] += float(ranking_kl(scores, data["targets"][sl])) * len(scores)
        totals["overlap_at_98"] += float(overlap_at(ranks, teacher, 98).sum())
        totals["overlap_at_100"] += float(overlap_at(ranks, teacher, 100).sum())
        totals["deleted_50_overlap"] += float(overlap_at(ranks, teacher, 50, deleted=True).sum())
        if save_scores:
            predictions.append(scores.cpu())
    metrics = {key: value / n for key, value in totals.items()}
    if not save_scores:
        return metrics
    scores = torch.cat(predictions)
    ranks = rank_scores(scores)
    return metrics, {"image_ids": data["image_ids"], "predicted_scores": scores.numpy(),
                     "predicted_rankings": ranks.numpy(),
                     "selected_indices": ranks[:, :keep_count].sort(dim=1).values.numpy()}


def write_history(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def train(config, output):
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
    started = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    batch_size = config["batch_size"]
    keep_count = config.get("keep_patch_tokens", 98)
    epochs = config["epochs"]
    data = {split: load_split(Path(config["feature_cache_dir"]), Path(config["preservation_dir"]),
                             split, device, batch_size) for split in ("train", "val")}
    assert set(data["train"]["image_ids"]).isdisjoint(data["val"]["image_ids"])
    synchronize(device)
    setup_seconds = time.perf_counter() - start
    summary = {
        "started_utc": started, "config": config,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source": "https://github.com/CEWu/PatchRanking/blob/main/train_predictor.py",
        "target": "Public KL target: equal logits for first 146 ranks; -5-0.1*rank for bottom 50.",
        "architecture": "AdaptiveAvgPool1d(64), timm MixerBlock(64,197), Linear(64,1), CLS omitted from output.",
        "input": "Frozen prefix channels pooled once in float32; all 197 token positions retained.",
        "counts": {split: len(v["image_ids"]) for split, v in data.items()},
        "environment": {"torch": torch.__version__, "timm": timm.__version__},
        "setup_seconds": setup_seconds, "seeds": {},
    }
    all_history = []
    for seed in config["seeds"]:
        seed_start = time.perf_counter()
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        model = PatchRankingPredictor().to(device)
        parameter_count = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                     weight_decay=config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=config["min_learning_rate"])
        seed_dir = output / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        initial = {split: evaluate(model, v, batch_size, keep_count) for split, v in data.items()}
        training_seconds = 0.
        diagnostic_seconds = 0.
        updates = 0
        history = []
        for epoch in range(1, epochs + 1):
            model.train()
            order = torch.randperm(len(data["train"]["tokens"]), generator=generator).to(device)
            synchronize(device)
            epoch_start = time.perf_counter()
            online_loss, visited = 0., 0
            lr = optimizer.param_groups[0]["lr"]
            for begin in range(0, len(order), batch_size):
                indices = order[begin:begin + batch_size]
                if len(indices) < batch_size and config.get("drop_last", False):
                    continue
                optimizer.zero_grad(set_to_none=True)
                scores = model(data["train"]["tokens"][indices])
                loss = ranking_kl(scores, data["train"]["targets"][indices])
                loss.backward()
                optimizer.step()
                online_loss += float(loss.detach()) * len(indices)
                visited += len(indices)
                updates += 1
            scheduler.step()
            synchronize(device)
            epoch_seconds = time.perf_counter() - epoch_start
            training_seconds += epoch_seconds
            diagnostic_start = time.perf_counter()
            metrics = {split: evaluate(model, v, batch_size, keep_count) for split, v in data.items()}
            synchronize(device)
            diagnostic_seconds += time.perf_counter() - diagnostic_start
            row = {"seed": seed, "epoch": epoch, "updates": updates, "learning_rate": lr,
                   "online_train_kl": online_loss / visited, "fit_seconds": epoch_seconds,
                   **{f"{split}_{key}": value for split, item in metrics.items()
                      for key, value in item.items()}}
            history.append(row)
            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                print(json.dumps(row), flush=True)
        final_val, predictions = evaluate(model, data["val"], batch_size, keep_count, save_scores=True)
        np.savez_compressed(seed_dir / "predictions_val.npz", **predictions)
        torch.save({"model_state": model.cpu().state_dict(), "config": config, "seed": seed,
                    "epoch": epochs, "parameter_count": parameter_count}, seed_dir / "model.pt")
        write_history(seed_dir / "training_history.csv", history)
        all_history.extend(history)
        summary["seeds"][str(seed)] = {"parameter_count": parameter_count, "updates": updates,
                                     "initial": initial, "final_train": metrics["train"],
                                     "final_val": final_val, "training_seconds": training_seconds,
                                     "diagnostic_seconds": diagnostic_seconds,
                                     "total_seconds": time.perf_counter() - seed_start}
        (output / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        write_history(output / "training_history.csv", all_history)
        print(json.dumps({"completed_seed": seed, **summary["seeds"][str(seed)]}), flush=True)
        del model, optimizer, scheduler
    summary["total_wall_seconds"] = time.perf_counter() - start
    summary["parameter_count"] = parameter_count
    summary["gpu"] = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    summary["peak_allocated_gpu_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.seeds is not None:
        config["seeds"] = args.seeds
    train(config, args.output)
