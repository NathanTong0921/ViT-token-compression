"""Fixed-update C/U training on balanced two- and five-class task panels."""

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from token_compression.nway import make_selector, training_order


def load_panel(path, device):
    with np.load(path) as f:
        panel = {k: f[k].copy() for k in f.files}
    tensors = {k: torch.as_tensor(panel[k], device=device)
               for k in ("image_indices", "candidates", "targets", "weights")}
    return panel, tensors


def losses(model, tokens, text, panel, indices, temperature):
    prefix = tokens[panel["image_indices"][indices]]
    logits = model(prefix, text[panel["candidates"][indices]])
    return F.kl_div(F.log_softmax(logits / temperature, dim=1),
                    panel["targets"][indices], reduction="none").sum(dim=1)


@torch.no_grad()
def evaluate(model, tokens, text, panel, batch_size, temperature):
    model.eval()
    total = torch.zeros((), dtype=torch.float64, device=tokens.device)
    for start in range(0, len(panel["targets"]), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(panel["targets"])), device=tokens.device)
        values = losses(model, tokens, text, panel, indices, temperature)
        total += (values.double() * panel["weights"][indices]).sum()
    return float(total)


def write_history(path, history):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)


def train(config, prepared, output, selected_seed=None, selected_way=None):
    start = time.perf_counter()
    device = torch.device(config["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    cache = Path(config["cache_dir"])
    tokens = {split: torch.from_numpy(np.load(cache / split / "prefix_tokens.npy")).to(device)
              for split in ("train", "val")}
    with np.load(cache / "train" / "evidence.npz") as f:
        text = torch.as_tensor(f["text_features"].copy(), device=device)
    token_dim, text_dim = tokens["train"].shape[-1], text.shape[-1]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    ways = [selected_way] if selected_way is not None else config["ways"]
    seeds = [selected_seed] if selected_seed is not None else config["training_seeds"]
    steps, interval = config["training_steps"], config["checkpoint_interval"]
    batch_size, temperature = config["batch_size"], config["temperature"]
    output.mkdir(parents=True, exist_ok=True)
    for way in ways:
        panels = {split: load_panel(prepared / "panels" / f"{way}way" / f"{split}.npz", device)
                  for split in ("train", "val")}
        train_numpy, train_panel = panels["train"]
        val_numpy, val_panel = panels["val"]
        for seed in seeds:
            folder = output / f"{way}way" / f"seed_{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            order_numpy = training_order(train_numpy, steps * batch_size, seed)
            order = torch.as_tensor(order_numpy, device=device)
            cell_counts = np.unique(np.column_stack((train_numpy["task_indices"][order_numpy],
                                                     train_numpy["labels"][order_numpy])),
                                    axis=0, return_counts=True)[1]
            history, records = [], []
            for arm in ("conditioned", "image_only_wide"):
                torch.manual_seed(seed)
                hidden_dim = config["hidden_dim"] if arm == "conditioned" else config["wide_hidden_dim"]
                model = make_selector(arm, token_dim, text_dim, hidden_dim).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                              weight_decay=config["weight_decay"])
                before = time.perf_counter()
                initial_train = evaluate(model, tokens["train"], text, train_panel, batch_size, temperature)
                initial_val = evaluate(model, tokens["val"], text, val_panel, batch_size, temperature)
                sync()
                initial_evaluation_seconds = time.perf_counter() - before
                history.append({"way": way, "seed": seed, "arm": arm, "step": 0,
                                "train_kl": initial_train, "val_kl": initial_val,
                                "train_seconds": 0., "validation_seconds": initial_evaluation_seconds,
                                "cumulative_train_seconds": 0.})
                training_seconds, validation_seconds = 0., 0.
                for block_start in range(0, steps, interval):
                    block_end = min(block_start + interval, steps)
                    model.train()
                    total = torch.zeros((), device=device)
                    sync()
                    before = time.perf_counter()
                    for step in range(block_start, block_end):
                        indices = order[step * batch_size:(step + 1) * batch_size]
                        optimizer.zero_grad(set_to_none=True)
                        loss = losses(model, tokens["train"], text, train_panel, indices, temperature).mean()
                        loss.backward()
                        optimizer.step()
                        total += loss.detach()
                    sync()
                    block_seconds = time.perf_counter() - before
                    training_seconds += block_seconds
                    train_kl = float(total / (block_end - block_start))
                    before = time.perf_counter()
                    val_kl = evaluate(model, tokens["val"], text, val_panel, batch_size, temperature)
                    sync()
                    val_seconds = time.perf_counter() - before
                    validation_seconds += val_seconds
                    history.append({"way": way, "seed": seed, "arm": arm, "step": block_end,
                                    "train_kl": train_kl, "val_kl": val_kl,
                                    "train_seconds": block_seconds, "validation_seconds": val_seconds,
                                    "cumulative_train_seconds": training_seconds})
                    write_history(folder / "training_history.csv", history)
                    print(f"way={way} seed={seed} arm={arm} step={block_end}/{steps} "
                          f"train_kl={train_kl:.6f} val_kl={val_kl:.6f} seconds={block_seconds:.2f}", flush=True)
                before = time.perf_counter()
                final_train = evaluate(model, tokens["train"], text, train_panel, batch_size, temperature)
                sync()
                final_evaluation_seconds = time.perf_counter() - before
                parameter_count = sum(p.numel() for p in model.parameters())
                record = {"way": way, "seed": seed, "arm": arm, "parameter_count": parameter_count,
                          "training_steps": steps, "training_draws": steps * batch_size,
                          "initial_train_kl": initial_train, "initial_val_kl": initial_val,
                          "final_train_kl": final_train, "final_val_kl": val_kl,
                          "training_seconds": training_seconds, "validation_seconds": validation_seconds,
                          "initial_evaluation_seconds": initial_evaluation_seconds,
                          "final_train_evaluation_seconds": final_evaluation_seconds}
                checkpoint = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                              "hidden_dim": hidden_dim, "selector_variant": arm, "conditioned": arm == "conditioned",
                              "way": way, "token_dim": token_dim, "text_dim": text_dim,
                              "num_patches": tokens["train"].shape[1] - 1,
                              "seed": seed, "step": steps, "parameter_count": parameter_count,
                              "config": config, "code_commit": commit,
                              "checkpoint_policy": "Fixed final update.", "cost": record}
                torch.save(checkpoint, folder / f"{arm}.pt")
                records.append(record)
                (folder / "training_summary.json").write_text(json.dumps({
                    "config": config, "code_commit": commit, "runs": records,
                    "train_images": int(len(np.unique(train_numpy["image_ids"]))),
                    "val_images": int(len(np.unique(val_numpy["image_ids"]))),
                    "train_examples": len(train_numpy["labels"]), "val_examples": len(val_numpy["labels"]),
                    "sampling": {"seed": seed, "task_class_draw_min": int(cell_counts.min()),
                                 "task_class_draw_max": int(cell_counts.max()),
                                 "unique_training_images_drawn": int(len(np.unique(train_numpy["image_ids"][order_numpy]))),
                                 "same_draws_between_arms": True},
                    "checkpoint_policy": "Fixed final update; validation KL is logged.",
                    "wall_seconds_since_process_start": time.perf_counter() - start,
                    "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                    "evaluation_split": "development validation images and unseen candidate sets"
                }, indent=2) + "\n")
                del model, optimizer
            del order
        del panels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--way", type=int)
    args = parser.parse_args()
    train(json.loads(args.config.read_text()), args.prepared, args.output, args.seed, args.way)
