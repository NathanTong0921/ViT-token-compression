"""Train matched conditioned and image-only selectors on cached frozen tokens."""

import argparse
import csv
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from token_compression.selector import (TokenSelector, build_examples,
                                        shared_descriptor, utility_targets)


def read_tasks(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def load_split(cache, split, device, temperature, batch_size):
    """Keep small pilot caches on one device and prepare the shared soft targets."""
    tasks = read_tasks(cache / f"{split}_tasks.csv")
    pairs = torch.tensor([[int(row["class_a"]), int(row["class_b"])] for row in tasks],
                         dtype=torch.long, device=device)
    folder = cache / split
    with np.load(folder / "evidence.npz") as evidence:
        image_ids = evidence["image_ids"].copy()
        labels = torch.tensor(evidence["labels"], dtype=torch.long, device=device)
        text = torch.tensor(evidence["text_features"], dtype=torch.float32, device=device)
    tokens = torch.from_numpy(np.load(folder / "prefix_tokens.npy")).to(device)
    with np.load(folder / "teacher.npz") as teacher:
        np.testing.assert_array_equal(image_ids, teacher["image_ids"])
        coefficients = torch.tensor(teacher["coefficients"], dtype=torch.float32, device=device)
    assert len(tokens) == len(labels) == len(coefficients)
    examples = build_examples(labels, pairs)
    targets = []
    for start in range(0, len(examples["image_indices"]), batch_size):
        indices = examples["image_indices"][start:start + batch_size]
        task_pairs = examples["class_pairs"][start:start + batch_size]
        targets.append(utility_targets(coefficients[indices], labels[indices], task_pairs,
                                       text, temperature=temperature))
    return {"tokens": tokens, "text": text, "pairs": pairs, "examples": examples,
            "targets": torch.cat(targets), "image_ids": image_ids.tolist(),
            "task_ids": [row["task_id"] for row in tasks]}


def batch_loss(model, data, indices, temperature):
    images = data["examples"]["image_indices"][indices]
    pairs = data["examples"]["class_pairs"][indices]
    logits = model(data["tokens"][images], data["text"][pairs])
    return F.kl_div(F.log_softmax(logits / temperature, dim=1),
                    data["targets"][indices], reduction="batchmean")


@torch.no_grad()
def evaluate_loss(model, data, batch_size, temperature):
    model.eval()
    count = len(data["targets"])
    total = torch.zeros((), device=data["tokens"].device)
    for start in range(0, count, batch_size):
        indices = torch.arange(start, min(start + batch_size, count), device=total.device)
        total += batch_loss(model, data, indices, temperature) * len(indices)
    return float(total / count)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train(config, output):
    started = datetime.now(timezone.utc).isoformat()
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    wall_start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config["device"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    cache = Path(config["cache_dir"])
    batch_size, temperature = config["batch_size"], config["temperature"]
    data = {split: load_split(cache, split, device, temperature, batch_size)
            for split in ("train", "val")}
    assert not set(data["train"]["image_ids"]) & set(data["val"]["image_ids"])
    assert not set(data["train"]["task_ids"]) & set(data["val"]["task_ids"])
    torch.testing.assert_close(data["train"]["text"], data["val"]["text"], rtol=0, atol=0)
    prior = shared_descriptor(data["train"]["text"], data["train"]["pairs"])
    synchronize(device)
    preparation_seconds = time.perf_counter() - wall_start
    count = len(data["train"]["targets"])
    history, results = [], []
    for seed in config["training_seeds"]:
        torch.manual_seed(seed)
        template = TokenSelector(prior, conditioned=True, hidden_dim=config["hidden_dim"]).to(device)
        initial_state = {key: value.detach().clone() for key, value in template.state_dict().items()}
        parameter_count = sum(parameter.numel() for parameter in template.parameters())
        del template
        generator = torch.Generator(device="cpu").manual_seed(seed)
        epoch_orders = [torch.randperm(count, generator=generator).to(device)
                        for _ in range(config["epochs"])]
        seed_output = output / f"seed_{seed}"
        seed_output.mkdir(parents=True, exist_ok=True)
        for arm, conditioned in (("conditioned", True), ("image_only", False)):
            model = TokenSelector(prior, conditioned=conditioned, hidden_dim=config["hidden_dim"]).to(device)
            model.load_state_dict(initial_state)
            assert all(torch.equal(value, initial_state[key]) for key, value in model.state_dict().items())
            assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                          weight_decay=config["weight_decay"])
            before = time.perf_counter()
            initial_train = evaluate_loss(model, data["train"], batch_size, temperature)
            initial_val = evaluate_loss(model, data["val"], batch_size, temperature)
            synchronize(device)
            initial_evaluation_seconds = time.perf_counter() - before
            training_seconds, validation_seconds = 0., 0.
            for epoch, order in enumerate(epoch_orders, start=1):
                model.train()
                total = torch.zeros((), device=device)
                synchronize(device)
                before = time.perf_counter()
                for start in range(0, count, batch_size):
                    indices = order[start:start + batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    loss = batch_loss(model, data["train"], indices, temperature)
                    loss.backward()
                    optimizer.step()
                    total += loss.detach() * len(indices)
                synchronize(device)
                epoch_seconds = time.perf_counter() - before
                training_seconds += epoch_seconds
                train_loss = float(total / count)
                before = time.perf_counter()
                val_loss = evaluate_loss(model, data["val"], batch_size, temperature)
                synchronize(device)
                val_seconds = time.perf_counter() - before
                validation_seconds += val_seconds
                history.append({"seed": seed, "arm": arm, "epoch": epoch,
                                "train_kl": train_loss, "val_kl": val_loss,
                                "train_seconds": epoch_seconds, "validation_seconds": val_seconds,
                                "cumulative_train_seconds": training_seconds})
                print(f"seed={seed} arm={arm} epoch={epoch}/{config['epochs']} "
                      f"train_kl={train_loss:.6f} val_kl={val_loss:.6f} "
                      f"train_seconds={epoch_seconds:.2f}", flush=True)
            before = time.perf_counter()
            final_train = evaluate_loss(model, data["train"], batch_size, temperature)
            synchronize(device)
            final_evaluation_seconds = time.perf_counter() - before
            checkpoint = {"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                          "hidden_dim": config["hidden_dim"], "conditioned": conditioned,
                          "seed": seed, "epoch": config["epochs"], "parameter_count": parameter_count,
                          "config": config, "code_commit": source_commit,
                          "checkpoint_policy": "Fixed final epoch; no validation-based selection."}
            torch.save(checkpoint, seed_output / f"{arm}.pt")
            results.append({"seed": seed, "arm": arm, "parameter_count": parameter_count,
                            "epochs": config["epochs"], "initial_train_kl": initial_train,
                            "initial_val_kl": initial_val, "final_train_kl": final_train,
                            "final_val_kl": val_loss, "training_seconds": training_seconds,
                            "validation_seconds": validation_seconds,
                            "initial_evaluation_seconds": initial_evaluation_seconds,
                            "final_train_evaluation_seconds": final_evaluation_seconds,
                            "initial_state_exactly_matches_shared_seed_state": True,
                            "epoch_orders_reused_between_arms": True,
                            "checkpoint": str(Path(f"seed_{seed}") / f"{arm}.pt")})
            del model, optimizer
    with (output / "training_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    synchronize(device)
    summary = {"run_type": "paired_selector_soft_utility_distillation_pilot",
               "started_utc": started, "code_commit": source_commit, "config": config,
               "torch_version": torch.__version__, "numpy_version": np.__version__,
               "python_version": platform.python_version(),
               "device": str(device),
               "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
               "train_image_count": len(data["train"]["image_ids"]),
               "val_image_count": len(data["val"]["image_ids"]),
               "train_task_count": len(data["train"]["task_ids"]),
               "val_task_count": len(data["val"]["task_ids"]),
               "train_example_count": count, "val_example_count": len(data["val"]["targets"]),
               "frozen_prefix_cache_bytes": sum(split["tokens"].numel() * split["tokens"].element_size() for split in data.values()),
               "preparation_seconds": preparation_seconds,
               "wall_seconds": time.perf_counter() - wall_start,
               "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
               "supervision_cost_record": str(cache / "prep_summary.json"),
               "training_policy": "Full float32 selector training from cached frozen prefix tokens; no CLIP updates; fixed final epoch.",
               "target_definition": "KL to softmax of centered, population-standardized geometric teacher utilities at temperature=1 (configured temperature recorded). Constant utility yields a uniform target.",
               "pairing": "Each seed uses one cloned initial state for both arms and exactly the same list of CPU-generator randperm epoch orders; every training example appears once per epoch.",
               "capacity_comparison": "Identical architecture and trainable parameter count; image-only receives the fixed mean training-pair descriptor, so effective input information differs.",
               "runs": results}
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"wall_seconds": summary["wall_seconds"], "train_examples": count,
                      "model_runs": len(results), "parameter_count": results[0]["parameter_count"]}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    train(json.loads(args.config.read_text()), args.output)
