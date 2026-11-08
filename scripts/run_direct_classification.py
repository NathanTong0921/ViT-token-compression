"""Continue paired selectors with actual frozen-CLIP classification or teacher KL."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from evaluate_selector_scale import selector_from_checkpoint
from run_semantic_identity_pilot import load_data, synchronize
from run_shared_residual import paired_order
from token_compression.selector import shared_descriptor


ARMS = ("conditioned", "image_only_wide")
OBJECTIVES = ("direct", "kl")


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def selected_rows(model, data, rows, keep):
    images = data["examples"]["image_indices"][rows]
    pairs = data["examples"]["class_pairs"][rows]
    tokens = data["tokens"][images].float()
    text = data["text"][pairs]
    scores = model(tokens, text)
    positions = scores.argsort(dim=1, descending=True, stable=True)[:, :keep].sort(dim=1).values
    labels = data["labels_tensor"][images]
    targets = (labels == pairs[:, 1]).long()
    return tokens, pairs, text, scores, positions, targets


def classification_metrics(features, text, targets, scale):
    features = F.normalize(features.float(), dim=-1)
    pair_scores = torch.einsum("bd,bcd->bc", features, text.float())
    logits = pair_scores * scale
    ce = F.cross_entropy(logits, targets, reduction="none")
    raw = pair_scores.gather(1, targets[:, None]).squeeze(1) - pair_scores.gather(1, (1-targets)[:, None]).squeeze(1)
    geom = raw / (text[:, 0] - text[:, 1]).norm(dim=-1)
    accuracy = (pair_scores.argmax(dim=1) == targets).float()
    return dict(ce=ce, raw_gap=raw, geometric_margin=geom, accuracy=accuracy)


def training_loss(model, data, rows, pruner, objective, config, scale):
    from token_compression.direct_classification import straight_through_topk

    tokens, pairs, text, scores, positions, targets = selected_rows(model, data, rows, config["keep_patch_tokens"])
    if objective == "kl":
        value = F.kl_div(F.log_softmax(scores/config["teacher_temperature"], dim=1),
                         data["targets"][rows], reduction="batchmean")
    else:
        policy, positions, _ = straight_through_topk(scores, config["keep_patch_tokens"], config["gate_temperature"])
        features = pruner.finish_policy(tokens, policy)
        value = classification_metrics(features, text, targets, scale)["ce"].mean()
    return value


@torch.no_grad()
def evaluate(model, data, pruner, config, scale, destination=None, rows=None):
    model.eval()
    rows = (torch.arange(len(data["weights"]), device=data["tokens"].device) if rows is None else rows)
    arrays = {name: [] for name in ("cosine_scores", "patch_indices", "ce", "accuracy", "raw_gap", "geometric_margin")}
    synchronize(data["tokens"].device)
    started = time.perf_counter()
    for start in range(0, len(rows), config["eval_batch_size"]):
        current = rows[start:start+config["eval_batch_size"]]
        tokens, pairs, text, scores, positions, targets = selected_rows(model, data, current, config["keep_patch_tokens"])
        features = pruner.finish(tokens, positions).float()
        metrics = classification_metrics(features, text, targets, scale)
        arrays["cosine_scores"].append((F.normalize(features, dim=-1) @ data["text"].T).cpu().numpy())
        arrays["patch_indices"].append(positions.cpu().numpy().astype(np.int16))
        for name, value in metrics.items():
            arrays[name].append(value.cpu().numpy())
    synchronize(data["tokens"].device)
    arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    if destination is not None:
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **arrays)
    indices = rows.cpu().numpy()
    weights = data["weights"][indices]
    weights = weights / weights.sum()
    summary = {name: float(weights @ arrays[name]) for name in ("ce", "accuracy", "raw_gap", "geometric_margin")}
    summary.update(rows=len(rows), elapsed_seconds=time.perf_counter()-started)
    return summary, arrays


def source_checkpoint(config, seed, arm):
    directory = config["conditioned_dir"] if arm == "conditioned" else config["shared_dir"]
    return Path(directory) / "full" / f"seed_{seed}" / f"{arm}.pt"


def restore_selector(config, seed, arm, prior, device):
    path = source_checkpoint(config, seed, arm)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = selector_from_checkpoint(checkpoint, prior, arm, 768, device).float().requires_grad_(True)
    return model, checkpoint, path


def optimizer_for(model, config):
    return torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])


def save_checkpoint(path, model, optimizer, step, metadata, seconds):
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = dict(metadata, state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
                      optimizer_state=optimizer.state_dict(), step=step, training_seconds=seconds)
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def continue_training(model, data, pruner, config, scale, seed, arm, objective, output, until_step, source):
    directory = output / "training" / f"seed_{seed}" / arm / objective
    directory.mkdir(parents=True, exist_ok=True)
    optimizer = optimizer_for(model, config)
    saved = sorted(directory.glob("step_*.pt"))
    start_step, elapsed = 0, 0.
    if saved:
        checkpoint = torch.load(saved[-1], map_location=data["tokens"].device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step, elapsed = checkpoint["step"], checkpoint["training_seconds"]
    order, stream_record = paired_order(data, config["training_steps"] * config["batch_size"], seed + 2710)
    write_json(directory / "training_stream.json", stream_record)
    metadata = dict(seed=seed, selector_variant=arm, conditioned=arm=="conditioned", hidden_dim=model.hidden_dim,
                    token_dim=768, objective=objective, source_checkpoint=str(source), config=config,
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    monitor = order[:min(256, len(order))]
    if start_step == 0:
        before, initial = evaluate(model, data, pruner, config, scale, rows=monitor)
        write_json(directory / "initial_train_monitor.json", before)
        np.save(directory / "initial_monitor_positions.npy", initial["patch_indices"])
    else:
        initial = {"patch_indices": np.load(directory / "initial_monitor_positions.npy")}
    history_path = directory / "training_history.csv"
    fieldnames = ["step", "objective_loss", "gradient_norm", "elapsed_training_seconds"]
    window_loss, window_grad, window_steps = 0., 0., 0
    synchronize(data["tokens"].device)
    clock = time.perf_counter()
    for step in range(start_step, until_step):
        model.train()
        rows = order[step*config["batch_size"]:(step+1)*config["batch_size"]]
        optimizer.zero_grad(set_to_none=True)
        loss = training_loss(model, data, rows, pruner, objective, config, scale)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
        if not torch.isfinite(loss) or not torch.isfinite(grad):
            raise FloatingPointError(f"Nonfinite loss or gradient: {seed}/{arm}/{objective}/{step+1}")
        optimizer.step()
        window_loss += float(loss.detach())
        window_grad += float(grad)
        window_steps += 1
        current = step + 1
        if current % config["log_interval"] == 0 or current == until_step:
            synchronize(data["tokens"].device)
            row = dict(step=current, objective_loss=window_loss/window_steps, gradient_norm=window_grad/window_steps,
                       elapsed_training_seconds=elapsed+time.perf_counter()-clock)
            new = not history_path.exists()
            with history_path.open("a") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
                if new: writer.writeheader()
                writer.writerow(row)
            print(json.dumps(dict(seed=seed, arm=arm, objective=objective, **row)), flush=True)
            window_loss, window_grad, window_steps = 0., 0., 0
        if current % config["checkpoint_interval"] == 0 or current == until_step:
            synchronize(data["tokens"].device)
            elapsed += time.perf_counter()-clock
            save_checkpoint(directory / f"step_{current:04d}.pt", model, optimizer, current, metadata, elapsed)
            monitor_summary, after = evaluate(model, data, pruner, config, scale, rows=monitor)
            old = initial["patch_indices"]
            changed = config["keep_patch_tokens"] - (old[:, :, None] == after["patch_indices"][:, None, :]).any(axis=2).sum(axis=1)
            monitor_summary["mean_changed_tokens_from_initial"] = float(changed.mean())
            write_json(directory / f"monitor_{current:04d}.json", monitor_summary)
            synchronize(data["tokens"].device)
            clock = time.perf_counter()
    return dict(step=max(start_step, until_step), training_seconds=elapsed)


def write_evaluation_manifest(output, data, scale):
    destination = output / "evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    rows = [dict(image_id=str(data["image_ids"][i]), image_index=int(i), class_id=int(data["labels"][i]),
                 task_id=data["task_ids"][p], class_a=int(data["pairs_array"][p, 0]),
                 class_b=int(data["pairs_array"][p, 1]), task_class_weight=float(w))
            for i, p, w in zip(data["image_ix"], data["pair_ix"], data["weights"])]
    with (destination / "examples.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(destination / "evidence.npz", image_ids=data["image_ids"], labels=data["labels"],
                        text_features=data["text_array"], full_scores=data["full_scores"], logit_scale=scale)


def smoke(config, output, train, val, backbone, pruner, prior, scale, steps):
    from token_compression.direct_classification import straight_through_topk

    device = train["tokens"].device
    rows, _ = paired_order(train, config["training_steps"] * config["batch_size"], config["training_seeds"][0]+2710)
    batch = rows[:config["batch_size"]]
    results = []
    for arm in ARMS:
        model, _, _ = restore_selector(config, config["training_seeds"][0], arm, prior, device)
        tokens, _, text, scores, positions, targets = selected_rows(model, train, batch, config["keep_patch_tokens"])
        policy, _, soft = straight_through_topk(scores, config["keep_patch_tokens"], config["gate_temperature"])
        dense = pruner.finish_policy(tokens, policy)
        with torch.no_grad(): native = pruner.finish(tokens, positions)
        torch.testing.assert_close(F.normalize(dense, dim=-1), F.normalize(native, dim=-1), atol=2e-5, rtol=2e-5)
        classification_metrics(dense, text, targets, scale)["ce"].mean().backward()
        gradient = sum(float(p.grad.square().sum()) for p in model.parameters() if p.grad is not None)**.5
        assert gradient > 0 and np.isfinite(gradient)
        assert all(p.grad is None and not p.requires_grad for p in backbone.parameters())
        error = float((F.normalize(dense.detach(), dim=-1)-F.normalize(native, dim=-1)).abs().max())
        del tokens, text, scores, positions, policy, soft, dense, native
        optimizer = optimizer_for(model, config)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            training_loss(model, train, batch, pruner, "direct", config, scale).backward()
        synchronize(device)
        before = time.perf_counter()
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(model, train, rows[step*config["batch_size"]:(step+1)*config["batch_size"]], pruner, "direct", config, scale)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
            optimizer.step()
        synchronize(device)
        result = dict(arm=arm, normalized_feature_max_error=error, selector_gradient_norm=gradient,
                      backbone_frozen=True, benchmark_steps=steps, seconds_per_step=(time.perf_counter()-before)/steps,
                      max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else 0)
        results.append(result)
        print(json.dumps(result), flush=True)
        del model, optimizer
    write_json(output / "smoke.json", dict(results=results, config=config, smoke_weights_saved=False,
               estimated_six_direct_training_seconds=sum(r["seconds_per_step"] for r in results)*3*config["training_steps"]))


def main(args):
    import clip
    from token_compression.direct_classification import DifferentiablePruner

    config = json.loads(args.config.read_text())
    if args.device: config["device"] = args.device
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    train, val = [load_data(Path(config["cache_dir"]), split, device, config["batch_size"], config["teacher_temperature"])
                  for split in ("train", "val")]
    for data in (train, val): data["labels_tensor"] = torch.as_tensor(data["labels"], dtype=torch.long, device=device)
    assert not set(train["image_ids"]) & set(val["image_ids"])
    assert not set(map(tuple, train["pairs_array"])) & set(map(tuple, val["pairs_array"]))
    backbone, _ = clip.load(config["model"], device=device, jit=False, download_root=config["model_cache"])
    backbone = backbone.float().eval().requires_grad_(False)
    pruner = DifferentiablePruner(backbone.visual, config["prune_after"])
    scale = float(backbone.logit_scale.exp().detach())
    prior = shared_descriptor(train["text"], train["pairs"])
    write_json(args.output / "execution_config.json", dict(config=config, logit_scale=scale,
               prefix_dtype=str(train["tokens"].dtype), backbone_dtype=str(next(backbone.parameters()).dtype)))
    if args.mode == "smoke":
        smoke(config, args.output, train, val, backbone, pruner, prior, scale, args.benchmark_steps)
        return
    write_evaluation_manifest(args.output, val, scale)
    seeds = args.seeds or config["training_seeds"]
    until = args.until_step or config["training_steps"]
    for seed in seeds:
        for arm in args.arms:
            for objective in args.objectives:
                model, _, source = restore_selector(config, seed, arm, prior, device)
                evaluation_dir = args.output / "evaluation" / f"seed_{seed}" / arm
                evaluation_dir.mkdir(parents=True, exist_ok=True)
                if args.mode in ("evaluate", "all") and not (evaluation_dir / "initial.npz").exists():
                    summary, _ = evaluate(model, val, pruner, config, scale, evaluation_dir / "initial.npz")
                    write_json(evaluation_dir / "initial.json", summary)
                if args.mode in ("train", "all"):
                    continue_training(model, train, pruner, config, scale, seed, arm, objective, args.output, until, source)
                if args.mode in ("evaluate", "all"):
                    checkpoint = torch.load(args.output / "training" / f"seed_{seed}" / arm / objective / f"step_{until:04d}.pt", map_location=device, weights_only=False)
                    model.load_state_dict(checkpoint["state_dict"])
                    stage = f"{objective}_step_{until:04d}"
                    summary, _ = evaluate(model, val, pruner, config, scale, evaluation_dir / f"{stage}.npz")
                    write_json(evaluation_dir / f"{stage}.json", summary)
                    print(json.dumps(dict(seed=seed, arm=arm, stage=stage, **summary)), flush=True)
                del model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "train", "evaluate", "all"), default="all")
    parser.add_argument("--device")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--objectives", nargs="+", choices=OBJECTIVES, default=list(OBJECTIVES))
    parser.add_argument("--until-step", type=int)
    parser.add_argument("--benchmark-steps", type=int, default=20)
    main(parser.parse_args())
