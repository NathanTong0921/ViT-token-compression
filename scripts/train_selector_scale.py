"""Compare training-image coverage with paired fixed-update selector training."""

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

from token_compression.selector import (TokenSelector, make_selector, build_examples,
                                        shared_descriptor, utility_targets)
from token_compression.selector_objectives import topk_rank_losses, teacher_topk_diagnostics
from token_compression.selector_scale import (balanced_example_weights,
                                              balanced_training_stream,
                                              map_stream_images,
                                              validate_image_partition)


def read_tasks(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def load_split(cache, split, device, temperature, batch_size):
    tasks = read_tasks(cache / f"{split}_tasks.csv")
    pairs_array = np.array([[int(row["class_a"]), int(row["class_b"])] for row in tasks])
    pairs = torch.as_tensor(pairs_array, dtype=torch.long, device=device)
    folder = cache / split
    with np.load(folder / "evidence.npz") as evidence:
        image_ids, labels_array = evidence["image_ids"].copy(), evidence["labels"].copy()
        labels = torch.as_tensor(labels_array, dtype=torch.long, device=device)
        text = torch.as_tensor(evidence["text_features"], dtype=torch.float32, device=device)
    tokens = torch.from_numpy(np.load(folder / "prefix_tokens.npy")).to(device)
    with np.load(folder / "teacher.npz") as teacher:
        np.testing.assert_array_equal(image_ids, teacher["image_ids"])
        coefficients = torch.as_tensor(teacher["coefficients"], dtype=torch.float32, device=device)
    assert len(tokens) == len(labels) == len(coefficients)
    examples = build_examples(labels, pairs)
    targets = []
    with torch.no_grad():
        for start in range(0, len(examples["image_indices"]), batch_size):
            indices = examples["image_indices"][start:start + batch_size]
            task_pairs = examples["class_pairs"][start:start + batch_size]
            targets.append(utility_targets(coefficients[indices], labels[indices], task_pairs,
                                           text, temperature=temperature))
    return {"tokens": tokens, "text": text, "pairs": pairs, "pairs_array": pairs_array,
            "examples": examples, "targets": torch.cat(targets), "image_ids": image_ids,
            "labels_array": labels_array, "task_ids": [row["task_id"] for row in tasks]}


def evaluation_view(data, allowed_ids=None):
    images = data["examples"]["image_indices"].cpu().numpy()
    tasks = data["examples"]["pair_indices"].cpu().numpy()
    selected = (np.ones(len(images), dtype=bool) if allowed_ids is None
                else np.isin(data["image_ids"][images], allowed_ids))
    weights = balanced_example_weights(data["labels_array"], data["pairs_array"],
                                       images[selected], tasks[selected])
    device = data["tokens"].device
    return {"indices": torch.as_tensor(np.flatnonzero(selected), device=device),
            "weights": torch.as_tensor(weights, dtype=torch.float64, device=device)}


def example_logits(model, data, indices):
    images = data["examples"]["image_indices"][indices]
    pairs = data["examples"]["class_pairs"][indices]
    return model(data["tokens"][images], data["text"][pairs])


def example_losses(model, data, indices, temperature, objective="kl", keep_count=None):
    logits = example_logits(model, data, indices)
    if objective == "topk_rank":
        return topk_rank_losses(logits, data["targets"][indices], keep_count)
    if objective != "kl":
        raise ValueError(f"Unknown selector objective: {objective}")
    return F.kl_div(F.log_softmax(logits / temperature, dim=1),
                    data["targets"][indices], reduction="none").sum(dim=1)


@torch.no_grad()
def evaluate_loss(model, data, view, batch_size, temperature, objective="kl", keep_count=None):
    model.eval()
    total = torch.zeros((), dtype=torch.float64, device=data["tokens"].device)
    for start in range(0, len(view["indices"]), batch_size):
        indices = view["indices"][start:start + batch_size]
        losses = example_losses(model, data, indices, temperature, objective, keep_count)
        total += (losses.double() * view["weights"][start:start + batch_size]).sum()
    return float(total)


@torch.no_grad()
def evaluate_teacher_diagnostics(model, data, view, batch_size, temperature, keep_count):
    """Use common metrics at both endpoints regardless of the training loss."""
    model.eval()
    totals = {key: torch.zeros((), dtype=torch.float64, device=data["tokens"].device)
              for key in ("kl", "topk_overlap", "topk_tie_aware_overlap", "teacher_boundary_tie_fraction")}
    for start in range(0, len(view["indices"]), batch_size):
        indices = view["indices"][start:start + batch_size]
        logits = example_logits(model, data, indices)
        targets = data["targets"][indices]
        metrics = teacher_topk_diagnostics(logits, targets, keep_count)
        metrics["kl"] = F.kl_div(F.log_softmax(logits / temperature, dim=1),
                                 targets, reduction="none").sum(dim=1)
        weights = view["weights"][start:start + batch_size]
        for key, values in metrics.items():
            totals[key] += (values.double() * weights).sum()
    return {key: float(value) for key, value in totals.items()}


def initial_gradient_diagnostics(model, data, order, batch_size, temperature, keep_count, batches=8):
    """Measure both losses on identical initial batches without updating weights."""
    was_training = model.training
    model.eval()
    parameters = tuple(model.parameters())
    rows = []
    for batch in range(min(batches, len(order) // batch_size)):
        indices = order[batch * batch_size:(batch + 1) * batch_size]
        logits = example_logits(model, data, indices)
        targets = data["targets"][indices]
        losses = {"kl": F.kl_div(F.log_softmax(logits / temperature, dim=1),
                                  targets, reduction="none").sum(dim=1).mean(),
                  "topk_rank": topk_rank_losses(logits, targets, keep_count).mean()}
        row = {"batch": batch, "examples": len(indices)}
        for name, loss in losses.items():
            gradients = torch.autograd.grad(loss, parameters, retain_graph=name == "kl")
            norm = torch.stack([g.detach().double().square().sum() for g in gradients]).sum().sqrt()
            row[f"{name}_loss"] = float(loss.detach())
            row[f"{name}_gradient_l2"] = float(norm)
        rows.append(row)
    model.train(was_training)
    return {"batches": rows, "batch_count": len(rows),
            "mean_kl_gradient_l2": float(np.mean([r["kl_gradient_l2"] for r in rows])),
            "mean_topk_rank_gradient_l2": float(np.mean([r["topk_rank_gradient_l2"] for r in rows])),
            "policy": "First eight balanced training minibatches; no optimizer update, rescaling or hyperparameter selection."}


def common_history_metrics(metrics=None):
    keys = ("kl", "topk_overlap", "topk_tie_aware_overlap", "teacher_boundary_tie_fraction")
    return {f"{split}_{key}": None if metrics is None else metrics[split][key]
            for split in ("train", "val") for key in keys}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def write_history(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def train(config, output):
    started = datetime.now(timezone.utc).isoformat()
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    wall_start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    cache, old_cache = Path(config["cache_dir"]), Path(config["source_cache_dir"])
    batch_size, temperature = config["batch_size"], config["temperature"]
    objective = config.get("objective", "kl")
    arms = config.get("selector_arms", ["conditioned", "image_only"])
    if objective not in ("kl", "topk_rank"):
        raise ValueError(f"Unknown selector objective: {objective}")
    # The legacy KL path does not require a retention budget during training.
    keep_count = config.get("keep_patch_tokens")
    loss_options = {"objective": objective, "keep_count": keep_count}
    train_key = "train_kl" if objective == "kl" else "train_objective_loss"
    val_key = "val_kl" if objective == "kl" else "val_objective_loss"
    steps, interval = config["training_steps"], config["checkpoint_interval"]
    assert steps % interval == 0
    data = {split: load_split(cache, split, device, temperature, batch_size)
            for split in ("train", "val")}
    with np.load(old_cache / "train" / "evidence.npz") as evidence:
        small_ids = evidence["image_ids"].copy()
    validate_image_partition(data["train"]["image_ids"], data["val"]["image_ids"], small_ids)
    assert not set(data["train"]["task_ids"]) & set(data["val"]["task_ids"])
    assert read_tasks(cache / "train_tasks.csv") == read_tasks(old_cache / "train_tasks.csv")
    torch.testing.assert_close(data["train"]["text"], data["val"]["text"], rtol=0, atol=0)
    assert all(not item.requires_grad for split in data.values()
               for item in (split["tokens"], split["text"], split["targets"]))
    available_coverages = {"small": small_ids, "full": data["train"]["image_ids"]}
    coverage_ids = {name: available_coverages[name] for name in config["coverages"]}
    train_views = {name: evaluation_view(data["train"], ids) for name, ids in coverage_ids.items()}
    val_view = evaluation_view(data["val"])
    prior = shared_descriptor(data["train"]["text"], data["train"]["pairs"])
    token_dim = data["train"]["tokens"].shape[-1]
    assert data["val"]["tokens"].shape[-1] == token_dim
    # Map a sampled (image, task) directly to the precomputed utility target.
    lookup = np.full((len(data["train"]["image_ids"]), len(data["train"]["pairs"])), -1, np.int64)
    train_images = data["train"]["examples"]["image_indices"].cpu().numpy()
    train_tasks = data["train"]["examples"]["pair_indices"].cpu().numpy()
    lookup[train_images, train_tasks] = np.arange(len(train_images))
    synchronize(device)
    preparation_seconds = time.perf_counter() - wall_start
    history, results, sampling_records, gradient_records = [], [], [], []
    for seed in config["training_seeds"]:
        torch.manual_seed(seed)
        template = TokenSelector(prior, conditioned=True, hidden_dim=config["hidden_dim"],
                                 token_dim=token_dim).to(device)
        initial_state = {key: value.detach().clone() for key, value in template.state_dict().items()}
        del template
        wide_initial_state = None
        if "image_only_wide" in arms:
            torch.manual_seed(seed)
            template = make_selector(prior, "image_only_wide", config["wide_hidden_dim"], token_dim).to(device)
            wide_initial_state = {key: value.detach().clone() for key, value in template.state_dict().items()}
            del template
        stream = balanced_training_stream(data["train"]["pairs_array"], steps * batch_size, seed)
        side_counts = np.bincount(2 * stream["pair_indices"] + stream["sides"])
        stream_record = {"seed": seed, "total_draws": steps * batch_size,
                         "task_side_count_min": int(side_counts.min()),
                         "task_side_count_max": int(side_counts.max()), "coverages": {}}
        for coverage in config["coverages"]:
            selected_images = map_stream_images(data["train"]["labels_array"],
                                                data["train"]["image_ids"], stream,
                                                data["train"]["pairs_array"], coverage_ids[coverage])
            example_indices = lookup[selected_images, stream["pair_indices"]]
            assert np.all(example_indices >= 0)
            order = torch.as_tensor(example_indices, dtype=torch.long, device=device)
            image_counts = np.bincount(selected_images, minlength=len(data["train"]["image_ids"]))
            allowed = np.isin(data["train"]["image_ids"], coverage_ids[coverage])
            assert image_counts[~allowed].sum() == 0
            stream_record["coverages"][coverage] = {
                "available_images": int(allowed.sum()), "unique_images_drawn": int((image_counts > 0).sum()),
                "draws_per_available_image_min": int(image_counts[allowed].min()),
                "draws_per_available_image_max": int(image_counts[allowed].max()),
                "first_batch": {"task_ids": [data["train"]["task_ids"][i] for i in stream["pair_indices"][:batch_size]],
                                "image_ids": data["train"]["image_ids"][selected_images[:batch_size]].tolist(),
                                "sides": stream["sides"][:batch_size].tolist()},
                "image_draw_counts": {str(i): int(n) for i, n in zip(data["train"]["image_ids"][allowed], image_counts[allowed])}}
            seed_output = output / coverage / f"seed_{seed}"
            seed_output.mkdir(parents=True, exist_ok=True)
            for arm in arms:
                hidden_dim = config["wide_hidden_dim"] if arm == "image_only_wide" else config["hidden_dim"]
                model = make_selector(prior, arm, hidden_dim, token_dim).to(device)
                arm_initial_state = wide_initial_state if arm == "image_only_wide" else initial_state
                model.load_state_dict(arm_initial_state)
                assert all(torch.equal(value, arm_initial_state[key]) for key, value in model.state_dict().items())
                parameter_count = sum(parameter.numel() for parameter in model.parameters())
                optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                              weight_decay=config["weight_decay"])
                before = time.perf_counter()
                initial_train = evaluate_loss(model, data["train"], train_views[coverage], batch_size, temperature,
                                              **loss_options)
                initial_val = evaluate_loss(model, data["val"], val_view, batch_size, temperature, **loss_options)
                initial_common, initial_gradients = None, None
                if objective == "topk_rank":
                    initial_common = {
                        split: evaluate_teacher_diagnostics(model, data[split], view, batch_size,
                                                            temperature, keep_count)
                        for split, view in (("train", train_views[coverage]), ("val", val_view))}
                    initial_gradients = initial_gradient_diagnostics(
                        model, data["train"], order, batch_size, temperature, keep_count)
                    gradient_records.append({"seed": seed, "coverage": coverage, "arm": arm,
                                             **initial_gradients})
                    (output / "initial_gradient_diagnostics.json").write_text(
                        json.dumps(gradient_records, indent=2, allow_nan=False) + "\n")
                synchronize(device)
                initial_evaluation_seconds = time.perf_counter() - before
                history.append({"seed": seed, "coverage": coverage, "arm": arm, "step": 0,
                                train_key: initial_train, val_key: initial_val, "train_seconds": 0.,
                                "validation_seconds": initial_evaluation_seconds, "cumulative_train_seconds": 0.})
                if objective == "topk_rank":
                    history[-1].update(objective=objective, **common_history_metrics(initial_common))
                training_seconds, validation_seconds = 0., 0.
                for block_start in range(0, steps, interval):
                    model.train()
                    total = torch.zeros((), device=device)
                    synchronize(device)
                    before = time.perf_counter()
                    for step in range(block_start, block_start + interval):
                        indices = order[step * batch_size:(step + 1) * batch_size]
                        optimizer.zero_grad(set_to_none=True)
                        loss = example_losses(model, data["train"], indices, temperature, **loss_options).mean()
                        loss.backward()
                        optimizer.step()
                        total += loss.detach()
                    synchronize(device)
                    block_seconds = time.perf_counter() - before
                    training_seconds += block_seconds
                    train_loss = float(total / interval)
                    before = time.perf_counter()
                    val_loss = evaluate_loss(model, data["val"], val_view, batch_size, temperature, **loss_options)
                    synchronize(device)
                    val_seconds = time.perf_counter() - before
                    validation_seconds += val_seconds
                    completed_steps = block_start + interval
                    history.append({"seed": seed, "coverage": coverage, "arm": arm, "step": completed_steps,
                                    train_key: train_loss, val_key: val_loss,
                                    "train_seconds": block_seconds, "validation_seconds": val_seconds,
                                    "cumulative_train_seconds": training_seconds})
                    if objective == "topk_rank":
                        history[-1].update(objective=objective, **common_history_metrics())
                    write_history(output / "training_history.csv", history)
                    print(f"coverage={coverage} seed={seed} arm={arm} step={completed_steps}/{steps} "
                          f"{train_key}={train_loss:.6f} {val_key}={val_loss:.6f} "
                          f"train_seconds={block_seconds:.2f}", flush=True)
                before = time.perf_counter()
                final_train = evaluate_loss(model, data["train"], train_views[coverage], batch_size, temperature,
                                            **loss_options)
                final_common = None
                if objective == "topk_rank":
                    final_common = {
                        split: evaluate_teacher_diagnostics(model, data[split], view, batch_size,
                                                            temperature, keep_count)
                        for split, view in (("train", train_views[coverage]), ("val", val_view))}
                    history[-1].update(**common_history_metrics(final_common))
                    write_history(output / "training_history.csv", history)
                synchronize(device)
                final_evaluation_seconds = time.perf_counter() - before
                checkpoint = {"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                              "hidden_dim": hidden_dim, "conditioned": model.conditioned,
                              "selector_variant": arm,
                              "token_dim": token_dim, "text_dim": data["train"]["text"].shape[-1],
                              "num_patches": data["train"]["tokens"].shape[1] - 1,
                              "seed": seed, "step": steps, "coverage": coverage, "parameter_count": parameter_count,
                              "objective": objective,
                              "config": config, "code_commit": source_commit,
                              "checkpoint_policy": "Fixed final update; no validation-based selection."}
                torch.save(checkpoint, seed_output / f"{arm}.pt")
                results.append({"seed": seed, "coverage": coverage, "arm": arm, "parameter_count": parameter_count,
                                "training_steps": steps, "training_draws": steps * batch_size,
                                "available_training_images": len(coverage_ids[coverage]),
                                "initial_train_kl": initial_train, "initial_val_kl": initial_val,
                                "final_train_kl": final_train, "final_val_kl": val_loss,
                                "training_seconds": training_seconds, "validation_seconds": validation_seconds,
                                "initial_evaluation_seconds": initial_evaluation_seconds,
                                "final_train_evaluation_seconds": final_evaluation_seconds,
                                "initial_state_exactly_matches_shared_seed_state": arm != "image_only_wide",
                                "initial_state_exactly_matches_own_architecture_seed_state": True,
                                "initialization_group": "image_only_wide" if arm == "image_only_wide" else "original_descriptor_network",
                                "task_class_quantile_stream_reused_between_all_four_runs": True,
                                "image_draws_reused_between_arms": True,
                                "clip_parameters_in_optimizer": 0,
                                "checkpoint": str(Path(coverage) / f"seed_{seed}" / f"{arm}.pt")})
                if "selector_arms" in config:
                    results[-1].pop("task_class_quantile_stream_reused_between_all_four_runs")
                    results[-1]["task_class_quantile_stream_reused_between_arms"] = True
                if objective == "topk_rank":
                    results[-1].update(
                        objective=objective,
                        initial_train_objective_loss=initial_train, initial_val_objective_loss=initial_val,
                        final_train_objective_loss=final_train, final_val_objective_loss=val_loss,
                        initial_train_kl=initial_common["train"]["kl"], initial_val_kl=initial_common["val"]["kl"],
                        final_train_kl=final_common["train"]["kl"], final_val_kl=final_common["val"]["kl"],
                        initial_teacher_diagnostics=initial_common, final_teacher_diagnostics=final_common,
                        initial_gradient_diagnostics=initial_gradients)
                (output / "completed_runs.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
                del model, optimizer
        sampling_records.append(stream_record)
        (output / "sampling_audit.json").write_text(json.dumps(sampling_records, indent=2, allow_nan=False) + "\n")
    synchronize(device)
    summary = {"run_type": "fixed_update_paired_selector_image_coverage", "started_utc": started,
               "experiment_name": config.get("experiment_name"),
               "code_commit": source_commit, "config": config,
               "environment": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()},
               "device": str(device),
               "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
               "train_image_counts": {name: len(ids) for name, ids in coverage_ids.items()},
               "val_image_count": len(data["val"]["image_ids"]),
               "train_task_count": len(data["train"]["task_ids"]), "val_task_count": len(data["val"]["task_ids"]),
               "selector_dimensions": {"token_dim": token_dim, "text_dim": data["train"]["text"].shape[-1],
                                       "num_patches": data["train"]["tokens"].shape[1] - 1},
               "train_example_counts": {name: len(view["indices"]) for name, view in train_views.items()},
               "val_example_count": len(val_view["indices"]),
               "frozen_prefix_cache_bytes": sum(split["tokens"].numel() * split["tokens"].element_size() for split in data.values()),
               "preparation_seconds": preparation_seconds, "wall_seconds": time.perf_counter() - wall_start,
               "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
               "supervision_cost_record": str(cache / "prep_summary.json"),
               "training_policy": "Float32 selector only, frozen cached CLIP features; fixed final update with no validation-based selection.",
               "sampling_policy": "One NumPy PCG64 CPU stream per seed: shuffle equal repetitions of each task/side, then draw uniform image quantiles. Map each quantile into alphabetically sorted available class images. Both arms reuse the identical mapped stream, and both coverages reuse tasks, sides and quantiles.",
               "sampling_audit": "sampling_audit.json",
               "evaluation_weighting": "Equal weight per task, equal weight per class within each task, then uniform over available images of that class.",
               "curve_train_kl": "Step 0 is balanced full-panel evaluation. Later points are average online losses from the preceding checkpoint_interval updates; final_train_kl in runs is evaluated over the full allowed panel.",
               "capacity_comparison": "Identical architecture and parameter count; image-only uses mean training-pair descriptor. Each seed shares its exact initial state across both coverages and both arms.",
               "runs": results}
    if objective == "topk_rank":
        summary.update(
            run_type="fixed_update_paired_selector_objective", objective=objective,
            objective_definition="Mean softplus(s_out-s_in) over strict teacher top-k boundary preferences; boundary ties expanded symmetrically, equal target values ignored, uniform targets have zero loss/gradient.",
            rank_temperature="Raw selector logits at unit scale; existing temperature is retained only for the common KL target and diagnostic.",
            curve_train_objective_loss="Step 0 is balanced full-panel evaluation. Later points average online rank losses from preceding updates; final_train_objective_loss is a separate balanced full-panel evaluation.",
            curve_train_kl="Common KL and top-k diagnostics are evaluated on full balanced train/validation panels at initialization and the final step only; intervening CSV values are empty.",
            initial_gradient_diagnostics="initial_gradient_diagnostics.json",
            initial_gradient_cost="Included in initial_evaluation_seconds; no rescaling or hyperparameter adjustment was performed.")
    if "selector_arms" in config:
        summary.update(
            run_type="fixed_update_selector_component_ablations", selector_arms=arms,
            capacity_comparison="Mean-only and difference-only retain the original network and its exact per-seed initial state; the missing raw descriptor half uses the training-pair mean before context normalization. The image-only-wide arm reallocates parameters to a wider image network, independently initialized with the same seed; its tensor shapes and parameter count differ.",
            sampling_policy="The same balanced task/class/image stream is reused by every selector arm. Existing full-coverage KL streams are unchanged.")
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"wall_seconds": summary["wall_seconds"], "model_runs": len(results),
                      "parameter_count": results[0]["parameter_count"]}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    train(json.loads(args.config.read_text()), args.output)
