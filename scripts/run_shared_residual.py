"""Learn task or image score corrections around frozen strong shared selectors."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from evaluate_core_protection import padded_batches, unique_image_masks
from evaluate_selector_scale import selector_from_checkpoint
from evaluate_selectors import binary_outcomes, write_rows
from run_semantic_identity_pilot import load_data, synchronize, write_json
from semantic_identity_protocol import fixed_step_stream
from token_compression.selector import shared_descriptor
from token_compression.selector_scale import map_stream_images
from token_compression.shared_residual import SharedResidualSelector, shared_features


SETTINGS = {
    "pets_b16_49": ("configs/pets_selector_ablations_49.json", "runs/pets-selector-ablations-49-training-v1"),
    "flowers_b16_49": ("configs/flowers_selector_ablations_49.json", "runs/flowers-selector-ablations-49-training-v1"),
    "flowers_b16_98": ("configs/flowers_selector_ablations_98.json", "runs/flowers-selector-ablations-98-training-v1"),
}
ARMS = ("task_residual", "image_residual")
METHODS = ("shared", *ARMS)


@torch.no_grad()
def cache_shared(shared, tokens, batch_size):
    """Normal detached tensors support later branch-weight gradients."""
    hidden = torch.empty((len(tokens), tokens.shape[1] - 1, shared.hidden_dim), device=tokens.device)
    scores = torch.empty(hidden.shape[:2], device=tokens.device)
    for start, end, rows in padded_batches(len(tokens), batch_size):
        h, s = shared_features(shared, tokens[torch.as_tensor(rows, device=tokens.device)])
        hidden[start:end], scores[start:end] = h[:end - start], s[:end - start]
    return hidden, scores


def cached_losses(model, data, cached, rows, temperature):
    ix = data["examples"]["image_indices"][rows]
    pairs = data["examples"]["class_pairs"][rows]
    logits = model.score_cached(cached[0][ix], cached[1][ix], data["text"][pairs])
    return F.kl_div(F.log_softmax(logits / temperature, dim=1), data["targets"][rows],
                    reduction="none").sum(1)


@torch.no_grad()
def mean_teacher_loss(model, data, cached, batch_size, temperature):
    model.eval()
    total = 0.
    for start in range(0, len(data["weights"]), batch_size):
        end = min(start + batch_size, len(data["weights"]))
        rows = torch.arange(start, end, device=data["tokens"].device)
        value = cached_losses(model, data, cached, rows, temperature).double().cpu().numpy()
        total += float(value @ data["weights"][start:end])
    return total


def paired_order(data, total_draws, seed):
    stream = fixed_step_stream(data["pairs_array"], total_draws, seed)
    image_ix = map_stream_images(data["labels"], data["image_ids"], stream, data["pairs_array"])
    lookup = np.full((len(data["image_ids"]), len(data["pairs_array"])), -1, dtype=np.int64)
    lookup[data["image_ix"], data["pair_ix"]] = np.arange(len(data["image_ix"]))
    order = lookup[image_ix, stream["pair_indices"]]
    assert np.all(order >= 0)
    counts = np.bincount(2 * stream["pair_indices"] + stream["sides"], minlength=2 * len(data["pairs_array"]))
    audit = {"seed": seed, "draws": len(order), "task_side_min": int(counts.min()),
             "task_side_max": int(counts.max()), "unique_training_images": len(np.unique(image_ix))}
    return torch.as_tensor(order, dtype=torch.long, device=data["tokens"].device), audit


@torch.inference_mode()
def evaluate(models, shared, data, pruner, k, batch_size, destination):
    """Use online base+branch forward, then deduplicate identical CLIP suffix masks."""
    count, patches = len(data["image_ix"]), data["tokens"].shape[1] - 1
    device = data["tokens"].device
    logits = np.empty((len(METHODS), count, patches), dtype=np.float32)
    selector_seconds = {}
    for mi, method in enumerate(METHODS):
        model = shared if method == "shared" else models[method]
        model.eval()
        unique = method != "task_residual"
        indices = np.arange(len(data["image_ids"])) if unique else data["image_ix"]
        saved = np.empty((len(indices), patches), np.float32)
        synchronize(device)
        before = time.perf_counter()
        for start, end, rows in padded_batches(len(indices), batch_size):
            positions = torch.as_tensor(rows, device=device)
            pairs = (data["pairs"][:1].expand(len(rows), -1) if unique
                     else data["examples"]["class_pairs"][positions])
            value = model(data["tokens"][torch.as_tensor(indices[rows], device=device)], data["text"][pairs])
            saved[start:end] = value[:end - start].float().cpu().numpy()
        synchronize(device)
        selector_seconds[method] = time.perf_counter() - before
        logits[mi] = saved[data["image_ix"]] if unique else saved
    masks = np.sort(np.argsort(-logits, axis=-1, kind="stable")[..., :k], axis=-1).astype(np.int16)
    keys, inverse = unique_image_masks(data["image_ix"], masks)
    features = np.empty((len(keys), data["text"].shape[1]), np.float32)
    synchronize(device)
    before = time.perf_counter()
    for start, end, rows in padded_batches(len(keys), batch_size):
        current = keys[rows]
        result = pruner.finish(data["tokens"][torch.as_tensor(current[:, 0], device=device)],
                              torch.as_tensor(current[:, 1:], dtype=torch.long, device=device)).float()
        result = (result / result.norm(dim=-1, keepdim=True)).cpu().numpy()
        features[start:end] = result[:end - start]
    synchronize(device)
    suffix_seconds = time.perf_counter() - before
    membership = np.zeros((*masks.shape[:-1], patches), bool)
    np.put_along_axis(membership, masks.astype(int), True, axis=-1)
    changes = k - (membership & membership[:1]).sum(axis=-1)
    residual = logits - logits[:1]
    np.savez_compressed(destination, methods=np.asarray(METHODS), cosine_scores=(features @ data["text_array"].T)[inverse],
                        patch_indices=masks, selector_logits=logits, unique_keys=keys,
                        unique_features=features, feature_index=inverse, changed_tokens_vs_shared=changes,
                        correction_patch_std=residual.std(axis=-1), correction_patch_rms=np.sqrt((residual ** 2).mean(axis=-1)),
                        shared_score_patch_std=logits[0].std(axis=-1))
    return {"online_selector_seconds": selector_seconds, "unique_suffix_seconds": suffix_seconds,
            "unique_image_masks": len(keys), "raw_image_masks": int(np.prod(masks.shape[:2])),
            "timing_scope": "Batched selector replay and deduplicated cached-prefix suffix evaluation; not end-to-end latency."}


def audit_shared_endpoint(data, predictions, reference, seed, tolerance):
    """Align the historical shared endpoint by image/task and compare predictions."""
    with (reference / "examples.csv").open() as handle:
        previous_rows = list(csv.DictReader(handle))
    lookup = {(r["image_id"], r["task_id"]): i for i, r in enumerate(previous_rows)}
    keys = [(str(data["image_ids"][i]), data["task_ids"][p]) for i, p in zip(data["image_ix"], data["pair_ix"])]
    assert set(keys) == set(lookup)
    order = np.asarray([lookup[key] for key in keys])
    with np.load(reference / "evidence.npz") as saved:
        np.testing.assert_array_equal(saved["image_ids"], data["image_ids"])
        np.testing.assert_array_equal(saved["full_scores"], data["full_scores"])
        np.testing.assert_array_equal(saved["text_features"], data["text_array"])
    with np.load(reference / f"seed_{seed}.npz") as old, np.load(predictions) as new:
        old_index, new_index = list(old["variants"]).index("wide"), list(new["methods"]).index("shared")
        previous_scores = old["cosine_scores"][old_index, order]
        current_scores = new["cosine_scores"][new_index]
        np.testing.assert_array_equal(old["patch_indices"][old_index, order], new["patch_indices"][new_index])
        np.testing.assert_allclose(current_scores, previous_scores, atol=tolerance, rtol=0)
    labels = data["labels"][data["image_ix"]]
    pairs = data["pairs_array"][data["pair_ix"]]
    old_margin, old_correct = binary_outcomes(previous_scores, labels, pairs, data["text_array"])
    new_margin, new_correct = binary_outcomes(current_scores, labels, pairs, data["text_array"])
    return {"source": str(reference.resolve()), "retained_masks_exact": True,
            "full_model_scores_exact": True, "cosine_score_max_absolute_error": float(abs(current_scores - previous_scores).max()),
            "accuracy_flip_rows": int(np.sum(old_correct != new_correct)),
            "balanced_accuracy_change": float(data["weights"] @ (new_correct.astype(float) - old_correct.astype(float))),
            "balanced_margin_change": float(data["weights"] @ (new_margin - old_margin))}


def run(config, shared_dir, output, setting):
    import clip
    from token_compression.pruning import SingleStagePruner

    output, shared_dir = Path(output), Path(shared_dir)
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    device = torch.device(config["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    temperature, batch_size = config["temperature"], config["batch_size"]
    train, val = [load_data(Path(config["cache_dir"]), split, device, batch_size, temperature)
                  for split in ("train", "val")]
    assert not set(train["image_ids"]) & set(val["image_ids"])
    train_pairs = {tuple(sorted(pair)) for pair in train["pairs_array"].tolist()}
    val_pairs = {tuple(sorted(pair)) for pair in val["pairs_array"].tolist()}
    assert not train_pairs & val_pairs
    torch.testing.assert_close(train["text"], val["text"], rtol=0, atol=0)
    backbone, _ = clip.load(config["model"], device=device, jit=False, download_root=config["model_cache"])
    backbone.eval().requires_grad_(False)
    pruner = SingleStagePruner(backbone.visual, config["prune_after"])
    prior = shared_descriptor(train["text"], train["pairs"])
    examples = [{"image_index": int(image_index), "image_id": str(val["image_ids"][image_index]),
                 "class_id": int(val["labels"][image_index]), "task_id": val["task_ids"][pair_index],
                 "class_a": int(val["pairs_array"][pair_index, 0]), "class_b": int(val["pairs_array"][pair_index, 1]),
                 "task_split": "unseen", "task_class_weight": float(weight)}
                for image_index, pair_index, weight in zip(val["image_ix"], val["pair_ix"], val["weights"])]
    write_rows(output / "examples.csv", examples)
    np.savez_compressed(output / "evidence.npz", image_ids=val["image_ids"], labels=val["labels"],
                        text_features=val["text_array"], full_scores=val["full_scores"])
    protocol = {"setting": setting, "config": config, "git_commit": commit,
                "shared_dir": str(shared_dir.resolve()), "training_seeds": config["training_seeds"],
                "methods": list(METHODS), "primary": "Task correction minus equally budgeted image-only correction on unseen validation task combinations; three settings.",
                "training": "Freeze trained shared selector. Zero-initialize correction output. Same original KL targets, paired image/task stream, optimizer and fixed steps for both corrections. No validation checkpoint selection.",
                "inference": "Compute shared199-dimensional penultimate features online, then add the branch score. No label input. Cached hidden features are used only for fitting.",
                "prior_cost": "Existing wide selector checkpoints and frozen CLIP/teacher caches are reused; recorded new training cost excludes their earlier creation.",
                "final_test_used": False}
    write_json(output / "protocol.json", protocol)
    records, curves, sampling, evaluation_records = [], [], [], []
    for seed in config["training_seeds"]:
        checkpoint_path = shared_dir / "full" / f"seed_{seed}/image_only_wide.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        shared = selector_from_checkpoint(checkpoint, prior, "image_only_wide", train["tokens"].shape[-1], device)
        synchronize(device)
        before = time.perf_counter()
        train_cache = cache_shared(shared, train["tokens"], config["image_batch_size"])
        val_cache = cache_shared(shared, val["tokens"], config["image_batch_size"])
        synchronize(device)
        cache_seconds = time.perf_counter() - before
        order, audit = paired_order(train, config["training_steps"] * batch_size, seed)
        sampling.append(audit)
        models, initial_losses = {}, []
        for arm in ARMS:
            torch.manual_seed(seed)
            model = SharedResidualSelector(shared, conditioned=arm == "task_residual",
                text_dim=train["text"].shape[1]).to(device)
            initial = mean_teacher_loss(model, val, val_cache, batch_size, temperature)
            initial_losses.append(initial)
            optimizer = torch.optim.AdamW(model.branch.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
            model.train()
            synchronize(device)
            before = time.perf_counter()
            for step in range(config["training_steps"]):
                rows = order[step * batch_size:(step + 1) * batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = cached_losses(model, train, train_cache, rows, temperature).mean()
                loss.backward()
                optimizer.step()
                if step == 0 or (step + 1) % config["checkpoint_interval"] == 0 or step + 1 == config["training_steps"]:
                    curves.append({"seed": seed, "arm": arm, "step": step + 1, "batch_teacher_kl": float(loss.detach())})
            synchronize(device)
            seconds = time.perf_counter() - before
            train_kl = mean_teacher_loss(model, train, train_cache, batch_size, temperature)
            val_kl = mean_teacher_loss(model, val, val_cache, batch_size, temperature)
            folder = output / f"seed_{seed}"
            folder.mkdir(exist_ok=True)
            torch.save({"branch_state_dict": {key: value.detach().cpu().clone() for key, value in model.branch.state_dict().items()},
                        "conditioned": model.conditioned, "hidden_dim": model.branch.hidden_dim,
                        "shared_hidden_dim": shared.hidden_dim, "text_dim": train["text"].shape[1],
                        "shared_checkpoint": str(checkpoint_path.resolve()), "seed": seed, "arm": arm,
                        "config": config, "git_commit": commit}, folder / f"{arm}.pt")
            record = {"seed": seed, "arm": arm, "initial_val_teacher_kl": initial,
                      "final_train_teacher_kl": train_kl, "final_val_teacher_kl": val_kl,
                      "training_seconds": seconds, "parameters": model.parameter_counts(),
                      "shared_checkpoint": str(checkpoint_path.resolve())}
            records.append(record)
            models[arm] = model.eval().requires_grad_(False)
            write_json(output / "training_records.json", records)
            write_rows(output / "training_curve.csv", curves)
            print(json.dumps({"setting": setting, **record}), flush=True)
        assert initial_losses[0] == initial_losses[1]
        measured = evaluate(models, shared, val, pruner, config["keep_patch_tokens"], config["image_batch_size"],
                            output / f"seed_{seed}.npz")
        if setting in SETTINGS:
            measured["historical_shared_endpoint"] = audit_shared_endpoint(val, output / f"seed_{seed}.npz",
                Path("runs/core-protection-v1") / setting, seed, config["identity_atol"])
        evaluation_records.append({"seed": seed, "shared_feature_cache_seconds": cache_seconds,
                                   "identical_initial_val_loss": True, **measured})
        write_json(output / "evaluation_records.json", evaluation_records)
        write_json(output / "sampling_audit.json", sampling)
        del train_cache, val_cache, models, shared, model, optimizer
    from analyze_shared_residual import analyze
    result = analyze(output, output / "analysis", config.get("bootstrap_replicates", 2000))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=SETTINGS, required=True)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path, shared_dir = SETTINGS[args.setting]
    config = json.loads(Path(config_path).read_text())
    if args.device:
        config["device"] = args.device
    run(config, shared_dir, args.output, args.setting)
