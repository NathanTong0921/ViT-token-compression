"""Retrain true versus fixed-permuted task descriptors on development caches."""

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from evaluate_selectors import binary_outcomes
from semantic_identity_protocol import (class_partition, class_permutation, fixed_step_stream,
                                        mapped_candidate_text, pair_strata, training_membership,
                                        weighted_image_interval, paired_stratum_interaction)
from token_compression.selector import build_examples, make_selector, shared_descriptor, utility_targets
from token_compression.selector_scale import balanced_example_weights, map_stream_images


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_data(cache, split, device, batch_size, temperature, seen_classes=None):
    with (cache / f"{split}_tasks.csv").open() as handle:
        tasks = list(csv.DictReader(handle))
    pairs = np.asarray([[int(t["class_a"]), int(t["class_b"])] for t in tasks])
    with np.load(cache / split / "evidence.npz") as saved:
        ids, labels = saved["image_ids"].copy(), saved["labels"].copy()
        text_array = saved["text_features"].copy()
        full_scores = saved["full_scores"].copy()
    image_keep, pair_keep = (np.ones(len(ids), bool), np.ones(len(pairs), bool))
    if seen_classes is not None:
        image_keep, pair_keep = training_membership(labels, pairs, seen_classes)
    pairs = pairs[pair_keep]
    tasks = [t for t, keep in zip(tasks, pair_keep) if keep]
    original_ids = ids
    ids, labels, full_scores = ids[image_keep], labels[image_keep], full_scores[image_keep]
    # Filter before placing image tensors/teacher coefficients on the GPU.
    tokens = torch.as_tensor(np.load(cache / split / "prefix_tokens.npy", mmap_mode="r")[image_keep].copy(), device=device)
    with np.load(cache / split / "teacher.npz") as saved:
        np.testing.assert_array_equal(original_ids, saved["image_ids"])
        coefficients = torch.as_tensor(saved["coefficients"][image_keep].copy(), dtype=torch.float32, device=device)
    text = torch.as_tensor(text_array, dtype=torch.float32, device=device)
    labels_tensor = torch.as_tensor(labels, dtype=torch.long, device=device)
    pairs_tensor = torch.as_tensor(pairs, dtype=torch.long, device=device)
    examples = build_examples(labels_tensor, pairs_tensor)
    targets = []
    with torch.no_grad():
        for start in range(0, len(examples["image_indices"]), batch_size):
            ix = examples["image_indices"][start:start + batch_size]
            ps = examples["class_pairs"][start:start + batch_size]
            targets.append(utility_targets(coefficients[ix], labels_tensor[ix], ps, text, temperature))
    image_ix = examples["image_indices"].cpu().numpy()
    pair_ix = examples["pair_indices"].cpu().numpy()
    weights = balanced_example_weights(labels, pairs, image_ix, pair_ix)
    return dict(tokens=tokens, text=text, text_array=text_array, labels=labels, image_ids=ids,
                pairs=pairs_tensor, pairs_array=pairs, examples=examples, targets=torch.cat(targets),
                weights=weights, image_ix=image_ix, pair_ix=pair_ix, full_scores=full_scores,
                task_ids=[t["task_id"] for t in tasks])


def selector_logits(model, data, indices, mapping):
    image_ix = data["examples"]["image_indices"][indices]
    pairs = data["examples"]["class_pairs"][indices]
    candidates = mapped_candidate_text(data["text"], pairs, mapping)
    return model(data["tokens"][image_ix], candidates)


def losses(model, data, indices, mapping, temperature):
    logits = selector_logits(model, data, indices, mapping)
    return F.kl_div(F.log_softmax(logits / temperature, dim=1), data["targets"][indices],
                    reduction="none").sum(1)


@torch.no_grad()
def teacher_loss(model, data, mapping, batch_size, temperature):
    model.eval()
    total = 0.
    for start in range(0, len(data["weights"]), batch_size):
        ix = torch.arange(start, min(start + batch_size, len(data["weights"])), device=data["tokens"].device)
        value = losses(model, data, ix, mapping, temperature).double().cpu().numpy()
        total += float(value @ data["weights"][start:start + len(ix)])
    return total


def make_model(prior, arm, config, token_dim, device):
    wide = arm == "image_only_wide"
    return make_selector(prior, "image_only_wide" if wide else "conditioned",
                         config.get("wide_hidden_dim", 199) if wide else config["hidden_dim"], token_dim).to(device)


@torch.inference_mode()
def evaluate_model(model, data, mapping, pruner, keep, batch_size, output):
    model.eval()
    unique = not model.conditioned
    image_indices = np.arange(len(data["image_ids"])) if unique else data["image_ix"]
    count, device = len(image_indices), data["tokens"].device
    features = np.empty((count, data["text"].shape[-1]), np.float32)
    positions = np.empty((count, keep), np.int16)
    for start in range(0, count, batch_size):
        valid = min(batch_size, count - start)
        rows = np.arange(start, start + valid)
        rows = np.pad(rows, (0, batch_size - valid), mode="edge")
        tokens = data["tokens"][torch.as_tensor(image_indices[rows], device=device)]
        pairs = (data["pairs"][:1].expand(batch_size, -1) if unique
                 else data["examples"]["class_pairs"][torch.as_tensor(rows, device=device)])
        values = model(tokens.float(), mapped_candidate_text(data["text"], pairs, mapping)).cpu().numpy()
        selected = np.sort(np.argsort(-values, axis=1, kind="stable")[:, :keep], axis=1)
        result = pruner.finish(tokens, torch.as_tensor(selected, dtype=torch.long, device=device)).float()
        result = (result / result.norm(dim=-1, keepdim=True)).cpu().numpy()
        features[start:start + valid] = result[:valid]
        positions[start:start + valid] = selected[:valid]
    if unique:
        features, positions = features[data["image_ix"]], positions[data["image_ix"]]
    # Original CLIP dictionary and original class pairs are always used here.
    margin, correct = binary_outcomes(features @ data["text_array"].T,
                                     data["labels"][data["image_ix"]],
                                     data["pairs_array"][data["pair_ix"]], data["text_array"])
    np.savez_compressed(output, image_features=features, patch_indices=positions,
                        margin=margin, correct=correct)
    return {"margin": margin, "correct": correct}


def summarize(data, predictions, arms, seeds, strata, replicates):
    ids = data["image_ids"][data["image_ix"]]
    weights = data["weights"]
    metrics = {metric: {arm: np.stack([predictions[(seed, arm)][metric] for seed in seeds])
                       for arm in arms} for metric in ("correct", "margin")}
    full_margin, full_correct = binary_outcomes(data["full_scores"][data["image_ix"]],
        data["labels"][data["image_ix"]], data["pairs_array"][data["pair_ix"]], data["text_array"])
    report = {"strata": {}, "interactions": {},
              "uncertainty": "Paired image bootstrap after averaging all fixed weight seeds and both mappings; conditional on this class/task partition. Mapping-specific and seed-specific estimates retained."}
    for group in np.unique(strata):
        selected = strata == group
        part = {"images": len(np.unique(ids[selected])), "examples": int(selected.sum()),
                "tasks": len(np.unique(data["pair_ix"][selected])),
                "full_model_accuracy_percent": float(np.average(full_correct[selected], weights=weights[selected]) * 100),
                "full_model_margin": float(np.average(full_margin[selected], weights=weights[selected])), "metrics": {}}
        for metric, arrays in metrics.items():
            means = {arm: value.mean(0) for arm, value in arrays.items()}
            perms = np.mean([means[a] for a in arms if a.startswith("permuted_")], axis=0)
            scale = 100. if metric == "correct" else 1.
            contrasts = {"real_minus_permuted_mean": means["real_text"] - perms,
                         "real_minus_wide": means["real_text"] - means["image_only_wide"],
                         "permuted_mean_minus_wide": perms - means["image_only_wide"]}
            part["metrics"][metric] = {
                "unit": "percentage points" if metric == "correct" else "geometric margin",
                "arms": {a: {"mean": float(np.average(v[:, selected].mean(0), weights=weights[selected]) * scale),
                             "per_seed": [float(np.average(row[selected], weights=weights[selected]) * scale) for row in v]}
                         for a, v in arrays.items()},
                "contrasts": {name: weighted_image_interval(ids[selected], weights[selected],
                                                             difference[selected] * scale, replicates)
                              for name, difference in contrasts.items()}}
        report["strata"][group] = part
    if {"known_known", "new_new"} <= set(strata):
        for metric, arrays in metrics.items():
            difference = arrays["real_text"].mean(0) - np.mean(
                [v.mean(0) for a, v in arrays.items() if a.startswith("permuted_")], axis=0)
            scale = 100. if metric == "correct" else 1.
            report["interactions"][metric] = paired_stratum_interaction(
                ids, weights, difference * scale, strata, "new_new", "known_known", replicates)
    return report


def run(config, output, regimes, permutation_seeds, partition_seed, replicates):
    import clip
    from token_compression.pruning import SingleStagePruner

    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    device = torch.device(config["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    cache = Path(config["cache_dir"])
    batch_size, temperature = config["batch_size"], config["temperature"]
    val = load_data(cache, "val", device, batch_size, temperature)
    classes = len(val["text"])
    seen, held = class_partition(classes, partition_seed)
    backbone, _ = clip.load(config["model"], device=device, jit=False, download_root=config["model_cache"])
    backbone.eval().requires_grad_(False)
    pruner = SingleStagePruner(backbone.visual, config["prune_after"])
    arms = ["real_text"] + [f"permuted_{seed}" for seed in permutation_seeds] + ["image_only_wide"]
    protocol = dict(config=config, code_commit=commit, regimes=regimes, arms=arms,
                    partition_seed=partition_seed, seen_classes=seen.tolist(), held_classes=held.tolist(),
                    permutation_seeds=permutation_seeds, training_seeds=config["training_seeds"],
                    final_test_used=False, official_test_used=False,
                    evaluation="Original cached validation images and validation task combinations only",
                    label_use="Original labels/text for teacher and CLIP scoring; only selector candidate encodings permuted.")
    write_json(output / "protocol.json", protocol)
    reports = {}
    for regime in regimes:
        folder = output / regime
        folder.mkdir(exist_ok=True)
        data = load_data(cache, "train", device, batch_size, temperature,
                         seen if regime == "class_holdout" else None)
        assert not set(data["image_ids"]) & set(val["image_ids"])
        assert not set(data["task_ids"]) & set(val["task_ids"])
        torch.testing.assert_close(data["text"], val["text"], rtol=0, atol=0)
        mappings = {"real_text": np.arange(classes), "image_only_wide": np.arange(classes)}
        for seed in permutation_seeds:
            mappings[f"permuted_{seed}"] = class_permutation(classes, seed,
                (seen, held) if regime == "class_holdout" else None)
        if regime == "class_holdout":
            assert not np.isin(data["labels"], held).any()
            assert not np.isin(data["pairs_array"], held).any()
        prior = shared_descriptor(data["text"], data["pairs"])
        token_dim = data["tokens"].shape[-1]
        mapping_tensors = {arm: torch.as_tensor(value, dtype=torch.long, device=device) for arm, value in mappings.items()}
        strata = (pair_strata(val["pairs_array"][val["pair_ix"]], held)
                  if regime == "class_holdout" else np.full(len(val["weights"]), "unseen_combinations"))
        base_margin, base_correct = binary_outcomes(val["full_scores"][val["image_ix"]],
            val["labels"][val["image_ix"]], val["pairs_array"][val["pair_ix"]], val["text_array"])
        np.savez_compressed(folder / "evaluation_panel.npz", image_ids=val["image_ids"][val["image_ix"]],
            class_pairs=val["pairs_array"][val["pair_ix"]], class_ids=val["labels"][val["image_ix"]],
            task_ids=np.asarray(val["task_ids"])[val["pair_ix"]], weights=val["weights"], strata=strata,
            full_margin=base_margin, full_correct=base_correct)
        write_json(folder / "mapping_audit.json", dict(
            mappings={a: m.tolist() for a, m in mappings.items()},
            fixed_points={a: int(np.sum(m == np.arange(classes))) for a, m in mappings.items()},
            training_images=len(data["image_ids"]), training_tasks=len(data["pairs_array"]),
            training_class_ids=np.unique(data["labels"]).tolist(),
            training_task_ids=data["task_ids"], training_image_ids=data["image_ids"].tolist(),
            training_vector_bank_preserved=True, final_test_used=False))
        lookup = np.full((len(data["image_ids"]), len(data["pairs_array"])), -1, np.int64)
        lookup[data["image_ix"], data["pair_ix"]] = np.arange(len(data["image_ix"]))
        predictions, records, sampling = {}, [], []
        for seed in config["training_seeds"]:
            stream = fixed_step_stream(data["pairs_array"], config["training_steps"] * batch_size, seed)
            image_ix = map_stream_images(data["labels"], data["image_ids"], stream, data["pairs_array"])
            order = torch.as_tensor(lookup[image_ix, stream["pair_indices"]], dtype=torch.long, device=device)
            assert (order >= 0).all()
            counts = np.bincount(2 * stream["pair_indices"] + stream["sides"], minlength=2 * len(data["pairs_array"]))
            sampling.append(dict(seed=seed, total_draws=len(order), task_side_min=int(counts.min()),
                                 task_side_max=int(counts.max()), unique_images=int(len(np.unique(image_ix)))))
            for arm in arms:
                torch.manual_seed(seed)
                model = make_model(prior, arm, config, token_dim, device)
                mapping = mapping_tensors[arm]
                optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
                synchronize(device)
                started = time.perf_counter()
                initial = teacher_loss(model, val, mapping, batch_size, temperature)
                model.train()
                for step in range(config["training_steps"]):
                    ix = order[step * batch_size:(step + 1) * batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    loss = losses(model, data, ix, mapping, temperature).mean()
                    loss.backward()
                    optimizer.step()
                synchronize(device)
                training_seconds = time.perf_counter() - started
                final_train = teacher_loss(model, data, mapping, batch_size, temperature)
                final_val = teacher_loss(model, val, mapping, batch_size, temperature)
                filename = f"seed_{seed}_{arm}"
                torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                                arm=arm, seed=seed, regime=regime, mapping=mappings[arm].tolist(),
                                config=config, code_commit=commit, token_dim=token_dim,
                                final_test_used=False), folder / f"{filename}.pt")
                before_eval = time.perf_counter()
                predictions[(seed, arm)] = evaluate_model(model, val, mapping, pruner,
                    config["keep_patch_tokens"], config["image_batch_size"], folder / f"{filename}.npz")
                synchronize(device)
                record = dict(seed=seed, regime=regime, arm=arm, initial_val_kl=initial,
                              final_train_kl=final_train, final_val_kl=final_val,
                              parameters=sum(p.numel() for p in model.parameters()),
                              training_seconds=training_seconds, evaluation_seconds=time.perf_counter() - before_eval)
                records.append(record)
                write_json(folder / "training_records.json", records)
                print(json.dumps(record), flush=True)
                del model, optimizer
        write_json(folder / "sampling_audit.json", sampling)
        reports[regime] = summarize(val, predictions, arms, config["training_seeds"], strata, replicates)
        write_json(folder / "summary.json", reports[regime])
        del data, predictions
    write_json(output / "summary.json", {"code_commit": commit, "final_test_used": False, "regimes": reports})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--regimes", nargs="+", choices=["known_classes", "class_holdout"], default=["known_classes", "class_holdout"])
    parser.add_argument("--permutation-seeds", nargs="+", type=int, default=[2097, 2098])
    parser.add_argument("--partition-seed", type=int, default=2099)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.device:
        config["device"] = args.device
    run(config, args.output, args.regimes, args.permutation_seeds, args.partition_seed, args.bootstrap_replicates)
