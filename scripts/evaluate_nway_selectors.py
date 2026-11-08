"""Evaluate 2/5-way selectors on development images and held-out candidate sets."""

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np


ARMS = ("conditioned", "image_only_wide")
PANEL_FIELDS = ("image_indices", "task_indices", "candidates", "labels",
                "image_ids", "task_ids", "weights")


def classification_outcomes(scores, labels, candidates, text_features):
    """Return classification and the closest raw/geometric decision margin.

    A score tie is resolved by class id, independent of candidate presentation.
    Geometric margin is the minimum signed distance to any rival boundary.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    candidates = np.sort(np.asarray(candidates, dtype=np.int64), axis=1)
    text = np.asarray(text_features, dtype=np.float64)
    rows = np.arange(len(labels))
    is_true = candidates == labels[:, None]
    assert np.all(is_true.sum(axis=1) == 1)
    assert np.all(np.diff(candidates, axis=1) > 0)
    candidate_scores = np.take_along_axis(scores, candidates, axis=1)
    predicted = candidates[rows, candidate_scores.argmax(axis=1)]
    rivals = candidates[~is_true].reshape(len(labels), candidates.shape[1] - 1)
    differences = scores[rows, labels, None] - np.take_along_axis(scores, rivals, axis=1)
    distances = np.linalg.norm(text[labels, None] - text[rivals], axis=-1)
    return {"predicted": predicted, "correct": (predicted == labels).astype(np.int8),
            "margin": differences.min(axis=1),
            "geometric_margin": (differences / distances).min(axis=1)}


def validate_panel(panel):
    """Check task/class balancing before using the saved evaluation weights."""
    weights, labels = panel["weights"], panel["labels"]
    tasks = np.unique(panel["task_ids"])
    k = panel["candidates"].shape[1]
    np.testing.assert_allclose(weights.sum(), 1., atol=1e-10, rtol=0)
    assert np.all((panel["candidates"] == labels[:, None]).sum(axis=1) == 1)
    for task in tasks:
        rows = np.flatnonzero(panel["task_ids"] == task)
        members = panel["candidates"][rows[0]]
        np.testing.assert_array_equal(panel["candidates"][rows],
                                      np.broadcast_to(members, (len(rows), k)))
        assert len(np.unique(members)) == k
        for label in members:
            selected = rows[labels[rows] == label]
            np.testing.assert_allclose(weights[selected],
                                      1. / (len(tasks) * k * len(selected)), atol=1e-12, rtol=0)


def packed_predictions(panel, predictions, text):
    """Pack complete row metadata beside every method's saved outcomes."""
    parts = []
    for method, seed, values in predictions:
        count = len(panel["labels"])
        parts.append({**{key: panel[key] for key in PANEL_FIELDS},
                      "method": np.full(count, method), "seed": np.full(count, seed),
                      **values})
    result = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    result["text_features"] = text
    return result


def read_panel(path):
    with np.load(path) as saved:
        return {key: saved[key].copy() for key in PANEL_FIELDS}


def run(config, prepared, models, output, ways=None, seeds=None,
        skip_references=False, references_only=False):
    import clip
    import torch

    from token_compression.nway import make_selector
    from token_compression.pruning import SingleStagePruner

    started = time.perf_counter()
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    ways = config.get("ways", [2, 5]) if ways is None else ways
    seeds = config["training_seeds"] if seeds is None else seeds
    cache = Path(config["cache_dir"]) / "val"
    with np.load(cache / "evidence.npz") as saved:
        evidence = {key: saved[key].copy() for key in saved.files}
    text = evidence["text_features"]
    np.testing.assert_allclose(evidence["full_features"] @ text.T,
                               evidence["full_scores"], atol=config.get("identity_atol", 1e-6), rtol=0)
    device = torch.device(config["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    backbone, _ = clip.load(config["model"], device=device, jit=False,
                            download_root=config["model_cache"])
    backbone.eval().requires_grad_(False)
    pruner = SingleStagePruner(backbone.visual, config["prune_after"])
    prefixes = torch.as_tensor(np.load(cache / "prefix_tokens.npy"), device=device)
    text_tensor = torch.as_tensor(text, device=device)
    batch_size = config.get("image_batch_size", 64)
    keep = config["keep_patch_tokens"]
    costs = []

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def predict(panel, way, method, seed):
        began = time.perf_counter()
        ix = panel["image_indices"]
        unique = method != "conditioned"
        image_indices = np.arange(len(prefixes)) if unique else ix
        count = len(image_indices)
        features = np.empty((count, text.shape[1]), np.float32)
        positions = np.full((count, pruner.num_patches), -1, np.int16)
        checkpoint_path, model = None, None
        if method in ARMS:
            checkpoint_path = models / f"{way}way" / f"seed_{seed}" / f"{method}.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            assert checkpoint["way"] == way and checkpoint["selector_variant"] == method
            model = make_selector(method, token_dim=checkpoint["token_dim"],
                                  text_dim=checkpoint["text_dim"],
                                  hidden_dim=checkpoint["hidden_dim"]).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval().requires_grad_(False)
        selector_seconds, suffix_seconds = 0., 0.
        physical = 0
        if method == "full_model":
            features[:] = evidence["full_features"]
            positions[:] = np.arange(pruner.num_patches)
        else:
            for start in range(0, count, batch_size):
                stop = min(start + batch_size, count)
                valid = stop - start
                rows = np.arange(start, stop)
                if valid < batch_size:
                    rows = np.concatenate([rows, np.repeat(rows[-1], batch_size - valid)])
                tokens = prefixes[torch.as_tensor(image_indices[rows], device=device)]
                if model is None:
                    logits = evidence["attention_scores"][image_indices[rows]]
                else:
                    candidates = (np.tile(panel["candidates"][0], (len(rows), 1))
                                  if unique else panel["candidates"][rows])
                    candidate_text = text_tensor[torch.as_tensor(candidates, device=device)]
                    synchronize()
                    begin_selector = time.perf_counter()
                    logits = model(tokens.float(), candidate_text).cpu().numpy()
                    synchronize()
                    selector_seconds += time.perf_counter() - begin_selector
                selected = np.sort(np.argsort(-logits, axis=1, kind="stable")[:, :keep], axis=1)
                retained = torch.as_tensor(selected, device=device, dtype=torch.long)
                synchronize()
                begin_suffix = time.perf_counter()
                z = pruner.finish(tokens, retained).float()
                z = (z / z.norm(dim=-1, keepdim=True)).cpu().numpy()
                synchronize()
                suffix_seconds += time.perf_counter() - begin_suffix
                features[start:stop] = z[:valid]
                positions[start:stop, :keep] = selected[:valid]
                physical += len(rows)
        if unique:
            features, positions = features[ix], positions[ix]
        scores = features @ text.T
        values = {"image_features": features, "cosine_scores": scores,
                  "patch_indices": positions,
                  **classification_outcomes(scores, panel["labels"], panel["candidates"], text)}
        costs.append({"way": way, "method": method, "seed": seed,
                      "checkpoint": str(checkpoint_path) if checkpoint_path else None,
                      "wall_seconds": time.perf_counter() - began,
                      "selector_seconds": selector_seconds, "suffix_seconds": suffix_seconds,
                      "logical_task_examples": len(ix), "unique_image_reuse": unique,
                      "logical_suffix_examples": 0 if method == "full_model" else count,
                      "physical_suffix_examples": physical})
        print(json.dumps(costs[-1]), flush=True)
        return values

    with torch.inference_mode():
        for way in ways:
            panel = read_panel(prepared / "panels" / f"{way}way" / "val.npz")
            validate_panel(panel)
            assert panel["candidates"].shape[1] == way
            np.testing.assert_array_equal(evidence["image_ids"][panel["image_indices"]], panel["image_ids"])
            np.testing.assert_array_equal(evidence["labels"][panel["image_indices"]], panel["labels"])
            folder = output / f"{way}way"
            folder.mkdir(parents=True, exist_ok=True)
            if not skip_references:
                values = [(method, -1, predict(panel, way, method, -1))
                          for method in ("full_model", "attention")]
                np.savez_compressed(folder / "references.npz", **packed_predictions(panel, values, text))
                (folder / "reference_costs.json").write_text(json.dumps(
                    {"code_commit": code_commit, "evaluation_split": "development",
                     "costs": [row for row in costs if row["way"] == way and row["seed"] == -1]}, indent=2) + "\n")
                del values
            if references_only:
                continue
            for seed in seeds:
                values = [(method, seed, predict(panel, way, method, seed)) for method in ARMS]
                np.savez_compressed(folder / f"seed_{seed}.npz", **packed_predictions(panel, values, text))
                (folder / f"seed_{seed}_costs.json").write_text(json.dumps(
                    {"code_commit": code_commit, "evaluation_split": "development",
                     "costs": [row for row in costs if row["way"] == way and row["seed"] == seed]}, indent=2) + "\n")
                del values
    summary = {"code_commit": code_commit, "evaluation_split": "development",
               "ways": ways, "seeds": [] if references_only else seeds,
               "wall_seconds": time.perf_counter() - started, "costs": costs}
    print(json.dumps({key: value for key, value in summary.items() if key != "costs"}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--way", type=int, choices=[2, 5])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--skip-references", action="store_true")
    parser.add_argument("--references-only", action="store_true")
    args = parser.parse_args()
    if args.skip_references and args.references_only:
        parser.error("--skip-references and --references-only cannot be combined")
    run(json.loads(args.config.read_text()), args.prepared, args.models, args.output,
        [args.way] if args.way else None, [args.seed] if args.seed is not None else None,
        skip_references=args.skip_references, references_only=args.references_only)
