"""Verify single-stage pruning, then evaluate random and attention selection."""

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import clip
import numpy as np
import torch
from torch.utils.data import DataLoader

from token_compression.baseline import PetsImages, write_csv
from token_compression.pets import load_manifests
from token_compression.pruning import SingleStagePruner
from token_compression.pruning_eval import evaluate_variant


def normalized(features):
    features = features.float()
    return (features / features.norm(dim=-1, keepdim=True)).cpu().numpy()


@torch.inference_mode()
def run(config, output):
    started = datetime.now(timezone.utc).isoformat()
    before_run = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config["seed"])
    images, classes, tasks = load_manifests(config["manifest_dir"])
    image_rows = [row for row in images if row["split"] == config["image_split"]]
    image_ids = np.array([row["image_id"] for row in image_rows])
    labels = np.array([row["class_id"] for row in image_rows])
    with np.load(Path(config["baseline_dir"]) / "embeddings.npz") as saved:
        assert np.array_equal(image_ids, saved["image_ids"])
        assert np.array_equal(labels, saved["labels"])
        reference_features = saved["image_features"].copy()
        reference_text = saved["text_features"].copy()
        reference_scores = saved["cosine_scores"].copy()

    device = torch.device(config["device"])
    model, preprocess = clip.load(config["model"], device=device, jit=False,
                                  download_root=config["model_cache"])
    model.eval().requires_grad_(False)
    pruner = SingleStagePruner(model.visual, prune_after=config["prune_after"])
    prompts = [config["prompt_template"].format(class_name=row["class_name"])
               for row in classes]
    text = normalized(model.encode_text(clip.tokenize(prompts).to(device)))
    np.testing.assert_allclose(text, reference_text, atol=config["identity_atol"], rtol=0)
    loader = DataLoader(PetsImages(config["data_root"], image_rows, preprocess),
                        batch_size=config["batch_size"], shuffle=False,
                        num_workers=config["num_workers"], pin_memory=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Only 223 MB of prefix states for this pilot; keep them in host RAM.
    # No pruning outcomes are evaluated until the full keep-all check passes.
    prefixes, attentions, native, keep_all, visited_ids = [], [], [], [], []
    before_identity = time.perf_counter()
    for pixels, _, batch_ids in loader:
        pixels = pixels.to(device, non_blocking=True)
        native.append(normalized(model.encode_image(pixels)))
        tokens, attention = pruner.prefix(pixels, return_attention=True)
        count = tokens.shape[1] - 1
        indices = torch.arange(count, device=device).expand(len(pixels), -1)
        keep_all.append(normalized(pruner.finish(tokens, indices)))
        prefixes.append(tokens.cpu())
        attentions.append(attention.cpu())
        visited_ids.extend(batch_ids)
    assert visited_ids == image_ids.tolist()
    native = np.concatenate(native)
    keep_all = np.concatenate(keep_all)
    keep_scores = keep_all @ text.T
    native_scores = native @ text.T
    attention_scores = torch.cat(attentions)
    n_patches = attention_scores.shape[1]
    identity = {
        "image_count": len(image_ids),
        "max_feature_error_vs_native": float(np.max(np.abs(keep_all - native))),
        "max_feature_error_vs_saved_baseline": float(np.max(np.abs(keep_all - reference_features))),
        "max_score_error_vs_saved_baseline": float(np.max(np.abs(keep_scores - reference_scores))),
        "max_score_error_vs_native": float(np.max(np.abs(keep_scores - native_scores))),
        "prediction_disagreements_37class": int(np.sum(keep_scores.argmax(1) != reference_scores.argmax(1))),
        "prediction_disagreements_37class_vs_native": int(np.sum(keep_scores.argmax(1) != native_scores.argmax(1))),
        "atol": config["identity_atol"],
    }
    for key in ("max_feature_error_vs_native", "max_feature_error_vs_saved_baseline",
                "max_score_error_vs_saved_baseline", "max_score_error_vs_native"):
        assert identity[key] <= config["identity_atol"], identity
    assert identity["prediction_disagreements_37class"] == 0, identity
    assert identity["prediction_disagreements_37class_vs_native"] == 0, identity
    evaluation_args = dict(baseline_scores=reference_scores, text_features=text, labels=labels,
                           image_ids=image_ids.tolist(), images=images, tasks=tasks,
                           image_split=config["image_split"], task_splits=config["task_splits"],
                           balance_seed=config["balance_seed"],
                           quartile_edges=config["margin_quartile_edges"])
    keep_metrics, keep_tasks, keep_strata = evaluate_variant(keep_scores, **evaluation_args)
    identity["prediction_disagreements_binary"] = {
        split: values["new_errors"] + values["recoveries"]
        for split, values in keep_metrics["binary"].items()
    }
    assert all(value == 0 for value in identity["prediction_disagreements_binary"].values()), identity
    native_metrics, _, _ = evaluate_variant(native_scores, **evaluation_args)
    identity["native_vs_saved_binary_disagreements"] = {
        split: values["new_errors"] + values["recoveries"]
        for split, values in native_metrics["binary"].items()
    }
    assert all(value == 0 for value in identity["native_vs_saved_binary_disagreements"].values()), identity
    identity["passed"] = True
    identity["wall_seconds"] = time.perf_counter() - before_identity
    (output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    print("Keep-all verification passed:", json.dumps(identity), flush=True)

    all_tasks, all_strata, variants, metric_rows = [], [], [], []

    def save_variant(name, method, k, seed, features, indices, metrics, task_rows, stratum_rows):
        descriptor = {"variant": name, "method": method, "keep_patch_tokens": k,
                      "keep_fraction": k / n_patches, "random_seed": seed}
        np.savez_compressed(output / (name + ".npz"), image_ids=image_ids, labels=labels,
                            image_features=features, text_features=text,
                            cosine_scores=features @ text.T,
                            patch_indices=np.sort(indices, axis=1).astype(np.int16))
        variants.append({**descriptor, "metrics": metrics})
        all_tasks.extend({**descriptor, **row} for row in task_rows)
        all_strata.extend({**descriptor, **row} for row in stratum_rows)
        for split, binary in metrics["binary"].items():
            metric_rows.append({**descriptor, "task_split": split,
                                "accuracy_37class": metrics["accuracy_37class"],
                                "binary_accuracy_macro": binary["accuracy_macro"],
                                "binary_accuracy_micro": binary["accuracy_micro"],
                                "new_errors": binary["new_errors"],
                                "recoveries": binary["recoveries"],
                                "new_error_rate_among_baseline_correct": binary["new_error_rate_among_baseline_correct"],
                                "mean_geometric_margin_loss": binary["mean_geometric_margin_loss"]})
        print(name, json.dumps({"accuracy_37class": metrics["accuracy_37class"],
                               "binary_val": metrics["binary"].get("val")}), flush=True)

    full_indices = np.broadcast_to(np.arange(n_patches), (len(image_ids), n_patches))
    save_variant("keep_all", "keep_all", n_patches, None, keep_all, full_indices,
                 keep_metrics, keep_tasks, keep_strata)
    np.savez_compressed(output / "attention_scores.npz", image_ids=image_ids,
                        attention_scores=attention_scores.numpy())
    attention_order = torch.argsort(attention_scores, dim=1, descending=True, stable=True).numpy()
    rankings = [("attention", None, attention_order)]
    for seed in config["random_seeds"]:
        rng = np.random.default_rng(seed)
        rankings.append(("random", seed,
                         np.stack([rng.permutation(n_patches) for _ in image_ids])))

    before_pruning = time.perf_counter()
    for method, seed, ranking in rankings:
        for k in config["keep_patch_tokens"]:
            indices = ranking[:, :k]
            outputs, offset = [], 0
            for tokens in prefixes:
                n = len(tokens)
                selected = torch.as_tensor(indices[offset:offset + n], device=device)
                outputs.append(normalized(pruner.finish(tokens.to(device), selected)))
                offset += n
            features = np.concatenate(outputs)
            metrics, task_rows, stratum_rows = evaluate_variant(features @ text.T, **evaluation_args)
            name = f"attention_k{k}" if seed is None else f"random_s{seed}_k{k}"
            save_variant(name, method, k, seed, features, indices, metrics, task_rows, stratum_rows)

    write_csv(output / "metrics.csv", metric_rows)
    write_csv(output / "per_task.csv", all_tasks)
    write_csv(output / "margin_strata.csv", all_strata)
    summary = {
        "run_type": "single_stage_pruning_development_pilot",
        "started_utc": started,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config": config, "identity": identity, "variants": variants,
        "prompts": prompts, "prune_after_completed_blocks": config["prune_after"],
        "total_blocks": len(model.visual.transformer.resblocks), "patch_tokens_before": n_patches,
        "cls_always_retained": True,
        "selection_scope": "One image-only selection per image, shared by all candidate-class tasks.",
        "attention_definition": "Mean over heads of CLS-to-patch attention in the last completed prefix block; scoring is recomputed beside the unchanged native block forward.",
        "attention_score_precision": pruner.attention_score_precision,
        "random_definition": "Per-image NumPy permutations in manifest order; nested top-k sets within each seed; selected tokens gathered in original spatial order.",
        "timing_scope": "Prefix states reused in RAM across variants; wall times include evaluation/output and cannot establish inference speedup.",
        "pruning_sweep_wall_seconds": time.perf_counter() - before_pruning,
        "total_wall_seconds": time.perf_counter() - before_run,
        "prefix_cache_bytes": sum(tokens.numel() * tokens.element_size() for tokens in prefixes),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Completed pruning pilot.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pets_pruning_pilot.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.output)


if __name__ == "__main__":
    main()
