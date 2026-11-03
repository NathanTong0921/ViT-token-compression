#!/usr/bin/env python3
"""Frozen-CLIP screen of shared evidence and two-object selection conflict.

Only development images and preassigned development candidate pairs are read.
The mask-library optima are label-informed diagnostics, not learned predictors.
"""
import argparse
import csv
import json
import subprocess
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from token_compression.pets import load_manifests
from token_compression.pruning import SingleStagePruner

SETTINGS = {
    "pets": ("data/raw/oxford-iiit-pet", "data/splits/pets-v1", "a photo of a {class_name}, a type of pet."),
    "flowers": ("data/raw/flowers102", "data/splits/flowers-v1", "a photo of a {class_name}, a type of flower."),
}


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize(features):
    features = features.float()
    return features / features.norm(dim=-1, keepdim=True)


def finish_with_cls_only(pruner, tokens):
    """Retain the contextual CLS and run the original suffix with length one."""
    hidden = tokens[:, :1, :].permute(1, 0, 2)
    for block in pruner.blocks[pruner.prune_after:]:
        hidden = block(hidden)
    features = pruner.visual.ln_post(hidden[0])
    return features if pruner.visual.proj is None else features @ pruner.visual.proj


def spread_indices(available, count, grid=14):
    """Deterministic spatial coverage within a specified region of the grid."""
    available = np.asarray(available, dtype=np.int64)
    coords = np.column_stack((available // grid, available % grid)).astype(float)
    first = int(np.argmin(((coords - coords.mean(0)) ** 2).sum(1)))
    selected, distances = [], np.full(len(available), np.inf)
    next_index = first
    for _ in range(count):
        selected.append(next_index)
        distances = np.minimum(distances, ((coords - coords[next_index]) ** 2).sum(1))
        distances[selected] = -1
        next_index = int(distances.argmax())
    return np.sort(available[selected])


def build_panel(images, tasks, text, count, seed, image_split="val", task_split="val"):
    """Choose sources and distractors without inspecting visual predictions."""
    rng = np.random.default_rng(seed)
    by_class, neighbors, pair_ids = defaultdict(list), defaultdict(list), {}
    for row in images:
        if row["split"] == image_split:
            by_class[row["class_id"]].append(row)
    for task in tasks:
        if task["split"] == task_split:
            a, b = task["class_a"], task["class_b"]
            neighbors[a].append(b)
            neighbors[b].append(a)
            pair_ids[tuple(sorted((a, b)))] = task["task_id"]
    eligible = [(a, b) for a, b in combinations(sorted(by_class), 2)
                if len(set(neighbors[a]) - {b}) >= 2
                and len(set(neighbors[b]) - {a}) >= 2]
    if not eligible:
        raise ValueError("No class pair has two non-present development distractors per object")
    order = rng.permutation(len(eligible))
    similarity = text @ text.T
    panel = []
    for index in range(count):
        labels = list(eligible[int(order[index % len(order)])])
        if rng.integers(2):
            labels.reverse()
        sources, task_rows = [], []
        for slot, label in enumerate(labels):
            available = sorted(by_class[label], key=lambda row: row["image_id"])
            source = available[int(rng.integers(len(available)))]
            sources.append({key: source[key] for key in ("image_id", "class_id", "split", "official_split")})
            # Both distractors are absent from the two designated source labels.
            distractors = sorted(set(neighbors[label]) - set(labels),
                                 key=lambda c: (-float(similarity[label, c]), c))[:2]
            for distractor in distractors:
                task_rows.append({"object_slot": slot, "label": label,
                                  "distractor": distractor,
                                  "task_id": pair_ids[tuple(sorted((label, distractor)))],
                                  "task_split": task_split})
        panel.append({"panel_id": f"composite_{index:04d}",
                      "layout": "horizontal" if rng.integers(2) else "vertical",
                      "sources": sources, "tasks": task_rows})
    return panel, {"eligible_source_class_pairs": len(eligible),
                   "source_classes": len(by_class), "candidate_task_split": task_split}


def render_composite(row, root):
    canvas = Image.new("RGB", (224, 224), (127, 127, 127))
    tile_size = (112, 224) if row["layout"] == "horizontal" else (224, 112)
    transforms = []
    for slot, source in enumerate(row["sources"]):
        with Image.open(Path(root) / "images" / (source["image_id"] + ".jpg")) as opened:
            original = opened.convert("RGB")
            fitted = ImageOps.contain(original, tile_size, Image.Resampling.BICUBIC)
            origin = (slot * 112, 0) if row["layout"] == "horizontal" else (0, slot * 112)
            offset = (origin[0] + (tile_size[0] - fitted.width) // 2,
                      origin[1] + (tile_size[1] - fitted.height) // 2)
            canvas.paste(fitted, offset)
            transforms.append({"original_size": list(original.size),
                               "resized_size": list(fitted.size), "paste_xy": list(offset)})
    row["transforms"] = transforms
    return canvas


def mask_bank(attention, panels, keep, random_masks, seed):
    n, patches = attention.shape
    rng = np.random.default_rng(seed)
    uniform = spread_indices(np.arange(patches), keep)
    # Random masks are shared across images/tasks; no task or label enters selection.
    random = [np.sort(rng.choice(patches, keep, replace=False)) for _ in range(random_masks)]
    names = [f"random_{i:02d}" for i in range(random_masks)] + ["uniform", "attention", "tile_0", "tile_1"]
    masks = np.empty((n, len(names), keep), dtype=np.int64)
    for i, panel in enumerate(panels):
        masks[i, :random_masks] = np.asarray(random)
        masks[i, random_masks] = uniform
        masks[i, random_masks + 1] = np.sort(np.argsort(-attention[i], kind="stable")[:keep])
        coords = np.arange(patches)
        region = coords % 14 < 7 if panel["layout"] == "horizontal" else coords // 14 < 7
        masks[i, -2] = spread_indices(coords[region], keep)
        masks[i, -1] = spread_indices(coords[~region], keep)
    return masks, names


def pair_diagnostics(margins, full_margins, panels, layer, keep, text_features):
    """Bank margins have shape (images, masks, four tasks)."""
    rows = []
    for i, panel in enumerate(panels):
        for a, b in combinations(range(4), 2):
            pair = margins[i][:, [a, b]]
            separate = pair.max(axis=0)
            shared_min = pair.min(axis=1).max()
            full = full_margins[i, [a, b]]
            directions = np.asarray([text_features[panel["tasks"][t]["label"]] -
                                     text_features[panel["tasks"][t]["distractor"]] for t in (a, b)])
            cosine = float(np.dot(*directions) / np.prod(np.linalg.norm(directions, axis=1)))
            angle = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
            separate_feasible = bool(np.all(separate > 0))
            common_feasible = bool(shared_min > 0)
            rows.append({"panel_id": panel["panel_id"], "layer": layer, "keep": keep,
                         "task_a": a, "task_b": b, "direction_angle_degrees": angle,
                         "source_image_0": panel["sources"][0]["image_id"],
                         "source_image_1": panel["sources"][1]["image_id"],
                         "layout": panel["layout"],
                         "different_objects": int(panel["tasks"][a]["object_slot"] != panel["tasks"][b]["object_slot"]),
                         "full_margin_min": float(full.min()), "full_margin_max": float(full.max()),
                         "full_both_correct": int(np.all(full > 0)),
                         "separate_feasible": int(separate_feasible),
                         "common_feasible": int(common_feasible),
                         "conflict": int(separate_feasible and not common_feasible),
                         "separate_mean_margin": float(separate.mean()),
                         "best_shared_mean_margin": float(pair.mean(axis=1).max()),
                         "mean_margin_gap": float(separate.mean() - pair.mean(axis=1).max()),
                         "separate_min_margin": float(separate.min()),
                         "best_shared_min_margin": float(shared_min),
                         "min_margin_gap": float(separate.min() - shared_min)})
    return rows


def summarize_pairs(rows):
    summaries = []
    for layer, keep in sorted({(r["layer"], r["keep"]) for r in rows}):
        for correct_only in (False, True):
            for different in (0, 1):
                subset = [r for r in rows if r["layer"] == layer and r["keep"] == keep
                          and r["different_objects"] == different
                          and (not correct_only or r["full_both_correct"])]
                by_image = defaultdict(list)
                for r in subset:
                    by_image[r["panel_id"]].append(r)
                result = {"layer": layer, "keep": keep, "full_both_correct_only": correct_only,
                          "different_objects": different, "pairs": len(subset), "images": len(by_image)}
                for key in ("separate_feasible", "common_feasible", "conflict", "mean_margin_gap", "min_margin_gap"):
                    result[key] = float(np.mean([np.mean([r[key] for r in group])
                                                for group in by_image.values()])) if by_image else None
                summaries.append(result)
    return summaries


def difficulty_adjustment(rows):
    """Exploratory within-image regression; no significance or causal claim."""
    results = []
    bins = np.array([-np.inf, 0, .01, .03, .06, .1, np.inf])
    for layer, keep in sorted({(r["layer"], r["keep"]) for r in rows}):
        subset = [r for r in rows if r["layer"] == layer and r["keep"] == keep]
        ids = sorted({r["panel_id"] for r in subset})
        identity = np.asarray([[float(r["panel_id"] == id_) for id_ in ids] for r in subset])
        difficulty = np.asarray([[r["full_margin_min"], r["full_margin_max"]] for r in subset])
        kind = np.asarray([r["different_objects"] for r in subset])
        design = np.column_stack((identity, kind, difficulty, difficulty ** 2))
        angles = np.asarray([r["direction_angle_degrees"] / 180 for r in subset])
        design_angle = np.column_stack((design, angles, angles ** 2))
        for outcome in ("mean_margin_gap", "min_margin_gap", "conflict"):
            y = np.asarray([r[outcome] for r in subset])
            beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
            beta_angle, _, rank_angle, _ = np.linalg.lstsq(design_angle, y, rcond=None)
            # A separate coarse exact-bin standardization exposes overlap directly.
            grouped = defaultdict(lambda: defaultdict(list))
            for r in subset:
                cell = tuple(np.digitize([r["full_margin_min"], r["full_margin_max"]], bins))
                grouped[cell][r["different_objects"]].append(r[outcome])
            shared = [g for g in grouped.values() if 0 in g and 1 in g]
            angle_cells = defaultdict(lambda: defaultdict(list))
            for r in subset:
                cell = tuple(np.digitize([r["full_margin_min"], r["full_margin_max"]], bins))
                cell += (int(r["direction_angle_degrees"] // 30),)
                angle_cells[cell][r["different_objects"]].append(r[outcome])
            common_angle_cells = [g for g in angle_cells.values() if 0 in g and 1 in g]
            weights = np.asarray([min(len(g[0]), len(g[1])) for g in shared], dtype=float)
            matched = float(np.average([np.mean(g[1]) - np.mean(g[0]) for g in shared], weights=weights)) if len(shared) else None
            results.append({"layer": layer, "keep": keep, "outcome": outcome,
                            "different_object_coefficient": float(beta[len(ids)]),
                            "design_full_rank": int(rank) == design.shape[1],
                            "angle_adjusted_different_object_coefficient": float(beta_angle[len(ids)]),
                            "angle_design_full_rank": int(rank_angle) == design_angle.shape[1],
                            "same_object_angle_min": float(angles[kind == 0].min() * 180),
                            "same_object_angle_max": float(angles[kind == 0].max() * 180),
                            "different_object_angle_min": float(angles[kind == 1].min() * 180),
                            "different_object_angle_max": float(angles[kind == 1].max() * 180),
                            "common_difficulty_cells": len(shared),
                            "common_difficulty_and_angle_cells": len(common_angle_cells),
                            "difficulty_and_angle_matched_pairs_per_group": sum(min(len(g[0]), len(g[1])) for g in common_angle_cells),
                            "matched_effective_pairs_per_group": int(weights.sum()),
                            "difficulty_bin_standardized_difference": matched})
    return results


def run_dataset(args, name, model, preprocess):
    import clip
    root, manifests, template = SETTINGS[name]
    output = args.output / name
    output.mkdir(parents=True, exist_ok=True)
    images, classes, tasks = load_manifests(manifests)
    with torch.inference_mode():
        text = normalize(model.encode_text(clip.tokenize([template.format(class_name=c["class_name"])
                                                        for c in classes]).to(args.device)))
    text_np = text.cpu().numpy()
    selected = []
    rng = np.random.default_rng(args.seed)
    for label in range(len(classes)):
        available = sorted([r for r in images if r["class_id"] == label and r["split"] == "val"], key=lambda r: r["image_id"])
        selected.extend(available[i] for i in rng.permutation(len(available))[:args.original_per_class])
    with_images = []
    for row in selected:
        with Image.open(Path(root) / "images" / (row["image_id"] + ".jpg")) as im:
            with_images.append(preprocess(im.convert("RGB")))
    original_pixels = torch.stack(with_images)
    panel, eligibility = build_panel(images, tasks, text_np, args.composites, args.seed + 1)
    thumbnails = output / "composites"
    thumbnails.mkdir(exist_ok=True)
    panel_pixels = []
    for row in panel:
        image = render_composite(row, root)
        image.save(thumbnails / (row["panel_id"] + ".png"))
        panel_pixels.append(preprocess(image))
    panel_pixels = torch.stack(panel_pixels)
    (output / "panel_manifest.json").write_text(json.dumps({"source_manifest": manifests,
        "image_split": "val", "task_split": "val", "distractor_rule": "two text-nearest valid development neighbors excluding both source labels",
        "background_rgb": [127, 127, 127], "eligibility": eligibility, "panels": panel}, indent=2) + "\n")
    write_csv(output / "original_images.csv", selected)

    def native_scores(pixels):
        features = []
        for batch in pixels.split(args.batch_size):
            features.append(normalize(model.encode_image(batch.to(args.device))))
        return (torch.cat(features) @ text.T).cpu().numpy()

    def prefix(pixels, pruner):
        pairs = [pruner.prefix(batch.to(args.device)) for batch in pixels.split(args.batch_size)]
        return torch.cat([p[0] for p in pairs]), torch.cat([p[1] for p in pairs])

    def score_masks(tokens, masks, pruner):
        n, bank_size, keep = masks.shape
        masks = torch.as_tensor(masks.reshape(n * bank_size, keep), device=args.device)
        output_scores = []
        for start in range(0, n * bank_size, args.batch_size):
            flat = torch.arange(start, min(start + args.batch_size, n * bank_size), device=args.device)
            hidden = tokens[flat // bank_size]
            features = (finish_with_cls_only(pruner, hidden) if keep == 0
                        else pruner.finish(hidden, masks[flat]))
            output_scores.append((normalize(features) @ text.T).cpu().numpy())
        return np.concatenate(output_scores).reshape(n, bank_size, -1)

    original_rows, pair_rows, fixed_pair_rows, identity_errors = [], [], [], []
    started = time.perf_counter()
    with torch.inference_mode():
        original_full, panel_full = native_scores(original_pixels), native_scores(panel_pixels)
        labels = np.asarray([[t["label"] for t in p["tasks"]] for p in panel])
        distractors = np.asarray([[t["distractor"] for t in p["tasks"]] for p in panel])
        direction_norms = np.linalg.norm(text_np[labels] - text_np[distractors], axis=-1)
        full_cosine_margins = np.take_along_axis(panel_full, labels, axis=1) - np.take_along_axis(panel_full, distractors, axis=1)
        full_margins = full_cosine_margins / direction_norms
        binary = [(i, r["class_id"], t["class_b"] if t["class_a"] == r["class_id"] else t["class_a"])
                  for i, r in enumerate(selected) for t in tasks
                  if t["split"] == "val" and r["class_id"] in (t["class_a"], t["class_b"])]
        def original_summary(scores, layer, keep, method):
            binary_margins = [scores[i, label] - scores[i, other] for i, label, other in binary]
            return {"layer": layer, "keep": keep, "method": method,
                    "images": len(selected), "binary_examples": len(binary),
                    "multiclass_accuracy": float(np.mean(scores.argmax(1) == [r["class_id"] for r in selected])),
                    "binary_accuracy": float(np.mean(np.asarray(binary_margins) > 0)),
                    "binary_mean_cosine_margin": float(np.mean(binary_margins))}
        original_rows.append(original_summary(original_full, 0, 196, "native_full"))
        np.savez_compressed(output / "native_scores.npz", original_scores=original_full,
                            panel_scores=panel_full, full_margins=full_margins, full_cosine_margins=full_cosine_margins,
                            direction_norms=direction_norms, text_features=text_np,
                            labels=labels, distractors=distractors)
        for layer in args.layers:
            print(f"{name}: layer {layer}, original-image and composite masks", flush=True)
            pruner = SingleStagePruner(model.visual, prune_after=layer)
            tokens, attention = prefix(original_pixels, pruner)
            for keep in args.original_budgets:
                if keep == 0:
                    names, masks = ["cls_only"], np.empty((len(selected), 1, 0), dtype=np.int64)
                elif keep == 196:
                    names, masks = ["all"], np.broadcast_to(np.arange(196), (len(selected), 1, 196)).copy()
                else:
                    rng = np.random.default_rng(args.seed + keep)
                    fixed = [spread_indices(np.arange(196), keep)] + [np.sort(rng.choice(196, keep, replace=False)) for _ in range(args.original_random_masks)]
                    masks = np.broadcast_to(np.asarray(fixed), (len(selected), len(fixed), keep)).copy()
                    masks = np.concatenate((masks, np.sort(np.argsort(-attention.cpu().numpy(), axis=1, kind="stable")[:, :keep], axis=1)[:, None, :]), axis=1)
                    names = ["uniform"] + [f"random_{i}" for i in range(args.original_random_masks)] + ["attention"]
                scores = score_masks(tokens, masks, pruner)
                np.savez_compressed(output / f"original_layer{layer}_k{keep}.npz", cosine_scores=scores, masks=masks, methods=np.asarray(names))
                original_rows.extend(original_summary(scores[:, j], layer, keep, method) for j, method in enumerate(names))
                if keep == 196:
                    error = float(np.max(np.abs(scores[:, 0] - original_full)))
                    identity_errors.append({"layer": layer, "max_cosine_error": error})
                    assert error < 1e-5, f"Keep-all identity failed: {error}"
            tokens, attention = prefix(panel_pixels, pruner)
            for keep in args.composite_budgets:
                masks, names = mask_bank(attention.cpu().numpy(), panel, keep, args.random_masks, args.seed + keep)
                scores = score_masks(tokens, masks, pruner)
                cosine_margins = np.take_along_axis(scores, np.broadcast_to(labels[:, None], (len(panel), len(names), 4)), axis=2) - np.take_along_axis(scores, np.broadcast_to(distractors[:, None], (len(panel), len(names), 4)), axis=2)
                margins = cosine_margins / direction_norms[:, None, :]
                np.savez_compressed(output / f"composite_layer{layer}_k{keep}.npz", cosine_scores=scores,
                                    margins=margins, cosine_margins=cosine_margins, masks=masks, methods=np.asarray(names))
                pair_rows.extend(pair_diagnostics(margins, full_margins, panel, layer, keep, text_np))
                fixed_indices = [j for j, method in enumerate(names) if method != "attention"]
                fixed_pair_rows.extend(pair_diagnostics(margins[:, fixed_indices], full_margins, panel, layer, keep, text_np))
    write_csv(output / "original_summary.csv", original_rows)
    write_csv(output / "pair_diagnostics.csv", pair_rows)
    write_csv(output / "difficulty_adjustment.csv", difficulty_adjustment(pair_rows))
    write_csv(output / "pair_diagnostics_fixed_library.csv", fixed_pair_rows)
    write_csv(output / "difficulty_adjustment_fixed_library.csv", difficulty_adjustment(fixed_pair_rows))
    summary = {"dataset": name,
               "composite_margin_units": "geometric signed margin: cosine score difference / norm(text_label - text_distractor)",
               "original_margin_units": "raw cosine score difference", "elapsed_seconds": time.perf_counter() - started,
               "full_composite_task_accuracy": float((full_margins > 0).mean()),
               "full_composite_all_four_correct_fraction": float((full_margins > 0).all(axis=1).mean()),
               "full_composite_margin_quantiles": np.quantile(full_margins, [0, .1, .5, .9, 1]).tolist(),
               "identity_checks": identity_errors, "pair_summaries": summarize_pairs(pair_rows),
               "fixed_library_pair_summaries": summarize_pairs(fixed_pair_rows),
               "interpretation": "Library optima are label-informed diagnostic bounds within this finite library. Difficulty adjustment is exploratory and descriptive."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("pair_summaries", "fixed_library_pair_summaries")}), flush=True)


def main():
    import clip
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["pets", "flowers", "both"], default="both")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2081)
    parser.add_argument("--composites", type=int, default=64)
    parser.add_argument("--original-per-class", type=int, default=2)
    parser.add_argument("--layers", type=int, nargs="+", default=[3, 6])
    parser.add_argument("--original-budgets", type=int, nargs="+", default=[0, 4, 16, 49, 98, 196])
    parser.add_argument("--composite-budgets", type=int, nargs="+", default=[16, 49])
    parser.add_argument("--random-masks", type=int, default=32)
    parser.add_argument("--original-random-masks", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert args.composites > 0 and args.original_per_class > 0
    assert all(1 <= k <= 98 for k in args.composite_budgets)
    torch.manual_seed(args.seed)
    model, preprocess = clip.load("ViT-B/16", device=args.device, jit=False, download_root="cache/clip")
    model.eval().requires_grad_(False)
    metadata = {**vars(args), "output": str(args.output), "model": "ViT-B/16",
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "torch_version": torch.__version__, "numpy_version": np.__version__,
                "composite_margin_units": "geometric signed margin: cosine score difference / norm(text_label - text_distractor)",
                "original_margin_units": "raw cosine score difference",
                "scope": "TRAIN/VAL development only; default uses val images and val candidate pairs"}
    (args.output / "protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for dataset in SETTINGS if args.dataset == "both" else [args.dataset]:
        run_dataset(args, dataset, model, preprocess)


if __name__ == "__main__":
    main()
