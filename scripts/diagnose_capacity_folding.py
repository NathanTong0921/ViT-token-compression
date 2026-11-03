"""Fold a trained constant-context selector into its equivalent image-only MLP.

This CPU diagnostic uses saved development prefix tokens. It tests functional
equivalence at fixed checkpoints, not equivalence of training trajectories.
The explicit forwards preserve float64 instead of the production float32 casts.
"""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from token_compression.selector import ImageOnlyWideSelector, TokenSelector


SETTINGS = (
    {
        "id": "pets_b16_49",
        "cache": "runs/pets-selector-scale-supervision-v1",
        "models": "runs/pets-selector-scale-training-v1",
        "keep_patch_tokens": 49,
    },
    {
        "id": "flowers_b16_98",
        "cache": "runs/flowers-selector-budget-98-supervision-v1",
        "models": "runs/flowers-selector-budget-98-training-v1",
        "keep_patch_tokens": 98,
    },
)
SEEDS = (2041, 2042, 2043)


def fold_constant_context(model):
    """Absorb the fixed normalized context contribution into the first bias."""
    assert not model.conditioned
    compact = ImageOnlyWideSelector(model.hidden_dim, model.token_dim).double().eval()
    image_dim = 2 * model.token_dim
    with torch.no_grad():
        compact.image_norm.load_state_dict(model.image_norm.state_dict())
        compact.scorer[0].weight.copy_(model.scorer[0].weight[:, :image_dim])
        context = model.context_norm(model.shared_descriptor)
        compact.scorer[0].bias.copy_(
            model.scorer[0].bias + model.scorer[0].weight[:, image_dim:] @ context
        )
        compact.scorer[2].load_state_dict(model.scorer[2].state_dict())
        compact.scorer[4].load_state_dict(model.scorer[4].state_dict())
    return compact


def scores_float64(model, tokens, with_context):
    tokens = model.image_norm(tokens)
    patches = tokens[:, 1:]
    cls = tokens[:, :1].expand(-1, patches.shape[1], -1)
    inputs = [patches, cls]
    if with_context:
        context = model.context_norm(model.shared_descriptor)
        inputs.append(context.expand(len(tokens), patches.shape[1], -1))
    return model.scorer(torch.cat(inputs, dim=-1)).squeeze(-1)


def diagnose_setting(setting, image_count):
    folder = Path(setting["cache"]) / "val"
    prefix_path = folder / "prefix_tokens.npy"
    prefix = np.load(prefix_path, mmap_mode="r")
    assert 0 < image_count <= len(prefix)
    tokens = torch.from_numpy(np.array(prefix[:image_count], copy=True)).double()
    with np.load(folder / "evidence.npz", allow_pickle=False) as data:
        ids = data["image_ids"][:image_count].astype(str).tolist()
    rows = []
    for seed in SEEDS:
        checkpoint_path = Path(setting["models"]) / "full" / f"seed_{seed}" / "image_only.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert not checkpoint["conditioned"] and checkpoint["seed"] == seed
        state = checkpoint["state_dict"]
        original = TokenSelector(
            state["shared_descriptor"], conditioned=False,
            hidden_dim=checkpoint["hidden_dim"], token_dim=tokens.shape[-1],
        ).double().eval()
        original.load_state_dict(state)
        compact = fold_constant_context(original)
        with torch.inference_mode():
            scores = scores_float64(original, tokens, with_context=True)
            folded = scores_float64(compact, tokens, with_context=False)
        error = (scores - folded).abs()
        torch.testing.assert_close(scores, folded, atol=1e-10, rtol=1e-10)
        count_original = sum(value.numel() for value in original.parameters())
        count_compact = sum(value.numel() for value in compact.parameters())
        k = setting["keep_patch_tokens"]
        selected = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]
        selected_folded = torch.argsort(folded, dim=1, descending=True, stable=True)[:, :k]
        mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, selected, True)
        mask_folded = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, selected_folded, True)
        rows.append({
            "seed": seed,
            "checkpoint": str(checkpoint_path),
            "checkpoint_code_commit": checkpoint.get("code_commit"),
            "hidden_dim": original.hidden_dim,
            "original_parameter_count": count_original,
            "folded_parameter_count": count_compact,
            "foldable_parameter_count": count_original - count_compact,
            "foldable_parameter_fraction": (count_original - count_compact) / count_original,
            "max_abs_score_error": float(error.max()),
            "mean_abs_score_error": float(error.mean()),
            "topk_identical_set_fraction": float((mask == mask_folded).all(dim=1).double().mean()),
            "mean_topk_overlap": float((mask & mask_folded).sum(dim=1).double().mean() / k),
        })
    return {
        "setting": setting,
        "prefix_source": str(prefix_path),
        "image_count": image_count,
        "image_ids": ids,
        "image_selection": "First fixed rows in the existing validation prefix cache.",
        "prefix_shape": list(tokens.shape),
        "seeds": rows,
    }


def run(output, image_count=32):
    started = datetime.now(timezone.utc).isoformat()
    before = time.perf_counter()
    torch.set_num_threads(2)
    results = {setting["id"]: diagnose_setting(setting, image_count) for setting in SETTINGS}
    summary = {
        "status": "passed",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - before,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "device": "cpu",
        "dtype": "float64",
        "reserved_test_used": False,
        "new_training": False,
        "identity": "W_image x + W_context LN(c0) + b = W_image x + b_effective",
        "interpretation": (
            "The fixed-context model and compact width-matched model have the same input-dependent "
            "function class. Folding preserves each checkpoint function; it does not preserve optimizer "
            "dynamics, regularization or training trajectories. The original constant-context baseline "
            "remains a valid task-information-removal control."
        ),
        "settings": results,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=int, default=32)
    arguments = parser.parse_args()
    result = run(arguments.output, arguments.images)
    print(json.dumps({"status": result["status"], "settings": list(result["settings"]),
                      "wall_seconds": result["wall_seconds"]}))
