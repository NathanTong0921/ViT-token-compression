"""Recover hybrid-follow-up evidence directly from downloaded ZIPs, using CPU only.

No extraction, original server paths, CLIP weights, network access, or output-file
writes are needed. A successful run prints a JSON reconstruction record.
"""

import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from token_compression.selector import ImageOnlyWideSelector
from token_compression.shared_residual import SharedResidualSelector


SETTINGS = ("pets_b16_49", "flowers_b16_49", "flowers_b16_98")
METHODS = ("shared", "task_residual", "image_residual")


class Archive:
    def __init__(self, source):
        self.owned = not isinstance(source, zipfile.ZipFile)
        self.zip = zipfile.ZipFile(source) if self.owned else source
        self.path = str(self.zip.filename or "<opened ZipFile>")

    def close(self):
        if self.owned:
            self.zip.close()

    def find(self, suffix):
        matches = [n for n in self.zip.namelist() if n == suffix or n.endswith("/" + suffix)]
        if len(matches) != 1:
            raise ValueError(f"Expected one ZIP member ending with {suffix!r}; found {matches}")
        return matches[0]

    def json(self, member):
        return json.loads(self.zip.read(member))

    def rows(self, member):
        return list(csv.DictReader(io.StringIO(self.zip.read(member).decode())))

    def arrays(self, member):
        # Reading every member also verifies the nested NPZ ZIP checksums.
        with np.load(io.BytesIO(self.zip.read(member)), allow_pickle=False) as data:
            result = {key: data[key].copy() for key in data.files}
        for key, value in result.items():
            if np.issubdtype(value.dtype, np.number):
                assert np.isfinite(value).all(), (member, key, "nonfinite array")
        return result

    def checkpoint(self, member):
        return torch.load(io.BytesIO(self.zip.read(member)), map_location="cpu", weights_only=True)


def selected_positions(logits, k):
    return np.sort(np.argsort(-logits, axis=-1, kind="stable")[..., :k], axis=-1)


@torch.inference_mode()
def reconstruct(base, branches, example):
    """Rebuild actual saved architectures rather than relying on server references."""
    tokens = torch.from_numpy(example["prefix_tokens"]).float()
    candidate_text = torch.from_numpy(example["candidate_text"]).float()
    assert base["selector_variant"] == "image_only_wide"
    shared = ImageOnlyWideSelector(hidden_dim=base["hidden_dim"], token_dim=tokens.shape[-1])
    shared.load_state_dict(base["state_dict"])
    shared.eval().requires_grad_(False)
    predictions = [shared(tokens, candidate_text).numpy()]
    for arm in METHODS[1:]:
        checkpoint = branches[arm]
        assert checkpoint["arm"] == arm
        assert checkpoint["conditioned"] == (arm == "task_residual")
        assert checkpoint["shared_hidden_dim"] == shared.hidden_dim
        assert checkpoint["text_dim"] == candidate_text.shape[-1]
        model = SharedResidualSelector(shared, checkpoint["conditioned"],
            text_dim=checkpoint["text_dim"], hidden_dim=checkpoint["hidden_dim"])
        model.branch.load_state_dict(checkpoint["branch_state_dict"])
        predictions.append(model.eval()(tokens, candidate_text).numpy())
    return np.stack(predictions)


def check_masks(values, k, patches, name):
    assert values.shape[-1] == k, (name, values.shape, k)
    assert np.issubdtype(values.dtype, np.integer), name
    assert ((values >= 0) & (values < patches)).all(), name
    assert (np.diff(values, axis=-1) > 0).all(), (name, "unsorted or repeated positions")


def verify_distillation(path, tolerance=1e-5):
    archive = Archive(path)
    try:
        member = archive.find("base_checkpoints/manifest.json")
        root = member[:-len("base_checkpoints/manifest.json")]
        manifest = archive.json(member)
        records = manifest["checkpoint_records"]
        assert len(records) == 9
        assert Counter(r["setting"] for r in records) == Counter({s: 3 for s in SETTINGS})
        examples = {r["setting"]: archive.arrays(root + r["path"])
                    for r in manifest["verification_examples"]}
        assert set(examples) == set(SETTINGS)
        protocols = {s: archive.json(root + s + "/protocol.json") for s in SETTINGS}
        forward_checks, target_checks, raw_checks = [], [], []
        for record in records:
            setting, seed = record["setting"], record["seed"]
            example, protocol = examples[setting], protocols[setting]
            np.testing.assert_array_equal(example["methods"], METHODS)
            np.testing.assert_array_equal(example["seeds"], protocol["training_seeds"])
            assert set(record["branch_checkpoints"]) == set(METHODS[1:])
            branches = {arm: archive.checkpoint(root + name)
                        for arm, name in record["branch_checkpoints"].items()}
            for checkpoint in branches.values():
                assert checkpoint["seed"] == seed
            logits = reconstruct(archive.checkpoint(root + record["bundled_base"]), branches, example)
            index = example["seeds"].tolist().index(seed)
            expected = example["expected_selector_logits"][index]
            k = protocol["config"]["keep_patch_tokens"]
            np.testing.assert_allclose(logits, expected, atol=tolerance, rtol=0,
                                       err_msg=f"{setting}/seed_{seed}: reconstructed logits")
            masks = selected_positions(logits, k)
            np.testing.assert_array_equal(masks, example["expected_patch_indices"][index])
            for mi, method in enumerate(METHODS):
                forward_checks.append({"setting": setting, "seed": seed, "method": method,
                    "logit_max_absolute_error": float(np.max(np.abs(logits[mi] - expected[mi]))),
                    "topk_exact": True})

            folder = root + setting + "/"
            raw = archive.arrays(folder + f"seed_{seed}.npz")
            np.testing.assert_array_equal(raw["methods"], METHODS)
            np.testing.assert_array_equal(raw["selector_logits"][:, :1], expected)
            np.testing.assert_array_equal(raw["patch_indices"][:, :1], example["expected_patch_indices"][index])
            np.testing.assert_array_equal(selected_positions(raw["selector_logits"], k), raw["patch_indices"])
            assert raw["cosine_scores"].shape[:2] == raw["selector_logits"].shape[:2]
            raw_checks.append({"setting": setting, "seed": seed,
                "validation_rows": int(raw["selector_logits"].shape[1]),
                "arrays_read": len(raw), "saved_topk_matches_logits": True})
            patches = logits.shape[-1]
            for split in ("train", "val"):
                name = folder + f"seed_{seed}/{split}_targets.npz"
                target = archive.arrays(name)
                for key in ("patch_indices", "conditioned_patch_indices", "shared_patch_indices"):
                    check_masks(target[key], k, patches, name + ":" + key)
                    assert target[key].shape == target["patch_indices"].shape
                count = len(target["patch_indices"])
                assert len(target["image_indices"]) == len(target["pair_indices"]) == count
                np.testing.assert_allclose(target["target_probability"], 1. / k, rtol=0, atol=1e-12)
                if split == "val":
                    assert count == raw["selector_logits"].shape[1]
                    np.testing.assert_array_equal(target["shared_patch_indices"], raw["patch_indices"][0])
                target_checks.append({"setting": setting, "seed": seed, "split": split,
                    "rows": count, "arrays_read": len(target), "fixed_k_and_probability_verified": True})
            teacher = archive.arrays(folder + f"seed_{seed}/teacher_reference.npz")
            np.testing.assert_array_equal(teacher["methods"], ("conditioned", "wide", "top_50"))
            assert teacher["cosine_scores"].shape == raw["cosine_scores"].shape
        return {"archive": archive.path, "passed": True, "device": "cpu", "atol": tolerance,
            "base_checkpoints_reconstructed": len(records), "branch_checkpoints_reconstructed": 2 * len(records),
            "verification_inputs": len(examples), "forward_checks": forward_checks,
            "maximum_logit_absolute_error": max(r["logit_max_absolute_error"] for r in forward_checks),
            "target_checks": target_checks, "raw_checks": raw_checks,
            "scope": "CPU selector reconstruction for one saved input per setting, all seeds/arms; all target/raw NPZ arrays readable and structurally checked. Does not rerun CLIP or training."}
    finally:
        archive.close()


def verify_archive(archive, tolerance=1e-5):
    """Verify a caller-owned ZipFile without opening/closing its underlying source."""
    return verify_distillation(archive, tolerance)


def verify_specificity(path):
    """Recover one Flowers49 pair's CU-minus-UU mean directly from raw scores."""
    archive = Archive(path)
    try:
        name = archive.find("flowers_b16_49/protocol.json")
        folder = name[:-len("protocol.json")]
        protocol = archive.json(name)
        summary = archive.json(folder + "analysis/summary.json")
        s, t = protocol["primary_seed_pairs"][0]
        data = archive.arrays(folder + f"pair_{s}_{t}.npz")
        rows = archive.rows(folder + "examples.csv")
        text = archive.arrays(folder + "evidence.npz")["text_features"].astype(float)
        labels = np.asarray([int(r["class_id"]) for r in rows])
        pairs = np.asarray([[int(r["class_a"]), int(r["class_b"])] for r in rows])
        other = np.where(labels == pairs[:, 0], pairs[:, 1], pairs[:, 0])
        counts = Counter((r["task_id"], r["class_id"]) for r in rows)
        tasks = len({r["task_id"] for r in rows})
        weights = np.asarray([1. / (2 * tasks * counts[(r["task_id"], r["class_id"])]) for r in rows])
        np.testing.assert_allclose(weights, [float(r["task_class_weight"]) for r in rows], atol=1e-12, rtol=0)
        values = []
        for method in ("cu_half", "uu_half"):
            score = data["cosine_scores"][data["methods"].tolist().index(method)].astype(float)
            own, competitor = score[np.arange(len(rows)), labels], score[np.arange(len(rows)), other]
            correct = (own > competitor) | ((own == competitor) & (labels < other))
            margin = (own - competitor) / np.linalg.norm(text[labels] - text[other], axis=1)
            values.append(np.stack((correct.astype(float), margin), axis=-1))
        mean = weights @ (values[0] - values[1]) / weights.sum()
        saved = next(r for r in summary["classification"]["contrasts"]["cu_minus_uu"]["per_seed_pair"]
                     if (r["recipient_seed"], r["donor_seed"]) == (s, t))
        expected = np.asarray([saved["accuracy"], saved["margin"]])
        np.testing.assert_allclose(mean, expected, atol=1e-12, rtol=0)
        return {"archive": archive.path, "passed": True, "setting": "flowers_b16_49",
            "seed_pair": [s, t], "validation_rows": len(rows),
            "cu_minus_uu": dict(zip(("accuracy", "margin"), mean.tolist())),
            "maximum_mean_error": float(np.max(np.abs(mean - expected))),
            "scope": "One pair's raw archived classification scores reproduce its reported weighted means; no model replay or bootstrap rerun."}
    finally:
        archive.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distillation", type=Path, required=True, help="Downloaded hybrid-distillation-v1.zip")
    parser.add_argument("--specificity", type=Path, help="Optional downloaded hybrid-specificity-v1.zip")
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = {"distillation": verify_distillation(args.distillation, args.atol)}
    if args.specificity:
        result["specificity"] = verify_specificity(args.specificity)
    print(json.dumps(result, indent=2))
