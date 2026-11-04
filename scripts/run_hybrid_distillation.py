"""Fit matched shared residuals to a fixed, label-free hybrid mask target."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from evaluate_core_protection import SETTINGS as TEACHERS, padded_batches
from evaluate_selector_scale import selector_from_checkpoint
from evaluate_selectors import write_rows
from run_semantic_identity_pilot import load_data, synchronize, write_json
from run_shared_residual import (ARMS, METHODS, SETTINGS, audit_shared_endpoint,
                                cache_shared, cached_losses, evaluate, paired_order)
from token_compression.selector import shared_descriptor
from token_compression.shared_residual import SharedResidualSelector


def hybrid_masks(recipient_scores, donor_scores, k):
    """Half the exclusive donor positions replace the weakest recipient-only ones."""
    c_rank = np.argsort(-recipient_scores, axis=-1, kind="stable")[:, :k]
    u_rank = np.argsort(-donor_scores, axis=-1, kind="stable")[:, :k]
    masks = np.empty_like(c_rank, dtype=np.int16)
    for i, (c, u) in enumerate(zip(c_rank, u_rank)):
        add = u[~np.isin(u, c)]
        removable = c[~np.isin(c, u)]
        d = len(add) // 2
        drop = removable[-d:] if d else removable[:0]
        masks[i] = np.sort(np.concatenate((c[~np.isin(c, drop)], add[:d])))
    return masks


def mask_distribution(masks, patches, device):
    target = torch.zeros((len(masks), patches), dtype=torch.float32, device=device)
    target.scatter_(1, torch.as_tensor(masks, dtype=torch.long, device=device), 1. / masks.shape[1])
    return target


@torch.no_grad()
def generate_targets(conditioned, data, shared_scores, k, batch_size, output):
    """Selector inputs are tokens and both texts; no correct-class label enters."""
    logits = np.empty((len(data["image_ix"]), data["tokens"].shape[1] - 1), np.float32)
    device = data["tokens"].device
    for start, end, rows in padded_batches(len(logits), batch_size):
        index = torch.as_tensor(rows, device=device)
        values = conditioned(data["tokens"][data["examples"]["image_indices"][index]],
                             data["text"][data["examples"]["class_pairs"][index]])
        logits[start:end] = values[:end-start].float().cpu().numpy()
    u = shared_scores[data["examples"]["image_indices"]].cpu().numpy()
    masks = hybrid_masks(logits, u, k)
    conditioned_masks = np.sort(np.argsort(-logits, axis=-1, kind="stable")[:, :k], axis=-1)
    shared_masks = np.sort(np.argsort(-u, axis=-1, kind="stable")[:, :k], axis=-1)
    np.savez_compressed(output, patch_indices=masks, conditioned_patch_indices=conditioned_masks,
                        shared_patch_indices=shared_masks, image_indices=data["image_ix"],
                        pair_indices=data["pair_ix"], image_ids=data["image_ids"],
                        class_pairs=data["pairs_array"], target_probability=1. / k)
    return {**data, "targets": mask_distribution(masks, logits.shape[-1], device)}, masks


@torch.no_grad()
def fit_diagnostics(model, data, cached, original_targets, target_masks, batch_size, temperature):
    model.eval()
    totals = np.zeros(3)
    for start in range(0, len(data["weights"]), batch_size):
        end = min(start + batch_size, len(data["weights"]))
        rows = torch.arange(start, end, device=data["tokens"].device)
        ix, pairs = data["examples"]["image_indices"][rows], data["examples"]["class_pairs"][rows]
        logits = model.score_cached(cached[0][ix], cached[1][ix], data["text"][pairs])
        logs = F.log_softmax(logits / temperature, dim=-1)
        losses = [F.kl_div(logs, target[rows], reduction="none").sum(1).double().cpu().numpy()
                  for target in (data["targets"], original_targets)]
        chosen = np.argsort(-logits.cpu().numpy(), axis=-1, kind="stable")[:, :target_masks.shape[1]]
        overlap = (chosen[:, :, None] == target_masks[start:end, None, :]).any(-1).mean(-1)
        totals += data["weights"][start:end] @ np.stack((*losses, overlap), axis=-1)
    return dict(zip(("hybrid_target_kl", "original_target_kl", "hybrid_mask_overlap"), totals.tolist()))


def fit_branch(shared, conditioned, data, cached, order, config, seed):
    torch.manual_seed(seed)
    model = SharedResidualSelector(shared, conditioned, text_dim=data["text"].shape[1]).to(data["tokens"].device)
    optimizer = torch.optim.AdamW(model.branch.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    model.train()
    device, batch_size = data["tokens"].device, config["batch_size"]
    synchronize(device)
    before = time.perf_counter()
    curves = []
    for step in range(config["training_steps"]):
        rows = order[step*batch_size:(step+1)*batch_size]
        optimizer.zero_grad(set_to_none=True)
        loss = cached_losses(model, data, cached, rows, config["temperature"]).mean()
        loss.backward()
        optimizer.step()
        if step == 0 or (step+1) % config["checkpoint_interval"] == 0 or step+1 == config["training_steps"]:
            curves.append({"step": step+1, "batch_teacher_kl": float(loss.detach())})
    synchronize(device)
    return model.eval().requires_grad_(False), curves, time.perf_counter()-before


def historical_teacher_reference(data, teacher_file, matched_source, seed, output):
    """Verify newly generated masks, then reuse their existing frozen CLIP outputs."""
    with (matched_source / "examples.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(r["image_id"], r["task_id"]): i for i,r in enumerate(rows)}
    keys = [(str(data["image_ids"][i]), data["task_ids"][p]) for i,p in zip(data["image_ix"], data["pair_ix"])]
    assert set(keys) == set(lookup)
    order = np.asarray([lookup[key] for key in keys])
    with np.load(teacher_file) as fresh, np.load(matched_source / f"seed_{seed}.npz") as old:
        names = list(old["variants"])
        methods = ("conditioned", "wide", "top_50")
        expected = ("conditioned_patch_indices", "shared_patch_indices", "patch_indices")
        for method, field in zip(methods, expected):
            np.testing.assert_array_equal(fresh[field], old["patch_indices"][names.index(method), order])
        scores = np.stack([old["cosine_scores"][names.index(m), order] for m in methods])
    np.savez_compressed(output, methods=np.asarray(methods), cosine_scores=scores)
    return {"source": str(matched_source.resolve()), "all_three_masks_exact": True,
            "classification_scores_reused_for_identical_masks": True}


def run(config, shared_dir, conditioned_dir, output, setting, original_source, matched_source):
    import clip
    from token_compression.pruning import SingleStagePruner

    output, shared_dir, conditioned_dir = Path(output), Path(shared_dir), Path(conditioned_dir)
    original_source, matched_source = Path(original_source), Path(matched_source)
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    device = torch.device(config["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    temp, batch_size, k = config["temperature"], config["batch_size"], config["keep_patch_tokens"]
    train, val = [load_data(Path(config["cache_dir"]), split, device, batch_size, temp) for split in ("train", "val")]
    assert not set(train["image_ids"]) & set(val["image_ids"])
    assert not set(map(tuple, train["pairs_array"])) & set(map(tuple, val["pairs_array"]))
    torch.testing.assert_close(train["text"], val["text"], rtol=0, atol=0)
    old_protocol = json.loads((original_source / "protocol.json").read_text())
    paired_fields = ("training_seeds", "training_steps", "batch_size", "temperature", "learning_rate", "weight_decay",
                     "keep_patch_tokens", "prune_after", "model", "cache_dir", "image_batch_size")
    for key in paired_fields:
        assert config[key] == old_protocol["config"][key], key
    old_records = {(r["seed"], r["arm"]):r for r in json.loads((original_source / "training_records.json").read_text())}
    old_audits = {r["seed"]:r for r in json.loads((original_source / "sampling_audit.json").read_text())}
    backbone, _ = clip.load(config["model"], device=device, jit=False, download_root=config["model_cache"])
    backbone.eval().requires_grad_(False)
    pruner = SingleStagePruner(backbone.visual, config["prune_after"])
    prior = shared_descriptor(train["text"], train["pairs"])
    examples = [{"image_index":int(i), "image_id":str(val["image_ids"][i]), "class_id":int(val["labels"][i]),
                 "task_id":val["task_ids"][p], "class_a":int(val["pairs_array"][p,0]), "class_b":int(val["pairs_array"][p,1]),
                 "task_split":"unseen", "task_class_weight":float(w)} for i,p,w in zip(val["image_ix"],val["pair_ix"],val["weights"])]
    write_rows(output / "examples.csv", examples)
    np.savez_compressed(output / "evidence.npz", image_ids=val["image_ids"], labels=val["labels"],
                        text_features=val["text_array"], full_scores=val["full_scores"])
    protocol = {"setting":setting, "config":config, "git_commit":commit, "training_seeds":config["training_seeds"],
                "methods":list(METHODS), "original_source":str(original_source.resolve()),
                "shared_dir":str(shared_dir.resolve()), "conditioned_dir":str(conditioned_dir.resolve()),
                "primary":"Task minus image correction under hybrid supervision, with three-setting adjusted intervals.",
                "target":"Uniform 1/k on fixed50%-exclusive hybrid mask from C_s and U_s. Same softmax KL loss; only target changes.",
                "secondary":"Target-by-conditioning interaction against historical original-utility KL, ordinary paired95% intervals.",
                "teacher_labels":"No true label input in hybrid generation. Frozen C/U teachers were previously trained with supervised utilities.",
                "validation":"Val targets only diagnose fitting; val examples never optimize weights. Final fixed step, no checkpoint selection.",
                "final_test_used":False}
    write_json(output / "protocol.json", protocol)
    records, curves, sampling, evaluations, fits = [], [], [], [], []
    for seed in config["training_seeds"]:
        folder = output / f"seed_{seed}"
        folder.mkdir(exist_ok=True)
        shared_path = shared_dir / "full" / f"seed_{seed}/image_only_wide.pt"
        c_path = conditioned_dir / "full" / f"seed_{seed}/conditioned.pt"
        shared = selector_from_checkpoint(torch.load(shared_path, map_location=device, weights_only=True), prior,
                                          "image_only_wide", train["tokens"].shape[-1], device)
        conditioned = selector_from_checkpoint(torch.load(c_path, map_location=device, weights_only=True), prior,
                                               "conditioned", train["tokens"].shape[-1], device)
        synchronize(device)
        before=time.perf_counter()
        caches=[cache_shared(shared, d["tokens"],config["image_batch_size"]) for d in (train,val)]
        synchronize(device)
        cache_seconds=time.perf_counter()-before
        before=time.perf_counter()
        target_data, masks = [], []
        for split, data, cached in zip(("train","val"),(train,val),caches):
            new_data, mask=generate_targets(conditioned,data,cached[1],k,config["image_batch_size"],folder/f"{split}_targets.npz")
            target_data.append(new_data);masks.append(mask)
        synchronize(device)
        target_seconds=time.perf_counter()-before
        reference=historical_teacher_reference(val,folder/"val_targets.npz",matched_source,seed,folder/"teacher_reference.npz")
        order,audit=paired_order(train, config["training_steps"]*batch_size,seed)
        assert audit==old_audits[seed]
        sampling.append(audit)
        models, initials={},[]
        for arm in ARMS:
            torch.manual_seed(seed)
            initial_model=SharedResidualSelector(shared, arm=="task_residual",text_dim=train["text"].shape[1]).to(device)
            initial=fit_diagnostics(initial_model,target_data[1],caches[1],val["targets"],masks[1],batch_size,temp)
            assert abs(initial["original_target_kl"]-old_records[(seed,arm)]["initial_val_teacher_kl"])<1e-10
            assert str(shared_path.resolve())==old_records[(seed,arm)]["shared_checkpoint"]
            initials.append(initial)
            model,curve,seconds=fit_branch(shared,arm=="task_residual",target_data[0],caches[0],order,config,seed)
            models[arm]=model
            curves.extend({"seed":seed,"arm":arm,**row} for row in curve)
            diagnostic={split:fit_diagnostics(model,d,c,old["targets"],mask,batch_size,temp)
                        for split,d,c,old,mask in zip(("train","val"),target_data,caches,(train,val),masks)}
            old_checkpoint=torch.load(original_source/f"seed_{seed}/{arm}.pt",map_location=device,weights_only=True)
            original_model=SharedResidualSelector(shared,arm=="task_residual",text_dim=train["text"].shape[1]).to(device)
            original_model.branch.load_state_dict(old_checkpoint["branch_state_dict"])
            original_diagnostic={split:fit_diagnostics(original_model,d,c,old["targets"],mask,batch_size,temp)
                                 for split,d,c,old,mask in zip(("train","val"),target_data,caches,(train,val),masks)}
            fits.append({"seed":seed,"arm":arm,"initial_val":initial,"hybrid_trained":diagnostic,"original_trained":original_diagnostic})
            torch.save({"branch_state_dict":{key:value.detach().cpu().clone() for key,value in model.branch.state_dict().items()},
                        "conditioned":model.conditioned,"hidden_dim":model.branch.hidden_dim,"shared_hidden_dim":shared.hidden_dim,
                        "text_dim":train["text"].shape[1],"shared_checkpoint":str(shared_path.resolve()),
                        "conditioned_teacher_checkpoint":str(c_path.resolve()),"seed":seed,"arm":arm,"config":config,
                        "target":"uniform_hybrid_mask","git_commit":commit},folder/f"{arm}.pt")
            record={"seed":seed,"arm":arm,"initial_val_teacher_kl":initial["hybrid_target_kl"],
                    "final_train_teacher_kl":diagnostic["train"]["hybrid_target_kl"],"final_val_teacher_kl":diagnostic["val"]["hybrid_target_kl"],
                    "training_seconds":seconds,"parameters":model.parameter_counts(),"shared_checkpoint":str(shared_path.resolve())}
            records.append(record)
            write_json(output/"training_records.json",records);write_json(output/"target_fit.json",fits)
            write_rows(output/"training_curve.csv",curves)
            print(json.dumps({"setting":setting,**record}),flush=True)
            del initial_model,original_model
        assert initials[0]==initials[1]
        measured=evaluate(models,shared,val,pruner,k,config["image_batch_size"],output/f"seed_{seed}.npz")
        measured["historical_shared_endpoint"]=audit_shared_endpoint(val,output/f"seed_{seed}.npz",Path("runs/core-protection-v1")/setting,seed,config["identity_atol"])
        evaluations.append({"seed":seed,"shared_feature_cache_seconds":cache_seconds,"target_generation_seconds":target_seconds,
                            "identical_initial_val_loss":True,"original_initial_loss_matches":True,"teacher_reference":reference,**measured})
        write_json(output/"evaluation_records.json",evaluations);write_json(output/"sampling_audit.json",sampling)
        del caches,target_data,models,shared,conditioned,model
    from analyze_hybrid_distillation import analyze
    return analyze(output,output/"analysis",config.get("bootstrap_replicates",2000))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting",choices=SETTINGS,required=True)
    parser.add_argument("--device")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--original-source",type=Path)
    parser.add_argument("--matched-source",type=Path)
    args=parser.parse_args()
    config_path,shared_dir=SETTINGS[args.setting]
    config=json.loads(Path(config_path).read_text())
    if args.device:config["device"]=args.device
    run(config,shared_dir,TEACHERS[args.setting]["conditioned_dir"],args.output,args.setting,
        args.original_source or Path("runs/shared-residual-v1")/args.setting,
        args.matched_source or Path("runs/matched-replacement-v1")/args.setting)
