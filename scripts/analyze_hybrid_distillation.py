"""Paired target-by-conditioning comparison and hybrid-teacher transfer."""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_core_protection import save_cases
from analyze_selector_ablations import binary_scores
from analyze_selector_budgets import weighted_estimates
from analyze_selector_scale import panel_weights, read_rows
from analyze_shared_residual import analyze as analyze_residual


METRICS = ("accuracy", "margin")


def target_interaction(new_task, new_image, old_task, old_image):
    return (new_task - new_image) - (old_task - old_image)


def summarize_effect(ids, weights, delta, seeds, replicates, family=1):
    summary = weighted_estimates(ids, weights, delta.mean(0), METRICS, replicates, 2321, family)
    points = np.einsum("n,snm->sm", weights, delta) / weights.sum()
    summary["per_seed"] = [{"seed":seed, **dict(zip(METRICS, point.tolist()))} for seed, point in zip(seeds, points)]
    return summary


def analyze(source, output, replicates=2000):
    source, output = Path(source), Path(output)
    result = analyze_residual(source, output, replicates)
    protocol = json.loads((source / "protocol.json").read_text())
    original = Path(protocol["original_source"])
    old_protocol = json.loads((original / "protocol.json").read_text())
    seeds, methods = protocol["training_seeds"], protocol["methods"]
    assert seeds == old_protocol["training_seeds"]
    base, old_rows = read_rows(source / "examples.csv"), read_rows(original / "examples.csv")
    keys = [(r["image_id"],r["task_id"]) for r in base]
    old_lookup = {(r["image_id"],r["task_id"]):i for i,r in enumerate(old_rows)}
    assert set(keys)==set(old_lookup)
    order=np.asarray([old_lookup[key] for key in keys])
    ids=np.asarray([r["image_id"] for r in base])
    labels=np.asarray([int(r["class_id"]) for r in base])
    pairs=np.asarray([[int(r["class_a"]),int(r["class_b"])] for r in base])
    others=np.where(labels==pairs[:,0],pairs[:,1],pairs[:,0])
    weights=panel_weights(base,"all")
    np.testing.assert_allclose(weights, panel_weights(old_rows,"all")[order], rtol=0,atol=1e-12)
    with np.load(source/"evidence.npz") as data:
        text=data["text_features"].astype(float)
    with np.load(original/"evidence.npz") as data:
        np.testing.assert_array_equal(text,data["text_features"])
    new_values,old_values,teacher_values=[],[],[]
    for seed in seeds:
        with np.load(source/f"seed_{seed}.npz") as new, np.load(original/f"seed_{seed}.npz") as old:
            np.testing.assert_array_equal(new["methods"],old["methods"])
            for method in ("shared",):
                mi=methods.index(method)
                np.testing.assert_array_equal(new["patch_indices"][mi],old["patch_indices"][mi,order])
                np.testing.assert_array_equal(new["cosine_scores"][mi],old["cosine_scores"][mi,order])
            new_values.append(binary_scores(new["cosine_scores"].astype(float),labels,others,text))
            old_values.append(binary_scores(old["cosine_scores"][:,order].astype(float),labels,others,text))
        with np.load(source/f"seed_{seed}/teacher_reference.npz") as teacher:
            assert list(teacher["methods"])==["conditioned","wide","top_50"]
            teacher_values.append(binary_scores(teacher["cosine_scores"].astype(float),labels,others,text))
    new_values,old_values,teacher_values=map(np.stack,(new_values,old_values,teacher_values))
    u,t,i=[methods.index(m) for m in ("shared","task_residual","image_residual")]
    values={"hybrid_target_task":new_values[:,t],"hybrid_target_image":new_values[:,i],
            "original_target_task":old_values[:,t],"original_target_image":old_values[:,i],
            "shared":new_values[:,u],"conditioned_teacher":teacher_values[:,0],"hybrid_teacher":teacher_values[:,2]}
    differences={
        "target_by_conditioning_interaction":target_interaction(new_values[:,t],new_values[:,i],old_values[:,t],old_values[:,i]),
        "task_target_change":new_values[:,t]-old_values[:,t],
        "image_target_change":new_values[:,i]-old_values[:,i],
        "hybrid_task_minus_hybrid_teacher":new_values[:,t]-teacher_values[:,2],
        "hybrid_image_minus_hybrid_teacher":new_values[:,i]-teacher_values[:,2],
        "hybrid_teacher_minus_shared":teacher_values[:,2]-new_values[:,u],
        "hybrid_teacher_minus_conditioned":teacher_values[:,2]-teacher_values[:,0]}
    result.update(target=protocol["target"],original_source=str(original),
                  target_comparisons={name:summarize_effect(ids,weights,delta,seeds,replicates) for name,delta in differences.items()},
                  target_comparison_table={name:summarize_effect(ids,weights,value,seeds,replicates) for name,value in values.items()},
                  target_fit=json.loads((source/"target_fit.json").read_text()),
                  interpretation="Same KL loss with different target distributions. Absolute KL values across targets have different entropy baselines; compare arms under the same target and use common mask overlap/classification metrics.")
    metadata={"image_id":ids,"task_id":np.asarray([r["task_id"] for r in base]),"class_id":labels,
              "class_a":pairs[:,0],"class_b":pairs[:,1]}
    save_cases(output,"target_comparisons",metadata,{**values,**differences},seeds,weights)
    np.savez_compressed(output/"target_classification_outcomes.npz",seeds=seeds,methods=methods,
                        hybrid_target=new_values,original_target=old_values,teacher=teacher_values,
                        teacher_methods=np.asarray(["conditioned","wide","top_50"]),metric_names=METRICS)
    (output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--replicates",type=int,default=2000)
    args=parser.parse_args()
    analyze(args.source,args.output or args.source/"analysis",args.replicates)
