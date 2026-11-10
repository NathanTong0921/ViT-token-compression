"""Post-result exploration of each selector's accuracy versus retained-token budget."""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_direct_classification import METHODS, METRICS, summarize, write_csv
from compare_direct_classification_budgets import load, paired_budget_effects


def tradeoff_values(low, high):
    primary, _, _ = paired_budget_effects(low, high)
    delta=high["outcomes"]-low["outcomes"]
    names=list(METHODS)+["objective_tradeoff_C","objective_tradeoff_U"]
    c=delta[:,METHODS.index("direct_C")]-delta[:,METHODS.index("kl_C")]
    u=delta[:,METHODS.index("direct_U")]-delta[:,METHODS.index("kl_U")]
    np.testing.assert_allclose(c-u,primary[:,1],atol=1e-14,rtol=1e-12)
    return np.concatenate((delta,np.stack((c,u),axis=1)),axis=1), names


def main(args):
    low,high=load(args.low),load(args.high)
    values,names=tradeoff_values(low,high)
    results,per_seed,_,_=summarize(values,low["weights"],low["image_ids"],names,
                                  args.replicates,args.bootstrap_seed)
    record=dict(status="Post-result exploratory analysis; not a primary hypothesis test.",
        direction="98-token accuracy minus 49-token accuracy for the same selector/training stage.",
        objective_tradeoff="(98 direct - 49 direct) - (98 KL - 49 KL), separately for C and U.",
        interval_scope="Ordinary 95% image-cluster bootstrap intervals; no equivalence or acceleration claim.",
        bootstrap_replicates=args.replicates,bootstrap_seed=args.bootstrap_seed,
        seeds=low["seeds"].tolist(),results=results,
        per_seed={name:per_seed[:,i,:].tolist() for i,name in enumerate(names)},metrics=list(METRICS))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"budget_tradeoff.json").write_text(json.dumps(record,indent=2)+"\n")
    rows=[]
    for name in names:
        for metric in METRICS:
            value=results[name][metric]
            rows.append(dict(contrast=name,metric=metric,mean=value["mean"],
                             low=value["ci95"][0],high=value["ci95"][1],
                             scope="post-result exploratory; ordinary 95% interval"))
    write_csv(args.output/"budget_tradeoff.csv",rows)
    print(json.dumps({name:results[name]["accuracy"] for name in names}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--low",type=Path,required=True)
    p.add_argument("--high",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--replicates",type=int,default=2000)
    p.add_argument("--bootstrap-seed",type=int,default=2717)
    main(p.parse_args())
