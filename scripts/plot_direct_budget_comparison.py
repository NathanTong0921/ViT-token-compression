"""Plot paired 49/98-token effects and their directly estimated budget changes."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EFFECTS=("direct_C_minus_direct_U", "direct_gap_minus_kl_gap")
CHANGES=("direct_gap_high_minus_low", "objective_interaction_high_minus_low")
LABELS=("Direct C − direct U", "Change in C−U gap\n(direct versus KL)")


def points(summary, names):
    values=[summary["contrasts"][name]["accuracy"] for name in names]
    means=np.asarray([v["mean"] for v in values])*100
    ci=np.asarray([v["two_primary_adjusted_ci95"] for v in values])*100
    return means, np.stack((means-ci[:,0],ci[:,1]-means)), ci


def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    low=json.loads(args.low.read_text());high=json.loads(args.high.read_text())
    change=json.loads(args.change.read_text())
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42,
                         "axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.5),sharey=True)
    fig.subplots_adjust(left=.25,right=.97,top=.72,bottom=.22,wspace=.22)
    fig.suptitle("Direct-training benefits at two token budgets",fontsize=16,fontweight="bold",y=.97)
    fig.text(.5,.865,"Flowers102 · same 6,149 images and 204 class combinations · three fixed seeds",
             ha="center",fontsize=10,color="#596573")
    ranges=[]
    for source,budget,color,offset in ((low,change["low_budget"],"#8A94A6",-.1),
                                     (high,change["high_budget"],"#2166A5",.1)):
        means,err,ci=points(source,EFFECTS)
        axes[0].errorbar(means,np.arange(2)+offset,xerr=err,fmt="o",color=color,
                         capsize=3,markersize=5,label=f"{budget} tokens")
        ranges.extend(ci.ravel())
    means,err,ci=points(change,CHANGES)
    axes[1].errorbar(means,np.arange(2),xerr=err,fmt="o",color="#2166A5",capsize=3,markersize=5)
    for ax,values in zip(axes,(ranges,ci.ravel())):
        span=max(abs(float(v)) for v in values)*1.15
        ax.set_xlim(-span,span);ax.set_ylim(1.45,-.45)
        ax.axvline(0,color="#737D8B",linestyle="--",linewidth=1)
        ax.grid(axis="x",color="#E7EBF0");ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    axes[0].set_yticks([0,1],LABELS)
    axes[0].set_xlabel("Accuracy effect (percentage points)")
    axes[1].set_xlabel(f"Paired change: {change['high_budget']} − {change['low_budget']} tokens (pp)")
    axes[0].legend(loc="upper center",bbox_to_anchor=(.5,1.15),ncol=2,frameon=False)
    fig.text(.25,.085,"Right panel directly subtracts paired image/task effects across budgets.\n"
             "Intervals use image-cluster bootstrap with adjustment for each pair of comparisons.",
             fontsize=9,color="#596573",va="center")
    for extension in ("png","pdf"):
        fig.savefig(args.output/f"budget_effects.{extension}",dpi=240,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--low",type=Path,default=Path("results/direct-classification-panel-v1/summary.json"))
    p.add_argument("--high",type=Path,default=Path("results/direct-classification-98-panel-v1/summary.json"))
    p.add_argument("--change",type=Path,default=Path("results/direct-classification-98-panel-v1/budget-comparison/summary.json"))
    p.add_argument("--output",type=Path,default=Path("results/direct-classification-98-panel-v1/budget-comparison/figures"))
    main(p.parse_args())
