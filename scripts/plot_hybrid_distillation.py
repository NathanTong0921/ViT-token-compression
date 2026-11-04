"""Plot classification and mask-transfer results of fixed hybrid supervision."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

SETTINGS=(("pets_b16_49","Pets · 49 tokens"),("flowers_b16_49","Flowers · 49 tokens"),("flowers_b16_98","Flowers · 98 tokens"))


def read_summary(root,setting):
    direct=root/setting/"summary.json"
    return json.loads((direct if direct.exists() else root/setting/"analysis/summary.json").read_text())


def plot(source,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    source,output=Path(source),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    results={s:read_summary(source,s) for s,_ in SETTINGS}
    plt.rcParams.update({"font.size":10,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    definitions=[("classification","task_residual_minus_image_residual","Task − image, hybrid target","#2166ac"),
                 ("classification","task_residual_minus_shared","Task − shared U","#d97925"),
                 ("classification","image_residual_minus_shared","Image − shared U","#75945c"),
                 ("target_comparisons","target_by_conditioning_interaction","Target × conditioning","#8e5ca2")]
    fig,axes=plt.subplots(1,2,figsize=(11,6.3),sharey=True)
    fig.subplots_adjust(left=.20,right=.98,bottom=.23,top=.77,wspace=.28)
    rows=[]
    for si,(setting,_) in enumerate(SETTINGS):
        for ci,(section,key,label,color) in enumerate(definitions):
            estimate=results[setting][section]
            if section=="classification":estimate=estimate["contrasts"]
            estimate=estimate[key]
            y=si*1.7+(ci-1.5)*.27
            row={"setting":setting,"contrast":key}
            for ax,metric,scale in zip(axes,("accuracy","margin"),(100.,1000.)):
                d=estimate[metric];mean=d["mean"]*scale;low,high=np.asarray(d["ci95"])*scale
                ax.plot([low,high],[y,y],color=color,lw=3);ax.scatter(mean,y,color=color,s=25,zorder=3)
                row.update({f"{metric}_mean":mean,f"{metric}_ci95_low":low,f"{metric}_ci95_high":high})
                if "bonferroni_family_ci95" in d:
                    a,b=np.asarray(d["bonferroni_family_ci95"])*scale
                    ax.plot([a,b],[y,y],color=color,lw=1,alpha=.5,marker="|")
                    row.update({f"{metric}_adjusted_low":a,f"{metric}_adjusted_high":b})
            for run in estimate["per_seed"]:
                row[f"seed_{run['seed']}_accuracy_pp"]=run["accuracy"]*100
                row[f"seed_{run['seed']}_margin_x1000"]=run["margin"]*1000
            rows.append(row)
    for ax,metric in zip(axes,("Accuracy difference (percentage points)","Geometric margin difference (×10⁻³)")):
        ax.axvline(0,color="#888888",ls="--",lw=.8)
        ax.set_yticks(np.arange(3)*1.7,[label for _,label in SETTINGS]);ax.set_ylim(4.15,-.7)
        ax.set_xlabel(metric);ax.grid(axis="x",color="#eeeeee");ax.set_axisbelow(True)
    fig.suptitle("Learning the shared / conditioned hybrid selection",fontsize=15,fontweight="bold",y=.97)
    fig.legend(handles=[Line2D([0],[0],color=color,marker="o",label=label) for _,_,label,color in definitions],
               loc="upper center",bbox_to_anchor=(.60,.90),ncol=2,frameon=False,fontsize=9)
    fig.text(.20,.09,"Same KL loss; target changes from soft utility to uniform hybrid-mask membership.\n"
             "Three paired seeds; thick bars: image-bootstrap 95% intervals. Thin blue: three-setting adjustment.\n"
             "Target × conditioning = (task − image) with hybrid target minus the same effect with original target.",
             fontsize=8.7,linespacing=1.5,va="center")
    for ext in ("png","pdf"):fig.savefig(output/f"hybrid_distillation.{ext}",dpi=220,facecolor="white")
    plt.close(fig)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (output/"classification_contrasts.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

    fit_rows=[]
    fig,axes=plt.subplots(1,3,figsize=(11.4,5.2),sharey=True)
    fig.subplots_adjust(left=.08,right=.98,bottom=.25,top=.70,wspace=.20)
    for ax,(setting,title) in zip(axes,SETTINGS):
        records=results[setting]["target_fit"]
        for ai,(arm,label,color) in enumerate((("task_residual","Task correction","#2166ac"),("image_residual","Image correction","#d97925"))):
            for ti,(training,style) in enumerate((("original_trained","--"),("hybrid_trained","-"))):
                selected=[r for r in records if r["arm"]==arm]
                means=[np.mean([r[training][split]["hybrid_mask_overlap"] for r in selected])*100 for split in ("train","val")]
                ax.plot([0,1],means,ls=style,marker="o",color=color,label=f"{label}, {'original' if ti==0 else 'hybrid'} target")
                for split,mean in zip(("train","val"),means):
                    fit_rows.append({"setting":setting,"arm":arm,"training_target":training,"split":split,"mean_overlap_percent":mean})
        ax.set_title(title,loc="left");ax.set_xticks([0,1],["Training","Validation"])
        ax.grid(axis="y",color="#eeeeee");ax.set_xlim(-.12,1.12)
    axes[0].set_ylabel("Top-k overlap with hybrid teacher (%)")
    fig.suptitle("Does the hybrid mask transfer to unseen images and class pairs?",fontsize=13,fontweight="bold",y=.97)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.52,.88),ncol=2,frameon=False,fontsize=8.8)
    fig.text(.08,.075,"Points are task/class-balanced overlap, averaged over all three fixed seeds.\n"
             "Dashed: original-utility-trained branches. Solid: hybrid-mask-trained branches. Both evaluated against the same hybrid masks.",fontsize=8.5,linespacing=1.5)
    for ext in ("png","pdf"):fig.savefig(output/f"hybrid_target_fit.{ext}",dpi=220,facecolor="white")
    plt.close(fig)
    with (output/"target_fit_table.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(fit_rows[0]));writer.writeheader();writer.writerows(fit_rows)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source",type=Path,default=Path("results/hybrid-distillation-v1"));p.add_argument("--output",type=Path,default=Path("results/hybrid-distillation-v1"))
    a=p.parse_args();plot(a.source,a.output)
