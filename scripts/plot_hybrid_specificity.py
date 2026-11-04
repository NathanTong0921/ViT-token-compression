"""Plot the fixed conditional/shared versus shared/shared mixture comparison."""

import argparse
import csv
import json
from pathlib import Path

SETTINGS=(('pets_b16_49','Pets · 49 tokens'),('pets_b16_98','Pets · 98 tokens'),
          ('flowers_b16_49','Flowers · 49 tokens'),('flowers_b16_98','Flowers · 98 tokens'))


def plot(source,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    source,output=Path(source),Path(output);output.mkdir(parents=True,exist_ok=True)
    data=[json.loads((source/setting/'summary.json').read_text()) for setting,_ in SETTINGS]
    rows=[]
    for (setting,label),summary in zip(SETTINGS,data):
        for contrast,values in summary['classification']['contrasts'].items():
            for metric in ('accuracy','margin'):
                v=values[metric]
                rows.append(dict(setting=setting,label=label,contrast=contrast,metric=metric,mean=v['mean'],
                    ci95_low=v['ci95'][0],ci95_high=v['ci95'][1],
                    adjusted_low=v.get('bonferroni_family_ci95',[None,None])[0],
                    adjusted_high=v.get('bonferroni_family_ci95',[None,None])[1]))
    with (output/'contrast_table.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10.8,5.2),sharey=True)
    fig.subplots_adjust(left=.20,right=.98,bottom=.25,top=.77,wspace=.25)
    for ax,metric,scale,title in zip(axes,('margin','accuracy'),(1000,100),
            ('Classification margin (×10⁻³)','Classification accuracy (percentage points)')):
        for i,summary in enumerate(data):
            v=summary['classification']['contrasts']['cu_minus_uu'][metric]
            ax.hlines(i,*[x*scale for x in v['bonferroni_family_ci95']],color='#9bb4cd',lw=1.5)
            ax.hlines(i,*[x*scale for x in v['ci95']],color='#2166ac',lw=4)
            ax.plot(v['mean']*scale,i,'o',color='#2166ac',ms=6)
        ax.axvline(0,color='#777777',ls='--',lw=1)
        ax.set_yticks(range(4),[label for _,label in SETTINGS]);ax.invert_yaxis()
        ax.grid(axis='x',color='#eeeeee');ax.set_axisbelow(True);ax.set_xlabel(title)
    fig.suptitle('Does candidate information add more than a second shared selector?',fontsize=14,fontweight='bold',y=.97)
    fig.legend(handles=[Line2D([0],[0],color='#2166ac',lw=4,label='Ordinary 95% interval'),
                        Line2D([0],[0],color='#9bb4cd',lw=1.5,label='4-setting adjusted interval')],
               loc='upper center',bbox_to_anchor=(.58,.90),ncol=2,frameon=False,fontsize=9)
    fig.text(.20,.10,'Conditional + shared minus shared + shared; equal retained-token budgets.\n'
        'Same 50%-of-exclusive-pool replacement rule; all six ordered distinct-seed pairs.\n'
        'Pair outcomes are averaged before paired image resampling. Positive favors conditional + shared.',
        fontsize=9,ha='left',va='center',linespacing=1.5)
    for extension in ('png','pdf'):fig.savefig(output/f'hybrid_specificity.{extension}',dpi=220,facecolor='white')
    plt.close(fig)
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path('results/hybrid-specificity-v1'))
    p.add_argument('--output',type=Path,default=Path('results/hybrid-specificity-v1'));a=p.parse_args();plot(a.source,a.output)
