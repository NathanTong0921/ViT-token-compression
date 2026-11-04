"""Test whether conditional/shared mixtures beat independent shared/shared mixtures."""

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from analyze_core_protection import exchange_outcomes
from analyze_selector_ablations import binary_scores
from analyze_selector_budgets import weighted_estimates
from analyze_selector_scale import panel_weights, read_rows
from evaluate_selectors import write_rows
from evaluate_hybrid_specificity import METHODS, FIXED_D

METRICS = ('accuracy', 'margin')


def paired_summary(ids, weights, delta, pairs, replicates, family=1):
    result = weighted_estimates(ids, weights, delta.mean(0), METRICS, replicates, 2311, family)
    result['per_seed_pair'] = [dict(recipient_seed=s, donor_seed=t,
        **dict(zip(METRICS, (weights @ values / weights.sum()).tolist())))
        for (s, t), values in zip(pairs, delta)]
    return result


def restricted_summary(ids, weights, delta, eligible, pairs, replicates):
    """Pool feasible pair/row observations, then resample images jointly across pairs."""
    counts = eligible.sum(0)
    retained = counts > 0
    effective_weights = weights * counts / len(pairs)
    per_pair = []
    for (s, t), values, keep in zip(pairs, delta, eligible):
        w = weights * keep
        per_pair.append(dict(recipient_seed=s, donor_seed=t, weighted_coverage=float(w.sum()/weights.sum()),
            **{key: float((w @ values)[i]/w.sum()) if w.sum() else None for i, key in enumerate(METRICS)}))
    result = dict(weighted_pair_row_coverage=float(effective_weights.sum()/weights.sum()),
        image_count=int(len(np.unique(ids[retained]))), row_count=int(retained.sum()),
        per_seed_pair=per_pair,
        estimand='Original task/class row weights times equal pair weights, restricted to jointly feasible pair/rows. Pair eligibility can differ; images are resampled jointly.')
    if retained.any():
        values = (delta[:, retained] * eligible[:, retained, None]).sum(0) / counts[retained, None]
        result.update(weighted_estimates(ids[retained], effective_weights[retained], values,
                                        METRICS, replicates, 2311))
    else:
        result.update({key: None for key in METRICS})
    return result


def analyze(source, output, replicates=2000):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((source/'protocol.json').read_text())
    pairs = protocol['primary_seed_pairs']
    base = read_rows(source/'examples.csv')
    ids, tasks = np.asarray([r['image_id'] for r in base]), np.asarray([r['task_id'] for r in base])
    ix = np.asarray([int(r['image_index']) for r in base])
    labels = np.asarray([int(r['class_id']) for r in base])
    candidates = np.asarray([[int(r['class_a']), int(r['class_b'])] for r in base])
    other = np.where(labels == candidates[:, 0], candidates[:, 1], candidates[:, 0])
    weights = panel_weights(base, 'all')
    np.testing.assert_allclose(weights, [float(r['task_class_weight']) for r in base], atol=1e-12, rtol=0)
    with np.load(source/'evidence.npz') as data:
        text, full = data['text_features'].astype(float), data['full_scores'][ix].astype(float)
        np.testing.assert_array_equal(data['image_ids'][ix], ids)
    scores, masks, eligibility, displacements, disagreements = [], [], [], [], []
    for s, t in pairs:
        with np.load(source/f'pair_{s}_{t}.npz') as data:
            np.testing.assert_array_equal(data['methods'], METHODS)
            scores.append(data['cosine_scores'].astype(float)); masks.append(data['patch_indices'].copy())
            eligibility.append(data['fixed_d_eligible'].copy())
            displacements.append(data['changed_tokens_vs_recipient'].copy()); disagreements.append(data['disagreement_sizes'].copy())
    scores, masks, eligibility = np.stack(scores), np.stack(masks), np.stack(eligibility)
    displacements, disagreements = np.stack(displacements), np.stack(disagreements)
    raw = binary_scores(scores.reshape(-1, len(ids), len(text)), labels, other, text).reshape(len(pairs), len(METHODS), len(ids), 2)
    differences = {'cu_minus_uu': raw[:,3]-raw[:,4], 'cu_minus_conditioned': raw[:,3]-raw[:,0],
                   'cu_minus_shared_recipient': raw[:,3]-raw[:,1], 'cu_minus_shared_donor': raw[:,3]-raw[:,2],
                   'uu_minus_shared_recipient': raw[:,4]-raw[:,1], 'uu_minus_shared_donor': raw[:,4]-raw[:,2],
                   'hybrid_increment_interaction': (raw[:,3]-raw[:,0])-(raw[:,4]-raw[:,1])}
    classification = {'table': [dict(method=METHODS[i], **paired_summary(ids, weights, raw[:,i], pairs, replicates)) for i in range(5)],
        'contrasts': {key: paired_summary(ids, weights, value, pairs, replicates, 4 if key=='cu_minus_uu' else 1) for key,value in differences.items()},
        'full_reference': weighted_estimates(ids, weights, binary_scores(full[None], labels, other, text)[0], METRICS, replicates, 2311)}
    fixed = {}
    for di,d in enumerate(FIXED_D):
        delta=raw[:,5+2*di]-raw[:,6+2*di]
        fixed[str(d)] = restricted_summary(ids, weights, delta, eligibility[:,di], pairs, replicates)
        for pi in range(len(pairs)):
            keep = eligibility[pi,di]
            np.testing.assert_array_equal(displacements[pi,5+2*di:7+2*di,keep], np.full((keep.sum(),2),d))
    by_image=defaultdict(list)
    for i,image in enumerate(ids):by_image[image].append(i)
    left,right=np.asarray([p for image in sorted(by_image) for p in combinations(by_image[image],2)],dtype=int).T
    ex_ids=ids[left]
    unique,counts=np.unique(ex_ids,return_counts=True);count_map=dict(zip(unique,counts))
    ex_weights=np.asarray([1/(len(unique)*count_map[i]) for i in ex_ids])
    own,swapped=exchange_outcomes(scores[:,:5], labels, other, text, left,right)
    effect=own-swapped
    for i in (1,2,4):
        np.testing.assert_array_equal(masks[:,i,left],masks[:,i,right])
        np.testing.assert_array_equal(effect[:,i],np.zeros_like(effect[:,i]))
    exchange={'table':[dict(method=METHODS[i],**paired_summary(ex_ids,ex_weights,effect[:,i],pairs,replicates)) for i in range(5)],
        'image_count':len(unique),'task_pair_count':len(left)}
    diagnostics=[]
    for pi,(s,t) in enumerate(pairs):
        diagnostics.append(dict(recipient_seed=s,donor_seed=t,
            cu_disagreement=float(weights@disagreements[pi,0]),uu_disagreement=float(weights@disagreements[pi,1]),
            cu_actual_replacements=float(weights@displacements[pi,3]),uu_actual_replacements=float(weights@displacements[pi,4])))
    diagonal=[]
    for s,t in protocol['reference_seed_pairs']:
        with np.load(source/f'pair_{s}_{t}.npz') as data:
            values=binary_scores(data['cosine_scores'][:5].astype(float),labels,other,text)
        diagonal.append(dict(seed=s,cu_minus_c=dict(zip(METRICS,(weights@(values[3]-values[0])).tolist())),
            cu_minus_u=dict(zip(METRICS,(weights@(values[3]-values[1])).tolist()))))
    metadata=dict(image_id=ids,task_id=tasks,class_id=labels,class_a=candidates[:,0],class_b=candidates[:,1])
    np.savez_compressed(output/'classification_cases.npz',**metadata,weights=weights,seed_pairs=np.asarray(pairs),
        methods=np.asarray(METHODS),outcomes=raw,eligible=eligibility,actual_replacements=displacements,disagreements=disagreements)
    np.savez_compressed(output/'exchange_cases.npz',image_id=ex_ids,weights=ex_weights,seed_pairs=np.asarray(pairs),
        methods=np.asarray(METHODS[:5]),own=own,swapped=swapped,left=left,right=right)
    image_ids,inverse=np.unique(ids,return_inverse=True)
    columns={'image_id':image_ids,'weight':np.bincount(inverse,weights=weights)}
    for name,delta in differences.items():
        for i,metric in enumerate(METRICS):columns[name+'_'+metric+'_weighted_sum']=np.bincount(inverse,weights=weights*delta.mean(0)[:,i])
    write_rows(output/'classification_images.csv',[dict(zip(columns,row)) for row in zip(*columns.values())])
    write_rows(output/'replacement_counts.csv',diagnostics)
    result=dict(setting=protocol['setting'],source=str(source.resolve()),git_commit=protocol['git_commit'],
        primary_seed_pairs=pairs,reference_seed_pairs=protocol['reference_seed_pairs'],primary_family_size=4,
        bootstrap_replicates=replicates,bootstrap_seed=2311,classification_image_count=len(image_ids),
        classification_task_count=len(set(tasks)),classification_image_task_count=len(ids),
        primary='C_s+U_t minus U_s+U_t; six ordered distinct-seed pairs, same 50%-exclusive rule; four dataset/budget settings.',
        uncertainty='Average outcomes across fixed seed pairs before paired image bootstrap; pairs are correlated, not independent replicates. Adjust four primary comparisons separately for accuracy and margin.',
        classification=classification,fixed_displacement=fixed,exchange=exchange,replacement_counts=diagnostics,
        historical_same_seed_reference=diagonal,execution=json.loads((source/'execution.json').read_text()))
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--replicates',type=int,default=2000)
    args=parser.parse_args()
    result=analyze(args.source,args.output,args.replicates)
    print(json.dumps({'setting':result['setting'],'primary':result['classification']['contrasts']['cu_minus_uu']},indent=2))
