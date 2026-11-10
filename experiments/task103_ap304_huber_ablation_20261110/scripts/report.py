"""Produce the four-way comparison only after all three selected models finish."""
import argparse
import collections
import csv
from pathlib import Path
import numpy as np
from common import BRANCHES, atomic, load

def main(root):
    root=Path(root);data={}
    for b in BRANCHES:
        if load(root/'runs'/b/'DONE.json')['phase']!='complete':raise ValueError('Incomplete branch')
        data[b]={r['id']:r for r in load(root/'runs'/b/'final_test.json')['rows']}
    ids=set(data[BRANCHES[0]])
    if any(set(v)!=ids for v in data.values()):raise ValueError('Test record denominators differ')
    checked=0
    for id in sorted(ids):
        paths=[root/'runs'/b/'test_predictions'/(id+'.npz') for b in BRANCHES]
        with np.load(paths[0]) as z:reference={k:z[k].copy() for k in ['mask','raw','cal','bsq','finger','t_ns']}
        for p in paths[1:]:
            with np.load(p) as z:
                if not all(np.array_equal(reference[k],z[k]) for k in reference):raise ValueError('Scoring points or GT differ: '+id)
        checked+=1
    experiments=[(1,'calibrated_huber','cal'),(2,'calibrated_huber','bsq'),(3,'raw_huber','bsq'),(4,'raw_original','bsq')]
    rows=[];people={}
    for n,b,target in experiments:
        subject=collections.defaultdict(list);stage=collections.defaultdict(list)
        for r in data[b].values():
            subject[r['subject']].append(r['metrics'][target]);stage[r['stage']].append(r['metrics'][target])
        def avg(records,key):
            values=[r[key] for r in records if r is not None and r[key] is not None]
            return float(np.mean(values)) if values else None
        by_person={s:{k:avg(rr,k) for k in ['mae_N','mape_percent','corr','huber_N2']} for s,rr in subject.items()}
        people[n]=by_person
        row=dict(experiment=n,training_branch=b,test_label=target,recordings=len(ids),subjects=len(subject))
        for k in ['mae_N','mape_percent','corr','huber_N2']:
            row['subject_equal_'+k]=avg(list(by_person.values()),k)
            row['trial_equal_'+k]=avg([r['metrics'][target] for r in data[b].values()],k)
        rows.append(row)
    paired=[]
    for a,b,name in [(2,3,'calibration: experiment2 minus3'),(3,4,'loss: experiment3 minus4')]:
        for s in sorted(set(people[a])&set(people[b])):
            paired.append(dict(comparison=name,subject=s,mae_difference_N=people[a][s]['mae_N']-people[b][s]['mae_N']))
    out=root/'results';out.mkdir(exist_ok=True)
    for name,rr in [('four_experiments.csv',rows),('paired_person_differences.csv',paired)]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    atomic(out/'FINAL_REPORT.json',dict(status='COMPLETE',experiments=rows,per_person=people,
        common_test_recordings=checked,point_and_ground_truth_identity_checked=True,
        calibration_role='first eligible complete press per finger, removed from scoring',
        caveats=['Experiment1 measures agreement with calibrated CMCU. Experiments2/3/4 use BSQ truth.',
                 'Metrics cover the common in-domain, routed, synchronized, EMG-valid subset; exclusions must be read with dataset and calibration QC.',
                 'This is within-person ordinary-stage training with held-out calibration stages, not subject-disjoint task103 generalization.']))
    lines=['Task103 four-way comparison (completed)', '', 'Subject-equal metrics on identical scoring points:',
           'experiment\ttest label\tMAE_N\tMAPE_%\tCorr']
    for r in rows:lines.append(f"{r['experiment']}\t{r['test_label']}\t{r['subject_equal_mae_N']:.6f}\t{r['subject_equal_mape_percent']:.6f}\t{r['subject_equal_corr']}")
    lines.extend(['','See FINAL_REPORT.json and paired_person_differences.csv.',
                  'Read calibration_quality.json and dataset exclusions before interpreting retained-point accuracy.'])
    (out/'REPORT.txt').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)
