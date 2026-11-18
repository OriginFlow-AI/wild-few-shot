"""Trial finger mean -> subject/stage trial mean -> subject stage mean -> people mean."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/test_finger_stage_subject_equal_20261116'
METRICS=['mae_n','mape_percent','corr','tau_b','pairwise_force_loss_n']
LEGACY={'ap304_stage2_ex45_A_oneepoch_seed43_20261107',
        'ap304_stage2_ex45_pairwise_oneepoch_seed43_20261112',
        'ap304_stage2_ex45_original_pairwise_oneepoch_seed43_20261113'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def mean(xs):
    good=[float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return math.fsum(good)/len(good) if good else None

def aggregate(rows):
    return dict(metrics={k:mean(r['metrics'][k] for r in rows) for k in METRICS},
        units=len(rows),valid_units={k:sum(r['metrics'][k] is not None for r in rows) for k in METRICS})

def fingers(row):
    stats=row['per_finger_statistics'];taus=row['per_finger_tau_b']
    assert len(stats)==len(taus)==5
    per=[]
    for s,tau in zip(stats,taus):
        corr=s['cross']/math.sqrt(s['m2_pred']*s['m2_target']) if s['m2_pred']>0 and s['m2_target']>0 else None
        per.append(dict(mae_n=s['absolute_error']/s['n'],
            mape_percent=100*s['ape_sum']/s['active_count'] if s['active_count'] else None,
            corr=max(-1.,min(1.,corr)) if corr is not None else None,tau_b=tau))
    metrics={k:mean(p[k] for p in per) for k in METRICS[:-1]}
    metrics['pairwise_force_loss_n']=row.get('pairwise_force_loss_n')
    return dict(trial_key=row['trial_key'],subject=row['subject'],stage=row['stage'],
        metrics=metrics,valid_fingers={k:sum(p[k] is not None for p in per) for k in METRICS[:-1]},per_finger=per)

def hierarchy(trials):
    groups=defaultdict(list)
    for t in trials:groups[(t['subject'],t['stage'])].append(t)
    cells=[dict(subject=s,stage=g,**aggregate(ts)) for (s,g),ts in sorted(groups.items())]
    people=defaultdict(list)
    for c in cells:people[c['subject']].append(c)
    subjects=[dict(subject=s,**aggregate(cs)) for s,cs in sorted(people.items())]
    return aggregate(subjects),cells,subjects

def main():
    # Known unbalanced cohort: stage and subject weighting must not follow trial counts.
    synthetic=[dict(subject=s,stage=g,metrics={k:v for k in METRICS})
        for s,g,v in [('a',1,0.),('a',1,2.),('a',2,5.),('b',1,9.)]]
    assert hierarchy(synthetic)[0]['metrics']['mae_n']==6.
    OUT.mkdir(exist_ok=True);summary=[]
    for run in sorted((ROOT/'runs').glob('ap304_*')):
        if not (run.name.startswith('ap304_filtered_') or run.name in LEGACY):continue
        if not (run/'resolved_config.yaml').exists():continue
        config=yaml.safe_load((run/'resolved_config.yaml').read_text())
        for receipt_path in sorted((run/'evaluations').glob('*test*/COMPLETE.json')):
            evaluation=receipt_path.parent;receipt=json.loads(receipt_path.read_text())
            for name,digest in receipt['files'].items():assert sha(evaluation/name)==digest,(evaluation,name)
            files=[evaluation/'trials.jsonl'] if (evaluation/'trials.jsonl').exists() else sorted(evaluation.glob('rank*_trials.jsonl'))
            records=[json.loads(line) for p in files for line in p.read_text().splitlines()]
            assert len(records)==receipt['trials'] and len({r['trial_key'] for r in records})==len(records)
            trials=[fingers(r) for r in records];overall,cells,people=hierarchy(trials)
            target=OUT/run.name/evaluation.name;target.mkdir(parents=True,exist_ok=True)
            detail=dict(overall=overall,per_subject=people,per_subject_stage=cells,
                source_evaluation=str(evaluation),receipt_sha256=sha(receipt_path),binding=receipt['binding'])
            (target/'result.json').write_text(json.dumps(detail,indent=2,allow_nan=False))
            with (target/'trials.jsonl').open('w') as f:
                for trial in trials:f.write(json.dumps(trial,allow_nan=False)+'\n')
            row=dict(run=run.name,evaluation=evaluation.name,
                train_cohort='five-rule filtered' if run.name.startswith('ap304_filtered_') else 'stage4/5 excluded only',
                family='original' if 'original' in run.name else 'A',pairwise_weight=config.get('pairwise_force_weight',0.),
                trials=len(records),subjects=len(people),stages=','.join(map(str,sorted({r['stage'] for r in records}))),
                **overall['metrics'],checkpoint=receipt['binding']['checkpoint'])
            summary.append(row)
            print(row['run'],row['evaluation'],overall['metrics'],flush=True)
    with (OUT/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    protocol=dict(order=['within trial: metrics per finger then equal valid-finger mean',
        'within person/stage: equal valid-trial mean','within person: equal valid-stage mean','equal valid-person mean'],
        undefined='null/excluded per metric; valid counts saved at every level; absent stages not zero-filled',
        mape='absolute target >= 0.5 N; percentage units',
        pairwise='original ten-pair mean, then same trial/stage/subject hierarchy; unavailable baseline values remain null',
        scope='all completed Test evaluations of filtered-data arms and the three earlier completed ex45 controls in this conversation; preserve each original cohort',
        loss='training objective and checkpoint selection unchanged; do not compare weighted loss between objectives',
        completed_evaluations=len(summary),synthetic_hierarchy_check_passed=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (OUT/'results.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    lines=['# Five-finger / stage / subject equal Test results','',
        'Each trial: mean per-finger metrics; each person-stage: mean trials; each person: mean stages; finally mean people.',
        'Undefined values excluded with counts retained. Original Test cohorts are preserved. Training/selection unchanged.','',
        '|Run|Evaluation|Trials|People|MAE N|MAPE %|Corr|Tau|Pairwise N|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        vals=['—' if r[k] is None else f'{r[k]:.6f}' for k in METRICS]
        lines.append('|'+ '|'.join([r['run'],r['evaluation'],str(r['trials']),str(r['subjects']),*vals])+'|')
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('COMPLETE',len(summary),OUT)

if __name__=='__main__':main()
