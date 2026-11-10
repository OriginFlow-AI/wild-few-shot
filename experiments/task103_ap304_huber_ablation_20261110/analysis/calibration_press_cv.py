"""Diagnostic leave-one-press-out calibration; never mutates training artifacts."""
import argparse
import collections
import csv
import json
import os
from pathlib import Path
import sys
import numpy as np
from scipy.stats import spearmanr


def main(root):
    root=Path(root)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from common import atomic,load
    from prepare import fit,pair,CALIBRATION_VERSION
    plan=load(root/'plan.json');assert plan['calibration_version']==CALIBRATION_VERSION
    metadata={r['id']:r for r in plan['rows']}
    output=root/'analysis_output';output.mkdir(exist_ok=True)
    with (output/'accepted_calibration_diagnostics.csv').open() as f:
        fit_errors={(r['id'],int(r['finger_index'])):float(r['source_fit_MAE_N']) for r in csv.DictReader(f)}
    records=[];fold_reasons=collections.Counter();total_abs=0.;total_scored=0
    for nr,path in enumerate(sorted((root/'cache/calibration').glob('*.json'))):
        model=load(path);row=metadata[model['id']]
        if row['stage']!='mix-test-5' or not model['channels']:continue
        assert model['calibration_version']==CALIBRATION_VERSION
        with np.load(root/'cache/native'/(row['id']+'.npz')) as a:paired=pair(a)
        with np.load(root/'cache/routes'/(row['id']+'.npz')) as a:
            reference=a['reference'].copy();sync=a['sync'].copy()
        events={e['event']:e for e in load(root/'cache/events'/(row['id']+'.json'))}
        for finger,m in model['channels'].items():
            j=int(finger);presses=[events[k] for k in m['presses']]
            record=dict(id=row['id'],subject=row['subject'],finger_index=j,presses=len(presses),
                source_fit_MAE_N=fit_errors[row['id'],j],scored_folds=0,reference_points=0,
                scored_points=0,coverage=None,press_equal_MAE_N=None,point_equal_MAE_N=None)
            if len(presses)<2:
                record['status']='one_press_no_independent_repeat';records.append(record);continue
            samples=[np.unique(np.r_[json.loads(e['quiet_indices']),np.arange(e['start_idx'],e['end_idx'])]).astype(int) for e in presses]
            core=[np.arange(e['start_idx'],e['end_idx']) for e in presses]
            fold_errors=[];point_errors=[]
            for held in range(len(presses)):
                evaluate=core[held]
                ok=sync[evaluate,j]&np.isfinite(reference[evaluate])&(reference[evaluate]>=-.2)&(reference[evaluate]<=100)
                evaluate=evaluate[ok];record['reference_points']+=len(evaluate)
                train=np.unique(np.concatenate([v for k,v in enumerate(samples) if k!=held]))
                train=np.setdiff1d(train,core[held],assume_unique=True)
                ok=sync[train,j]&np.isfinite(reference[train])&(reference[train]>=-.2)&(reference[train]<=100)
                train=train[ok];x=paired['cmcu_ad'][train,j];y=reference[train]
                if len(x)<20 or len(np.unique(x))<8 or np.ptp(x)<100:
                    fold_reasons['remaining_presses_insufficient_AD_points_or_range']+=1;continue
                if np.ptp(y)<1:
                    fold_reasons['remaining_presses_insufficient_force_range']+=1;continue
                rho=float(spearmanr(x,y).statistic)
                if not np.isfinite(rho) or rho<.2:
                    fold_reasons['remaining_presses_weak_monotonicity']+=1;continue
                xx,yy=fit(x,y);span=float(np.ptp(yy))
                if span<1:
                    fold_reasons['remaining_presses_collapsed_fit_range']+=1;continue
                p90=float(np.quantile(abs(np.interp(x,xx,yy)-y),.9))
                if p90>10 and p90>.5*span:
                    fold_reasons['remaining_presses_gross_fit_inconsistency']+=1;continue
                ex=paired['cmcu_ad'][evaluate,j]
                inside=(ex>=xx[0])&(ex<=xx[-1]);evaluate=evaluate[inside];ex=ex[inside]
                if not len(evaluate):
                    fold_reasons['held_press_entirely_outside_remaining_AD_domain']+=1;continue
                error=abs(np.interp(ex,xx,yy)-reference[evaluate])
                point_errors.append(error);fold_errors.append(float(np.mean(error)))
                record['scored_points']+=len(error);record['scored_folds']+=1
                total_abs+=float(np.sum(error));total_scored+=len(error)
                fold_reasons['scored_in_domain']+=1
            record['coverage']=record['scored_points']/max(1,record['reference_points'])
            record['status']='scored' if point_errors else 'no_scorable_repeat_fold'
            if point_errors:
                error=np.concatenate(point_errors)
                record['press_equal_MAE_N']=float(np.mean(fold_errors))
                record['point_equal_MAE_N']=float(np.mean(error))
            records.append(record)
        if (nr+1)%200==0:print('PRESS_CV',nr+1,'curves',len(records),flush=True)
    with (output/'calibration_press_cv.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');writer.writeheader();writer.writerows(records)
    repeated=[r for r in records if r['presses']>=2];scored=[r for r in repeated if r['point_equal_MAE_N'] is not None]
    subjects=collections.defaultdict(list)
    for r in scored:subjects[r['subject']].append(r['point_equal_MAE_N'])
    quantiles={}
    for field,rows in [('coverage',repeated),('point_equal_MAE_N',scored)]:
        quantiles[field]={str(q):float(np.quantile([r[field] for r in rows],q)) for q in [0,.1,.5,.9,.99,1]}
    summary=dict(calibration_version=CALIBRATION_VERSION,accepted_curves=len(records),
        single_press_curves=len(records)-len(repeated),repeat_curves=len(repeated),scored_repeat_curves=len(scored),
        fold_outcomes=dict(fold_reasons),quantiles=quantiles,
        repeat_reference_points=sum(r['reference_points'] for r in repeated),
        repeat_scored_points=total_scored,scored_subjects=len(subjects),
        reference_point_coverage=total_scored/max(1,sum(r['reference_points'] for r in repeated)),
        subject_equal_curve_MAE_N=float(np.mean([np.mean(v) for v in subjects.values()])),
        pooled_point_MAE_N=total_abs/max(1,total_scored),
        scope='Leave one complete press out within the same test5 and finger. Fit other presses with the frozen calibration QC; score only held core points inside their AD domain. This is conditional on whole-source channel acceptance and fixed finger routing. Whole-record fixed tare is shared. It measures repeated-press consistency, not subject-disjoint, cross-session or neural-model Test accuracy.',
        interpretation='Report coverage with error: high-force held presses may exceed the AD range of remaining presses. One-press channels have no independent repetition check. Fit residual and repetition error can both miss shared reference scale or finger-routing errors.',
        neural_Test_metrics_read=False,training_or_label_artifacts_changed=False)
    atomic(output/'calibration_press_cv.json',summary)
    diagnostics=load(output/'accepted_calibration_diagnostics.json')
    assert diagnostics['calibration_version']==CALIBRATION_VERSION
    diagnostics['leave_one_press_out']=summary
    atomic(output/'accepted_calibration_diagnostics.json',diagnostics)
    os.environ['MPLCONFIGDIR']=str(root/'tmp/mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    axes[0].hist([r['coverage'] for r in repeated],bins=np.linspace(0,1,31),color='#236c98')
    axes[0].set(xlabel='Held core point coverage',ylabel='Finger curves',title='Remaining-press AD domain and QC')
    axes[1].scatter([r['source_fit_MAE_N'] for r in scored],[r['point_equal_MAE_N'] for r in scored],s=3,alpha=.15,color='#236c98')
    axes[1].plot([.01,100],[.01,100],ls='--',lw=1,color='#b74338')
    axes[1].set(xscale='log',yscale='log',xlabel='Whole-source fit MAE (N)',ylabel='Leave-one-press-out MAE (N)',title='Within-file repetition diagnostic')
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(output/'calibration_press_cv.png',dpi=180);plt.close(fig)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)
