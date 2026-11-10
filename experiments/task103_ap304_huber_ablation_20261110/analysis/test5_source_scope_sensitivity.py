"""Compare self/other curve coverage using calibration inputs, no neural outputs."""
import argparse
import collections
import csv
from pathlib import Path
import sys
import numpy as np


def main(root):
    root=Path(root)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from common import atomic,load,group
    from prepare import pair,CALIBRATION_VERSION,wall_center
    plan=load(root/'plan.json');assert plan['calibration_version']==CALIBRATION_VERSION
    domains=collections.defaultdict(list)
    for record in plan['rows']:
        if record['stage']=='mix-test-5':domains[group(record)].append(record)
    rows=[];unavailable=[]
    def predict(ad,models):
        out=np.full(ad.shape,np.nan)
        for finger,m in models.items():
            j=int(finger);x=np.asarray(m['x']);y=np.asarray(m['y'])
            valid=np.isfinite(ad[:,j])&(ad[:,j]>=x[0])&(ad[:,j]<=x[-1])
            out[valid,j]=np.interp(ad[valid,j],x,y)
        return out
    for r in plan['rows']:
        if r['stage']!='mix-test-5':continue
        try:
            with np.load(root/'cache/native'/(r['id']+'.npz')) as a:paired=pair(a)
            with np.load(root/'cache/routes'/(r['id']+'.npz')) as a:route={k:a[k].copy() for k in ['t','identity','score','sync','reference']}
            assert np.array_equal(route['t'],paired['sync_ts_ns'])
            self_model=load(root/'cache/calibration'/(r['id']+'.json'))['channels']
            candidates=[x for x in domains[group(r)] if x['id']!=r['id']]
            nearest=min(candidates,key=lambda x:(abs(wall_center(x)-wall_center(r)),x['id'])) if candidates else None
            clock_ok=nearest is not None and abs(((r['start_ns']+r['end_ns'])-(nearest['start_ns']+nearest['end_ns']))/2e9-(wall_center(r)-wall_center(nearest)))<=5
            other_path=root/'cache/calibration'/((nearest['id'] if nearest else 'None')+'.json')
            other_model=load(other_path)['channels'] if clock_ok and other_path.exists() else {}
            own=predict(paired['cmcu_ad'],self_model);other=predict(paired['cmcu_ad'],other_model)
            raw=paired['cmcu_force_N'];target=np.repeat(route['reference'][:,None],10,axis=1)
            base=(route['identity'][:,None]==np.arange(10)[None,:])&route['score'][:,None]&route['sync']
            base&=np.isfinite(raw)&(raw>=0)&(raw<=100)&np.isfinite(target)
            own_valid=base&np.isfinite(own);other_valid=base&np.isfinite(other);common=own_valid&other_valid
            row=dict(id=r['id'],subject=r['subject'],split=r['split'],base_points=int(base.sum()),
                self_curve_points=int(own_valid.sum()),other_curve_points=int(other_valid.sum()),common_points=int(common.sum()),
                self_curve_MAE_on_common_N=None,other_curve_MAE_on_common_N=None,curve_difference_MAE_on_common_N=None)
            if common.any():
                row.update(self_curve_MAE_on_common_N=float(np.mean(abs(own[common]-target[common]))),
                    other_curve_MAE_on_common_N=float(np.mean(abs(other[common]-target[common]))),
                    curve_difference_MAE_on_common_N=float(np.mean(abs(own[common]-other[common]))))
            rows.append(row)
        except (OSError,KeyError,ValueError) as exc:unavailable.append(dict(id=r['id'],reason=str(exc)))
    output=root/'analysis_output';output.mkdir(exist_ok=True)
    with (output/'test5_source_scope_sensitivity.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    totals={k:sum(r[k] for r in rows) for k in ['base_points','self_curve_points','other_curve_points','common_points']}
    metrics={}
    for key in ['self_curve_MAE_on_common_N','other_curve_MAE_on_common_N','curve_difference_MAE_on_common_N']:
        people=collections.defaultdict(list)
        for r in rows:
            if r[key] is not None:people[r['subject']].append(r[key])
        metrics['subject_equal_'+key]=float(np.mean([np.mean(v) for v in people.values()]))
    summary=dict(calibration_version=CALIBRATION_VERSION,checked_test5_records=len(rows),unavailable=unavailable,
        totals=totals,records_with_self_curve=sum(r['self_curve_points']>0 for r in rows),
        records_with_other_curve=sum(r['other_curve_points']>0 for r in rows),metrics=metrics,
        scope='Calibration-input sensitivity only, at native BSQ paired times with fixed routing and raw-force/QC guards. No EMG/token screening or neural outputs. Self-fit agreement is not independent accuracy. Explicitly contrast self with the nearest other test5, regardless of primary plan policy. This analysis does not modify training, labels or selection.',
        training_artifacts_modified=False,neural_Test_metrics_read=False)
    atomic(output/'test5_source_scope_sensitivity.json',summary)
    print(totals,metrics,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)
