"""Describe source-curve residuals; never change masks or use neural Test errors."""
import argparse
import collections
import csv
import json
import os
from pathlib import Path
import sys
import numpy as np


def main(root):
    root=Path(root)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from common import atomic, load
    from prepare import pair
    meta={r['id']:r for r in load(root/'plan.json')['rows']}
    rows=[]
    for p in sorted((root/'cache/calibration').glob('*.json')):
        model=load(p);record=meta[model['id']]
        if record['stage']!='mix-test-5' or not model['channels']:continue
        with np.load(root/'cache/native'/(model['id']+'.npz')) as a:paired=pair(a)
        with np.load(root/'cache/routes'/(model['id']+'.npz')) as a:
            ref=a['reference'].copy();sync=a['sync'].copy()
        events={e['event']:e for e in load(root/'cache/events'/(model['id']+'.json'))}
        for finger,m in model['channels'].items():
            j=int(finger);presses=[events[k] for k in m['presses']]
            ix=np.unique(np.concatenate([np.r_[json.loads(e['quiet_indices']),np.arange(e['start_idx'],e['end_idx'])] for e in presses])).astype(int)
            good=sync[ix,j]&np.isfinite(ref[ix])&(ref[ix]>=-.2)&(ref[ix]<=100)
            ix=ix[good];x=paired['cmcu_ad'][ix,j];y=ref[ix]
            prediction=np.interp(x,m['x'],m['y']);error=prediction-y
            mae=float(np.mean(abs(error)));p90=float(np.quantile(abs(error),.9))
            span=float(np.ptp(m['y']));flags=[]
            if m['spearman']<.8:flags.append('source_Spearman_below_0.8_review')
            if p90>3 and p90/span>.2:flags.append('source_fit_p90_above_3N_and_20pct_range_review')
            rows.append(dict(id=model['id'],subject=record['subject'],date=record['key'].split('/')[2],
                finger_index=j,presses=len(presses),points=len(ix),Spearman=m['spearman'],
                AD_min=m['AD_min'],AD_max=m['AD_max'],fit_range_N=span,
                reference_max_N=m['reference_max_N'],source_fit_MAE_N=mae,source_fit_p90_N=p90,
                source_fit_bias_N=float(np.mean(error)),review_flags=';'.join(flags)))
    out=root/'analysis_output';out.mkdir(exist_ok=True)
    with (out/'accepted_calibration_diagnostics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    distributions={}
    for key in ['Spearman','fit_range_N','source_fit_MAE_N','source_fit_p90_N','presses']:
        distributions[key]={str(q):float(np.quantile([r[key] for r in rows],q)) for q in [0,.1,.5,.9,.99,1]}
    flags=collections.Counter(x for r in rows for x in r['review_flags'].split(';') if x)
    atomic(out/'accepted_calibration_diagnostics.json',dict(
        calibration_version=load(root/'plan.json')['calibration_version'],accepted_curves=len(rows),
        flagged_curves=sum(bool(r['review_flags']) for r in rows),review_flags=dict(flags),
        distributions=distributions,worst_p90_examples=sorted(rows,key=lambda r:r['source_fit_p90_N'],reverse=True)[:20],
        scope='Descriptive source calibration fit residuals, not independent accuracy or neural Test errors. Review flags do not alter source selection, masks, or model training.',
        interpretation='Low residual cannot rule out shared scale error, wrong reference units or drift. Large residual may indicate hysteresis, inconsistent repeated presses, incorrect finger routing or reference faults; inspect traces before attributing a cause.'))
    os.environ['MPLCONFIGDIR']=str(root/'tmp/mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
    for axis,row in zip(axes,sorted(rows,key=lambda r:r['source_fit_p90_N'],reverse=True)[:3]):
        id=row['id'];j=row['finger_index'];m=load(root/'cache/calibration'/(id+'.json'))['channels'][str(j)]
        events={e['event']:e for e in load(root/'cache/events'/(id+'.json'))}
        presses=[events[k] for k in m['presses']]
        ix=np.unique(np.concatenate([np.r_[json.loads(e['quiet_indices']),np.arange(e['start_idx'],e['end_idx'])] for e in presses])).astype(int)
        with np.load(root/'cache/native'/(id+'.npz')) as a:paired=pair(a)
        with np.load(root/'cache/routes'/(id+'.npz')) as a:ref=a['reference'].copy()
        t=(paired['sync_ts_ns']-paired['sync_ts_ns'][0])/1e9
        axis[0].plot(t,ref,color='#b74338',label='BSQ N',lw=1)
        right=axis[0].twinx();right.plot(t,paired['cmcu_ad'][:,j],color='#236c98',label='AD',lw=.8,alpha=.6)
        axis[0].scatter(t[ix],ref[ix],s=7,color='#b74338');right.set_ylabel('AD')
        axis[0].set(title=f'{id}, finger {j}',xlabel='Seconds',ylabel='BSQ (N)')
        axis[1].scatter(paired['cmcu_ad'][ix,j],ref[ix],c=t[ix],s=7,cmap='viridis')
        axis[1].plot(m['x'],m['y'],color='black',lw=1)
        axis[1].set(title=f"Source-fit P90={row['source_fit_p90_N']:.1f} N; inspect",xlabel='AD',ylabel='BSQ (N)')
        for ax in axis:ax.grid(alpha=.2)
    fig.savefig(out/'accepted_calibration_worst_residuals.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(curves=len(rows),flagged=sum(bool(r['review_flags']) for r in rows),flags=dict(flags))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)
