"""Show reference failures and a fit using calibration-role points only."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

def main(root):
    root=Path(root);out=root/'analysis_output';out.mkdir(exist_ok=True)
    cache=root/'tmp/mpl';cache.mkdir(parents=True,exist_ok=True);os.environ['MPLCONFIGDIR']=str(cache)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from prepare import pair
    fig,axes=plt.subplots(3,1,figsize=(10,10),layout='constrained')
    examples=[]
    bad='a3f698914701e1d10da4'
    with np.load(root/'cache/native'/(bad+'.npz')) as a:
        b=a['bsq_force'];t=(a['bsq_t']-a['bsq_t'][0])/1e9
        axes[0].plot(t,b,color='#b74338',lw=1)
        axes[0].set(title=f'A. Abnormal BSQ reference (median {np.median(b):.1f} N)',xlabel='Elapsed seconds',ylabel='BSQ force (N)')
        examples.append(dict(id=bad,reason='BSQ median above100N',min_N=float(b.min()),median_N=float(np.median(b)),max_N=float(b.max())))
    flat='eb25b5f3b25e32dfd01b'
    with np.load(root/'cache/native'/(flat+'.npz')) as a:
        axes[1].plot((a['bsq_t']-a['bsq_t'][0])/1e9,a['bsq_force'],color='#b74338',label='BSQ')
        axes[1].set(title='B. Constant BSQ despite changing CMCU AD',xlabel='Elapsed seconds',ylabel='BSQ force (N)',ylim=(0,1))
        right=axes[1].twinx()
        for side,color in [('left','#236c98'),('right','#2a8b71')]:
            right.plot((a[side+'_t']-a['bsq_t'][0])/1e9,np.max(a[side+'_ad'],axis=1),alpha=.65,lw=.8,color=color,label=side+' max AD')
        right.set_ylabel('AD count');right.legend(loc='upper right')
        examples.append(dict(id=flat,reason='No BSQ variation; cannot obtain calibration truth',BSQ_value_N=float(a['bsq_force'][0])))
    good='0013360f9c16da7ce471';j=0
    model=json.loads((root/'cache/calibration'/(good+'.json')).read_text())['channels'][str(j)]
    events=json.loads((root/'cache/events'/(good+'.json')).read_text())
    selected=[e for e in events if e['event'] in model.get('presses',[model['press']])]
    ix=np.unique(np.concatenate([np.r_[json.loads(e['quiet_indices']),np.arange(e['start_idx'],e['end_idx'])] for e in selected])).astype(int)
    with np.load(root/'cache/native'/(good+'.npz')) as a:paired=pair(a)
    with np.load(root/'cache/routes'/(good+'.npz')) as a:reference=a['reference']
    time=(paired['sync_ts_ns'][ix]-paired['sync_ts_ns'][ix[0]])/1e9
    scatter=axes[2].scatter(paired['cmcu_ad'][ix,j]/1000,reference[ix],c=time,s=9,cmap='viridis',alpha=.75)
    axes[2].plot(np.array(model['x'])/1000,model['y'],color='#1c2433',lw=1.5,label='PAVA fit')
    axes[2].set(title='C. Accepted whole test5: eligible calibration inputs',xlabel='AD count / 1000',ylabel='Zero-corrected BSQ (N)')
    axes[2].legend();fig.colorbar(scatter,ax=axes[2],label='Seconds within calibration segment')
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(out/'calibration_quality_examples.png',dpi=180);plt.close(fig)
    examples.append(dict(id=good,finger='left_thumb',presses=model.get('presses',[model['press']]),scope='complete source test5 calibration inputs; no model-Test accuracy feedback'))
    (out/'calibration_quality_examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)
