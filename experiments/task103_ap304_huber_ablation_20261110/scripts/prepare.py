"""Freeze recording splits and independent calibration/scoring press roles."""
import collections
import datetime
import functools
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from common import EXPERIMENT, SIDES, atomic, group, interpolate, load, nearest, sha, windows, source_binding
from routing import fixed_bsq_zero, route_press_events, routed_labels, runs

def freeze_plan(root):
    root = Path(root); path = root/'plan.json'
    if path.exists():
        plan = load(path)
        if plan['source_sha256'] != sha(EXPERIMENT/'manifests/sources.json'): raise ValueError('Changed source manifest')
        return plan
    rows = load(EXPERIMENT/'manifests/sources.json')
    assert len({r['recording_id'] for r in rows}) == len(rows)
    groups = collections.defaultdict(list)
    for r in rows:
        if r['stage'] == 'mix-test-5': groups[group(r)].append(r)
    val, odd = set(), []
    def rank(r): return hashlib.sha256(('43|'+r['id']).encode()).hexdigest()
    for key, rr in groups.items():
        rr.sort(key=rank); val.update(r['id'] for r in rr[:len(rr)//2])
        if len(rr)%2: odd.append(rr[len(rr)//2])
        rr.sort(key=lambda r:r['end_ns'])
    target = sum(len(rr) for rr in groups.values())//2
    val.update(r['id'] for r in sorted(odd, key=rank)[:target-len(val)])
    for r in rows:
        rr = [q for q in groups[group(r)] if q['end_ns'] < r['start_ns']]
        source = rr[-1] if rr else None
        status = 'no_prior_same_domain'
        if source:
            age = (r['start_ns']-source['end_ns'])/1e9
            wall = (datetime.datetime.fromisoformat(r['start'])-datetime.datetime.fromisoformat(source['end'])).total_seconds()
            status = 'candidate' if abs(age-wall) <= 5 else 'clock_discontinuity'
        r['source_id'] = source['id'] if source else None; r['route_status'] = status
        r['split'] = ('val' if r['id'] in val else 'test') if r['stage']=='mix-test-5' else 'test' if r['stage']=='mix-test-7' else 'train'
        r['active'] = r['stage']=='mix-test-5' or status=='candidate'
    plan = dict(source_sha256=sha(EXPERIMENT/'manifests/sources.json'), rows=rows,
                split_seed=43, test5_val=len(val), test5_test=sum(len(rr) for rr in groups.values())-len(val),
                calibration_role='first complete eligible press per finger within every test5; its points are never scored',
                nearest_source_policy='completed preceding test5, same subject/date/PC/boot; no fallback',
                test5_scoring='remaining eligible presses; use previous completed test5 calibration, never own-record mapping')
    atomic(path, plan); return plan

def pair(a):
    t = a['bsq_t']; ad, raw, delta = [], [], []
    for side in SIDES:
        k, d = nearest(a[side+'_t'], t)
        ad.append(a[side+'_ad'][k]); raw.append(a[side+'_force'][k]); delta.append(d)
    return dict(sync_ts_ns=t, bsq_force_N=a['bsq_force'], cmcu_ad=np.column_stack(ad),
                cmcu_force_N=np.column_stack(raw), nearest_delta_ns=np.column_stack(delta))

def stable_zero(a, row):
    b, t = a['bsq_force_N'], a['sync_ts_ns']
    if np.nanmedian(b)>100 or not np.isfinite(b).all(): raise ValueError('BSQ median above100N or nonfinite')
    if row['stage'] != 'mix-test-7': return fixed_bsq_zero(b), 'low_decile_fixed_zero'
    anchor = load(EXPERIMENT/'manifests/test7_tare_anchors.json').get(row['id'])
    if anchor:
        if anchor.get('zero_valid') is False: raise ValueError('Existing test7 zero marked invalid')
        return float(anchor['zero_N']), 'existing_reviewed_stable_zero'
    q = np.quantile(b, .1); candidates = []
    for lo, hi in runs(abs(b-q)<=.15):
        if hi-lo>=20 and t[hi-1]-t[lo]>=400_000_000:
            candidates.append((hi-lo, lo, hi))
    if not candidates: raise ValueError('No stable post-positioning BSQ tare segment')
    _, lo, hi = max(candidates)
    return float(np.median(b[lo:hi])), 'automatic_longest_low_level_stable_segment'

def stage7_events(a, zero):
    events = route_press_events(a, {}, zero)
    ad = a['cmcu_ad']; t = (a['sync_ts_ns']-a['sync_ts_ns'][0])/1e9
    b = a['bsq_force_N']-zero
    for e in events:
        if e['reason'] not in ['', 'weak_or_ambiguous_rawAD_identity']: continue
        quiet = np.asarray(json.loads(e['quiet_indices']), int)
        high = np.flatnonzero((np.arange(len(t))>=max(e['start_idx'],e['first3_idx'])) &
            (np.arange(len(t))<e['end_idx']) & (t<=t[max(e['start_idx'],e['first3_idx'])]+1.5) & (b>=3) & (b<=25))
        if len(high)<5: continue
        response = np.median(ad[high], axis=0)-np.median(ad[quiet], axis=0)
        hand = int(np.argmax([np.max(response[:5]), np.max(response[5:])]))
        candidates = np.array([hand*5+j for j in range(1,5)])
        order = candidates[np.argsort(response[candidates])]; j = int(order[-1])
        top, second = float(response[j]), max(0., float(response[order[-2]]))
        reason = '' if top>=100 and top>=3*max(1.,second) else 'weak_or_ambiguous_test_finger'
        e.update(target_index=j, target=('left_' if j<5 else 'right_')+['thumb','index','middle','ring','little'][j%5],
                 eligible=not reason, reason=reason)
    return events

def fit(x, y):
    order = np.lexsort((y, x)); x, y = x[order], y[order]
    xx, start, counts = np.unique(x, return_index=True, return_counts=True)
    med = (y[start+(counts-1)//2]+y[start+counts//2])*.5; blocks = []
    for i,(v,n) in enumerate(zip(med, counts)):
        blocks.append([i,i+1,float(v*n),float(n)])
        while len(blocks)>1 and blocks[-2][2]/blocks[-2][3] > blocks[-1][2]/blocks[-1][3]:
            b=blocks.pop();a=blocks.pop();blocks.append([a[0],b[1],a[2]+b[2],a[3]+b[3]])
    yy=np.empty(len(xx))
    for lo,hi,s,n in blocks: yy[lo:hi]=max(0.,s/n)
    return xx,yy

def prepare_routes(root, rows):
    root=Path(root); summary=[]
    for nr,r in enumerate(rows):
        dest=root/'cache/routes'/(r['id']+'.npz'); model_file=root/'cache/calibration'/(r['id']+'.json')
        receipt=root/'cache/routes'/(r['id']+'.json')
        if receipt.exists(): summary.append(load(receipt)); continue
        d=dict(id=r['id'],stage=r['stage'],subject=r['subject'],status='unavailable',channels={},events=0)
        try:
            meta=load(root/'cache/emg'/r['id']/'done.json')
            if not meta['clock_ok']: raise ValueError('Force receive/sync offset disagreement')
            with np.load(root/'cache/native'/(r['id']+'.npz')) as a: paired=pair(a)
            zero,method=stable_zero(paired,r);ref=paired['bsq_force_N']-zero
            events=stage7_events(paired,zero) if r['stage']=='mix-test-7' else route_press_events(paired,{},zero)
            identity,event,core,sync=routed_labels(paired,events)
            reserve=np.zeros(len(ref),bool);chosen={};models={};problems={}
            if r['stage']=='mix-test-5':
                for e in events:
                    j=e['target_index']
                    if not e['eligible'] or j in chosen: continue
                    chosen[j]=e['event']
                    ix=np.unique(np.r_[json.loads(e['quiet_indices']),np.arange(e['start_idx'],e['end_idx'])]).astype(int)
                    reserve[ix]=True
                    good=sync[ix,j]&np.isfinite(ref[ix])&(ref[ix]>=-.2)&(ref[ix]<=100)
                    ix=ix[good];x=paired['cmcu_ad'][ix,j];y=ref[ix]
                    if len(ix)<20 or len(np.unique(x))<8 or np.ptp(x)<100:
                        problems[str(j)]='insufficient_calibration_points_or_AD_range';continue
                    rho=float(spearmanr(x,y).statistic)
                    if not np.isfinite(rho) or rho<.2:
                        problems[str(j)]='nonpositive_or_weak_calibration_monotonicity';continue
                    xx,yy=fit(x,y)
                    if np.ptp(yy)<1:
                        problems[str(j)]='collapsed_calibration_force_range';continue
                    models[str(j)]=dict(x=xx.tolist(),y=yy.tolist(),press=e['event'],points=len(ix),
                        spearman=rho,AD_min=float(xx[0]),AD_max=float(xx[-1]),reference_max_N=float(y.max()))
            score=np.zeros(len(ref),bool)
            for e in events:
                if e['eligible'] and chosen.get(e['target_index'])!=e['event']:
                    score[e['context_start_idx']:e['context_end_idx']]=True
            score &= ~reserve
            dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest,t=paired['sync_ts_ns'],reference=ref,identity=identity,score=score,
                                calibration=reserve,sync=sync,event=event)
            atomic(model_file,dict(id=r['id'],channels=models,problems=problems,zero_N=zero,zero_method=method))
            d.update(status='prepared',channels={j:{k:v for k,v in m.items() if k not in ['x','y']} for j,m in models.items()},
                     events=len(events),scoring_points=int(score.sum()),reserved_points=int(reserve.sum()),
                     rejected_event_reasons=dict(collections.Counter(e['reason'] for e in events if not e['eligible'])),
                     problems=problems,zero_N=zero,zero_method=method)
            d['orientation']=load(EXPERIMENT/'manifests/test7_tare_anchors.json').get(r['id'],{}).get('orientation','not_reviewed') if r['stage']=='mix-test-7' else 'standard_test5'
            atomic(root/'cache/events'/(r['id']+'.json'),events)
        except (ValueError,OSError,KeyError) as exc: d['reason']=str(exc)
        atomic(receipt,d);summary.append(d)
        if (nr+1)%50==0: print('CALIBRATION_QC',nr+1,'/',len(rows),flush=True)
    atomic(root/'checks/calibration_quality.json',dict(records=summary,
        statuses=dict(collections.Counter(r['status'] for r in summary))))
    return {r['id']:r for r in summary}

@functools.lru_cache(maxsize=32)
def get_model(path): return load(path)['channels']

def labels_one(root, r):
    root=Path(root); out=root/'cache/labels'/(r['id']+'.npz')
    if out.exists(): return
    source=root/'cache/calibration'/(str(r['source_id'])+'.json')
    if r['route_status']!='candidate' or not source.exists(): raise ValueError('No eligible previous calibration source')
    models=get_model(str(source))
    if not models: raise ValueError('Nearest calibration has no valid channels; no fallback')
    meta=load(root/'cache/emg'/r['id']/'done.json')
    if not meta['clock_ok']: raise ValueError('Force clock consistency failed')
    route=None
    if r['split']!='train':
        route_path=root/'cache/routes'/(r['id']+'.npz')
        if not route_path.exists(): raise ValueError('Missing reference/finger route')
        route=dict(np.load(route_path))
    output={}
    with np.load(root/'cache/native'/(r['id']+'.npz')) as a:
        for h,side in enumerate(SIDES):
            t=a[side+'_t'];ad=a[side+'_ad'];raw=a[side+'_force'];cal=np.full_like(raw,np.nan)
            for j in range(5):
                m=models.get(str(h*5+j))
                if m is None: continue
                x,y=np.asarray(m['x']),np.asarray(m['y'])
                good=np.isfinite(ad[:,j])&(ad[:,j]>=x[0])&(ad[:,j]<=x[-1])
                cal[good,j]=np.interp(ad[good,j],x,y)
            valid=np.isfinite(cal)&np.isfinite(raw)&(raw>=0)&(raw<=100)
            bsq=np.zeros_like(raw)
            if route is not None:
                k,delta=nearest(route['t'],t)
                selected=(route['identity'][k,None]==(np.arange(5)+h*5)[None,:])
                selected &= route['score'][k,None]&route['sync'][k,h*5:h*5+5]&(delta<=20_000_000)[:,None]
                valid &= selected
                bsq[:]=route['reference'][k,None]
            output.update({side+'_t':t,side+'_raw':raw.astype(np.float32),side+'_cal':np.nan_to_num(cal).astype(np.float32),
                           side+'_valid':valid,side+'_bsq':bsq.astype(np.float32)})
    out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**output)

def build_dataset(root, rows):
    root=Path(root);entries=[];exclusions=[];window_index={s:[] for s in ['train','val','test']}
    sums={k:np.zeros(5,np.float64) for k in ['raw','cal','raw_sq','cal_sq','count']}
    for nr,r in enumerate(rows):
        if not r['active']: continue
        try:
            labels_one(root,r)
            meta=load(root/'cache/emg'/r['id']/'done.json')
            label_sha=sha(root/'cache/labels'/(r['id']+'.npz'))
            with np.load(root/'cache/labels'/(r['id']+'.npz')) as z:
                for side in SIDES:
                    t=z[side+'_t'];valid=z[side+'_valid'];raw=z[side+'_raw'];cal=z[side+'_cal']
                    if not valid.any(): continue
                    m=meta['hands'][side]
                    ev=np.load(root/'cache/emg'/r['id']/(side+'_valid.npy'),mmap_mode='r')
                    prefix=np.r_[0,np.cumsum(~ev)];starts=[]
                    for start in windows(m['samples']):
                        if prefix[start+4000]!=prefix[start]: continue
                        query=m['start_ns']+(start+470+40*np.arange(89,dtype=np.int64))*500000
                        _,ok=interpolate(t,cal,valid,query)
                        if ok.any(): starts.append(start)
                    if not starts: continue
                    index=len(entries);entries.append(dict(id=r['id'],side=side,split=r['split'],subject=r['subject'],
                        stage=r['stage'],source_id=r['source_id'],start_ns=m['start_ns'],samples=m['samples']))
                    entries[-1].update(EMG=meta.get('processed_files',{}).get(side+'.npy'),labels_sha256=label_sha)
                    window_index[r['split']].extend((index,s) for s in starts)
                    if r['split']=='train':
                        sums['count']+=valid.sum(axis=0)
                        for name,a in [('raw',raw),('cal',cal)]:
                            sums[name]+=np.where(valid,a,0).sum(axis=0,dtype=np.float64)
                            sums[name+'_sq']+=np.where(valid,a.astype(np.float64)**2,0).sum(axis=0)
        except (ValueError,OSError,KeyError) as exc: exclusions.append(dict(id=r['id'],split=r['split'],reason=str(exc)))
        if (nr+1)%100==0: print('BUILD_WINDOWS',nr+1,'/',len(rows),flush=True)
    if not np.all(sums['count']>0): raise ValueError('No train labels for one or more fingers')
    norms={}
    for name in ['raw','cal']:
        mean=sums[name]/sums['count'];std=np.sqrt(np.maximum(0,sums[name+'_sq']/sums['count']-mean**2))
        if np.any(std<1e-6): raise ValueError('Degenerate force normalization')
        norms[name]=dict(mean=mean.tolist(),std=std.tolist(),count=sums['count'].astype(int).tolist())
    out=root/'prepared';out.mkdir(exist_ok=True)
    for split, pairs in window_index.items():
        if not pairs: raise ValueError('No usable windows: '+split)
        np.save(out/(split+'_windows.npy'),np.asarray(pairs,np.int32))
    atomic(out/'dataset.json',dict(entries=entries,force_norm=norms,
        windows={s:len(v) for s,v in window_index.items()},exclusions=exclusions,
        split_recordings={s:len({r['id'] for r in entries if r['split']==s}) for s in window_index},
        plan_sha256=sha(root/'plan.json'),config_sha256=sha(EXPERIMENT/'config.json'),source_code=source_binding()))
    atomic(root/'checks/dataset_prepared.json',dict(status='PREPARED',windows={s:len(v) for s,v in window_index.items()},
        excluded_records=len(exclusions),force_statistics='common finite in-domain native CMCU timestamps, once per point; train only'))
    return load(out/'dataset.json')
