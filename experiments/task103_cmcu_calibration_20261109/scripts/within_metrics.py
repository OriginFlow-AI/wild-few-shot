"""Metrics for one recording × one physical force channel.

Inputs must be in native units: raw AD counts and zero-corrected BSQ N.  The
reference trace must include releases; ``valid`` is the channel-specific pairing
mask, not a prefilter of the time series.  No automatic zeroing/AD normalization.

Public API: analyze_mapping(ad, bsq_N, *, t_s=None, order=None, valid=None,
event_labels=None, params=None) -> JSON-clean dict.  No I/O and no dependencies
outside NumPy.  Caller supplies source identity and group hierarchy.
"""
from __future__ import annotations
import numpy as np

DEFAULTS=dict(force_min_N=2.,force_max_N=25.,event_threshold_N=.3,
              event_peak_min_N=3.,bridge_seconds=0.,bridge_rows=0,
              min_event_points=8,min_unique_AD=2,grid_AD_step=64.,
              min_common_grid=5,blocks=4,return_points=False)
BANDS=[('0-0.5',0.,.5),('0.5-1',.5,1.),('1-2',1.,2.),
       ('2-5',2.,5.),('5-10',5.,10.),('10-15',10.,15.),
       ('15-20',15.,20.),('20-25',20.,25.0000001)]

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):x=x.item()
    return None if isinstance(x,float) and not np.isfinite(x) else x

def runs(mask):
    mask=np.asarray(mask,bool)
    if not len(mask):return []
    return list(zip(np.flatnonzero(mask&~np.r_[False,mask[:-1]]),
                    np.flatnonzero(mask&~np.r_[mask[1:],False])+1))

def fit(x,y):
    order=np.argsort(x,kind='stable');x=x[order];y=y[order]
    xx,start,counts=np.unique(x,return_index=True,return_counts=True)
    med=np.array([np.median(y[a:a+n]) for a,n in zip(start,counts)])
    blocks=[]
    for i,(v,w) in enumerate(zip(med,counts)):
        blocks.append([i,i+1,float(v*w),float(w)])
        while len(blocks)>1 and blocks[-2][2]/blocks[-2][3]>blocks[-1][2]/blocks[-1][3]:
            b=blocks.pop();a=blocks.pop();blocks.append([a[0],b[1],a[2]+b[2],a[3]+b[3]])
    yy=np.empty(len(xx))
    for lo,hi,s,w in blocks:yy[lo:hi]=max(0.,s/w)
    return xx,yy

def predict(x,model):
    xx,yy=model;z=np.full(len(x),np.nan);ok=np.isfinite(x)&(x>=xx[0])&(x<=xx[-1])
    z[ok]=np.interp(x[ok],xx,yy);return z

def model_possible(ix,ad,p):
    return len(ix)>=p['min_event_points'] and len(np.unique(ad[ix]))>=p['min_unique_AD']

def wquantile(values,weights,q):
    values=np.asarray(values,float);weights=np.asarray(weights,float)
    ok=np.isfinite(values)&np.isfinite(weights)&(weights>0)
    if not ok.any():return np.nan
    values=values[ok];weights=weights[ok];ix=np.argsort(values)
    v=values[ix];w=weights[ix];return float(v[np.searchsorted(np.cumsum(w),q*w.sum(),side='left')])

def model_gap(a,b,p):
    lo=max(a[0][0],b[0][0]);hi=min(a[0][-1],b[0][-1]);step=p['grid_AD_step']
    grid=np.arange(np.ceil(lo/step)*step,hi+1,step)
    if not len(grid):return dict(common_grid_n=0,gap_N=np.nan,signed_gap_N=np.nan,AD_min=np.nan,AD_max=np.nan)
    aa=predict(grid,a);bb=predict(grid,b)
    ok=(aa>=p['force_min_N'])&(aa<=p['force_max_N'])&(bb>=p['force_min_N'])&(bb<=p['force_max_N'])
    enough=ok.sum()>=p['min_common_grid'];d=aa[ok]-bb[ok]
    return dict(common_grid_n=int(ok.sum()),gap_N=float(np.mean(np.abs(d)))if enough else np.nan,
                signed_gap_N=float(np.mean(d))if enough else np.nan,
                AD_min=float(grid[ok].min())if ok.any()else np.nan,
                AD_max=float(grid[ok].max())if ok.any()else np.nan)

def detect_events(ref,time,p,labels=None):
    """Full events include observed <=threshold samples on both sides.

    Endpoints remain incomplete even when externally labelled.  This prevents a
    truncated ramp from being presented as a repeatable full press.
    """
    n=len(ref);smooth=ref.copy()
    if n>=3:smooth=np.nanmedian(np.stack([np.r_[ref[0],ref[:-1]],ref,np.r_[ref[1:],ref[-1]]]),axis=0)
    active=np.isfinite(smooth)&(smooth>p['event_threshold_N'])
    if labels is None:
        for lo,hi in runs(~active):
            if lo==0 or hi==n:continue
            if time is None:bridge=(hi-lo)<=p['bridge_rows']
            else:bridge=time[hi]-time[lo]<p['bridge_seconds']
            if bridge and np.all(np.isfinite(smooth[lo:hi])):active[lo:hi]=True
        rr=runs(active)
    else:
        rr=[]
        for lab in [x for x in np.unique(labels) if np.isfinite(x) and x>0]:
            ix=np.flatnonzero(labels==lab)
            # Noncontiguous external event IDs must not connect separate presses.
            for run in np.split(ix,np.flatnonzero(np.diff(ix)>1)+1):
                if len(run):rr.append((int(run[0]),int(run[-1]+1)))
        rr.sort()
    out=[]
    for event,(lo,hi) in enumerate(rr,1):
        segment=smooth[lo:hi]
        if not len(segment) or not np.isfinite(segment).any():continue
        peak=float(np.nanmax(segment));ip=lo+int(np.nanargmax(segment))
        complete=lo>0 and hi<n and np.isfinite(smooth[lo-1]) and np.isfinite(smooth[hi]) and smooth[lo-1]<=p['event_threshold_N'] and smooth[hi]<=p['event_threshold_N']
        reason=''
        if not complete:reason='truncated_or_no_observed_release'
        elif peak<p['event_peak_min_N']:reason='peak_below_minimum'
        pre=smooth[lo:ip+1];post=smooth[ip:hi]
        drawdown=float(np.nanmax(np.maximum.accumulate(pre)-pre))if len(pre)else np.nan
        rebound=float(np.nanmax(post-np.minimum.accumulate(post)))if len(post)else np.nan
        out.append(dict(event=event,lo=int(lo),hi=int(hi),peak_index=ip,complete=complete,
                        peak_BSQ_N=peak,event_detection_reason=reason,
                        pre_peak_max_drawdown_N=drawdown,post_peak_max_rebound_N=rebound,
                        branch_nonmonotonic_warning=bool(drawdown>max(1.,.2*peak)or rebound>max(1.,.2*peak)),
                        start=float(time[lo])if time is not None else int(lo),
                        end=float(time[hi-1])if time is not None else int(hi-1)))
    return out

def score_folds(folds,ad,ref,valid,p,prefix):
    rows=[];pointrows=[];candidate=0;scored=0;all_e=[];all_w=[];event_maes=[];coverages=[]
    band_values={b:[]for b,_,_ in BANDS};band_counts={b:[0,0]for b,_,_ in BANDS}
    band_errors={b:[]for b,_,_ in BANDS};band_weights={b:[]for b,_,_ in BANDS}
    for fold,train,held in folds:
        train=train[valid[train]];held=held[valid[held]]
        score=held[(ref[held]>=p['force_min_N'])&(ref[held]<=p['force_max_N'])]
        candidate+=len(score);model=fit(ad[train],ref[train])if model_possible(train,ad,p)else None
        pred=predict(ad[held],model)if model is not None else np.full(len(held),np.nan)
        main=(ref[held]>=p['force_min_N'])&(ref[held]<=p['force_max_N']);ok=main&np.isfinite(pred)
        err=np.abs(pred[ok]-ref[held[ok]]);scored+=len(err)
        if len(score):coverages.append(len(err)/len(score))
        if len(err):
            event_maes.append(float(np.mean(err)));all_e.extend(err);all_w.extend(np.full(len(err),1./len(err)))
        row=dict(protocol=prefix,fold=int(fold),train_points=len(train),held_points=len(held),
                 candidate_n=len(score),scored_n=len(err),coverage_percent=100*len(err)/len(score)if len(score)else np.nan,
                 MAE_N=float(np.mean(err))if len(err)else np.nan,
                 P95_N=float(np.quantile(err,.95))if len(err)else np.nan)
        rows.append(row)
        for b,lo,hi in BANDS:
            bm=(ref[held]>=lo)&(ref[held]<hi);bo=bm&np.isfinite(pred)
            band_counts[b][0]+=int(bm.sum());band_counts[b][1]+=int(bo.sum())
            if bo.any():
                be=np.abs(pred[bo]-ref[held[bo]])
                band_values[b].append(float(np.mean(be)));band_errors[b].extend(be);band_weights[b].extend(np.full(len(be),1./len(be)))
        if p['return_points']:
            for idx,yhat in zip(held,pred):pointrows.append(dict(protocol=prefix,fold=int(fold),index=int(idx),AD=float(ad[idx]),reference_N=float(ref[idx]),prediction_N=float(yhat)))
    # A one-fold or one-press score is not a repeatability estimate.
    enough=len(event_maes)>=2 and scored>=p['min_event_points']
    result={prefix+'_MAE_N':float(np.mean(event_maes))if enough else np.nan,
            prefix+'_P95_N':wquantile(all_e,all_w,.95)if enough else np.nan,
            prefix+'_point_pooled_MAE_N':float(np.mean(all_e))if enough else np.nan,
            prefix+'_coverage_percent':100*scored/candidate if candidate else np.nan,
            prefix+'_unit_equal_coverage_percent':100*float(np.mean(coverages))if coverages else np.nan,
            prefix+'_candidate_n':candidate,prefix+'_scored_n':scored,prefix+'_scored_events':len(event_maes),
            prefix+'_status':'available'if enough else 'insufficient_independent_scored_units'}
    bands=[dict(protocol=prefix,band=b,candidate_n=band_counts[b][0],test_n=band_counts[b][0],scored_n=band_counts[b][1],
                scored_units=len(band_values[b]),MAE_N=float(np.mean(band_values[b]))if len(band_values[b])>=2 else np.nan,
                P95_N=wquantile(band_errors[b],band_weights[b],.95)if len(band_values[b])>=2 else np.nan,
                coverage_percent=100*band_counts[b][1]/band_counts[b][0]if band_counts[b][0]else np.nan)
           for b,_,_ in BANDS]
    return result,rows,bands,pointrows

def analyze_mapping(ad,bsq_N,*,t_s=None,order=None,valid=None,event_labels=None,params=None):
    p=DEFAULTS.copy();p.update(params or {})
    ad=np.asarray(ad,float).ravel();ref=np.asarray(bsq_N,float).ravel();n=len(ad)
    if len(ref)!=n:raise ValueError('ad and bsq_N lengths differ')
    v=np.ones(n,bool)if valid is None else np.asarray(valid,bool).ravel()
    if len(v)!=n:raise ValueError('valid length differs')
    time=None if t_s is None else np.asarray(t_s,float).ravel()
    labels=None if event_labels is None else np.asarray(event_labels).ravel()
    if time is not None and len(time)!=n:raise ValueError('t_s length differs')
    if labels is not None and len(labels)!=n:raise ValueError('event_labels length differs')
    if order is not None:
        seq=np.argsort(np.asarray(order),kind='stable');ad=ad[seq];ref=ref[seq];v=v[seq]
        if time is not None:time=time[seq]
        if labels is not None:labels=labels[seq]
    time_status='actual_seconds'if time is not None else 'row_order_only'
    if time is not None and (not np.isfinite(time).all()or np.any(np.diff(time)<=0)):
        time=None;time_status='invalid_seconds_fell_back_to_row_order'
    v=v&np.isfinite(ad)&np.isfinite(ref)
    summary=dict(input_n=n,valid_n=int(v.sum()),time_protocol=time_status,
                 force_min_N=p['force_min_N'],force_max_N=p['force_max_N'])
    if not n:
        summary['event_status']='empty_input'
        return clean(dict(summary=summary,events=[],folds=[],branch_events=[],repeat_pairs=[],forceband_rows=[],points=[]))
    events=detect_events(ref,time,p,labels);good=[];branch=[];repeat=[]
    for ev in events:
        ix=np.arange(ev['lo'],ev['hi']);ix=ix[v[ix]]
        eligible=not ev['event_detection_reason'] and model_possible(ix,ad,p)
        ev['valid_points']=len(ix);ev['unique_AD']=len(np.unique(ad[ix]));ev['fit_eligible']=eligible
        ev['exclusion']=ev['event_detection_reason']or(''if eligible else 'insufficient_pairs_or_AD_variation')
        if eligible:good.append((ev,ix))
    all_ix=np.flatnonzero(v);eval_ix=all_ix[(ref[all_ix]>=p['force_min_N'])&(ref[all_ix]<=p['force_max_N'])]
    if model_possible(all_ix,ad,p):
        model=fit(ad[all_ix],ref[all_ix]);e=np.abs(predict(ad[eval_ix],model)-ref[eval_ix])
        summary.update(train_MAE_N=float(np.mean(e))if len(e)else np.nan,train_P95_N=float(np.quantile(e,.95))if len(e)else np.nan,train_eval_n=len(e),constant_fit=bool(np.ptp(model[1])<1e-8))
    else:summary.update(train_MAE_N=np.nan,train_P95_N=np.nan,train_eval_n=0,constant_fit=None)
    folds=[]
    if len(good)>=2:
        for ev,held in good:
            train=np.concatenate([ix for e,ix in good if e['event']!=ev['event']])
            folds.append((ev['event'],train,held))
    cv,cvrows,cvbands,cvpoints=score_folds(folds,ad,ref,v,p,'cycle_loo');summary.update(cv)
    complete_ix=np.concatenate([np.arange(e['lo'],e['hi'])for e in events if e['complete'] and e['peak_BSQ_N']>=p['event_peak_min_N']])if any(e['complete'] and e['peak_BSQ_N']>=p['event_peak_min_N']for e in events)else np.array([],int)
    all_complete_candidates=int(np.sum(v[complete_ix]&(ref[complete_ix]>=p['force_min_N'])&(ref[complete_ix]<=p['force_max_N'])))
    summary.update(detected_events=len(events),complete_events=sum(e['complete']and e['peak_BSQ_N']>=p['event_peak_min_N']for e in events),eligible_events=len(good),
                   all_valid_2_25_n=len(eval_ix),complete_event_2_25_n=all_complete_candidates,
                   cycle_loo_all_complete_coverage_percent=100*cv['cycle_loo_scored_n']/all_complete_candidates if all_complete_candidates else np.nan,
                   event_status='available'if len(good)>=2 else 'fewer_than_two_eligible_complete_presses')
    # Block split on full recording order, not on a shuffled/filtered point array.
    blockfolds=[]
    if n>=p['blocks']:
        segments=np.array_split(np.arange(n),p['blocks'])
        for i,held in enumerate(segments):
            train=np.concatenate([ix for j,ix in enumerate(segments)if j!=i]);blockfolds.append((i+1,train,held))
    bc,br,bb,bp=score_folds(blockfolds,ad,ref,v,p,'block_cv');summary.update(bc)
    models={}
    for ev,ix in good:
        maps={}
        for phase,mask in [('loading',ix<=ev['peak_index']),('unloading',ix>ev['peak_index'])]:
            ii=ix[mask]
            if model_possible(ii,ad,p):maps[phase]=fit(ad[ii],ref[ii]);models[ev['event'],phase]=maps[phase]
        if len(maps)==2:branch.append(dict(event=ev['event'],pre_peak_max_drawdown_N=ev['pre_peak_max_drawdown_N'],post_peak_max_rebound_N=ev['post_peak_max_rebound_N'],branch_nonmonotonic_warning=ev['branch_nonmonotonic_warning'],**model_gap(maps['loading'],maps['unloading'],p)))
    ids=[ev['event']for ev,_ in good]
    for i,a in enumerate(ids):
        for b in ids[i+1:]:
            for phase in ['loading','unloading']:
                if (a,phase)in models and(b,phase)in models:repeat.append(dict(event_a=a,event_b=b,phase=phase,**model_gap(models[a,phase],models[b,phase],p)))
    values=[r['gap_N']for r in branch if np.isfinite(r['gap_N'])]
    summary.update(hysteresis_gap_N=float(np.mean(values))if values else np.nan,hysteresis_events=len(values),
                   hysteresis_nonmonotonic_events=sum(r['branch_nonmonotonic_warning']for r in branch if np.isfinite(r['gap_N'])))
    for phase in ['loading','unloading']:
        values=[r['gap_N']for r in repeat if r['phase']==phase and np.isfinite(r['gap_N'])]
        summary['repeat_'+phase+'_gap_N']=float(np.mean(values))if values else np.nan;summary['repeat_'+phase+'_pairs']=len(values)
    return clean(dict(summary=summary,events=events,folds=cvrows+br,branch_events=branch,repeat_pairs=repeat,forceband_rows=cvbands+bb,points=cvpoints+bp,params=p))
