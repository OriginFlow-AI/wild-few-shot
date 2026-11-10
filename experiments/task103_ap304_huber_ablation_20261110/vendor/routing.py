"""Finger routing from a later raw-AD response; never default CMCU force.

The BSQ zero is fixed for a whole file.  Identity is assigned after observing
later >=3N response, then used retrospectively for the complete press and a
fixed0.2s onset/release context.  Context samples are flagged separately.
"""
from pathlib import Path
import json
import numpy as np

TARGETS=[h+'_'+f for h in ['left','right']for f in ['thumb','index','middle','ring','little']]

def runs(mask):
    return list(zip(np.flatnonzero(mask&~np.r_[False,mask[:-1]]),np.flatnonzero(mask&~np.r_[mask[1:],False])+1))

def fixed_bsq_zero(b):
    finite=np.isfinite(b)
    if finite.sum()<25:return np.nan
    q10=np.quantile(b[finite],.1);quiet=finite&(b<=q10+.15)
    return float(np.median(b[quiet]))

def route_press_events(a,meta=None,bsq_zero_N=None,context_seconds=.2):
    meta=meta or {};b=np.asarray(a['bsq_force_N'],float);ad=np.asarray(a['cmcu_ad'],float)
    time=(a['sync_ts_ns']-a['sync_ts_ns'][0])/1e9;n=len(b)
    zero=fixed_bsq_zero(b)if bsq_zero_N is None else float(bsq_zero_N);ref=b-zero
    smooth=np.nanmedian(np.stack([np.r_[ref[0],ref[:-1]],ref,np.r_[ref[1:],ref[-1]]]),axis=0)
    active=np.isfinite(smooth)&(smooth>.3);rr=runs(active);out=[]
    sync=np.isfinite(ad)
    if'nearest_delta_ns'in a:
        dd=np.asarray(a['nearest_delta_ns']);sync&=np.repeat(dd<=20_000_000,5,axis=1)
    for ev,(lo,hi)in enumerate(rr,1):
        peak=float(np.nanmax(smooth[lo:hi]));peakidx=lo+int(np.nanargmax(smooth[lo:hi]));reason=''
        complete=lo>0 and hi<n and np.isfinite(smooth[lo-1])and np.isfinite(smooth[hi])and smooth[lo-1]<=.3 and smooth[hi]<=.3
        quiet=np.flatnonzero((time>=time[lo]-1)&(time<=time[lo]-.2)&(smooth>=-.2)&(smooth<=.2)&sync.all(axis=1))
        if not complete:reason='incomplete_press_at_boundary'
        elif peak<3:reason='peak_below3N_cannot_confirm_finger'
        elif len(quiet)<5 or time[quiet[-1]]-time[quiet[0]]<.3 or time[lo]-time[quiet[-1]]>.4:reason='no_local_quiet_for_AD_identity'
        first3idx=np.flatnonzero((np.arange(n)>=lo)&(np.arange(n)<hi)&(smooth>=3))
        first3=int(first3idx[0])if len(first3idx)else -1
        baseline=np.median(ad[quiet],axis=0)if len(quiet)else np.full(10,np.nan)
        noise=np.median(np.abs(ad[quiet]-baseline),axis=0)*1.4826 if len(quiet)else np.full(10,np.nan)
        later=(np.arange(n)>=max(lo,first3))&(np.arange(n)<hi)&(time<=time[max(lo,first3)]+1.5)&(smooth>=3)&sync.all(axis=1)
        high=np.flatnonzero(later&(smooth<=min(10.,peak)))
        band='preferred_3_to10'
        if len(high)<5:high=np.flatnonzero(later&(smooth<=min(25.,peak)));band='fallback_3_to25'
        response=np.median(ad[high],axis=0)-baseline if len(high)>=5 else np.full(10,np.nan)
        j=-1;top=np.nan;second=np.nan
        if not reason:
            if len(high)<5 or not np.isfinite(response).all():reason='insufficient_highforce_rawAD_identity'
            else:
                order=np.argsort(response);j=int(order[-1]);top=float(response[j]);second=max(0.,float(response[order[-2]]))
                if top<100 or top<3*max(1.,second):reason='weak_or_ambiguous_rawAD_identity'
        # Context cannot overlap the neighboring observed active press.  It is
        # an onset/release neighborhood, not evidence of contact force below.3N.
        cstart=max(0,int(np.searchsorted(time,time[lo]-context_seconds)))
        cend=min(n,int(np.searchsorted(time,time[hi-1]+context_seconds,side='right')))
        if ev>1:cstart=max(cstart,rr[ev-2][1])
        if ev<len(rr):cend=min(cend,rr[ev][0])
        if ev>1:
            mid=(rr[ev-2][1]+lo)//2;cstart=max(cstart,mid)
        if ev<len(rr):
            mid=(hi+rr[ev][0])//2;cend=min(cend,mid)
        e=dict(meta,event=ev,start_idx=int(lo),end_idx=int(hi),context_start_idx=cstart,context_end_idx=cend,
               first3_idx=first3,peak_idx=peakidx,start_s=float(time[lo]),end_s=float(time[hi-1]),peak_BSQ_N=peak,
               BSQ_zero_N=zero,target_index=j,target=TARGETS[j]if j>=0 else'',eligible=not reason,reason=reason,
               complete_press=complete,AD_baseline=float(baseline[j])if j>=0 else np.nan,
               AD_noise_sigma=float(noise[j])if j>=0 else np.nan,AD_quiet_robust_sigma=float(noise[j])if j>=0 else np.nan,
               AD_identity_top_delta=top,AD_identity_second_delta=second,identity_high_points=len(high),identity_band_rule=band,
               quiet_indices=json.dumps(quiet.tolist(),separators=(',',':')),
               pre_peak_max_drawdown_N=float(np.nanmax(np.maximum.accumulate(smooth[lo:peakidx+1])-smooth[lo:peakidx+1])),
               post_peak_max_rebound_N=float(np.nanmax(smooth[peakidx:hi]-np.minimum.accumulate(smooth[peakidx:hi]))))
        out.append(e)
    return out

def routed_labels(a,events):
    """Returns full-trace identity, event ID, core flag and sync-valid masks."""
    n=len(a['bsq_force_N']);identity=np.full(n,-1,int);labels=np.zeros(n,int);core=np.zeros(n,bool)
    for e in events:
        if not e['eligible']:continue
        lo=e['context_start_idx'];hi=e['context_end_idx'];identity[lo:hi]=e['target_index'];labels[lo:hi]=e['event'];core[e['start_idx']:e['end_idx']]=True
    ad=np.asarray(a['cmcu_ad'],float);sync=np.isfinite(ad)
    if'nearest_delta_ns'in a:sync&=np.repeat(np.asarray(a['nearest_delta_ns'])<=20_000_000,5,axis=1)
    return identity,labels,core,sync
