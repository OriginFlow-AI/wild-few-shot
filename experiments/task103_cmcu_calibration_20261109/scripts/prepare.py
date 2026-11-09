import os
from pathlib import Path
import json,sys,collections
import numpy as np,pandas as pd
from routing import route_press_events,routed_labels,fixed_bsq_zero,TARGETS,runs
from common import ROOT, MODEL_DIR, EXPERIMENT
SRC=ROOT
NORMAL='ecc7e67d1e6aa84e533e'
def nearest(s,t):
 i=np.searchsorted(s,t);lo=np.clip(i-1,0,len(s)-1);hi=np.clip(i,0,len(s)-1);k=np.where(abs(s[lo]-t)<=abs(s[hi]-t),lo,hi);return k,abs(s[k]-t)
def stage7(a,z,inverted=True):
 events=route_press_events(a,{},z);ad=a['cmcu_ad'];t=(a['sync_ts_ns']-a['sync_ts_ns'][0])/1e9;ref=a['bsq_force_N']-z;n=len(ref);smooth=np.nanmedian(np.stack([np.r_[ref[0],ref[:-1]],ref,np.r_[ref[1:],ref[-1]]]),axis=0);sync=np.repeat(a['nearest_delta_ns']<=20e6,5,axis=1)&np.isfinite(ad)
 for e in events:
  if e['reason'] not in ['', 'weak_or_ambiguous_rawAD_identity']:continue
  quiet=np.asarray(json.loads(e['quiet_indices']),int);baseline=np.median(ad[quiet],axis=0);first3=e['first3_idx'];lo=e['start_idx'];hi=e['end_idx'];peak=e['peak_BSQ_N'];later=(np.arange(n)>=max(lo,first3))&(np.arange(n)<hi)&(t<=t[max(lo,first3)]+1.5)&(smooth>=3)&sync.all(axis=1);high=np.flatnonzero(later&(smooth<=min(10.,peak)))
  if len(high)<5:high=np.flatnonzero(later&(smooth<=min(25.,peak)))
  if len(high)<5:continue
  response=np.median(ad[high],axis=0)-baseline;hand=int(np.argmax([np.max(response[:5]),np.max(response[5:])]))
  candidates=np.array([5*hand+x for x in [1,2,3,4]]) if inverted else np.array([j for j in range(10) if j%5])
  order=candidates[np.argsort(response[candidates])];j=int(order[-1]);top=float(response[j]);second=max(0.,float(response[order[-2]]));reason='weak_or_ambiguous_test_finger' if top<100 or top<3*max(1.,second) else ''
  e.update(target_index=j,target=TARGETS[j],eligible=not reason,reason=reason,AD_identity_top_delta=top,AD_identity_second_delta=second)
 return events
if __name__=='__main__':
 rows=json.loads((ROOT/'sources.json').read_text());(ROOT/'prepared').mkdir(exist_ok=True);quality=[];events_all=[]
 anchors={a['id']:a for a in json.loads((EXPERIMENT/'manifests/side_rest_tare.json').read_text())}
 for nr,r in enumerate(rows):
  a0=np.load(SRC/'native'/(r['id']+'.npz'));t=a0['bsq_t'];ad=[];force=[];delta=[]
  for s in ['left','right']:
   ix,dd=nearest(a0[s+'_t'],t);ad.append(a0[s+'_ad'][ix]);force.append(a0[s+'_force'][ix]);delta.append(dd)
  a=dict(sync_ts_ns=t,bsq_force_N=a0['bsq_force'],cmcu_ad=np.column_stack(ad),cmcu_force_N=np.column_stack(force),nearest_delta_ns=np.column_stack(delta));b=a['bsq_force_N'];z=fixed_bsq_zero(b);mode='low_decile_mode';anchor_lo=-1;anchor_hi=-1
  z3=np.nan;zroute=z
  if r['stage']=='mix-test-7':
   tt=(t-t[0])/1e9;first4=(tt<4)&np.isfinite(b);first3=(tt<3)&np.isfinite(b)
   assert first4.sum()>=25 and first3.sum()>=25
   z=float(np.median(b[first4]));z3=float(np.median(b[first3]));mode='first4s_median_user_requested';anchor_lo=0;anchor_hi=int(first4.sum())
   zroute=float(anchors[r['id']]['zero_N'])
  ref=b-z;events=stage7(a,zroute,r['id']!=NORMAL) if r['stage']=='mix-test-7' else route_press_events(a,{},z);identity,labels,core,sync=routed_labels(a,events)
  np.savez_compressed(ROOT/'prepared'/(r['id']+'.npz'),t=t,ref=ref,ref3=b-z3,refstable=b-zroute,ad=a['cmcu_ad'],raw=a['cmcu_force_N'],identity=identity,event=labels,core=core,sync=sync)
  quality.append(dict(id=r['id'],subject=r['subject'],date=r['key'].split('/')[2],stage=r['stage'],start=r['start'],zero_N=z,stable_side_rest_zero_N=zroute,first4_minus_stable_N=z-zroute,zero_first3_N=z3,first4_minus_first3_N=z-z3,first4_P95_P05_N=float(np.quantile(b[:anchor_hi],.95)-np.quantile(b[:anchor_hi],.05)) if anchor_hi>0 else np.nan,tare_method=mode,rest_lo=anchor_lo,rest_hi=anchor_hi,clock_ok=r['clock_ok'],known_upright=r['id']==NORMAL,events=len(events),eligible_events=sum(e['eligible'] for e in events),eligible_fingers=len({e['target'] for e in events if e['eligible']})))
  events_all.extend([dict(trial=r['id'],subject=r['subject'],stage=r['stage'],**e) for e in events])
 pd.DataFrame(quality).to_csv(ROOT/'quality.csv',index=False);pd.DataFrame(events_all).to_csv(ROOT/'events.csv',index=False)
 pairs=[]
 for a in sorted([x for x in rows if x['stage']=='mix-test-7'],key=lambda x:(x['subject'],x['start'])):
  same=sorted([x for x in rows if x['subject']==a['subject'] and x['key'].split('/')[2]==a['key'].split('/')[2] and x['stage']=='mix-test-5'],key=lambda x:x['start']);pp=[x for x in same if x['end']<a['start']];nn=[x for x in same if x['start']>a['end']];p=pp[-1] if pp else None;n=nn[0] if nn else None
  ps=bool(p and p['pc']==a['pc'] and p['boot']==a['boot']);ns=bool(n and n['pc']==a['pc'] and n['boot']==a['boot']);base=dict(test7=a['id'],subject=a['subject'],date=a['key'].split('/')[2],start7=a['start'],prev5=p['id'] if p else None,next5=n['id'] if n else None,prev_same_device=ps,next_same_device=ns,known_upright=a['id']==NORMAL)
  base['prev_gap_min']=(pd.Timestamp(a['start'])-pd.Timestamp(p['end'])).total_seconds()/60 if p else np.nan;base['next_gap_min']=(pd.Timestamp(n['start'])-pd.Timestamp(a['end'])).total_seconds()/60 if n else np.nan
  base['paired']=ps and ns and a['id']!=NORMAL;pairs.append(base)
 pd.DataFrame(pairs).to_csv(ROOT/'pairs.csv',index=False);(ROOT/'prepared_ready.json').write_text(json.dumps(dict(trials=len(rows),routing='side_rest_baseline',evaluation='first4s_and_side_rest')));print('PREPARED',len(rows),'PAIRS',sum(x['paired'] for x in pairs),'reasons',dict(collections.Counter((x['prev_same_device'],x['next_same_device'],x['known_upright']) for x in pairs)),flush=True)
