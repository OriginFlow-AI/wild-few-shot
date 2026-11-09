import os
from pathlib import Path
from functools import lru_cache
import json,time,sys
import numpy as np,pandas as pd
from scipy.stats import kendalltau
import within_metrics as w
from prepare import nearest
from common import ROOT, MODEL_DIR, EXPERIMENT;J=[j for j in range(10) if j%5];BANDS=[('0-2',0,2),('0+-2',1e-6,2),('2-10',2,10),('10-25',10,25.0000001),('2-25',2,25.0000001),('0-25',0,25.0000001)]
TARGETS=[h+'_'+f for h in ['left','right'] for f in ['thumb','index','middle','ring','little']]
@lru_cache(maxsize=12)
def prep(id):return dict(np.load(ROOT/'prepared'/(id+'.npz')))
@lru_cache(maxsize=12)
def native_scores(id,j):
 a=prep(id);p=np.load(ROOT/'predictions'/(id+'.npz'));s='left' if j<5 else 'right';tt=p[s+'_t'];ix,delta=nearest(a['t'],tt);ok=(delta<=12e6)&(a['identity'][ix]==j)&a['sync'][ix,j]&np.isfinite(a['ref'][ix])&bool(p['clock_ok'])
 return dict(t=tt[ok],ix=ix[ok],x=p[s+'_scores'][ok,j%5],ref=a['ref'][ix[ok]],ref3=a['ref3'][ix[ok]],refstable=a['refstable'][ix[ok]],raw=a['raw'][ix[ok],j],ad=a['ad'][ix[ok],j],event=a['event'][ix[ok]],core=a['core'][ix[ok]])
@lru_cache(maxsize=2000)
def sensor(id,j):
 a=prep(id);ok=(a['identity']==j)&a['sync'][:,j]&np.isfinite(a['ref'])
 return w.fit(a['ad'][ok,j],a['ref'][ok]) if ok.sum()>=8 and np.unique(a['ad'][ok,j]).size>=2 else None

def fit(x,y,method):
 if len(x)<8 or np.unique(x).size<2:return None
 if method=='isotonic':return w.fit(x,y)
 xm=x.mean();ym=y.mean();slope=max(0.,float(np.sum((x-xm)*(y-ym))/np.sum((x-xm)**2)));return np.array([x.min(),x.max()]),np.array([ym+slope*(x.min()-xm),ym+slope*(x.max()-xm)])
def predict(x,m,policy='domain'):
 if m is None:return np.full(len(x),np.nan)
 if policy=='clamp':return np.interp(x,m[0],m[1])
 return w.predict(x,m)
metrics=[];audit=[];maps=[];smetrics=[];qa=[]
def score(base,truth,pred,fold,band,lo,hi,candidate=None):
 candidate=(np.isfinite(truth)&(truth>=lo)&(truth<hi)) if candidate is None else candidate
 ok=candidate&np.isfinite(pred);event_mae=[];event_rmse=[];event_bias=[];event_mape=[]
 for e in np.unique(fold[ok]):
  q=ok&(fold==e);err=pred[q]-truth[q];event_mae.append(np.mean(abs(err)));event_rmse.append(np.mean(err**2));event_bias.append(np.mean(err));p=q&(truth>1e-6)
  if p.any():event_mape.append(np.mean(abs(pred[p]-truth[p])/truth[p])*100)
 enough=int(ok.sum())>=8;tau=float(kendalltau(truth[ok],pred[ok]).statistic) if enough and np.unique(truth[ok]).size>1 and np.unique(pred[ok]).size>1 else np.nan
 return dict(base,band=band,MAE_N=float(np.mean(event_mae)) if enough and event_mae else np.nan,RMSE_N=float(np.sqrt(np.mean(event_rmse))) if enough and event_rmse else np.nan,bias_N=float(np.mean(event_bias)) if enough and event_bias else np.nan,MAPE_pct=float(np.mean(event_mape)) if enough and event_mape else np.nan,tau_b=tau,candidate_n=int(candidate.sum()),scored_n=int(ok.sum()),events=int(np.unique(fold[ok]).size),coverage_pct=100*ok.sum()/candidate.sum() if candidate.any() else np.nan)
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1);args=parser.parse_args();OUT=ROOT/('score_shard_'+str(args.shard));OUT.mkdir(exist_ok=True)
 pairs=pd.read_csv(ROOT/'pairs.csv');quality=pd.read_csv(ROOT/'quality.csv').set_index('id');sources={r['id']:r for r in json.loads((ROOT/'sources.json').read_text())};pairs=pairs[pairs.paired].copy().iloc[args.shard::args.shards];(ROOT/'points').mkdir(exist_ok=True)
 for nr,p in enumerate(pairs.itertuples()):
  ids=[p.prev5,p.next5,p.test7]
  if not all((ROOT/'predictions'/(id+'.npz')).exists() for id in ids):audit.append(dict(test7=p.test7,reason='missing_prediction'));continue
  if not all(quality.loc[id,'clock_ok'] for id in ids):audit.append(dict(test7=p.test7,reason='force_clock_mismatch'));continue
  for j in J:
   base=dict(subject=p.subject,date=p.date,test7=p.test7,start7=p.start7,target=TARGETS[j],prev5=p.prev5,next5=p.next5);test=native_scores(p.test7,j);train={id:native_scores(id,j) for id in ids[:2]};sm={id:sensor(id,j) for id in ids[:2]}
   if not len(test['x']):audit.append(dict(base,reason='no_routed_test_outputs'));continue
   for scope,training_ids in [('both',[p.prev5,p.next5]),('previous',[p.prev5]),('next',[p.next5])]:
    if any(sm[id] is None for id in training_ids):audit.append(dict(base,scope=scope,reason='missing_sensor_map'));continue
    xx=[];yy={k:[] for k in ['raw_cmcu','bsq_calibrated_cmcu','direct_bsq']}
    for id in training_ids:
     d=train[id];cal=predict(d['ad'],sm[id]);matched=np.isfinite(d['x'])&np.isfinite(d['raw'])&np.isfinite(cal)&np.isfinite(d['ref']);xx.append(d['x'][matched]);yy['raw_cmcu'].append(d['raw'][matched]);yy['bsq_calibrated_cmcu'].append(cal[matched]);yy['direct_bsq'].append(d['ref'][matched])
    x=np.concatenate(xx)
    if len(x)<8 or np.unique(x).size<2:audit.append(dict(base,scope=scope,reason='insufficient_output_calibration'));continue
    columns={k:np.concatenate(v) for k,v in yy.items()};matched_test=None
    for method in ['isotonic','affine']:
     models={k:fit(x,y,method) for k,y in columns.items()}
     # Identical training x yields identical score-domain coverage for the two ablation arms.
     for policy in ['domain','clamp']:
      pp={k:predict(test['x'],model,policy) for k,model in models.items()};assert np.array_equal(np.isfinite(pp['raw_cmcu']),np.isfinite(pp['bsq_calibrated_cmcu']))
      for arm,pred in pp.items():
       bb=dict(base,tare='first4s',scope=scope,method=method,policy=policy,arm=arm,train_n=len(x),train_score_min=float(x.min()),train_score_max=float(x.max()))
       for band,lo,hi in BANDS:metrics.append(score(bb,test['ref'],pred,test['event'],band,lo,hi))
       if method=='isotonic' and policy=='domain':
        for band,lo,hi in BANDS:metrics.append(score(dict(bb,tare='stable_side_rest_same_routing'),test['refstable'],pred,test['event'],band,lo,hi))
        if scope=='both':
         for band,lo,hi in BANDS:metrics.append(score(dict(bb,tare='first3s_same_routing'),test['ref3'],pred,test['event'],band,lo,hi))
      if scope=='both' and method=='isotonic' and policy=='domain':
       np.savez_compressed(ROOT/'points'/(p.test7+'_'+str(j)+'.npz'),**test,**pp)
     for arm,model in models.items():
      maps.append(dict(base,scope=scope,method=method,arm=arm,x=model[0].tolist(),y=model[1].tolist()))
    # Separate sensor-only comparison on identical AD-domain points, without the EMG output stage.
    cal_test=np.column_stack([predict(test['ad'],sm[id]) for id in training_ids]);valid=np.isfinite(cal_test).all(axis=1);avg=np.full(len(valid),np.nan);avg[valid]=cal_test[valid].mean(axis=1)
    for band,lo,hi in BANDS:
     candidates=(test['ref']>=lo)&(test['ref']<hi)
     for arm,pred in [('raw_cmcu',np.where(valid,test['raw'],np.nan)),('bsq_calibrated_cmcu',avg)]:
      smetrics.append(score(dict(base,tare='first4s',scope=scope,arm=arm),test['ref'],pred,test['event'],band,lo,hi,candidate=candidates))
      smetrics.append(score(dict(base,tare='stable_side_rest_same_routing',scope=scope,arm=arm),test['refstable'],pred,test['event'],band,lo,hi))
    qa.append(dict(base,scope=scope,train_n=len(x),test_n=len(test['x']),sensor_train_abs_error_N=float(np.mean(abs(columns['bsq_calibrated_cmcu']-columns['direct_bsq']))),raw_train_abs_error_N=float(np.mean(abs(columns['raw_cmcu']-columns['direct_bsq'])))))
  if (nr+1)%20==0:print('SCORE',nr+1,len(pairs),'metrics',len(metrics),flush=True)
 pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(smetrics).to_csv(OUT/'sensor_metrics.csv',index=False);pd.DataFrame(audit).to_csv(OUT/'score_exclusions.csv',index=False);pd.DataFrame(qa).to_csv(OUT/'train_audit.csv',index=False)
 import gzip
 with gzip.open(OUT/'calibration_maps.json.gz','wt') as f:json.dump(maps,f)
 print('DONE',len(pairs),'pairs',len(metrics),'metric rows',len(audit),'exclusions',flush=True)
