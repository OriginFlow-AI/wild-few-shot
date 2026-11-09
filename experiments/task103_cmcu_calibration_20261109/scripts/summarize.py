import os
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from common import ROOT, MODEL_DIR, EXPERIMENT;METRICS=['MAE_N','RMSE_N','bias_N','MAPE_pct','tau_b','coverage_pct']
KEY=['tare','scope','method','policy','arm','band']
def hierarchical(d,keys):
 # Already equally weighted presses inside trial × finger. First trials, then fingers, then people.
 first=d.groupby(keys+['date','subject','target'],dropna=False)[METRICS].mean().reset_index()
 return first.groupby(keys+['date','subject'],dropna=False)[METRICS].mean().reset_index()
def aggregate(p,keys):
 rows=[]
 for label,s in [('all',p),*list(p.groupby('date'))]:
  for vals,q in s.groupby(keys,dropna=False):
   vals=vals if isinstance(vals,tuple) else (vals,);r=dict(zip(keys,vals));r.update(date_scope=label,subjects=q.subject.nunique());r.update({m:q[m].mean() for m in METRICS});rows.append(r)
 return pd.DataFrame(rows)
def paired_ci(p):
 rows=[];keys=['tare','scope','method','policy','band'];rng=np.random.default_rng(20261109)
 for date,s in [('all',p),*list(p.groupby('date'))]:
  for vals,q in s.groupby(keys):
   wide=q.pivot(index='subject',columns='arm',values='MAE_N');wide=wide.dropna(subset=['raw_cmcu','bsq_calibrated_cmcu']);n=len(wide)
   if not n:continue
   raw=wide.raw_cmcu.to_numpy();cal=wide.bsq_calibrated_cmcu.to_numpy();delta=cal-raw;indices=rng.integers(0,n,(10000,n));means=delta[indices].mean(axis=1)
   rows.append(dict(zip(keys,vals),date_scope=date,subjects=n,raw_MAE_N=raw.mean(),calibrated_MAE_N=cal.mean(),delta_N=delta.mean(),relative_reduction_pct=100*(raw.mean()-cal.mean())/raw.mean(),ci_low_N=np.quantile(means,.025),ci_high_N=np.quantile(means,.975),improved_subjects=int((delta<0).sum())))
 return pd.DataFrame(rows)
if __name__=='__main__':
 d=pd.read_csv(ROOT/'metrics.csv');p=hierarchical(d,KEY);p.to_csv(ROOT/'per_subject.csv',index=False);s=aggregate(p,KEY);s.to_csv(ROOT/'summary.csv',index=False);ci=paired_ci(p);ci.to_csv(ROOT/'paired_effect.csv',index=False)
 ss=pd.read_csv(ROOT/'sensor_metrics.csv');ss['method']='sensor_only';ss['policy']='domain';sp=hierarchical(ss,KEY);sp.to_csv(ROOT/'sensor_per_subject.csv',index=False);aggregate(sp,KEY).to_csv(ROOT/'sensor_summary.csv',index=False)
 pairs=pd.read_csv(ROOT/'pairs.csv');q=pd.read_csv(ROOT/'quality.csv');train=pd.read_csv(ROOT/'train_audit.csv');src=json.loads((ROOT/'sources.json').read_text());main=d[(d.tare=='first4s')&(d.scope=='both')&(d.method=='isotonic')&(d.policy=='domain')&(d.arm=='raw_cmcu')&(d.band=='0-25')];valid=main[main.MAE_N.notna()]
 qa=dict(source_trials=len(src),test5=sum(r['stage']=='mix-test-5' for r in src),test7=sum(r['stage']=='mix-test-7' for r in src),subjects=len(set(r['subject'] for r in src)),eligible_test7=int(pairs.paired.sum()),scored_test7=valid.test7.nunique(),scored_subjects=valid.subject.nunique(),scored_trial_fingers=len(valid),calibration_points=int(train[train.scope=='both'].train_n.sum()),test_points=int(main.scored_n.sum()),candidate_test_points=int(main.candidate_n.sum()),same_coverage=bool(np.array_equal(d[d.arm=='raw_cmcu'].scored_n.to_numpy(),d[d.arm=='bsq_calibrated_cmcu'].scored_n.to_numpy())),force_clock_ok=int(q.clock_ok.sum()),tare_methods=q.tare_method.value_counts().to_dict(),dates={date:dict(test7=int((pp.stage=='mix-test-7').sum()),test5=int((pp.stage=='mix-test-5').sum()),subjects=pp.subject.nunique()) for date,pp in q.groupby('date')})
 hashes={}
 for f in [MODEL_DIR/'ap304_rank_best_model_state_dict.pt',MODEL_DIR/'train_norm.npz',ROOT/'sources.json']:
  hashes[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
 qa['eligible_trial_fingers']=int(pairs.paired.sum())*8;qa['trial_finger_available_pct']=100*qa['scored_trial_fingers']/qa['eligible_trial_fingers'];qa['tare_first4_vs_first3_abs_max_N']=float(q.first4_minus_first3_N.abs().max());qa['tare_first4_vs_first3_abs_median_N']=float(q.first4_minus_first3_N.abs().median());qa['input_preprocessing']='canonical force-training MCAP preprocessing once: 50Hz harmonic notches Q30, order4 40-850Hz bandpass; model normalization unchanged' if (ROOT/'filter_result.json').exists() else 'unfiltered diagnostic';qa['filter_result']=json.loads((ROOT/'filter_result.json').read_text()) if (ROOT/'filter_result.json').exists() else None;qa['sha256']=hashes;qa['extraction']=json.loads((ROOT/'extraction_result.json').read_text());qa['inference']=json.loads((ROOT/'inference_result.json').read_text());(ROOT/'result_QA.json').write_text(json.dumps(qa,indent=2))
 print(json.dumps(qa,indent=2));print(ci[(ci.scope=='both')&(ci.method=='isotonic')&(ci.policy=='domain')].to_string(index=False))
