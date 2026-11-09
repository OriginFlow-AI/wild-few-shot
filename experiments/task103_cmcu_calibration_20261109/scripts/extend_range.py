from pathlib import Path
import os,time,sys,subprocess
import numpy as np,pandas as pd
from score import score,ROOT,TARGETS
if __name__=='__main__':
 while not (ROOT/'index.html').exists():time.sleep(2)
 d=pd.read_csv(ROOT/'metrics.csv');d=d[~d.band.isin(['0-all','25+'])];base=d[(d.tare=='first4s')&(d.scope=='both')&(d.method=='isotonic')&(d.policy=='domain')&(d.band=='0-25')].set_index(['test7','target','arm']);extra=[]
 for f in (ROOT/'points').glob('*.npz'):
  trial,j=f.stem.rsplit('_',1);target=TARGETS[int(j)];a=np.load(f)
  for arm in ['raw_cmcu','bsq_calibrated_cmcu','direct_bsq']:
   if (trial,target,arm) not in base.index:continue
   bb=base.loc[(trial,target,arm)].to_dict();bb.update(test7=trial,target=target,arm=arm)
   for tare,ref in [('first4s',a['ref']),('stable_side_rest_same_routing',a['refstable'])]:
    for band,lo,hi in [('0-all',0,np.inf),('25+',25,np.inf)]:extra.append(score(dict(bb,tare=tare),ref,a[arm],a['event'],band,lo,hi))
 pd.concat([d,pd.DataFrame(extra)],ignore_index=True).to_csv(ROOT/'metrics.csv',index=False)
 subprocess.run([sys.executable,str(Path(__file__).resolve().parent/'summarize.py')],check=True,stdout=open(ROOT/'summary.log','w'))
 subprocess.run([sys.executable,str(Path(__file__).resolve().parent/'report.py')],check=True)
 (ROOT/'full_range_ready.txt').write_text('complete');print('ADDITIONAL FULL RANGE METRICS',len(extra),flush=True)
