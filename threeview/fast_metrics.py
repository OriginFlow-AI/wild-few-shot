"""Same exact scipy tau-b and FP64 moments; reuse pointwise arrays across bins."""
import sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
if __package__:
    from . import dev_metrics as ref
else:
    import dev_metrics as ref
def window_stats(args):
    person,stage,trial,pr,yr=args
    tau=[];eligible=bad=constant=0
    for f in range(5):
        if np.ptp(yr[:,f])==0:constant+=1;continue
        eligible+=1
        if np.ptp(pr[:,f])==0:bad+=1;continue
        t=float(ref.kendalltau(yr[:,f],pr[:,f],variant='b').statistic)
        assert np.isfinite(t);tau.append(t)
    p=pr.astype(np.float64);y=yr.astype(np.float64);ay=abs(y);err=p-y;ae=abs(err);se=err*err
    active=ay>.5;ape=ae/np.maximum(ay,.5)*active;fp=p>1
    out={};all_valid=np.ones_like(y,dtype=bool)
    for name in ref.BINS:
        m=ref.mask(yr,name) if name!='all' else all_valid
        if not m.any():out[name]=np.zeros((13,5),dtype=np.float64);continue
        if name!='all' and m.all():out[name]=out['all'];continue
        n=m.sum(0);safe=np.maximum(n,1);mp=(p*m).sum(0)/safe;my=(y*m).sum(0)/safe
        pc=p-mp;yc=y-my
        out[name]=np.stack([n,mp,my,(pc*pc*m).sum(0),(yc*yc*m).sum(0),(pc*yc*m).sum(0),(ay*m).sum(0),(ae*m).sum(0),(se*m).sum(0),
                            (ape*m).sum(0),(active&m).sum(0),(err*m).sum(0),(fp&m).sum(0)])
    return person,stage,trial,out,(float(np.mean(tau)) if eligible and len(tau)==eligible else None),eligible,bad,constant
