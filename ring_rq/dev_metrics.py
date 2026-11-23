"""Exact window tau-b and subject/stage/force-bin sufficient statistics."""
from collections import defaultdict
import numpy as np
from scipy.stats import kendalltau
BINS=['all','le0.5','0.5to2','2to5','ge5','ge10']
def mask(y,name):
    a=np.abs(y)
    if name=='all':return np.ones_like(y,dtype=bool)
    if name=='le0.5':return a<=.5
    if name=='0.5to2':return (a>.5)&(a<2)
    if name=='2to5':return (a>=2)&(a<5)
    if name=='ge5':return a>=5
    return a>=10
def moments(p,y,valid):
    # Array [13,5]. Fixed 0.5N MAPE evaluation floor, as frozen reference Dev.
    p=p.astype(np.float64);y=y.astype(np.float64);m=valid;active=m&(abs(y)>.5)
    err=p-y
    n=m.sum(0);safe=np.maximum(n,1);mp=(p*m).sum(0)/safe;my=(y*m).sum(0)/safe
    pc=p-mp;yc=y-my
    return np.stack([n,mp,my,(pc*pc*m).sum(0),(yc*yc*m).sum(0),(pc*yc*m).sum(0),
                     (abs(y)*m).sum(0),(abs(err)*m).sum(0),(err*err*m).sum(0),(abs(err)/np.maximum(abs(y),.5)*active).sum(0),active.sum(0),(err*m).sum(0),((p>1)&m).sum(0)])
def merge_moments(a,b):
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64);na=a[0];nb=b[0];n=na+nb;safe=np.maximum(n,1)
    dp=b[1]-a[1];dy=b[2]-a[2];cross=na*nb/safe;out=a+b
    out[1]=a[1]+dp*nb/safe;out[2]=a[2]+dy*nb/safe
    out[3]=a[3]+b[3]+dp*dp*cross;out[4]=a[4]+b[4]+dy*dy*cross;out[5]=a[5]+b[5]+dp*dy*cross
    return out
def metric(z):
    n,mp,my,vp,vy,cy,ay,ae,se,ape,na,bias,fp=np.asarray(z,np.float64)
    good=n>0;safe=np.maximum(n,1)
    w=ay/safe;cv=good&(vp>1e-8)&(vy>1e-8);cw=w*cv;mw=w*(na>0)
    corr=float((cw*cy/np.sqrt(np.maximum(vp*vy,1e-8))).sum()/cw.sum()) if cw.sum()>1e-8 else None
    mape=float((mw*ape/np.maximum(na,1)).sum()/mw.sum()) if mw.sum()>1e-8 else None
    mae=float((ae[good]/safe[good]).mean()) if good.any() else None
    r2=float((1-se[good&(vy>1e-8)]/vy[good&(vy>1e-8)]).mean()) if np.any(good&(vy>1e-8)) else None
    return dict(L_A=.5*(1-corr)+.1*mape+.15*mae if corr is not None and mape is not None and mae is not None else None,
                MAE=mae,MAPE_percent=100*mape if mape is not None else None,Corr=corr,R2=r2,
                bias_N=float(bias.sum()/n.sum()) if n.sum() else None,
                prediction_gt1N_rate=float(fp.sum()/n.sum()) if n.sum() else None,
                valid_finger_points=int(n.sum()),per_finger_points=n.astype(int).tolist(),
                null_reason=None if good.any() else 'no eligible reference points',
                MAPE_null_reason='no reference force above 0.5N' if mape is None else None,
                Corr_null_reason='no defined nonconstant weighted finger' if corr is None else None,
                R2_null_reason='no reference finger with variance above1e-8' if r2 is None else None)
def window_stats(args):
    person,stage,trial,p,y=args
    out={};tau=[];eligible=bad=constant=0
    for f in range(5):
        if np.ptp(y[:,f])==0:constant+=1;continue
        eligible+=1
        if np.ptp(p[:,f])==0:bad+=1;continue
        t=float(kendalltau(y[:,f],p[:,f],variant='b').statistic)
        assert np.isfinite(t);tau.append(t)
    for name in BINS:out[name]=moments(p,y,mask(y,name))
    return person,stage,trial,out,(float(np.mean(tau)) if eligible and len(tau)==eligible else None),eligible,bad,constant
class Accumulator:
    def __init__(self):self.cells={};self.windows=0;self.bad=0;self.eligible=0;self.constant=0
    def add(self,result):
        person,stage,trial,bins,tau,eligible,bad,constant=result
        c=self.cells.setdefault((person,stage),dict(bins={k:np.zeros((13,5),np.float64) for k in BINS},trials={k:set() for k in BINS},tau_sum=0.,tau_count=0,tau_eligible_windows=0,windows=0))
        for k,z in bins.items():
            c['bins'][k]=merge_moments(c['bins'][k],z)
            if z[0].sum():c['trials'][k].add(trial)
        if tau is not None:c['tau_sum']+=tau;c['tau_count']+=1
        c['tau_eligible_windows']+=int(eligible>0);c['windows']+=1
        self.windows+=1;self.bad+=bad;self.eligible+=eligible;self.constant+=constant
    def payload(self):
        return dict(windows=self.windows,bad=self.bad,eligible=self.eligible,constant=self.constant,cells=[dict(person=p,stage=s,bins={k:v.tolist() for k,v in c['bins'].items()},trials={k:sorted(v) for k,v in c['trials'].items()},**{k:v for k,v in c.items() if k not in ('bins','trials')}) for (p,s),c in self.cells.items()])
    def merge(self,payload):
        self.windows+=payload['windows'];self.bad+=payload['bad'];self.eligible+=payload['eligible'];self.constant+=payload['constant']
        for c in payload['cells']:
            key=(c['person'],c['stage']);d=self.cells.setdefault(key,dict(bins={k:np.zeros((13,5)) for k in BINS},trials={k:set() for k in BINS},tau_sum=0.,tau_count=0,tau_eligible_windows=0,windows=0))
            for k in BINS:d['bins'][k]=merge_moments(d['bins'][k],c['bins'][k]);d['trials'][k].update(c['trials'][k])
            for k in ('tau_sum','tau_count','tau_eligible_windows','windows'):d[k]+=c[k]
    def summarize(self,stages=None):
        cells={k:v for k,v in self.cells.items() if stages is None or k[1] in stages}
        people={};perstage={}
        def aggregate(items):
            z={k:np.zeros((13,5)) for k in BINS};tr={k:set() for k in BINS};ts=tn=te=windows=0
            for c in items:
                for k in BINS:z[k]=merge_moments(z[k],c['bins'][k]);tr[k].update(c['trials'][k])
                ts+=c['tau_sum'];tn+=c['tau_count'];te+=c['tau_eligible_windows'];windows+=c['windows']
            return dict(bins={k:dict(**metric(v),recordings=len(tr[k])) for k,v in z.items()},tau=ts/tn if tn else None,tau_windows=tn,tau_eligible_windows=te,windows=windows)
        for person in sorted({p for p,s in cells}):people[person]=aggregate(c for (p,s),c in cells.items() if p==person)
        for stage in sorted({s for p,s in cells}):
            pp={p:aggregate([c]) for (p,s),c in cells.items() if s==stage};perstage[stage]=dict(per_subject=pp,macro=self.macro(pp))
        return dict(per_subject=people,per_stage=perstage,macro=self.macro(people),windows=sum(c['windows'] for c in cells.values()),aggregation='within-person concatenated windows/five-finger amplitude weighted; then persons equal; bins retain valid-point and recording denominators')
    @staticmethod
    def macro(people):
        def avg(values):
            v=[x for x in values if x is not None];return float(np.mean(v)) if v else None
        bins={}
        for b in BINS:
            z={k:avg(p['bins'][b][k] for p in people.values()) for k in ('MAE','MAPE_percent','Corr','R2','bias_N','prediction_gt1N_rate')}
            z['people']=sum(p['bins'][b]['valid_finger_points']>0 for p in people.values());z['valid_finger_points']=sum(p['bins'][b]['valid_finger_points'] for p in people.values());z['recordings']=sum(p['bins'][b]['recordings'] for p in people.values())
            z['L_A']=.5*(1-z['Corr'])+.1*z['MAPE_percent']/100+.15*z['MAE'] if all(z[k] is not None for k in ('Corr','MAPE_percent','MAE')) else None
            z['null_reasons']={k:('A-loss needs defined Corr, MAPE and MAE' if k=='L_A' else 'no defined per-subject statistic; see per_subject bins for physical denominator reason') for k,v in z.items() if v is None}
            bins[b]=z
        result=dict(bins=bins,people=len(people),tau=avg(p['tau'] for p in people.values()),tau_windows=sum(p['tau_windows'] for p in people.values()),tau_eligible_windows=sum(p['tau_eligible_windows'] for p in people.values()))
        result.update(bins['all']);return result
