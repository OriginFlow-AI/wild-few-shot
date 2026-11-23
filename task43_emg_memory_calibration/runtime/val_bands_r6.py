"""Reporting-only force bands; never used for loss, scheduling or selection."""
from collections import defaultdict
import numpy as np

METRICS=('MAE','MAPE','Corr','R2','bias','low_FP1')
BANDS={'abs_le_0p5':(None,.5),'abs_gt_0p5_lt2':(.5,2.),'abs_ge2_lt5':(2.,5.),
       'abs_ge5':(5.,None),'abs_ge10':(10.,None)}


def stats(x,y,low=False):
    x=x.astype(np.float64);y=y.astype(np.float64)
    if not len(y):return None
    err=x-y;active=abs(y)>=.5
    vx=np.sum((x-x.mean())**2);vy=np.sum((y-y.mean())**2)
    eps=np.finfo(np.float64).eps
    good_x=vx>eps*max(1.,float(np.sum(x*x)))
    good_y=vy>eps*max(1.,float(np.sum(y*y)))
    return {'MAE':float(abs(err).mean()),'MAPE':float(100*np.mean(abs(err[active])/abs(y[active]))) if active.any() else None,
            'Corr':float(np.clip(np.sum((x-x.mean())*(y-y.mean()))/np.sqrt(vx*vy),-1,1)) if good_x and good_y else None,
            'R2':float(1-np.sum(err*err)/vy) if good_y else None,'bias':float(err.mean()),
            'low_FP1':float((x>1).mean()) if low else None,'points':int(len(y)),'mape_points':int(active.sum())}


def mean(rows):
    return {k:float(np.mean(v)) if (v:=[r[k] for r in rows if r[k] is not None]) else None for k in METRICS}


class ForceBands:
    def __init__(self):
        self.rows={name:defaultdict(list) for name in BANDS}
        self.counts={name:defaultdict(lambda:{'points':0,'mape_points':0,'records':set(),'undefined':defaultdict(int)}) for name in BANDS}

    def add(self,query,hand,pred,target,valid):
        person,stage,rid=query['identity_group'],query['stage'],query['rid']
        for finger in range(5):
            mask=valid[:,finger];x=pred[:,finger][mask];y=target[:,finger][mask];a=abs(y)
            selectors={'abs_le_0p5':a<=.5,'abs_gt_0p5_lt2':(a>.5)&(a<2),
                       'abs_ge2_lt5':(a>=2)&(a<5),'abs_ge5':a>=5,'abs_ge10':a>=10}
            for name,selected in selectors.items():
                row=stats(x[selected],y[selected],name=='abs_le_0p5')
                if row is None:continue
                self.rows[name][person,stage,hand['hand'],finger].append(row)
                c=self.counts[name][person,stage];c['points']+=row['points'];c['mape_points']+=row['mape_points'];c['records'].add(rid)
                for metric in METRICS:
                    if row[metric] is None:c['undefined'][metric]+=1

    def finish(self):
        result={'definition':'abs calibrated reference N; MAPE only abs(reference)>=0.5 N; errors signed prediction-reference',
                'aggregation':'trial within hand/finger, then fingers/hands within person-stage, then stages within person, then people equal',
                'null_reasons':{'MAPE':'no abs(reference)>=0.5 N points','Corr':'near-zero prediction/reference variance','R2':'near-zero reference variance','low_FP1':'not the <=0.5 N band'},'bands':{}}
        for name,groups in self.rows.items():
            stage_rows=defaultdict(list)
            for (person,stage,_,_),rows in groups.items():stage_rows[person,stage].append(mean(rows))
            by_stage={key:mean(rows) for key,rows in stage_rows.items()}
            people=sorted({k[0] for k in by_stage});stages=sorted({k[1] for k in by_stage})
            by_person={person:mean([v for (who,_),v in by_stage.items() if who==person]) for person in people}
            def totals(keys):
                rows=[self.counts[name][key] for key in keys];records=set().union(*(r['records'] for r in rows))
                return {'points':sum(r['points'] for r in rows),'mape_points':sum(r['mape_points'] for r in rows),
                        'recordings':len(records),'people':len({k[0] for k in keys}),
                        'undefined_finger_trial_counts':{metric:sum(r['undefined'][metric] for r in rows) for metric in METRICS}}
            result['bands'][name]={'summary':dict(mean(list(by_person.values())),**totals(list(by_stage))),
                'per_person':{person:dict(v,**totals([k for k in by_stage if k[0]==person])) for person,v in by_person.items()},
                'per_stage':{str(stage):dict(mean([v for (who,st),v in by_stage.items() if st==stage]),**totals([k for k in by_stage if k[1]==stage])) for stage in stages},
                'per_person_stage':{person:{str(st):dict(v,**totals([(who,st)])) for (who,st),v in by_stage.items() if who==person} for person in people}}
        return result
