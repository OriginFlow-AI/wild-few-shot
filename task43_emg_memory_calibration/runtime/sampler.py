"""Shared stateless subject/stage/recording/time sampler, with replacement."""
from collections import defaultdict
import numpy as np

def plan(data,epoch,total,seed):
    by_person=defaultdict(lambda:defaultdict(list))
    for g,(q,h) in enumerate(data.groups):by_person[q['identity_group']][q['stage']].append(g)
    people=sorted(by_person);rng=np.random.default_rng(seed+epoch)
    out=np.empty(total,dtype=np.int64)
    # Exact 75/12.5/12.5 quotas, independently shuffled over the update stream.
    bands=np.tile(np.array([0,0,0,0,0,0,1,2],np.int8),(total+7)//8)[:total]
    rng.shuffle(bands)
    fallback=0
    band_cache={}
    for i,band in enumerate(bands):
        p=people[int(rng.integers(len(people)))];stages=sorted(by_person[p])
        st=stages[int(rng.integers(len(stages)))];groups=by_person[p][st]
        g=groups[int(rng.integers(len(groups)))];q,h=data.groups[g]
        count=h['windows'];eligible=None
        if band:
            if q['rid'] not in band_cache:
                with np.load(data.targets_path(q['rid']),allow_pickle=False) as z:
                    band_cache[q['rid']]=(np.flatnonzero(z['high']),np.flatnonzero(z['rest']))
            eligible=band_cache[q['rid']][int(band)-1]
        if eligible is None or not len(eligible):
            offset=int(rng.integers(count));fallback+=int(bool(band))
        else:offset=int(eligible[int(rng.integers(len(eligible)))])
        out[i]=data.ends[g]+offset
    return out,{'draws':total,'high_quota':int((bands==1).sum()),'rest_quota':int((bands==2).sum()),'uniform_fallbacks':fallback}
