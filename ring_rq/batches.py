"""Bounded ordered group prefetch; deterministic crop and augmentation identity."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np
from io_local import LocalReader,Index,closed_rows,ROOT,read
class Batcher:
    def __init__(self,role,norm,workers=8):
        self.rows=closed_rows()[role];self.role=role;self.norm=norm
        self.index=Index(self.rows,None,prepared=read(ROOT/'data/INDEX.json.gz')[role])
        self.pool=ThreadPoolExecutor(max_workers=workers);self.local=threading.local()
    def _group(self,args):
        i,items,epoch,augment=args
        if not hasattr(self.local,'reader'):self.local.reader=LocalReader(max_bytes=512<<20)
        row=self.rows[i];out=[]
        for position,start,identity in items:
            e,f=self.local.reader.window(row,start)
            e=(e-self.norm['emg_mean'])/self.norm['emg_std']
            if augment:
                rng=np.random.default_rng(np.random.SeedSequence([42,epoch,identity,0xB47C]))
                # Sampling uses its own stream, augmentation sequence is explicit.
                e*=rng.uniform(.8,1.25,(1,16)).astype(np.float32)
                if rng.random()<.05:e[:,int(rng.integers(16))]=0
                e+=rng.normal(0,.02,e.shape).astype(np.float32)
            out.append((position,e.T.copy(),f.T.copy(),row['subject'],row['stage'],row['id'],start))
        return out
    def prepare(self,specs,epoch=0,augment=False):
        groups=defaultdict(list)
        for position,(i,start,identity) in enumerate(specs):groups[i].append((position,start,identity))
        results=self.pool.map(self._group,[(i,v,epoch,augment) for i,v in groups.items()])
        ordered=[None]*len(specs)
        for group in results:
            for item in group:ordered[item[0]]=item
        return dict(emg=np.stack([v[1] for v in ordered]),raw=np.stack([v[2] for v in ordered]),subjects=[v[3] for v in ordered],stages=[v[4] for v in ordered],ids=[v[5] for v in ordered],starts=[v[6] for v in ordered])
    def train(self,epoch,first,n):
        specs=[(*self.index.sample(epoch,k),k) for k in range(first,first+n)]
        return self.prepare(specs,epoch=epoch,augment=True)
    def close(self):self.pool.shutdown(wait=True)
