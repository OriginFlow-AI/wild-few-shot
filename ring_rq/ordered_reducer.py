"""Eight logical FP32 micro gradients, summed in rank order on each rank.

No local-mean regrouping: all micro losses are divided by eight BEFORE backward,
just as on the single-rank reference. Physical rank count affects only which
GPU computes each logical micro. One gather per pair of micros, no extra steps.
"""
import hashlib
import json
from pathlib import Path
import torch


class Reducer:
    def __init__(self,model,world,rank,trace=None):
        self.params=[p for p in model.parameters() if p.requires_grad];self.world=world;self.rank=rank
        self.trace=Path(trace) if trace else None
        self.total=None

    def add(self,logical):
        assert all(p.grad is not None and p.grad.dtype==torch.float32 for p in self.params)
        flat=torch.cat([p.grad.detach().reshape(-1) for p in self.params])
        if self.trace:
            self.trace.parent.mkdir(parents=True,exist_ok=True)
            with self.trace.open('a') as f:
                f.write(json.dumps(dict(logical_micro=logical,sha256=hashlib.sha256(flat.cpu().numpy().tobytes()).hexdigest()),separators=(',',':'))+'\n')
        if self.world==1:values=[flat]
        else:
            values=[torch.empty_like(flat) for _ in range(self.world)]
            torch.distributed.all_gather(values,flat)
        for value in values:
            if self.total is None:self.total=value.clone()
            else:self.total.add_(value)
        for p in self.params:p.grad=None

    def finish(self):
        assert self.total is not None
        cursor=0
        for p in self.params:
            n=p.numel();p.grad=self.total[cursor:cursor+n].view_as(p);cursor+=n
        assert cursor==self.total.numel()
