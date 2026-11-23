"""Exact rank-ordered FP32 sum, gathered only at rank0 then broadcast."""
from pathlib import Path
import hashlib,json,torch
class Reducer:
 def __init__(self,model,world,rank,trace=None):
  self.model=model;self.params=[p for p in model.parameters() if p.requires_grad];self.world=world;self.rank=rank;self.trace=Path(trace) if trace else None;self.total=None
 def add(self,logical):
  if __package__:
   from .streams import wait_gradients
  else:
   from streams import wait_gradients
  wait_gradients(self.model)
  assert all(p.grad is not None and p.grad.dtype==torch.float32 for p in self.params)
  flat=torch.cat([p.grad.detach().reshape(-1) for p in self.params])
  if self.trace:
   self.trace.parent.mkdir(parents=True,exist_ok=True)
   with self.trace.open('a') as f:f.write(json.dumps(dict(logical_micro=logical,sha256=hashlib.sha256(flat.cpu().numpy().tobytes()).hexdigest()),separators=(',',':'))+'\n')
  if self.world==1:values=[flat]
  elif self.rank==0:
   values=[torch.empty_like(flat) for _ in range(self.world)];torch.distributed.gather(flat,gather_list=values,dst=0)
  else:
   values=None;torch.distributed.gather(flat,dst=0)
  if self.rank==0:
   for value in values:
    if self.total is None:self.total=value.clone()
    else:self.total.add_(value)
  else:self.total=torch.empty_like(flat)
  if self.world>1:torch.distributed.broadcast(self.total,src=0)
  for p in self.params:p.grad=None
 def finish(self):
  assert self.total is not None;cursor=0
  for p in self.params:
   n=p.numel();p.grad=self.total[cursor:cursor+n].view_as(p);cursor+=n
  assert cursor==self.total.numel()
