"""Overlap independent physical16 blocks without changing CUDA RNG call order."""
from types import MethodType
import torch
from torch.utils.checkpoint import checkpoint
def members_prediction(self,x):
 assert x.dtype==torch.float32 and x.shape[1:]==(16,4000)
 if torch.is_autocast_enabled('cuda') or torch.is_autocast_enabled('cpu'):raise RuntimeError('FP32 only')
 device=x.device;current=torch.cuda.current_stream(device)
 if not hasattr(self,'_speed_streams'):self._speed_streams=tuple(torch.cuda.Stream(device=device) for _ in range(2))
 ready=torch.cuda.Event();ready.record(current)
 for stream in self._speed_streams:stream.wait_event(ready)
 out=[]
 for i,part in enumerate(x.split(self.physical)):
  stream=self._speed_streams[i%2]
  # CPU invokes identical physical blocks, stochastic layers and checkpoint
  # captures in their original order; only independent CUDA work overlaps.
  with torch.cuda.stream(stream):
   part.record_stream(stream)
   if self.training and torch.is_grad_enabled():value=checkpoint(self.chunk,part,use_reentrant=False,preserve_rng_state=True)
   else:value=self.chunk(part)
   value.record_stream(current);out.append(value)
 for stream in self._speed_streams:current.wait_stream(stream)
 return torch.cat(out,dim=1)
def install(model):
 model.members_prediction=MethodType(members_prediction,model);return model
def wait_gradients(model):
 stream=torch.cuda.current_stream()
 for s in getattr(model,'_speed_streams',()):stream.wait_stream(s)
