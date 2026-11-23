"""Parameter-identical five-finger assembly; no RNG or precision changes."""
from types import MethodType
import torch
def frame_concat(self,z):
 assert self.groups==[[0],[1],[2],[3],[4]] and self.total_channels==5
 return torch.cat([net(value) for net,value in zip(self.nets,z.split(self.latent_channels,dim=1))],dim=1)
def install(model,*,concat=True,fill_off=True):
 if concat:model.core.tokenizer.frame.forward=MethodType(frame_concat,model.core.tokenizer.frame)
 if fill_off:torch.utils.deterministic.fill_uninitialized_memory=False
 return model
