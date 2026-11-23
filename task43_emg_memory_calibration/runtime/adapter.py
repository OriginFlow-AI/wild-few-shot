"""Per-finger prefix memory for the frozen W304 Affine head, not RQ weights."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class PrefixMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.finger=nn.Embedding(5,8)
        self.conv1=nn.Conv1d(2,16,9,stride=4)
        self.conv2=nn.Conv1d(16,32,5,stride=4)
        self.force_projection=nn.Linear(512,64)
        self.key=nn.Linear(320,128)
        self.query=nn.Linear(328,128)
        self.value=nn.Sequential(nn.Linear(396,256),nn.SiLU(),nn.Linear(256,128))
        self.delta=nn.Sequential(nn.Linear(460,256),nn.SiLU(),nn.Linear(256,304))
        self.gate=nn.Linear(460,1)
        nn.init.zeros_(self.delta[-1].weight);nn.init.zeros_(self.delta[-1].bias)
        nn.init.zeros_(self.gate.bias)

    def encode(self,e,force,amplitude,valid):
        b,g,k,_=e.shape
        assert g==5 and e.shape[-1]==320 and force.shape==(b,g,k,2,256)
        assert valid.dtype==torch.bool and valid.shape==(b,g,k)
        assert bool(torch.isfinite(e[valid]).all()) and bool(torch.isfinite(force[valid]).all())
        assert bool((force[...,1,:][valid]==1).all())
        z=F.silu(self.conv1(F.pad(force.reshape(-1,2,256),(5,0))))
        z=F.silu(self.conv2(F.pad(z,(1,0))))
        z=self.force_projection(z.flatten(1)).reshape(b,g,k,64)
        emb=self.finger.weight[None,:,None,:].expand(b,g,k,8)
        return self.key(e),self.value(torch.cat((e,z,amplitude,emb),-1)),valid

    def forward(self,e,memory,coverage,head_weight,head_bias):
        b,t,_=e.shape;k,v,valid=memory
        assert head_weight.shape==(15,304) and coverage.shape==(b,5,4)
        base=F.linear(e[...,:304],head_weight,head_bias).reshape(b,t,3,5).transpose(-1,-2)
        emb=self.finger.weight[None,None].expand(b,t,5,8)
        qe=e[:,:,None].expand(b,t,5,320)
        q=self.query(torch.cat((qe,emb),-1)).reshape(b,t,5,4,32).permute(0,2,3,1,4)
        kk=k.reshape(b,5,-1,4,32).permute(0,1,3,2,4)
        vv=v.reshape(b,5,-1,4,32).permute(0,1,3,2,4)
        active=valid.any(-1);safe=valid.clone();safe[:,:,0]|=~active
        scores=(q@kk.transpose(-2,-1))/math.sqrt(32)
        scores=scores.masked_fill(~safe[:,:,None,None,:],float('-inf'))
        r=(scores.softmax(-1)@vv).permute(0,3,1,2,4).reshape(b,t,5,128)
        u=torch.cat((qe,r,emb,coverage[:,None].expand(b,t,5,4)),-1)
        d=self.delta(u)*self.gate(u).sigmoid()
        # Original head is ordered [position5, velocity5, gate5].
        correction=torch.einsum('btgd,god->btgo',d,head_weight.reshape(3,5,304).permute(1,0,2))
        result=torch.where(active[:,None,:,None],base+correction,base)
        return result.transpose(-1,-2).flatten(-2)
