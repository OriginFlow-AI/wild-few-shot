"""Fixed-window FP32 CUDA Graph: shared A/F features, merged LoRA, frozen K/V cache."""
from inference_models import models
import torch
import numpy as np
from torch.nn.utils import parametrize
from base_model import diagonal_affine_scan
from train_data import POSITIONS,HISTORY


class FastShared:
    def __init__(self,device,batch=64,backend='torch'):
        self.device,self.batch=device,batch
        self.f,self.bl=models(device)
        assert self.f.base.config==self.bl.base.config
        aa=self.f.base.state_dict();bb=self.bl.base.state_dict()
        for key in aa:
            if key.startswith(('patch_embedding.','encoder_blocks.','encoder_norm.')) or key=='electrode_fourier':
                assert torch.equal(aa[key],bb[key]),key
        for key in ('emg_mean','emg_std','force_mean','force_std'):
            assert torch.equal(getattr(self.f,key),getattr(self.bl,key)),key
        for path,field in self.bl.lora_targets:
            parametrize.remove_parametrizations(self.bl.base.get_submodule(path),field,leave_parametrized=True)
        if backend!='torch':raise ValueError('Only the qualified Torch backend is exported')
        self.backend=backend
        self.raw=torch.zeros(batch,4000,16,device=device)
        self.offsets=torch.as_tensor(POSITIONS[:,None]+HISTORY,device=device)
        self.keys=torch.zeros(1,5,128,128,device=device);self.values=torch.zeros_like(self.keys)
        self.valid=torch.zeros(1,5,128,device=device,dtype=torch.bool)
        self.coverage=torch.zeros(1,5,4,device=device)
        self.enabled=torch.zeros((),device=device,dtype=torch.bool)
        self.index=torch.zeros(batch,device=device,dtype=torch.long)
        self.encoded={};self.capture_count=0
        warm=torch.cuda.Stream();warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(3):self.outputs=self.compute()
        torch.cuda.current_stream().wait_stream(warm);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.outputs=self.compute()
        self.capture_count+=1
        self.addresses=tuple(x.data_ptr() for x in [self.raw,self.keys,self.values,self.valid,self.coverage,self.enabled])

    def compute(self):
        x=self.f.normalized(self.raw)
        encoded=self.f.base.encode(x.transpose(1,2).contiguous())
        fused=self.f.base.fuse_streams(self.f.base.decode_streams(encoded))[0]
        rms=x[:,self.offsets].square().mean(dim=2).sqrt()
        features=torch.cat((fused,rms),-1)
        a=self.f.physical(self.f.base.affine_decode(fused)[0])
        selected=tuple(t.index_select(0,self.index) for t in (self.keys,self.values,self.valid))
        logits=self.f.memory(features,selected,self.coverage.index_select(0,self.index),
            self.f.base.output_head.weight,self.f.base.output_head.bias)
        position,velocity,gate=logits.chunk(3,-1)
        position=position*self.f.base.config.output_scale;velocity=velocity*self.f.base.config.output_scale
        gate=gate.sigmoid();n=self.f.base.config.position_anchor_tokens
        gate=torch.cat((torch.zeros_like(gate[:,:n]),gate[:,n:]),1)
        bias=gate*velocity+(1-gate)*position
        fp=self.f.physical(diagonal_affine_scan(gate,bias,bias.new_zeros(len(bias),5)).transpose(1,2))
        fp=torch.where(self.enabled,fp,a)
        bl_fused=self.bl.base.fuse_streams(self.bl.base.decode_streams(encoded))[0]
        bl=self.bl.physical(self.bl.base.affine_decode(bl_fused)[0])
        return dict(A=a,B_L=bl,F=fp)

    def bind(self,support,memory=None):
        if support is None:
            self.enabled.zero_();self.keys.zero_();self.values.zero_();self.valid.zero_();self.coverage.zero_();return
        if support not in self.encoded:
            assert memory is not None
            features,force,amplitude,valid,coverage=memory
            self.encoded[support]=(*self.f.memory.encode(features,force,amplitude,valid),coverage)
        keys,values,valid,coverage=self.encoded[support]
        self.keys.copy_(keys);self.values.copy_(values);self.valid.copy_(valid);self.coverage.copy_(coverage);self.enabled.fill_(True)

    @torch.inference_mode()
    def predict(self,raw):
        pieces={k:[] for k in self.outputs}
        for start in range(0,len(raw),self.batch):
            stop=min(len(raw),start+self.batch);self.raw.zero_()
            piece=np.ascontiguousarray(raw[start:stop])
            if not piece.flags.writeable:piece=piece.copy()
            self.raw[:stop-start].copy_(torch.from_numpy(piece).to(self.device))
            assert tuple(x.data_ptr() for x in [self.raw,self.keys,self.values,self.valid,self.coverage,self.enabled])==self.addresses
            self.graph.replay()
            for key,v in self.outputs.items():pieces[key].append(v[:stop-start].cpu().numpy().copy())
        return {k:np.concatenate(v) for k,v in pieces.items()}
