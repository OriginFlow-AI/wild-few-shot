"""Independent2-second/reset inference. No streaming-W100 claim or ReLU."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import json
import numpy as np
import torch
from arms import Banks
from train_data import Dataset
from calibration.calibration_tools import Calibration
from inference_models import ROOT,memory_model,lora_model


def setup(device):
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    return torch.device(device)


class SingleSupport(Dataset):
    """Use the original prefix selector and slot consumer for one explicit support."""
    def __init__(self,record):
        rid=record['key']
        if record['stage']!=3 or record['layout']!='1x5x1' or record['n']<120000:
            raise ValueError('This release uses a complete CMCU Stage3 first60s support')
        self.role='dev';self.records={rid:record}
        self.supports={rid:dict(record=rid,hand=record['hand'],stop=120000)}
        self.calibration=Calibration(ROOT/'calibration')
        self._support=Dataset.support.__wrapped__(self,rid)

    def support(self,rid):
        if rid not in self.records:raise KeyError(rid)
        return self._support


@torch.inference_mode()
def build_memory(arm,record,chunk=32):
    data=SingleSupport(record);banks=Banks(arm,None,data,None,chunk)
    values=Banks.one.__wrapped__(banks,'dev',record['key'],record['hand'])
    return tuple(t[None].to(arm.emg_mean.device) for t in values)


class Predictor:
    def __init__(self,device='cpu',arms=('A','B_L','F'),batch=16):
        if not arms or set(arms)-{'A','B_L','F'}:raise ValueError('Select A/B_L/F')
        self.device=setup(device);self.arms=tuple(arms);self.batch=int(batch)
        if self.batch<1:raise ValueError('Positive batch required')
        self.f=memory_model(self.device,trained='F' in arms) if {'A','F'}&set(arms) else None
        self.bl=lora_model(self.device) if 'B_L' in arms else None
        self.memory=None;self.support_id=None

    def bind(self,record=None):
        self.support_id=None if record is None else record['key']
        self.memory=None if record is None else build_memory(self.f,record)

    @torch.inference_mode()
    def predict(self,raw):
        raw=check_windows(raw);out={k:[] for k in self.arms}
        for start in range(0,len(raw),self.batch):
            piece=np.ascontiguousarray(raw[start:start+self.batch])
            if not piece.flags.writeable:piece=piece.copy()
            z=torch.from_numpy(piece).to(self.device)
            a=self.f(z) if self.f is not None else None
            if 'A' in out:out['A'].append(a.cpu().numpy())
            if 'B_L' in out:out['B_L'].append(self.bl(z).cpu().numpy())
            if 'F' in out:
                pred=a if self.memory is None else self.f(z,self.memory,torch.zeros(len(z),device=self.device,dtype=torch.long))
                out['F'].append(pred.cpu().numpy())
        result={k:np.concatenate(v) for k,v in out.items()}
        if not all(v.shape==(len(raw),5,89) and np.isfinite(v).all() for v in result.values()):raise ValueError('Prediction shape/finite gate')
        return result


def check_windows(raw):
    raw=np.asarray(raw)
    if raw.dtype!=np.float32 or raw.ndim!=3 or raw.shape[1:]!=(4000,16) or not len(raw) or not np.isfinite(raw).all():
        raise ValueError('Expected finite float32 [windows,4000,16], no silent tail padding')
    return raw


def graph_predictor(device,batch=64):
    setup(device)
    if torch.device(device).type!='cuda':raise ValueError('CUDA Graph requires CUDA')
    from fast_shared import FastShared
    return FastShared(torch.device(device),batch=batch,backend='torch')
