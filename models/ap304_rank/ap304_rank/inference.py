from __future__ import annotations
import contextlib
from pathlib import Path
import numpy as np
import torch
from .model import load_model
from .normalization import Normalization

FINGERS=('thumb','index','middle','ring','little')


def window_starts(length):
    if not isinstance(length,(int,np.integer)) or length<4000:raise ValueError('Need at least 4000 samples (2 seconds at 2000 Hz)')
    starts=np.arange(0,length-3999,4000,dtype=np.int64)
    if starts[-1]!=length-4000:starts=np.append(starts,length-4000)
    return starts


class RankPredictor:
    def __init__(self,*,variant='best',device='cpu',precision='fp32',root=None):
        self.root=Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]
        self.device=torch.device(device);self.variant=variant;self.precision=precision
        if precision not in ('fp32','bf16'):raise ValueError('precision must be fp32 or bf16')
        if precision=='bf16' and self.device.type!='cuda':raise ValueError('This bundle exposes BF16 only on CUDA; use fp32 for CPU')
        self.model=load_model(variant=variant,device=self.device,root=self.root)
        self.normalization=Normalization(self.root)

    @torch.inference_mode()
    def predict_windows(self,raw_windows):
        """[B,4000,16] raw EMG -> [B,89,5] ranking scores."""
        raw_windows=np.asarray(raw_windows)
        if raw_windows.ndim!=3 or raw_windows.shape[1:]!=(4000,16) or not len(raw_windows):
            raise ValueError('Expected nonempty raw_windows [B,4000,16]')
        normalized=self.normalization.normalize_emg(raw_windows)
        x=torch.from_numpy(np.ascontiguousarray(normalized.transpose(0,2,1))).to(self.device)
        ctx=torch.autocast('cuda',dtype=torch.bfloat16) if self.precision=='bf16' else contextlib.nullcontext()
        with ctx:scores=self.model(x)
        scores=scores.transpose(1,2).cpu().numpy()
        if scores.shape!=(len(raw_windows),89,5) or not np.isfinite(scores).all():raise ValueError('Invalid model scores')
        return scores

    def predict_trial(self,raw_emg,*,batch_size=4):
        """Match production reset-window inference and chronological tail stitching.

        A native window is 50Hz, but context gaps and the tail can make the full
        returned time axis nonuniform. Always use sample_indices/time_seconds.
        """
        raw=np.asarray(raw_emg)
        if raw.ndim!=2 or raw.shape[1]!=16 or raw.dtype.kind!='f' or not np.isfinite(raw).all():
            raise ValueError('Expected finite floating-point raw_emg [T,16]')
        if not isinstance(batch_size,int) or batch_size<1:raise ValueError('batch_size must be a positive integer')
        starts=window_starts(len(raw));kept_scores=[];kept_ticks=[];last=-1
        for begin in range(0,len(starts),batch_size):
            subset=starts[begin:begin+batch_size]
            windows=np.stack([raw[s:s+4000] for s in subset])
            predictions=self.predict_windows(windows)
            for start,scores in zip(subset,predictions):
                ticks=start+470+40*np.arange(89,dtype=np.int64)
                keep=ticks>last
                if keep.any():
                    kept_scores.append(scores[keep]);kept_ticks.append(ticks[keep]);last=int(ticks[keep][-1])
        indices=np.concatenate(kept_ticks)
        return dict(scores=np.concatenate(kept_scores),sample_indices=indices,time_seconds=indices/2000.0,
                    window_starts=starts,finger_order=np.asarray(FINGERS),
                    output_units=np.asarray('dimensionless_ranking_score'),variant=np.asarray(self.variant))
