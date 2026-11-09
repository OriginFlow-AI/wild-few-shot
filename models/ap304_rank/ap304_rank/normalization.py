from __future__ import annotations
from pathlib import Path
import numpy as np


class Normalization:
    """The frozen complete-Train EMG statistics. There is no force norm."""
    def __init__(self,root=None):
        root=Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]
        with np.load(root/'train_norm.npz',allow_pickle=False) as z:
            self.mean=np.asarray(z['emg_mean'],np.float32).copy()
            self.std=np.asarray(z['emg_std'],np.float32).copy()
        if self.mean.shape!=(16,) or self.std.shape!=(16,) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std<=0):
            raise ValueError('Invalid paired EMG normalization')

    def normalize_emg(self,raw):
        raw=np.asarray(raw)
        if raw.ndim<2 or raw.shape[-1]!=16 or raw.dtype.kind!='f':
            raise ValueError('EMG must be floating-point [...,time,16], in the original training channel order and units')
        if not np.isfinite(raw).all():raise ValueError('Nonfinite EMG')
        x=np.asarray(raw,np.float32)
        # Match production NumPy float32 subtract then divide; no clipping/filter.
        result=(x-self.mean)/self.std
        if not np.isfinite(result).all():raise ValueError('EMG overflows float32 normalization')
        return np.ascontiguousarray(result,dtype=np.float32)
