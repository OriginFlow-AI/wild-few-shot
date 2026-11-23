"""Vectorized coarse membrane-force recalibration, NumPy and optional PyTorch.

Input order: thumb, index, middle, ring, little. Shared across people and hands.
Output is always N. This is a descriptive remapping, not a validated force standard.
"""
from pathlib import Path
import json,hashlib
import numpy as np

class MembraneCalibrator:
    def __init__(self, directory=None):
        root=Path(directory) if directory is not None else Path(__file__).resolve().parent
        self.spec=json.loads((root/'coefficients.json').read_text(encoding='utf-8'))
        if hashlib.sha256((root/'tables.npz').read_bytes()).hexdigest()!=self.spec['tables_sha256']:
            raise ValueError('Calibration table SHA mismatch')
        with np.load(root/'tables.npz',allow_pickle=False) as f:
            self.table=f['calibrated_N'].copy();self.max_gf=f['max_gf'].copy();self.inverse_AD=f['inverse_AD'].copy()
        self._torch_cache={}

    @staticmethod
    def _scale(unit):
        try:return {'N':1/.00980665,'gf':1.,'mN':1/9.80665}[unit]
        except KeyError:raise ValueError("unit must be 'N', 'gf', or 'mN'") from None

    def _finger(self,finger):
        if isinstance(finger,str):
            if finger=='pinky':finger='little'
            try:return self.spec['finger_order'].index(finger)
            except ValueError:raise ValueError('Unknown finger') from None
        if isinstance(finger,(int,np.integer)) and not isinstance(finger,bool) and 0<=finger<5:return int(finger)
        raise ValueError('finger must be 0..4 or an English finger name')

    def numpy(self,force,*,finger_axis=-1,finger=None,unit='N',bounds='nan',return_mask=False):
        """Map any (...,5,...) array, or arbitrary shape with finger explicitly set.

        Default invalid/out-of-coverage -> NaN; mask indicates original validity.
        bounds='clip' explicitly clips finite outliers; NaN/Inf remain NaN.
        Use a long-lived instance and memory-mapped blocks for very large datasets.
        """
        if bounds not in ('nan','clip','raise'):raise ValueError('Invalid bounds policy')
        original=np.asarray(force)
        if original.dtype.kind not in 'fiu':raise TypeError('Numeric force array required')
        dtype=np.float64 if original.dtype==np.float64 else np.float32
        x=original.astype(dtype,copy=False)
        if finger is None:
            if x.ndim==0 or x.shape[finger_axis]!=5:raise ValueError('finger_axis must have length5')
            x=np.moveaxis(x,finger_axis,-1);ch=np.arange(5);maximum=self.max_gf.astype(dtype)
        else:
            ch=self._finger(finger);maximum=dtype(self.max_gf[ch])
        q=x*dtype(self._scale(unit));finite=np.isfinite(q)
        valid=finite & (q>=0) & (q<=maximum+4*np.finfo(dtype).eps*np.maximum(maximum,1))
        if bounds=='raise' and not np.all(valid):raise ValueError('Nonfinite, negative or outside calibrated force coverage')
        safe=np.where(finite,np.minimum(np.maximum(q,0),maximum),0)
        lo=safe.astype(np.int64);hi=np.minimum(lo+1,len(self.table)-1);w=safe-lo
        a=self.table[lo,ch];b=self.table[hi,ch]
        out=(a+(b-a)*w).astype(dtype)
        out=np.where(finite if bounds=='clip' else valid,out,np.nan).astype(dtype)
        if finger is None:out=np.moveaxis(out,-1,finger_axis);valid=np.moveaxis(valid,-1,finger_axis)
        return (out,valid) if return_mask else out

    def torch(self,force,*,finger_axis=-1,finger=None,unit='N',bounds='nan',return_mask=False):
        """Device-local vectorized interpolation; tables cached per device/dtype.

        Accepts CUDA tensors; no CPU copy or synchronization for nan/clip policies.
        Each worker/GPU creates its own instance. Output fp32, or fp64 for fp64 input.
        """
        import torch
        if bounds not in ('nan','clip','raise'):raise ValueError('Invalid bounds policy')
        if not isinstance(force,torch.Tensor) or force.dtype==torch.bool or force.is_complex():raise TypeError('Real numeric torch.Tensor required')
        dtype=torch.float64 if force.dtype==torch.float64 else torch.float32
        key=(str(force.device),dtype)
        if key not in self._torch_cache:
            self._torch_cache[key]=(torch.as_tensor(self.table,device=force.device,dtype=dtype),torch.as_tensor(self.max_gf,device=force.device,dtype=dtype),torch.arange(5,device=force.device))
        table,maximum,channels=self._torch_cache[key];x=force.to(dtype=dtype)
        if finger is None:
            if x.ndim==0 or x.shape[finger_axis]!=5:raise ValueError('finger_axis must have length5')
            x=torch.movedim(x,finger_axis,-1);ch=channels
        else:
            ch=self._finger(finger);maximum=maximum[ch]
        q=x*self._scale(unit);finite=torch.isfinite(q)
        valid=finite & (q>=0) & (q<=maximum+4*torch.finfo(dtype).eps*maximum.clamp_min(1))
        if bounds=='raise' and not bool(valid.all()):raise ValueError('Nonfinite, negative or outside calibrated force coverage')
        safe=torch.where(finite,torch.minimum(q.clamp_min(0),maximum),torch.zeros_like(q))
        lo=safe.to(torch.int64);hi=(lo+1).clamp_max(len(self.table)-1);w=safe-lo
        a=table[lo,ch];b=table[hi,ch];out=a+(b-a)*w
        out=torch.where(finite if bounds=='clip' else valid,out,torch.full_like(out,float('nan')))
        if finger is None:out=torch.movedim(out,-1,finger_axis);valid=torch.movedim(valid,-1,finger_axis)
        return (out,valid) if return_mask else out

    def direct_numpy(self,force,*,finger,unit='N'):
        """Float64 two-stage reference, one finger; useful for verifying the fast LUT."""
        j=self._finger(finger);q=np.asarray(force,dtype=np.float64)*self._scale(unit)
        valid=np.isfinite(q)&(q>=0)&(q<=self.max_gf[j]);safe=np.where(valid,q,0)
        ad=np.interp(safe,np.arange(len(self.inverse_AD)),self.inverse_AD);u=ad/60000
        p=self.spec['rows'][j];y=p['a']*u+p['b']*np.expm1(p['k']*u)/np.expm1(p['k'])
        return np.where(valid,y,np.nan)
