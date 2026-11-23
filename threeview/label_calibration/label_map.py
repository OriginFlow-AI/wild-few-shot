"""Frozen five-finger LUT with user-authorized tangent continuation above coverage."""
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'calibration_assets'))
from calibration import MembraneCalibrator

class LabelMap:
    def __init__(self):
        self.base=MembraneCalibrator(HERE/'calibration_assets')
        self.xmax=self.base.max_gf.astype(np.float64)*.00980665
        j=np.arange(5);g=self.base.max_gf.astype(np.int64)
        self.ymax=self.base.table[g,j]
        # The shipped mapping is piecewise linear on integer-gf knots.
        # Continue its LAST actual segment, not an unrelated analytic AD slope.
        self.slope=(self.ymax-self.base.table[g-1,j])/.00980665
        assert np.all(self.slope>0)

    def __call__(self, force):
        x=np.asarray(force)
        if x.dtype not in (np.float32,np.float64) or x.ndim<1 or x.shape[-1]!=5:
            raise ValueError('Expected float32/64 [...,5] force in N, thumb to little')
        if not np.isfinite(x).all() or np.any(x<0):
            raise ValueError('Negative or nonfinite label: no silent clipping/dropping')
        # Preserve ALL in-domain outputs of the original frozen mapping.
        y=self.base.numpy(x,bounds='clip')
        high=x.astype(np.float64)>self.xmax
        tail=self.ymax+(x.astype(np.float64)-self.xmax)*self.slope
        return np.where(high,tail,y).astype(x.dtype)

    def spec(self):
        return dict(input_unit='N',output_unit='N',finger_order=self.base.spec['finger_order'],
                    boundary_N=self.xmax.tolist(),boundary_output_N=self.ymax.tolist(),
                    slope_N_per_N=self.slope.tolist(),
                    tail='T(xmax) + last_LUT_segment_slope * (x-xmax)',
                    lower_policy='reject negative/nonfinite; zero maps to zero',
                    refit_on_benchmark=False,shared_across=['people','hands','sessions','train','val','test'])
