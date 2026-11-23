"""Frozen CMCU calibration and inverse; Arm loads its separate matched A_ep1 norm.

NumPy only. Force is [...,5] in N unless an axis is explicitly supplied.
Finger order is thumb, index, middle, ring, little, for both hands.
The original LUT and last-segment tail remain the numerical authority.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

GF_TO_N = 0.00980665
TABLE_SHA256 = '3ff7e7e67ebe2ce7b05ec1d32dcb418e0cf0a985b7a7c1bfdfcc8d751b7b5b77'
COEFFICIENT_SHA256 = 'c6417c60fa6706b4f624e2323ac3153e6aeb318179c8870764a632f3eb8101c0'


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def to_newton(value, unit):
    """Explicit unit conversion, not calibration. Never guess the input unit."""
    factors = {'N': 1., 'mN': .001, 'gf': GF_TO_N}
    if unit not in factors:
        raise ValueError('unit must be N, mN, or gf')
    x = np.asarray(value, np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite force')
    return x*factors[unit]


def _array(value, axis, channels):
    x = np.asarray(value)
    if x.dtype not in (np.float32, np.float64) or x.ndim < 1:
        raise ValueError('Expected float32/float64 array')
    if x.shape[axis] != channels or not np.isfinite(x).all():
        raise ValueError('Wrong channel count or nonfinite input')
    return np.moveaxis(x, axis, -1)


def _layout(layout):
    names = {'membrane': '1x5x1', 'gen3': '1x5x3', '1x5x1': '1x5x1', '1x5x3': '1x5x3'}
    if layout not in names:
        raise ValueError('Specify authoritative layout: membrane/1x5x1 or gen3/1x5x3')
    return names[layout]


class Calibration:
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else Path(__file__).resolve().parent
        folder = self.root/'calibration_assets'
        if _sha(folder/'tables.npz') != TABLE_SHA256 or _sha(folder/'coefficients.json') != COEFFICIENT_SHA256:
            raise ValueError('Frozen calibration asset SHA mismatch')
        self.coefficients = json.loads((folder/'coefficients.json').read_text(encoding='utf-8'))
        with np.load(folder/'tables.npz', allow_pickle=False) as z:
            self.table = z['calibrated_N'].copy()
            self.max_gf = z['max_gf'].copy()
            self.inverse_AD = z['inverse_AD'].copy()
        self.finger_order = self.coefficients['finger_order']
        self.xmax = self.max_gf.astype(np.float64)*GF_TO_N
        j = np.arange(5)
        self.ymax = self.table[self.max_gf.astype(np.int64), j]
        self.slope = (self.ymax-self.table[self.max_gf.astype(np.int64)-1, j])/GF_TO_N
        assert self.table.shape[1] == 5 and np.all(self.slope > 0)
        for j, g in enumerate(self.max_gf):
            assert self.table[0, j] == 0 and np.all(np.diff(self.table[:int(g)+1, j]) > 0)

    def forward(self, raw_force_N, *, layout, axis=-1):
        """Raw extracted N -> calibrated N. gen3 is identity, including signed GT.

        Membrane negative/nonfinite labels are rejected, not clipped or dropped.
        Matches the original production float32/float64 arithmetic and tail.
        """
        x = _array(raw_force_N, axis, 5)
        if _layout(layout) == '1x5x3':
            return np.moveaxis(x.copy(), -1, axis)
        if np.any(x < 0):
            raise ValueError('Negative membrane label: no silent clipping')
        dtype = x.dtype.type
        maximum = self.max_gf.astype(dtype)
        q = x*dtype(1/GF_TO_N)
        safe = np.minimum(np.maximum(q, 0), maximum)
        lo = safe.astype(np.int64)
        hi = np.minimum(lo+1, len(self.table)-1)
        w = safe-lo
        j = np.arange(5)
        a, b = self.table[lo, j], self.table[hi, j]
        within = (a+(b-a)*w).astype(dtype)
        tail = self.ymax+(x.astype(np.float64)-self.xmax)*self.slope
        result = np.where(x.astype(np.float64) > self.xmax, tail, within).astype(dtype)
        if not np.isfinite(result).all():
            raise ValueError('Output overflow; use an appropriate physical range/dtype')
        return np.moveaxis(result, -1, axis)

    def inverse(self, calibrated_force_N, *, layout, axis=-1, negative_policy='preserve_gen3'):
        """Calibrated N -> old raw N; returns float64.

        preserve_gen3: signed gen3 identity; negative membrane is rejected.
        raise: reject any negative input, including gen3.
        clip_zero: clip predictions of BOTH layouts to zero before inverse;
                   this is the authorized EP2 60 report's posthoc policy.
        Never use clip_zero to alter ground truth. No signed membrane extension.
        """
        x = _array(calibrated_force_N, axis, 5).astype(np.float64)
        kind = _layout(layout)
        if negative_policy not in ('preserve_gen3', 'raise', 'clip_zero'):
            raise ValueError('Unknown negative policy')
        if negative_policy == 'clip_zero':
            x = np.maximum(x, 0.)
        elif np.any(x < 0) and (kind == '1x5x1' or negative_policy == 'raise'):
            raise ValueError('Negative value outside selected inverse policy')
        if kind == '1x5x3':
            return np.moveaxis(x.copy(), -1, axis)
        flat = x.reshape(-1, 5)
        result = np.empty(flat.shape, np.float64)
        for j, maximum in enumerate(self.max_gf):
            old_knots = np.arange(int(maximum)+1, dtype=np.float64)*GF_TO_N
            new_knots = self.table[:int(maximum)+1, j]
            result[:, j] = np.interp(flat[:, j], new_knots, old_knots)
            high = flat[:, j] > self.ymax[j]
            result[high, j] = self.xmax[j]+(flat[high, j]-self.ymax[j])/self.slope[j]
        return np.moveaxis(result.reshape(x.shape), -1, axis)

    def parameters(self):
        return dict(finger_order=self.finger_order, GF_TO_N=GF_TO_N,
            rows=[dict(finger=name, a=row['a'], b=row['b'], k=row['k'], max_gf=int(self.max_gf[j]),
                       raw_boundary_N=float(self.xmax[j]), calibrated_boundary_N=float(self.ymax[j]),
                       last_LUT_segment_slope_N_per_N=float(self.slope[j]))
                  for j, (name, row) in enumerate(zip(self.finger_order, self.coefficients['rows']))],
            authority='Production uses the frozen integer-gf LUT plus its last actual segment tail, not direct analytic evaluation or analytic derivative tails.')
