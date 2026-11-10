"""One causal canonical EMG filter per continuous valid segment."""
import numpy as np
from scipy.signal import butter, iirnotch, tf2sos, sosfilt, sosfilt_zi

def regularize(t, x):
    if len(t) < 4000 or not np.isfinite(x).all(): raise ValueError('Invalid/short EMG')
    grid = t[0] + np.arange((int(t[-1])-int(t[0]))//500000+1, dtype=np.int64)*500000
    if len(grid) >= 2*len(t): raise ValueError('Implausible EMG sampling span')
    raw = np.column_stack([np.interp(grid-t[0], t-t[0], x[:, j]) for j in range(16)]).astype(np.float32)
    hi = np.clip(np.searchsorted(t, grid), 0, len(t)-1); lo = np.maximum(hi-1, 0)
    valid = (t[hi]-t[lo] <= 5_000_000) | (t[hi] == grid)
    sections = [tf2sos(*iirnotch(f, 30., fs=2000.)) for f in range(50, 1000, 50)]
    sections.append(butter(4, (40., 850.), btype='bandpass', fs=2000., output='sos'))
    sos = np.vstack(sections); output = np.zeros_like(raw)
    starts = np.flatnonzero(valid & ~np.r_[False, valid[:-1]])
    stops = np.flatnonzero(valid & ~np.r_[valid[1:], False])+1
    for a, b in zip(starts, stops):
        if b-a < 1000: valid[a:b] = False; continue
        mean = raw[a:a+1000].mean(axis=0, keepdims=True)
        zi = np.ascontiguousarray(sosfilt_zi(sos)[:, :, None]*mean)
        output[a:b] = sosfilt(sos, raw[a:b], zi=zi, axis=0)[0]
        valid[a:a+1000] = False  # Filter warmup is not training input.
    return int(grid[0]), output, valid
