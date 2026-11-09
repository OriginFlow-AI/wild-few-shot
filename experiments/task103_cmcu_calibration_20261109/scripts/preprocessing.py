"""MCAP uV -> training-format uV; apply ONCE before bundle normalization."""
import numpy as np
from scipy.signal import butter, iirnotch, tf2sos, sosfilt, sosfilt_zi


def filter_emg(x):
    sections = []
    for frequency in range(50, 1000, 50):
        b, a = iirnotch(frequency, 30., fs=2000.)
        sections.append(tf2sos(b, a))
    sections.append(butter(4, (40., 850.), btype='bandpass', fs=2000., output='sos'))
    sos = np.vstack(sections)
    mean = np.mean(x[:min(1000, len(x))], axis=0, keepdims=True)
    zi = np.ascontiguousarray(sosfilt_zi(sos)[:, :, None] * mean)
    result, _ = sosfilt(sos, x, zi=zi, axis=0)
    return result.astype(np.float32)


def regularize(t, x):
    grid = t[0] + np.arange((int(t[-1]) - int(t[0])) // 500000 + 1, dtype=np.int64) * 500000
    if len(grid) >= 2 * len(t):
        raise ValueError('Implausible EMG time span')
    source_time = (t - t[0]).astype(float)
    grid_time = (grid - t[0]).astype(float)
    raw = np.column_stack([np.interp(grid_time, source_time, x[:, j]) for j in range(16)]).astype(np.float32)
    hi = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
    lo = np.maximum(hi - 1, 0)
    valid = (t[hi] - t[lo] <= 5000000) | (t[hi] == grid)
    return int(grid[0]), filter_emg(raw), valid
