"""Shared utilities. Runtime artifacts are rooted in --root, never in source."""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / 'vendor'))
VERSION = 'task103-k2-5p56m-four-way-subject-disjoint-v1'
SIDES = ['left', 'right']
BRANCHES = ['calibrated_huber', 'raw_huber', 'raw_original']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()

def atomic(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    def clean(v):
        if isinstance(v, np.ndarray):return clean(v.tolist())
        if isinstance(v, np.generic):return clean(v.item())
        if isinstance(v, dict):return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v, (list, tuple)):return [clean(x) for x in v]
        if isinstance(v, float) and not np.isfinite(v):return None
        return v
    tmp.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(tmp, path)

def load(path): return json.loads(Path(path).read_text())

def settings(): return load(EXPERIMENT / 'config.json')

def source_binding():
    return {str(p.relative_to(EXPERIMENT)):sha(p) for p in
            sorted((EXPERIMENT/'scripts').glob('*.py'))+sorted((EXPERIMENT/'vendor').glob('*.py'))}

def nearest(source, target):
    source, target = np.asarray(source), np.asarray(target)
    if len(source) == 0: raise ValueError('Empty timestamp stream')
    hi = np.clip(np.searchsorted(source, target), 0, len(source)-1)
    lo = np.maximum(hi-1, 0)
    k = np.where(abs(source[lo]-target) <= abs(source[hi]-target), lo, hi)
    return k, abs(source[k]-target)

def interpolate(t, x, valid, query, max_gap_ns=40_000_000):
    """No extrapolation and no interpolation across a masked or long gap."""
    hi = np.clip(np.searchsorted(t, query), 0, len(t)-1)
    lo = np.maximum(hi-1, 0)
    lo = np.where(t[hi] == query, hi, lo)
    den = t[hi]-t[lo]
    alpha = np.divide(query-t[lo], den, out=np.zeros(len(query), float), where=den>0)
    y = x[lo]*(1-alpha[:, None]) + x[hi]*alpha[:, None]
    good = valid[lo] & valid[hi] & np.isfinite(y)
    good &= ((query >= t[0]) & (query <= t[-1]) & (den <= max_gap_ns))[:, None]
    return np.where(good, y, 0).astype(np.float32), good

def windows(n):
    if n < 4000: return []
    result = list(range(0, n-3999, 4000))
    if result[-1] != n-4000: result.append(n-4000)
    return result

def group(r): return r['subject'], r['key'].split('/')[2], r['pc'], r['boot']
