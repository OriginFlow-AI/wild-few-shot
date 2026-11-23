"""Explicit source-domain routing. Calibrate one fresh array on cache miss only."""
import gzip,json,hashlib
from pathlib import Path
from functools import lru_cache
import numpy as np
ROOT=Path(__file__).resolve().parent

@lru_cache(maxsize=1)
def bindings():
    with gzip.open(ROOT/'data/LABEL_BINDINGS.json.gz','rt') as f:return json.load(f)['rows']

@lru_cache(maxsize=1)
def mapping():
    policy=json.loads((ROOT/'LABEL_POLICY.json').read_bytes())
    for name,digest in policy['frozen_ex45_assets'].items():
        assert hashlib.sha256((ROOT/'label_calibration'/name).read_bytes()).hexdigest()==digest
    import sys
    sys.path.insert(0,str(ROOT/'label_calibration'))
    from label_map import LabelMap
    return LabelMap()

def transform(force,*,layout,domain,policy_sha256=None):
    value=np.asarray(force)
    assert value.dtype in (np.float32,np.float64) and value.ndim==2 and value.shape[1]==5 and np.isfinite(value).all()
    if domain=='ex45_calibrated_N':
        assert policy_sha256==hashlib.sha256((ROOT/'label_calibration/LABEL_POLICY.json').read_bytes()).hexdigest(),'Unbound pre-calibrated cache rejected'
        return value.copy()  # Never feed a known calibrated array through the LUT again.
    assert domain=='raw_extracted_N','Unknown label domain must not default to raw'
    if layout=='1x5x3':return value.copy()  # Signed gen3 native Fz is never LUT-mapped.
    assert layout=='1x5x1','Unknown sensor layout must not default to membrane'
    return mapping()(value)  # Frozen old ex45 NumPy arithmetic and exact tail.

def apply_source_force(force,row):
    bound=bindings()[row['id']]
    assert bound['source_uri']==row['source_uri'] and bound['input_unit']=='N'
    # Source arrays are readonly. The returned independent calibrated array is
    # cached by LocalReader, so repeated windows do not transform its outputs.
    return transform(force,layout=bound['layout'],domain=bound['label_domain'],policy_sha256=bound.get('policy_sha256'))
