"""Generate synthetic raw-like EMG only; not an accuracy benchmark."""
from pathlib import Path
import argparse
import numpy as np

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('synthetic_emg.npy'))
    args=parser.parse_args();root=Path(__file__).resolve().parent
    with np.load(root/'train_norm.npz',allow_pickle=False) as z:
        x=(np.random.default_rng(43).normal(size=(8010,16)).astype(np.float32)*z['emg_std']+z['emg_mean']).astype(np.float32)
    args.output.parent.mkdir(parents=True,exist_ok=True);np.save(args.output,x)
    print(f'SYNTHETIC ONLY: {args.output}; shape={x.shape}, dtype={x.dtype}; no force ground truth')
