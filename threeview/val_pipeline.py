"""Bounded prefetch and metric overlap, fixed64 predictions and ordered merges."""
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np,torch
from dev_metrics import Accumulator
from fast_metrics import window_stats
from fast_batches import Batcher
def evaluate_batches(model,norm,specs,*,progress=None):
    batcher=Batcher('val',norm);acc=Accumulator();cpu=ThreadPoolExecutor(max_workers=8);reader=ThreadPoolExecutor(max_workers=1)
    mean=torch.as_tensor(norm['force_mean'],device='cuda').view(1,5,1);scale=torch.as_tensor(norm['force_std'],device='cuda').view(1,5,1)
    queued=[];count=0;start=time.perf_counter();pred_hashes=[]
    import hashlib
    def collect():
        nonlocal count
        futures=queued.pop(0)
        for f in futures:acc.add(f.result());count+=1
        if progress:progress(count)
    future=reader.submit(batcher.prepare,specs[:64]) if specs else None
    try:
        with torch.inference_mode():
            for off in range(0,len(specs),64):
                batch=future.result()
                if off+64<len(specs):future=reader.submit(batcher.prepare,specs[off+64:off+128])
                x=torch.from_numpy(batch['emg']).cuda(non_blocking=True)
                pred=model(x);p=(pred*scale+mean).cpu().numpy().transpose(0,2,1)
                y=batch['raw'][...,np.arange(55)*40+1829].transpose(0,2,1)
                assert np.isfinite(p).all() and np.isfinite(y).all()
                pred_hashes.append(hashlib.sha256(p.tobytes()).hexdigest())
                queued.append([cpu.submit(window_stats,(batch['subjects'][j],batch['stages'][j],batch['ids'][j],p[j],y[j])) for j in range(len(p))])
                if len(queued)>=2:collect()
            while queued:collect()
    finally:reader.shutdown();cpu.shutdown();batcher.close()
    return acc,pred_hashes,time.perf_counter()-start
