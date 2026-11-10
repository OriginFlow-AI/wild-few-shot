"""Benchmark real train windows; freeze one microbatch shared by all objectives."""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from common import BRANCHES, atomic, load, settings
from train import Windows, network, objective, optimizer

def attempt(root,branch,micro,limit):
    torch.manual_seed(43);torch.cuda.manual_seed(43)
    torch.cuda.set_per_process_memory_fraction(limit*2**30/torch.cuda.get_device_properties(0).total_memory,0)
    net=network();net.load_state_dict(torch.load(root/'assets/common_stage2_initial_state_dict.pt',map_location='cpu',weights_only=True))
    net=net.cuda();net.train();opt=optimizer(net,settings());ds=Windows(root,'train',branch)
    batch=[ds[i%len(ds)] for i in range(micro)]
    x=torch.from_numpy(np.stack([s[0] for s in batch])).cuda();y=torch.from_numpy(np.stack([s[1] for s in batch])).cuda()
    valid=torch.from_numpy(np.stack([s[2] for s in batch])).cuda()
    n=ds.data['force_norm'][ds.kind]
    mean=torch.tensor(n['mean'],device='cuda',dtype=torch.float32)[None,:,None]
    std=torch.tensor(n['std'],device='cuda',dtype=torch.float32)[None,:,None]
    times=[];torch.cuda.reset_peak_memory_stats()
    for step in range(3):
        opt.zero_grad(set_to_none=True);torch.cuda.synchronize();tick=time.monotonic()
        with torch.autocast('cuda',dtype=torch.bfloat16):z=net(x)
        loss,count=objective(z.float()*std+mean,y,valid,branch)
        if count==0 or not torch.isfinite(loss):raise ValueError('Invalid real-data probe loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
        torch.cuda.synchronize();times.append(time.monotonic()-tick)
    return dict(branch=branch,microbatch=micro,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                windows_per_second=micro/float(np.median(times[1:])),step_seconds=times,loss=float(loss.detach()))

def main(root,limit):
    root=Path(root);torch.set_num_threads(2);out=root/'checks/gpu_tuning.json'
    for key,name in [('CUDA_CACHE_PATH','cuda_kernels'),('TORCH_HOME','torch'),('XDG_CACHE_HOME','xdg')]:
        path=root/'cache'/name;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    if out.exists():return
    rows=[];valid=[]
    for micro in [8,16,32,64]:
        candidate=[]
        for branch in BRANCHES:
            atomic(root/'checks/gpu_tuning_progress.json',dict(microbatch=micro,branch=branch,process_memory_limit_GiB=limit))
            try:r=attempt(root,branch,micro,limit)
            except torch.OutOfMemoryError:
                r=dict(branch=branch,microbatch=micro,status='OOM');rows.append(r);break
            rows.append(r);candidate.append(r);gc.collect();torch.cuda.empty_cache()
        if len(candidate)!=len(BRANCHES):gc.collect();torch.cuda.empty_cache();break
        valid.append(dict(microbatch=micro,windows_per_second=min(r['windows_per_second'] for r in candidate),
                          peak_allocated_GiB=max(r['peak_allocated_GiB'] for r in candidate)))
    if not valid:raise RuntimeError('No candidate microbatch passed all three objectives')
    chosen=max(valid,key=lambda r:r['windows_per_second'])
    atomic(out,dict(status='PASS_REAL_TRAIN_GPU_BENCHMARK',chosen=chosen,attempts=rows,
                   process_memory_limit_GiB=limit,GPU=torch.cuda.get_device_name(0),
                   formal_weights_changed=False,test_metrics_read=False,
                   note='Probe optimizer steps are disposable. All formal branches start from the common initial state.'))
    print(json.dumps(chosen),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--limit',type=float,required=True)
    a=p.parse_args();main(a.root,a.limit)
