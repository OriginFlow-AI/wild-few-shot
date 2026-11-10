"""A disposable real-Train update followed by model/Adam/RNG checkpoint restore."""
import argparse
import os
from pathlib import Path
import numpy as np
import torch
from common import atomic, load, settings
from gpu_selector import choose
from train import network, Windows, optimizer, objective, save

def main(root):
    root=Path(root);torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    gpu=choose(32,4,minimum_limit_GiB=8)
    if gpu is None:raise RuntimeError('No shared GPU headroom for checkpoint verification')
    os.environ['CUDA_VISIBLE_DEVICES']=gpu['uuid']
    cache=root/'cache/cuda_kernels';cache.mkdir(exist_ok=True);os.environ['CUDA_CACHE_PATH']=str(cache)
    torch.cuda.set_per_process_memory_fraction(gpu['process_memory_limit_GiB']*2**30/torch.cuda.get_device_properties(0).total_memory,0)
    torch.manual_seed(43);torch.cuda.manual_seed(43)
    net=network();net.load_state_dict(torch.load(root/'assets/common_stage2_initial_state_dict.pt',map_location='cpu',weights_only=True));net.cuda()
    c=settings();opt=optimizer(net,c);ds=Windows(root/'real_data_smoke','train','calibrated_huber')
    batch=[ds[i] for i in range(min(16,len(ds)))];x=torch.from_numpy(np.stack([r[0] for r in batch])).cuda()
    y=torch.from_numpy(np.stack([r[1] for r in batch])).cuda();valid=torch.from_numpy(np.stack([r[2] for r in batch])).cuda()
    norm=ds.data['force_norm']['cal'];mean=torch.tensor(norm['mean'],device='cuda',dtype=torch.float32)[None,:,None]
    std=torch.tensor(norm['std'],device='cuda',dtype=torch.float32)[None,:,None]
    with torch.autocast('cuda',dtype=torch.bfloat16):z=net(x)
    loss,count=objective(z.float()*std+mean,y,valid,'calibrated_huber');loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
    path=root/'checks/disposable_checkpoint_roundtrip.pt'
    state=dict(epoch=0,next_update=1,global_update=1,best=None,bad_epochs=0,history=[])
    binding=dict(scope='disposable_real_Train_check_only',force_norm=norm)
    save(path,net,opt,state,binding)
    ck=torch.load(path,map_location='cpu',weights_only=True)
    restored=network().cuda();restored.load_state_dict(ck['model'],strict=True)
    restored_optimizer=optimizer(restored,c);restored_optimizer.load_state_dict(ck['optimizer'])
    assert all(torch.equal(v.cpu(),restored.state_dict()[k].cpu()) for k,v in net.state_dict().items())
    a,b=opt.state_dict()['state'],restored_optimizer.state_dict()['state']
    assert a.keys()==b.keys()
    for p,v in a.items():
        for key,tensor in v.items():assert torch.equal(tensor.cpu(),b[p][key].cpu())
    torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state(ck['cuda_rng'])
    assert torch.equal(torch.get_rng_state(),ck['torch_rng'])
    assert torch.equal(torch.cuda.get_rng_state(),ck['cuda_rng'])
    assert ck['state']==state and ck['binding']==binding
    size=path.stat().st_size;path.unlink()
    atomic(root/'checks/checkpoint_roundtrip.json',dict(status='PASS_REAL_TRAIN_MODEL_ADAM_RNG_CHECKPOINT_ROUNDTRIP',
        GPU=gpu,microbatch=len(batch),valid_points=count,loss=float(loss.detach()),checkpoint_bytes=size,
        common_initial_model_unchanged=True,Test_metrics_read=False,disposable_checkpoint_deleted=True))
    print('PASS: real Train update and exact model/Adam/RNG checkpoint restoration')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)
