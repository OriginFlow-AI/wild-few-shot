"""Single shared GPU stage2 training with exact common point masks and resume."""
import argparse
import collections
import contextlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import kendalltau
from common import BRANCHES, EXPERIMENT, atomic, interpolate, load, settings, sha, source_binding
from force_losses import force_huber_loss, force_objective

def network():
    spec=importlib.util.spec_from_file_location('task103_model',EXPERIMENT/'vendor/model_source.py')
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m.K2RingNetwork(m.K2RingConfig())

class Windows(Dataset):
    def __init__(self,root,split,branch):
        self.root=Path(root);self.data=load(self.root/'prepared/dataset.json')
        self.index=np.load(self.root/'prepared'/(split+'_windows.npy'),mmap_mode='r')
        self.kind='cal' if branch=='calibrated_huber' else 'raw';self.cache=collections.OrderedDict()
        with np.load(EXPERIMENT/'reference/stage1_emg_train_norm.npz') as n:
            self.em=np.asarray(n['emg_mean'],np.float32);self.es=np.asarray(n['emg_std'],np.float32)
        if not np.isfinite(self.es).all() or not np.all(self.es>0):raise ValueError('Bad pretrained EMG norm')
    def __len__(self):return len(self.index)
    def one(self,entry,start):
        r=self.data['entries'][entry];key=(r['id'],r['side'])
        if key not in self.cache:
            a=np.load(self.root/'cache/emg'/r['id']/(r['side']+'.npy'),mmap_mode='r')
            with np.load(self.root/'cache/labels'/(r['id']+'.npz')) as z:
                label={k:z[r['side']+'_'+k].copy() for k in ['t','raw','cal','bsq','valid']}
                for k in ['reference_t','reference_N','reference_valid']:label[k]=z[k].copy()
            self.cache[key]=(a,label)
            if len(self.cache)>8:self.cache.popitem(last=False)
        else:self.cache.move_to_end(key)
        a,label=self.cache[key]
        x=((a[start:start+4000]-self.em)/self.es).T.copy()
        query=r['start_ns']+(start+470+40*np.arange(89,dtype=np.int64))*500000
        target,valid=interpolate(label['t'],label[self.kind],label['valid'],query)
        if not np.isfinite(x).all():raise ValueError('Nonfinite normalized EMG')
        return x,target.T.copy(),valid.T.copy()
    def __getitem__(self,i):
        entry,start=map(int,self.index[i]);return self.one(entry,start)

def objective(p,y,mask,kind):
    mask=mask & torch.isfinite(y) & torch.isfinite(p)
    # Nonfinite predictions are model failures, not silently excluded labels.
    if not torch.isfinite(p).all():raise ValueError('Nonfinite model output')
    count=int(mask.sum())
    if not count:return p.sum()*0,0
    a,b=p[mask],y[mask]
    if kind=='raw_original':
        loss=force_objective(a[None],b[None],corr_weight=1.,mape_weight=.3,mae_weight=.03,mape_threshold_n=.5)['loss']
    else:loss=force_huber_loss(a,b,delta_n=1.)
    return loss,count

def measures(p,y,valid):
    x=np.asarray(p,dtype=np.float64)[valid];t=np.asarray(y,dtype=np.float64)[valid]
    if not len(x):return None
    e=x-t;corr=float(np.corrcoef(x,t)[0,1]) if np.ptp(x)>0 and np.ptp(t)>0 else None
    active=abs(t)>=.5;mape=float(np.mean(abs(e[active])/abs(t[active]))) if active.any() else 0.
    huber=float(np.mean(np.where(abs(e)<=1,.5*e**2,abs(e)-.5)))
    return dict(points=len(x),mae_N=float(abs(e).mean()),mape_percent=100*mape if active.any() else None,
                mape_points=int(active.sum()),corr=corr,
                huber_N2=huber,original_loss=1-(corr if corr is not None else 0.)+.3*mape+.03*float(abs(e).mean()))

@torch.no_grad()
def evaluate(net,root,branch,split,mean,std,device,batch_size,save_predictions=False):
    ds=Windows(root,split,branch);by_entry=collections.defaultdict(list)
    for entry,start in ds.index:by_entry[int(entry)].append(int(start))
    records={};net.eval()
    for entry,starts in sorted(by_entry.items()):
        r=ds.data['entries'][entry];parts={k:[] for k in ['pred','raw','cal','bsq','mask','finger','t_ns']};last=-1
        for begin in range(0,len(starts),batch_size):
            chunk=starts[begin:begin+batch_size];samples=[ds.one(entry,s) for s in chunk]
            x=torch.from_numpy(np.stack([s[0] for s in samples])).to(device)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):z=net(x)
            p=(z.float()*std+mean).cpu().numpy().transpose(0,2,1)
            if not np.isfinite(p).all():raise ValueError('Nonfinite evaluation predictions')
            _,label=ds.cache[(r['id'],r['side'])]
            for prediction,start in zip(p,chunk):
                offsets=start+470+40*np.arange(89,dtype=np.int64);keep=offsets>last
                query=r['start_ns']+offsets*500000
                outputs={k:interpolate(label['t'],label[k],label['valid'],query) for k in ['raw','cal','bsq']}
                h=0 if r['side']=='left' else 5
                outputs['bsq']=interpolate(label['reference_t'],np.repeat(label['reference_N'][:,None],5,axis=1),
                    label['reference_valid'][:,h:h+5],query)
                valid=outputs['raw'][1]&outputs['cal'][1]&outputs['bsq'][1]
                parts['pred'].append(prediction[keep]);parts['mask'].append(valid[keep])
                for k in ['raw','cal','bsq']:parts[k].append(outputs[k][0][keep])
                parts['finger'].append(np.broadcast_to(np.arange(5)+(0 if r['side']=='left' else 5),(int(keep.sum()),5)))
                parts['t_ns'].append(np.broadcast_to(query[keep,None],(int(keep.sum()),5)))
                if keep.any():last=int(offsets[keep][-1])
        a={k:np.concatenate(v) for k,v in parts.items()}
        record=records.setdefault(r['id'],dict(meta=r,parts=collections.defaultdict(list)))
        for k,v in a.items():record['parts'][k].append(v)
    results=[]
    for id,record in records.items():
        a={k:np.concatenate(v) for k,v in record['parts'].items()};r=record['meta']
        metrics={k:measures(a['pred'],a[k],a['mask']) for k in ['raw','cal','bsq']}
        target='cal' if branch=='calibrated_huber' else 'raw'
        primary=metrics[target]
        if primary is None:continue
        loss=primary['original_loss'] if branch=='raw_original' else primary['huber_N2']
        row=dict(id=id,subject=r['subject'],stage=r['stage'],loss=loss,metrics=metrics)
        if split=='test':
            row['by_finger']={str(j):measures(a['pred'],a['bsq'],a['mask']&(a['finger']==j)) for j in range(10)}
            row['BSQ_force_bands']={name:measures(a['pred'],a['bsq'],a['mask']&(a['bsq']>=lo)&(a['bsq']<hi))
                for name,lo,hi in [('0-2',0,2),('2-10',2,10),('10-25',10,25),('25+',25,float('inf'))]}
            taus=[]
            for j in range(10):
                mask=a['mask']&(a['finger']==j);x=a['pred'][mask];y=a['bsq'][mask]
                if len(x)>=8 and np.ptp(x)>0 and np.ptp(y)>0:taus.append(float(kendalltau(x,y).statistic))
            row['tau_BSQ_finger_mean']=float(np.mean(taus)) if taus else None
        results.append(row)
        if save_predictions:
            dest=Path(root)/'runs'/branch/'test_predictions';dest.mkdir(exist_ok=True)
            np.savez_compressed(dest/(id+'.npz'),**a)
    if not results:raise ValueError('Empty evaluation')
    aggregate=dict(records=len(results),loss=float(np.mean([r['loss'] for r in results])),rows=results)
    net.train();return aggregate

def optimizer(net,c):
    groups={}
    for name,p in net.named_parameters():
        key=(name.startswith('output_head.'),p.ndim>1 and not name.endswith('.bias'))
        groups.setdefault(key,[]).append(p)
    return torch.optim.AdamW([dict(params=ps,lr=c['lr_head'] if head else c['lr_body'],
        base_lr=c['lr_head'] if head else c['lr_body'],weight_decay=c['weight_decay'] if decay else 0.)
        for (head,decay),ps in groups.items()],betas=(.9,.999),eps=1e-8)

def save(path,net,opt,state,binding):
    payload=dict(model=net.state_dict(),optimizer=opt.state_dict(),state=state,binding=binding,
                 torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
    tmp=path.with_suffix('.partial.pt');torch.save(payload,tmp);os.replace(tmp,path)

def physical_forward_windows(c,global_update,logical_microbatch):
    schedule=c.get('physical_forward_schedule',[dict(start_global_update=0,windows=logical_microbatch)])
    starts=[s['start_global_update'] for s in schedule]
    if starts!=sorted(set(starts)) or starts[0]!=0:raise ValueError('Invalid common forward schedule')
    if any(s['windows']<logical_microbatch or s['windows']%logical_microbatch for s in schedule):
        raise ValueError('Physical forwards must preserve complete logical loss groups')
    return next(s['windows'] for s in reversed(schedule) if global_update>=s['start_global_update'])

def run(root,branch,smoke=False):
    root=Path(root);c=settings();torch.set_num_threads(2)
    for key,name in [('CUDA_CACHE_PATH','cuda_kernels'),('TORCH_HOME','torch'),('XDG_CACHE_HOME','xdg')]:
        path=root/'cache'/name;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    if not torch.cuda.is_available():raise RuntimeError('GPU required for actual trainer')
    device=torch.device('cuda:0')
    total=torch.cuda.get_device_properties(device).total_memory
    limit=float(os.environ.get('TASK103_GPU_MEMORY_GIB',c['gpu_memory_limit_GiB']))
    torch.cuda.set_per_process_memory_fraction(limit*2**30/total,0)
    random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed']);torch.cuda.manual_seed(c['seed'])
    net=network();init=root/'assets/common_stage2_initial_state_dict.pt'
    if sha(init)!=c['common_initial_sha256']:raise ValueError('Common initialization hash mismatch')
    if sha(EXPERIMENT/'vendor/model_source.py')!=c['model_source_sha256']:raise ValueError('Model source hash mismatch')
    net.load_state_dict(torch.load(init,map_location='cpu',weights_only=True),strict=True);net.to(device)
    if not all(p.requires_grad for p in net.parameters()):raise ValueError('Unexpected frozen parameter')
    ds=Windows(root,'train',branch);kind=ds.kind;norm=ds.data['force_norm'][kind]
    mean=torch.tensor(norm['mean'],device=device,dtype=torch.float32)[None,:,None]
    std=torch.tensor(norm['std'],device=device,dtype=torch.float32)[None,:,None]
    out=root/'runs'/branch;out.mkdir(parents=True,exist_ok=True)
    if smoke:out=root/'checks'/('smoke_'+branch);out.mkdir(parents=True,exist_ok=True)
    micro=int(os.environ.get('TASK103_MICROBATCH',c['microbatch']))
    if micro!=c['common_microbatch']:raise ValueError('All branches require the fixed common logical loss group')
    binding=dict(config_sha256=sha(EXPERIMENT/'config.json'),dataset_sha256=sha(root/'prepared/dataset.json'),
                 source_code=source_binding(),
                 common_initial_sha256=sha(init),branch=branch,force_norm=norm,
                 effective_microbatch=micro,EMG_norm_sha256=sha(EXPERIMENT/'reference/stage1_emg_train_norm.npz'),
                 physical_forward_schedule=c.get('physical_forward_schedule',[dict(start_global_update=0,windows=micro)]))
    atomic(out/'CONTRACT.json',dict(**binding,configuration=c))
    opt=optimizer(net,c);state=dict(epoch=0,next_update=0,global_update=0,best=None,early_reference=None,bad_epochs=0,history=[])
    last=out/'last.pt';best=out/'best.pt'
    if last.exists() and not smoke:
        ck=torch.load(last,map_location='cpu',weights_only=True)
        if ck['binding']!=binding:raise ValueError('Resume binding mismatch')
        net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);state=ck['state']
        torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state(ck['cuda_rng']);del ck
    updates=math.ceil(len(ds)/c['global_batch']);total_updates=updates*c['max_epochs']
    for epoch in range(state['epoch'],c['max_epochs']):
        # A retry during final Test must not restart training after early stop.
        if state['bad_epochs']>=c['patience']:break
        order=np.random.default_rng(c['seed']+epoch).permutation(len(ds))
        sampler=order[state['next_update']*c['global_batch']:].tolist()
        if smoke:sampler=sampler[:int(os.environ.get('TASK103_SMOKE_WINDOWS',micro))]
        loader=DataLoader(ds,batch_size=c['global_batch'],sampler=sampler,num_workers=c['train_workers'],pin_memory=True,
                          timeout=120 if c['train_workers'] else 0)
        tick=time.monotonic();net.train()
        for u,batch in enumerate(loader,start=state['next_update']):
            x,y,valid=batch;den=int(valid.sum());opt.zero_grad(set_to_none=True);value=0.
            physical=physical_forward_windows(c,state['global_update'],micro)
            for start in range(0,len(x),physical):
                xx=x[start:start+physical].to(device,non_blocking=True);yy=y[start:start+physical].to(device,non_blocking=True)
                mask=valid[start:start+physical].to(device,non_blocking=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):z=net(xx)
                pred=z.float()*std+mean
                if physical==micro:
                    # Keep the original operation/RNG order through the common
                    # epoch-3 boundary, including exact prefix resume semantics.
                    loss,count=objective(pred,yy,mask,branch)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                    (loss*(count/max(1,den))).backward();value+=float(loss.detach())*count/max(1,den)
                else:
                    terms=[]
                    for lo in range(0,len(xx),micro):
                        loss,count=objective(pred[lo:lo+micro],yy[lo:lo+micro],mask[lo:lo+micro],branch)
                        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                        terms.append(loss*(count/max(1,den)))
                    combined=sum(terms);combined.backward();value+=float(combined.detach())
            step=state['global_update'];warm=c['warmup_updates']
            factor=(step+1)/warm if step<warm else .1+.9*(1+math.cos(math.pi*min(1.,(step-warm)/max(1,total_updates-warm-1))))/2
            for g in opt.param_groups:g['lr']=g['base_lr']*factor
            grad=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True));opt.step()
            state.update(next_update=u+1,global_update=step+1)
            status=dict(phase='training',branch=branch,epoch=epoch+1,epoch_update=u+1,updates_per_epoch=updates,
                global_update=step+1,loss=value,valid_points=den,gradient_norm=grad,microbatch=micro,
                physical_forward_windows=physical,
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,elapsed_epoch_s=time.monotonic()-tick)
            atomic(out/'STATUS.json',status)
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(status)+'\n')
            if smoke:
                atomic(out/'PASS.json',{**status,'phase':'real_data_GPU_smoke_passed','test_metrics_read':False})
                print(json.dumps(status),flush=True);return
            if (step+1)%c['save_every_updates']==0:save(last,net,opt,state,binding)
        save(last,net,opt,state,binding)
        result=evaluate(net,root,branch,'val',mean,std,device,micro)
        loss=result['loss'];is_best=state['best'] is None or loss<state['best']
        if is_best:state['best']=loss
        if state['early_reference'] is None or loss<state['early_reference']-c['min_delta']:
            state['early_reference']=loss;state['bad_epochs']=0
        else:state['bad_epochs']+=1
        state['history'].append(dict(epoch=epoch+1,val_loss=loss,global_update=state['global_update']))
        atomic(out/f'epoch{epoch+1:03d}_val.json',result)
        state.update(epoch=epoch+1,next_update=0)
        if is_best:save(best,net,opt,state,binding)
        # Persist best before the epoch-advanced resume state, so interruption
        # cannot claim a new minimum whose weights were never saved.
        save(last,net,opt,state,binding)
        if state['bad_epochs']>=c['patience']:break
    checkpoint=torch.load(best,map_location='cpu',weights_only=True);net.load_state_dict(checkpoint['model']);del checkpoint
    atomic(out/'STATUS.json',dict(phase='final_test',branch=branch,selected_checkpoint_sha256=sha(best)))
    result=evaluate(net,root,branch,'test',mean,std,device,micro,save_predictions=True)
    atomic(out/'final_test.json',result)
    atomic(out/'DONE.json',dict(phase='complete',branch=branch,best_sha256=sha(best),history=state['history'],
        test_recordings=result['records'],test_not_used_for_selection=True,binding=binding))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--branch',choices=BRANCHES,required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args();run(a.root,a.branch,a.smoke)
