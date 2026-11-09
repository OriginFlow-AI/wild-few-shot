"""Eight-GPU AP304 full-raw ranking pretraining and automatic final Val/Test."""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime
import json
import math
import os
import platform
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from scipy.stats import kendalltau
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .ap304_rank_data import (
    GroupDataset, TrialDataset, atomic_json, build_model, load_index, load_spec,
    make_pairs, make_plan, prepare, seed_for, sha,
)
from .ap304_rank_loss import ranking_objective


def capture_rng():
    n = np.random.get_state()
    return dict(python=random.getstate(), numpy=(n[0], n[1].tolist(), n[2], n[3], n[4]),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state(state['cuda'].cpu())


def distributed_setup():
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.set_num_threads(2)
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', timeout=datetime.timedelta(hours=12))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return rank, dist.get_world_size(), torch.device('cuda', local)


def move(batch, device):
    return {k: torch.as_tensor(v).to(device, non_blocking=True) for k, v in batch.items()}


def autocast():
    return torch.autocast('cuda', dtype=torch.bfloat16)


def score_model(net, x):
    groups = len(x)
    with autocast():
        out = net(x.reshape(-1, 16, 4000))
    # Do not de-normalize arbitrary ranking scores as newtons.
    return out.float().reshape(groups, 2, 5, 89) / 0.01


def optimizer_for(net, cfg):
    decay, no_decay = [], []
    for name, parameter in net.named_parameters():
        (no_decay if parameter.ndim <= 1 or name.endswith('.bias') else decay).append(parameter)
    o = cfg['optimization']
    return torch.optim.AdamW([dict(params=decay, weight_decay=o['weight_decay']),
                             dict(params=no_decay, weight_decay=0.0)], lr=o['learning_rate'],
                            betas=tuple(o['betas']), eps=o['epsilon'])


def learning_rate(cfg, completed):
    o = cfg['optimization']
    warm = o['warmup_optimizer_updates']
    total = cfg['execution']['max_optimizer_updates']
    if completed < warm:
        factor = (completed + 1) / warm
    else:
        ratio = min(1.0, (completed - warm) / max(1, total - warm - 1))
        factor = o['minimum_lr_ratio'] + (1 - o['minimum_lr_ratio']) * (1 + math.cos(math.pi * ratio)) / 2
    return o['learning_rate'] * factor


def tau_for_selection(y, pred):
    if np.ptp(y) == 0:
        return None, None
    if np.ptp(pred) == 0:
        return None, 0.0
    tau = float(kendalltau(y, pred, variant='b').statistic)
    if not math.isfinite(tau):
        raise ValueError('unexpected nonfinite tau for nonconstant vectors')
    return tau, tau


@torch.no_grad()
def force_diagnostic_statistics(scores, batch, cfg):
    """Additive window/finger diagnostics; MAPE compares uncalibrated scores to raw."""
    x, y = scores.detach().float(), batch['y'].float()
    real = (batch['starts'] >= 0)[:, :, None].expand_as(x[..., 0])
    centered_x, centered_y = x-x.mean(-1,keepdim=True), y-y.mean(-1,keepdim=True)
    vx, vy = centered_x.square().sum(-1), centered_y.square().sum(-1)
    varying_y = y.amax(-1) > y.amin(-1)
    varying_x = x.amax(-1) > x.amin(-1)
    corr_mask = real & varying_y
    corr = (centered_x*centered_y).sum(-1)/(vx*vy).sqrt().clamp_min(1e-30)
    corr = torch.where(varying_x,corr.clamp(-1,1),torch.zeros_like(corr))
    threshold=cfg['evaluation'].get('diagnostic_mape_min_raw_abs',0.5)
    active=(y.abs()>=threshold) & real[...,None]
    count=active.sum(-1)
    ape=(x-y).abs()/y.abs().clamp_min(threshold)*100.0
    mape=(ape*active).sum(-1)/count.clamp_min(1)
    mape_mask=real & (count>0)
    return torch.stack([(corr*corr_mask).sum(),corr_mask.sum().float(),
                        (mape*mape_mask).sum(),mape_mask.sum().float(),
                        (real & ~varying_y).sum().float(),
                        (corr_mask & ~varying_x).sum().float(),
                        active.sum().float(),real.sum().float()*x.shape[-1]])


def trial_force_diagnostics(y, pred, threshold=0.5):
    y,pred=np.asarray(y,np.float64),np.asarray(pred,np.float64)
    correlations,mapes,active_counts=[],[],[]
    for f in range(5):
        truth,estimate=y[f],pred[f]
        if np.ptp(truth)==0:
            corr=None
        elif np.ptp(estimate)==0:
            corr=0.0
        else:
            a,b=truth-truth.mean(),estimate-estimate.mean()
            corr=float(np.clip(np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b)),-1,1))
        active=np.abs(truth)>=threshold
        mape=float(np.mean(np.abs(estimate[active]-truth[active])/np.abs(truth[active]))*100) if active.any() else None
        correlations.append(corr);mapes.append(mape);active_counts.append(int(active.sum()))
    return dict(corr=correlations,mape_score_vs_raw_percent=mapes,mape_active_points=active_counts)


def append_epoch_metrics(out, cfg, epoch, progress, update, split, label, corr, mape, tau,
                         corr_count, mape_count, aggregation, loss=None):
    path=Path(out)/'epoch_metrics.csv'
    row=dict(epoch=epoch,epoch_progress=progress,global_update=update,split=split,label=label,
             corr=corr,mape_score_vs_raw_percent=mape,tau_b=tau,rank_objective=loss,
             corr_count=corr_count,mape_count=mape_count,aggregation=aggregation,
             mape_min_raw_abs=cfg['evaluation'].get('diagnostic_mape_min_raw_abs',0.5),
             mape_is_physical_force_error=False)
    exists=path.exists() and path.stat().st_size>0
    with path.open('a',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(row))
        if not exists:writer.writeheader()
        writer.writerow(row)


def stitch_trial(predictions, targets, starts):
    order = np.argsort(starts)
    pp, yy, tt = [], [], []
    last = -1
    for i in order:
        if starts[i] < 0:
            continue
        ticks = starts[i] + 470 + 40 * np.arange(89)
        keep = ticks > last
        if np.any(keep):
            pp.append(predictions[i][:, keep]); yy.append(targets[i][:, keep]); tt.append(ticks[keep])
            last = int(ticks[keep][-1])
    return np.concatenate(pp, 1), np.concatenate(yy, 1), np.concatenate(tt)


def summarize_eval(records, statistics, primary_stages):
    stage = {}
    groups = {key: {} for key in ('stage', 'hand', 'finger', 'subject')}
    undefined_target = constant_prediction = 0
    for row in records:
        for finger, (tau, selection) in enumerate(zip(row['tau_b'], row['selection_tau_b'])):
            if selection is None:
                undefined_target += 1
                continue
            constant_prediction += tau is None
            for axis, label in [('stage', str(row['stage'])), ('hand', row['hand']),
                                ('finger', str(finger)), ('subject', row['subject'])]:
                values = groups[axis].setdefault(label, [0.0, 0])
                values[0] += selection; values[1] += 1
    breakdown = {axis: {key: dict(tau_b=value[0]/value[1], count=value[1]) for key,value in values.items()}
                 for axis,values in groups.items()}
    for s in primary_stages:
        if str(s) in breakdown['stage']:
            stage[str(s)] = breakdown['stage'][str(s)]['tau_b']
    if not stage:
        raise ValueError('all validation targets undefined')
    a = statistics
    mean = a[8] / max(a[10],1)
    diagnostic={}
    for source,key,count_key in [('corr','corr_stage_macro','corr_valid_entries'),
                                  ('mape_score_vs_raw_percent','mape_score_vs_raw_percent_stage_macro','mape_valid_entries')]:
        values={};count=0
        for row in records:
            for value in row.get(source,[]):
                if value is not None:
                    values.setdefault(int(row['stage']),[]).append(value);count+=1
        per_stage={str(s):float(np.mean(v)) for s,v in values.items()}
        selected=[value for s,value in per_stage.items() if int(s) in primary_stages]
        diagnostic[key]=float(np.mean(selected)) if selected else None
        diagnostic[key+'_by_stage']=per_stage
        diagnostic[count_key]=count
    return dict(tau_stage_macro=float(np.mean(list(stage.values()))), trials=len(records),
                stage_metrics=stage, breakdown=breakdown, undefined_target_entries=undefined_target,
                constant_prediction_entries=constant_prediction,
                group_objective=a[0]/max(a[1],1), valid_groups=int(a[1]), total_groups=int(a[11]),
                supervised_group_fraction=a[1]/max(a[11],1),
                within_pair_accuracy=a[2]/a[3] if a[3] else None,
                cross_pair_accuracy=a[4]/a[5] if a[5] else None,
                pair_logistic_loss=a[6]/a[7] if a[7] else None,
                local_pairs=int(a[3]), cross_pairs=int(a[5]),
                score_mean=mean, score_std=math.sqrt(max(0,a[9]/max(a[10],1)-mean**2)),**diagnostic)


@torch.inference_mode()
def evaluate(net, cfg, rows, split, rank, world, device, label, limit=None,
             epoch=0,global_update=0,epoch_progress=0.0):
    net.eval()
    out = Path(cfg['workspace']) / cfg['artifacts']['root']
    directory = out / 'evaluations' / label
    if rank == 0:
        directory.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    ds = TrialDataset(cfg, rows, split, rank, world, limit)
    generator = torch.Generator().manual_seed(seed_for(43, split, rank, 'eval_loader') % (2**63-1))
    workers = cfg['execution']['validation_workers_per_device']
    loader = DataLoader(ds, batch_size=None, num_workers=workers, pin_memory=True,
                        multiprocessing_context='spawn' if workers else None, generator=generator)
    all_stats = torch.zeros(12, dtype=torch.float64, device=device)
    records = []
    abs_max = 0.0
    started = time.time()
    path = directory / f'rank{rank}.jsonl'
    with path.open('w') as handle:
        for number, item in enumerate(loader):
            trial = int(item['trial'])
            source = rows[trial]
            group_batch = item['groups']
            pred = []
            for begin in range(0, len(group_batch['x']), cfg['execution']['groups_per_device_microbatch']):
                b = move({k:v[begin:begin+cfg['execution']['groups_per_device_microbatch']] for k,v in group_batch.items()},device)
                scores = score_model(net, b['x'])
                if not torch.isfinite(scores).all():
                    raise ValueError('nonfinite evaluation scores')
                _,_,s = ranking_objective(scores,b,cfg)
                all_stats += s.double()
                abs_max = max(abs_max,float(scores.abs().max()))
                pred.append(scores.cpu().numpy())
            p = np.concatenate(pred).reshape(-1,5,89)
            y = group_batch['y'].numpy().reshape(-1,5,89)
            starts = group_batch['starts'].numpy().reshape(-1)
            p,y,_ = stitch_trial(p,y,starts)
            tau,selection = zip(*(tau_for_selection(y[f],p[f]) for f in range(5)))
            row = dict(trial_key=source['trial_key'], subject=source['subject'], stage=int(source['stage']),
                       hand=source.get('hand',source.get('side','unknown')), tau_b=list(tau),
                       selection_tau_b=list(selection),native_points=p.shape[1],
                       **trial_force_diagnostics(y,p,cfg['evaluation'].get('diagnostic_mape_min_raw_abs',0.5)))
            records.append(row);handle.write(json.dumps(row,allow_nan=False)+'\n')
            if (number+1)%50==0:
                handle.flush()
                print(json.dumps(dict(phase='evaluating',split=split,label=label,rank=rank,
                                      trials=number+1,total=len(ds),seconds=round(time.time()-started,1))),flush=True)
    dist.all_reduce(all_stats)
    max_tensor=torch.tensor(abs_max,device=device);dist.all_reduce(max_tensor,op=dist.ReduceOp.MAX)
    # Only JSON records are gathered, never large prediction arrays.
    gathered=[None]*world if rank==0 else None
    dist.gather_object(records,gathered,dst=0)
    metric=torch.zeros(1,device=device,dtype=torch.float64)
    if rank==0:
        all_records=[r for part in gathered for r in part]
        expected=len(rows) if limit is None else sum(min(limit,len(rows[r::world])) for r in range(world))
        if len(all_records)!=expected or len({r['trial_key'] for r in all_records})!=expected:
            raise ValueError('evaluation dropped or duplicated trials')
        result=summarize_eval(all_records,all_stats.cpu().tolist(),cfg['evaluation']['primary_stage_values'])
        result.update(split=split,label=label,output_units='dimensionless_ranking_score',
                      absolute_force_metrics=False,score_abs_max=float(max_tensor),seconds=time.time()-started,
                      epoch=epoch,epoch_progress=epoch_progress,global_update=global_update,
                      mape_diagnostic=dict(prediction='uncalibrated_dimensionless_score',target='raw_force',
                          min_raw_abs=cfg['evaluation'].get('diagnostic_mape_min_raw_abs',0.5),
                          units_comparable=False,physical_force_error=False))
        atomic_json(directory/'summary.json',result)
        append_epoch_metrics(out,cfg,epoch,epoch_progress,global_update,split,label,
                             result['corr_stage_macro'],result['mape_score_vs_raw_percent_stage_macro'],
                             result['tau_stage_macro'],result['corr_valid_entries'],result['mape_valid_entries'],
                             'stage_macro_of_trial_finger_means',result['group_objective'])
        metric[0]=result['tau_stage_macro']
        print(json.dumps(dict(phase='evaluation_complete',split=split,label=label,
                              epoch=epoch,global_update=global_update,
                              tau_stage_macro=result['tau_stage_macro'],corr=result['corr_stage_macro'],
                              mape_score_vs_raw_percent=result['mape_score_vs_raw_percent_stage_macro'],
                              trials=len(all_records))),flush=True)
    dist.broadcast(metric,src=0)
    net.train()
    return float(metric)


def save_checkpoint(net, optimizer, state, cfg, path, rank, world, extra_paths=()):
    local=capture_rng();rng=[None]*world if rank==0 else None
    dist.gather_object(local,rng,dst=0)
    if rank==0:
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(path.parent).free < 2*1024**3:
            raise RuntimeError('less than 2 GiB free before checkpoint')
        payload=dict(model=net.state_dict(),optimizer=optimizer.state_dict(),state=state,
                     rng_per_rank=rng,config=cfg,output_units='dimensionless_ranking_score',
                     model_source_sha256=cfg['architecture']['source_sha256'])
        tmp=path.with_name(path.name+'.tmp')
        torch.save(payload,tmp);os.replace(tmp,path)
        for other in extra_paths:
            other=Path(other);temp=other.with_name(other.name+'.tmp')
            shutil.copyfile(path,temp);os.replace(temp,other)
    dist.barrier()


def initial_state():
    return dict(epoch=0,next_update=0,global_update=0,best_tau=-2.0,
                early_stop_reference=-2.0,rounds_without_improvement=0,stop=False,epoch_statistics=[0.0]*20)


def record_training(out,state,stats,lr,seconds,rank,grad_norm):
    if rank:
        return
    a=stats.cpu().tolist()
    row=dict(phase='training',epoch=state['epoch']+1,update_in_epoch=state['next_update'],
             global_update=state['global_update'],loss=a[0]/max(a[1],1),valid_groups=int(a[1]),
             local_pair_accuracy=a[2]/a[3] if a[3] else None,
             cross_pair_accuracy=a[4]/a[5] if a[5] else None,
             score_std=math.sqrt(max(0,a[9]/max(a[10],1)-(a[8]/max(a[10],1))**2)),
             corr=a[12]/a[13] if a[13] else None,
             mape_score_vs_raw_percent=a[14]/a[15] if a[15] else None,
             learning_rate=lr,grad_norm=grad_norm,seconds=seconds)
    with (out/'train_metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    atomic_json(out/'status.json',row)
    print(json.dumps(row,allow_nan=False),flush=True)


def train(cfg, smoke=False):
    rank,world,device=distributed_setup()
    if world!=cfg['execution']['devices']:
        raise ValueError('world size does not match executable configuration')
    seed=cfg['experiment']['seed']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed(seed)
    net=build_model(cfg).to(device)
    model=DDP(net,device_ids=[device.index],broadcast_buffers=False)
    torch.manual_seed(seed+rank);torch.cuda.manual_seed(seed+rank)
    optimizer=optimizer_for(net,cfg)
    out=Path(cfg['workspace'])/cfg['artifacts']['root']
    if smoke:
        out=out/'smoke'
    if rank==0:out.mkdir(parents=True,exist_ok=True)
    dist.barrier()
    state=initial_state()
    checkpoint=out/'checkpoints/last.pt'
    if checkpoint.exists() and not smoke:
        ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if ck['config']!=cfg or ck['output_units']!='dimensionless_ranking_score':
            raise ValueError('resume config mismatch')
        net.load_state_dict(ck['model'],strict=True);optimizer.load_state_dict(ck['optimizer'])
        state=ck['state'];restore_rng(ck['rng_per_rank'][rank]);del ck
    if rank==0:
        atomic_json(out/'environment_and_architecture.json',dict(torch=torch.__version__,numpy=np.__version__,
                    python=platform.python_version(),gpu=torch.cuda.get_device_name(),world_size=world,
                    parameters=sum(p.numel() for p in net.parameters()),encoder_frozen=False))
    if smoke:
        run_smoke(model,net,optimizer,cfg,out,rank,world,device)
        dist.destroy_process_group();return
    complete=out/('prepared/TRAIN_READY.json' if cfg['execution'].get('overlap_validation_preparation')
                  else 'prepared/COMPLETE.json')
    if not complete.exists():raise ValueError('raw preparation incomplete')
    ready=json.loads(complete.read_text())
    if ready['index_sha256']!=cfg['dataset']['index_sha256'] or ready['norm_sha256']!=sha(Path(cfg['workspace'])/cfg['normalization']['output']):
        raise ValueError('preparation binding mismatch')
    index=load_index(cfg)
    if state['global_update']==0 and cfg['evaluation'].get('initial_validation',True):
        if rank==0:atomic_json(out/'status.json',dict(phase='initial_validation',global_update=0))
        evaluate(net,cfg,index['val'],'val',rank,world,device,'initial_val')
    execution=cfg['execution'];accumulate=execution['accumulate_microbatches']
    updates=execution['expected_optimizer_updates_per_epoch']
    validation_at={math.ceil(updates*f) for f in cfg['evaluation']['fractions_of_each_train_epoch']}
    while state['epoch']<execution['max_epochs'] and not state['stop']:
        epoch=state['epoch'];start_update=state['next_update']
        plan_path=out/'plans'/f'epoch{epoch:03d}.npy'
        if rank==0 and not plan_path.exists():
            plan=make_plan(index['train'],seed,epoch,world,execution['groups_per_device_microbatch'],accumulate)
            if len(plan)//execution['effective_groups_per_update']!=updates:
                raise ValueError('unexpected number of full-data optimizer updates')
            plan_path.parent.mkdir(exist_ok=True);tmp=plan_path.with_suffix('.tmp.npy');np.save(tmp,plan);os.replace(tmp,plan_path)
        dist.barrier()
        ds=GroupDataset(cfg,index['train'],plan_path,epoch,rank,world,start_update)
        generator=torch.Generator().manual_seed(seed_for(seed,epoch,rank,'train_loader')%(2**63-1))
        workers=execution['train_workers_per_device']
        loader=DataLoader(ds,batch_size=execution['groups_per_device_microbatch'],shuffle=False,
                          num_workers=workers,pin_memory=True,persistent_workers=workers>0,
                          multiprocessing_context='spawn' if workers else None,generator=generator)
        iterator=iter(loader);model.train()
        epoch_statistics=torch.tensor(state.get('epoch_statistics',[0.0]*20),dtype=torch.float64,device=device)
        for update in range(start_update,updates):
            tick=time.time();optimizer.zero_grad(set_to_none=True)
            statistics=torch.zeros(20,dtype=torch.float64,device=device)
            lr=learning_rate(cfg,state['global_update'])
            for pg in optimizer.param_groups:pg['lr']=lr
            for micro in range(accumulate):
                batch=move(next(iterator),device)
                ctx=model.no_sync() if micro<accumulate-1 else contextlib.nullcontext()
                with ctx:
                    scores=score_model(model,batch['x'])
                    loss_sum,_,s=ranking_objective(scores,batch,cfg)
                    if not torch.isfinite(loss_sum):raise ValueError('nonfinite ranking loss')
                    loss_sum.backward()
                diagnostic=force_diagnostic_statistics(scores,batch,cfg)
                statistics+=torch.cat((s,diagnostic)).double()
            dist.all_reduce(statistics)
            epoch_statistics+=statistics
            denominator=float(statistics[1])
            grad_norm=0.0
            if denominator:
                factor=world/denominator
                for p in model.parameters():
                    if p.grad is not None:p.grad.mul_(factor)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['optimization']['gradient_clip_norm'],error_if_nonfinite=True)
                grad_norm=float(norm);optimizer.step()
            state['global_update']+=1;state['next_update']=update+1
            if state['global_update']%20==0 or state['global_update']==1:
                record_training(out,state,statistics,lr,time.time()-tick,rank,grad_norm)
            validate=update+1 in validation_at
            is_best=False
            if validate:
                if rank==0:atomic_json(out/'status.json',dict(phase='validation',**state))
                rng=capture_rng()
                tau=evaluate(net,cfg,index['val'],'val',rank,world,device,f'update{state["global_update"]:06d}_val',
                             epoch=epoch+1,global_update=state['global_update'],epoch_progress=(update+1)/updates)
                restore_rng(rng)
                is_best=tau>state['best_tau']
                if is_best:state['best_tau']=tau
                if tau>state['early_stop_reference']+cfg['evaluation']['early_stop_min_delta']:
                    state['early_stop_reference']=tau;state['rounds_without_improvement']=0
                else:state['rounds_without_improvement']+=1
                completed_epochs=epoch+(update+1)/updates
                state['stop']=(completed_epochs>=execution['min_epochs_before_early_stop'] and
                               state['rounds_without_improvement']>=cfg['evaluation']['early_stop_patience_validation_rounds'])
            end_epoch=update+1==updates
            if end_epoch or state['stop']:
                if rank==0:
                    a=epoch_statistics.cpu().tolist()
                    append_epoch_metrics(out,cfg,epoch+1,(update+1)/updates,state['global_update'],'train',
                                         'train_epoch' if end_epoch else 'train_epoch_partial',
                                         a[12]/a[13] if a[13] else None,a[14]/a[15] if a[15] else None,None,
                                         int(a[13]),int(a[15]),'window_finger_macro',a[0]/max(a[1],1))
                    print(json.dumps(dict(phase='epoch_complete' if end_epoch else 'epoch_partial',epoch=epoch+1,
                                          corr=a[12]/a[13] if a[13] else None,
                                          mape_score_vs_raw_percent=a[14]/a[15] if a[15] else None)),flush=True)
            if end_epoch:
                state['epoch']=epoch+1;state['next_update']=0
                state['epoch_statistics']=[0.0]*20
            elif validate or state['global_update']%1000==0:
                state['epoch_statistics']=epoch_statistics.cpu().tolist()
            if validate or state['global_update']%1000==0 or end_epoch:
                extra=[]
                if is_best:extra.append(out/'checkpoints/best_tau.pt')
                if end_epoch:extra.append(out/f'checkpoints/epoch{epoch+1:03d}.pt')
                save_checkpoint(net,optimizer,state,cfg,checkpoint,rank,world,extra)
            if state['stop']:break
        del iterator,loader,ds
    # Freeze the Val-selected best model before touching Test labels.
    dist.barrier()
    best=out/'checkpoints/best_tau.pt'
    ck=torch.load(best,map_location='cpu',weights_only=True)
    net.load_state_dict(ck['model'],strict=True)
    selected=ck['state'];del ck
    if rank==0:
        atomic_json(out/'status.json',dict(phase='final_evaluation',selected=selected))
        atomic_json(out/'selected_checkpoint.json',dict(path=str(best),sha256=sha(best),state=selected,
                    selection='full Val stage-macro tau-b; Test not used for selection'))
    post=cfg.get('post_training',{})
    if post.get('enabled'):
        for split in post['splits']:
            evaluate(net,cfg,index[split],split,rank,world,device,'final_'+split,
                     epoch=selected['epoch']+int(selected['next_update']>0),global_update=selected['global_update'],
                     epoch_progress=(selected['next_update']/updates if selected['next_update'] else 1.0))
    if rank==0:atomic_json(out/'status.json',dict(phase='complete',state=state,selected_checkpoint=str(best),
                         final_validation=str(out/'evaluations/final_val/summary.json'),
                         final_test=str(out/'evaluations/final_test/summary.json')))
    dist.destroy_process_group()


def run_smoke(model,net,optimizer,cfg,out,rank,world,device):
    # Actual AP304, actual per-GPU batch and accumulation; synthetic labels only.
    torch.manual_seed(800+rank)
    statistics=torch.zeros(12,dtype=torch.float64,device=device)
    began=time.time()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True);statistics.zero_()
        for micro in range(cfg['execution']['accumulate_microbatches']):
            groups=[]
            for g in range(cfg['execution']['groups_per_device_microbatch']):
                y=np.broadcast_to(np.arange(178,dtype=np.float32).reshape(2,1,89),(2,5,89)).copy()
                pairs=make_pairs(y,[0,4000],np.full(5,100.),np.zeros(5),np.random.default_rng(seed_for(rank,step,micro,g)),cfg)
                groups.append(dict(x=np.random.default_rng(seed_for(rank,step,micro,g,'x')).normal(size=(2,16,4000)).astype(np.float32),
                                   starts=np.array([0,4000]),**pairs))
            b=move({k:np.stack([g[k] for g in groups]) for k in groups[0]},device)
            ctx=model.no_sync() if micro<cfg['execution']['accumulate_microbatches']-1 else contextlib.nullcontext()
            with ctx:
                loss,_,s=ranking_objective(score_model(model,b['x']),b,cfg);loss.backward()
            statistics+=s.double()
        dist.all_reduce(statistics)
        for p in model.parameters():
            if p.grad is not None:p.grad.mul_(world/float(statistics[1]))
        enc=sum(float(p.grad.abs().sum()) for n,p in net.named_parameters() if n.startswith('encoder_blocks.') and p.grad is not None)
        dec=sum(float(p.grad.abs().sum()) for n,p in net.named_parameters() if n.startswith('decoder_blocks.') and p.grad is not None)
        if not enc>0 or not dec>0:raise AssertionError('encoder/decoder gradients missing')
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);optimizer.step()
    state=initial_state();state['global_update']=2
    path=out/'checkpoints/last.pt'
    save_checkpoint(net,optimizer,state,cfg,path,rank,world)
    ck=torch.load(path,map_location='cpu',weights_only=True)
    for name,tensor in net.state_dict().items():
        if not torch.equal(tensor.cpu(),ck['model'][name]):raise AssertionError('checkpoint mismatch')
    restore_rng(ck['rng_per_rank'][rank])
    if rank==0:
        result=dict(status='PASS',world_size=world,updates=2,parameters=sum(p.numel() for p in net.parameters()),
                    effective_window_slots=cfg['execution']['effective_window_slots_per_update'],
                    all_rank_valid_groups=float(statistics[1]),loss=float(statistics[0]/statistics[1]),
                    encoder_gradient_nonzero=enc>0,decoder_gradient_nonzero=dec>0,
                    grad_norm=float(norm),checkpoint_safe_reload=True,
                    peak_memory_gb=torch.cuda.max_memory_allocated()/1024**3,seconds=time.time()-began)
        atomic_json(out/'PASS.json',result);print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--mode',choices=['prepare','train','smoke'],default='train')
    parser.add_argument('--prepare-workers',type=int,default=8)
    args=parser.parse_args();cfg=load_spec(args.config)
    if args.mode=='prepare':prepare(cfg,args.prepare_workers)
    else:train(cfg,smoke=args.mode=='smoke')


if __name__=='__main__':main()
