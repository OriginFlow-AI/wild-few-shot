"""Stage-2 regression from the best ranking checkpoint; six recoverable reports."""
from __future__ import annotations

import argparse
import contextlib
import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from scipy.stats import kendalltau

from .ap304_rank_data import atomic_json, sha, window_starts
from .ap304_stage2_data import TrainWindows, WindowReader, indices, load_config, preparation_binding, window_counts
from .losses import force_objective, force_objective_from_statistics, force_sufficient_statistics, high_force_mae_weights, force_huber_loss, paired_force_difference_loss
from .ap304_six_metrics import moments, summarize_six, METHODS


def network(c):
    spec=importlib.util.spec_from_file_location('stage2_ap304_model',c['model_source'])
    module=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module.K2RingNetwork(module.K2RingConfig())


def rng_state():
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=(n[0],n[1].tolist(),n[2],n[3],n[4]),
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())


def restore_rng(s):
    random.setstate(s['python']);n=s['numpy']
    np.random.set_state((n[0],np.asarray(n[1],np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(s['torch'].cpu());torch.cuda.set_rng_state(s['cuda'].cpu())


def initialize(net,c):
    p=Path(c['source_checkpoint'])
    if sha(p)!=c['source_checkpoint_sha256']:raise ValueError('source checkpoint hash mismatch')
    ck=torch.load(p,map_location='cpu',weights_only=True)
    if ck['output_units']!='dimensionless_ranking_score':raise ValueError('wrong source checkpoint domain')
    if any(ck['state'].get(k)!=v for k,v in c['source_checkpoint_state'].items()):
        raise ValueError('source is not the frozen best ranking checkpoint')
    net.load_state_dict(ck['model'],strict=True)
    net.output_head.reset_parameters()
    with torch.no_grad():net.output_head.bias[10:].fill_(4.0)
    loaded=net.state_dict()
    assert all(torch.equal(v,ck['model'][k]) for k,v in loaded.items() if not k.startswith('output_head.'))
    assert any(not torch.equal(v,ck['model'][k]) for k,v in loaded.items() if k.startswith('output_head.'))
    assert all(p.requires_grad for p in net.parameters())
    return dict(source=str(p),source_sha256=c['source_checkpoint_sha256'],source_state=ck['state'],
                retained_non_head_tensors=sum(not k.startswith('output_head.') for k in loaded),
                reinitialized=['output_head.weight','output_head.bias'],gate_bias=4.0,
                all_parameters_trainable=True,restored_optimizer_scheduler=False,
                output_domain='normalized_calibrated_force')


def optimizer_for(net,c):
    groups={}
    for name,p in net.named_parameters():
        head=name.startswith('output_head.');decay=p.ndim>1 and not name.endswith('.bias')
        key=(head,decay);groups.setdefault(key,[]).append(p)
    return torch.optim.AdamW([dict(params=ps,lr=c['lr_output_head'] if head else c['lr_backbone'],
        base_lr=c['lr_output_head'] if head else c['lr_backbone'],weight_decay=c['weight_decay'] if decay else 0.)
        for (head,decay),ps in groups.items()],betas=(.9,.999),eps=1e-8)


def lr_factor(c,step,total):
    warm=c['warmup_updates']
    if step<warm:return (step+1)/warm
    p=min(1.,max(0.,(step-warm)/max(1,total-warm-1)))
    return c['minimum_lr_ratio']+(1-c['minimum_lr_ratio'])*(1+math.cos(math.pi*p))/2


def regression(pred,y,valid,c):
    count=valid.sum()
    if not bool(count):
        zero=pred.sum()*0.
        return zero,torch.stack((zero.detach(),zero.detach(),zero.detach(),zero.detach(),count))
    weight=valid[:,None,None].expand_as(y)
    q=force_objective(pred,y,weight=weight,corr_weight=c['corr_weight'],mape_weight=c['mape_weight'],
                      mae_weight=c['mae_weight'],mape_threshold_n=c['mape_threshold_n'],
                      mae_high_force=c.get('mae_high_force'))
    if c.get('objective') == 'pure_huber':
        q['loss'] = force_huber_loss(pred, y, weight=weight, delta_n=c['huber_delta_n'])
    if c.get('objective') == 'corr_mae_huber':
        q['loss'] = q['loss'] + c['huber_weight'] * force_huber_loss(pred, y, weight=weight, delta_n=c['huber_delta_n'])
    if c.get('pairwise_force_weight', 0.0):
        q['pairwise_force_loss'] = paired_force_difference_loss(pred, y, weight=weight)
        q['loss'] = q['loss'] + c['pairwise_force_weight'] * q['pairwise_force_loss']
    stats=torch.stack((q['loss'].detach()*count,q['correlation'].detach()*count,
                       q['mape'].detach()*count,q['mae'].detach()*count,count))
    return q['loss']*count,stats


def summarize(records):
    def average(key,rs):
        values=[r[key] for r in rs if r[key] is not None]
        return float(np.mean(values)) if values else None
    def metrics(rs):
        return dict(trials=len(rs),loss=average('loss',rs),mae_n=average('mae_n',rs),
            mape_percent=average('mape_percent',rs),corr=average('corr',rs),
            corr_diagnostic=average('corr_diagnostic',rs),tau_b=average('tau_b',rs),
            tau_valid_trials=sum(r['tau_b'] is not None for r in rs),
            corr_valid_trials=sum(r['corr_diagnostic'] is not None for r in rs))
    result=metrics(records)
    if records and 'reference_loss' in records[0]:
        result['reference_loss'] = average('reference_loss', records)
    if records and 'huber_loss' in records[0]:
        result['huber_loss'] = average('huber_loss', records)
    if records and 'pairwise_force_loss_n' in records[0]:
        result['pairwise_force_loss_n'] = average('pairwise_force_loss_n', records)
    for key in ('stage','subject','layout'):
        result['by_'+key]={str(v):metrics([r for r in records if r[key]==v]) for v in sorted({r[key] for r in records})}
    primary=[s for stage,s in result['by_stage'].items() if 1<=int(stage)<=15]
    result['stage_macro']={k:float(np.mean([s[k] for s in primary if s[k] is not None]))
                           if any(s[k] is not None for s in primary) else None
                           for k in ('loss','mae_n','mape_percent','corr','tau_b')}
    result['stage_macro_stage_values']=list(range(1,16))
    result.update(aggregation='complete-trial statistics then equal-weight trial mean',
                  domain='calibrated_N',prediction_clipping=False,
                  tau_protocol='same finger within complete trial scored lattice, valid fingers then trials; stage macro separately',
                  correlation_protocol='pooled five fingers/time within trial; diagnostics omit undefined correlations',
                  mape_threshold_N=0.5,inference_precision='bf16_mixed')
    if records and 'high_force_statistics' in records[0]:
        defined = [r for r in records if r['high_force_statistics']['count'] > 0]
        def high_mae(r):
            s = r['high_force_statistics']
            return s['abs_error_sum'] / s['count']
        subjects = sorted({r['subject'] for r in defined})
        count = sum(r['high_force_statistics']['count'] for r in defined)
        result['high_force'] = dict(valid_trials=len(defined), valid_subjects=len(subjects), count=count,
            trial_equal_mae_n=float(np.mean([high_mae(r) for r in defined])) if defined else None,
            subject_equal_trial_mae_n=float(np.mean([
                np.mean([high_mae(r) for r in defined if r['subject'] == s]) for s in subjects])) if subjects else None,
            pooled_mae_n=sum(r['high_force_statistics']['abs_error_sum'] for r in defined)/count if count else None,
            pooled_bias_n=sum(r['high_force_statistics']['signed_error_sum'] for r in defined)/count if count else None)
    return result


@torch.no_grad()
def evaluate(net,c,rows,split,label,rank,world,device,mean,std,checkpoint):
    out=Path(c['run_root'])/'evaluations'/label
    if rank==0:out.mkdir(parents=True,exist_ok=True)
    dist.barrier();net.eval();reader=WindowReader(c,rows,split);records=[];tick=time.time()
    binding=[None]
    if rank==0:
        (out/'COMPLETE.json').unlink(missing_ok=True)
        binding[0]=dict(label=label,split=split,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
            source_checkpoint_sha256=c['source_checkpoint_sha256'],norm_sha256=sha(Path(c['run_root'])/'train_norm.npz'),
            config_sha256=sha(Path(c['run_root'])/'resolved_config.yaml'),index_sha256=c['index_sha256'],
            precision=c['precision'],mape_threshold_n=c['mape_threshold_n'],trial_count=len(rows),
            scoring_grid='4000-sample reset windows; start+470+40*k,k=0..88; tail timestamps deduplicated',
            metric_policy='six aggregate reports use float64, undefined=null/exclude/count; selection uses configured objective',
            objective=dict(corr_weight=c['corr_weight'], mape_weight=c['mape_weight'], mae_weight=c['mae_weight'],
                           mae_high_force=c.get('mae_high_force')),
            high_force_diagnostics=c.get('high_force_diagnostics', c.get('mae_high_force')))
        atomic_json(out/'protocol.json',binding[0])
        if c.get('objective') == 'pure_huber':
            binding[0]['objective'] = dict(kind='pure_huber', delta_n=c['huber_delta_n'], units='N_squared',
                                          reduction='mean within complete trial, then trial equal')
            atomic_json(out/'protocol.json',binding[0])
        if c.get('objective') == 'corr_mae_huber':
            binding[0]['objective'].update(kind='corr_mae_huber', huber_weight=c['huber_weight'], huber_delta_n=c['huber_delta_n'])
            atomic_json(out/'protocol.json',binding[0])
    dist.broadcast_object_list(binding,src=0)
    path=out/f'rank{rank}_trials.jsonl';temporary=path.with_suffix('.jsonl.partial')
    handle=temporary.open('w')
    for number,trial in enumerate(range(rank,len(rows),world),1):
        row=rows[trial];starts=window_starts(row['n_samples']);ps=[];ys=[];last=-1
        for begin in range(0,len(starts),c['eval_batch']):
            samples=[reader.window(trial,j) for j in range(begin,min(len(starts),begin+c['eval_batch']))]
            x=torch.from_numpy(np.stack([s[0] for s in samples])).to(device)
            with torch.autocast('cuda',dtype=torch.bfloat16):z=net(x)
            pred=(z.float()*std+mean).cpu().numpy()
            for p,(_,y,t) in zip(pred,samples):
                keep=t>last
                if keep.any():ps.append(p[:,keep].T);ys.append(y[:,keep].T);last=int(t[keep][-1])
        p=np.concatenate(ps);y=np.concatenate(ys)
        stats=force_sufficient_statistics(torch.from_numpy(p.T[None]),torch.from_numpy(y.T[None]),threshold_n=c['mape_threshold_n'])
        q=force_objective_from_statistics(stats,corr_weight=c['corr_weight'],mape_weight=c['mape_weight'],mae_weight=c['mae_weight'])
        mae_objective = float(q['mae'][0])
        if c.get('mae_high_force') is not None:
            target = torch.from_numpy(y.T).double()
            weights = high_force_mae_weights(target, c['mae_high_force'])
            mae_objective = float((weights * (torch.from_numpy(p.T).double() - target).abs()).sum() / weights.sum())
            q['loss'] = (c['corr_weight'] * (1 - q['correlation']) + c['mape_weight'] * q['mape']
                         + c['mae_weight'] * mae_objective)
        taus=[];finger_taus=[None]*5
        for f in range(5):
            if np.ptp(y[:,f])>0 and np.ptp(p[:,f])>0:
                t=float(kendalltau(y[:,f],p[:,f],variant='b').statistic)
                if math.isfinite(t):taus.append(t);finger_taus[f]=t
        varying=bool(np.ptp(p)>0 and np.ptp(y)>0)
        record=dict(trial_key=row['trial_key'],subject=row['subject'],stage=int(row['stage']),layout=row['ftype'],
                    loss=float(q['loss'][0]),mae_n=float(q['mae'][0]),mape_percent=float(q['mape'][0])*100,
                    corr=float(q['correlation'][0]),corr_diagnostic=float(q['correlation'][0]) if varying else None,
                    tau_b=float(np.mean(taus)) if taus else None,tau_valid_fingers=len(taus),scored_timestamps=len(y))
        record['reference_loss'] = float(1-q['correlation'][0]+.3*q['mape'][0]+.03*q['mae'][0])
        if c.get('objective') in ('pure_huber', 'corr_mae_huber'):
            huber = float(force_huber_loss(torch.from_numpy(p).double(), torch.from_numpy(y).double(),
                                           delta_n=c['huber_delta_n']))
            record['huber_loss'] = huber
            record['loss'] = huber if c['objective'] == 'pure_huber' else record['loss'] + c['huber_weight'] * huber
            record['huber_statistics'] = dict(sum=huber*p.size, count=int(p.size), delta_n=c['huber_delta_n'])
        if c.get('pairwise_force_weight', 0.0):
            pairwise = float(paired_force_difference_loss(
                torch.from_numpy(p.T).double(), torch.from_numpy(y.T).double()))
            record['pairwise_force_loss_n'] = pairwise
            record['loss'] += c['pairwise_force_weight'] * pairwise
        record.update(statistics=moments(p,y,c['mape_threshold_n']),
            per_finger_statistics=[moments(p[:,f],y[:,f],c['mape_threshold_n']) for f in range(5)],
            per_finger_tau_b=finger_taus)
        if c.get('mae_high_force') is not None:
            record['mae_objective_n'] = mae_objective
        high_policy = c.get('high_force_diagnostics', c.get('mae_high_force'))
        if high_policy is not None:
            # Keep raw metrics unchanged. Save additive high-force errors separately.
            high = np.abs(y) >= np.asarray(high_policy['full_n'])[None, :]
            error = p.astype(np.float64) - y.astype(np.float64)
            record['high_force_statistics'] = dict(count=int(high.sum()),
                abs_error_sum=float(np.abs(error[high]).sum()), signed_error_sum=float(error[high].sum()),
                per_finger_count=high.sum(0).tolist(),
                per_finger_abs_error_sum=(np.abs(error)*high).sum(0).tolist())
        if not all(math.isfinite(record[k]) for k in ('loss','mae_n','mape_percent','corr')):raise ValueError('nonfinite evaluation')
        records.append(record)
        handle.write(json.dumps(record,allow_nan=False)+'\n')
        if number%50==0:handle.flush()
        if number%100==0:print(json.dumps(dict(phase='evaluating',label=label,rank=rank,trials=number,seconds=time.time()-tick)),flush=True)
    handle.flush();os.fsync(handle.fileno());handle.close();os.replace(temporary,path)
    atomic_json(out/f'rank{rank}_complete.json',dict(rank=rank,trials=len(records),sha256=sha(path),
        protocol_sha256=sha(out/'protocol.json'),checkpoint_sha256=binding[0]['checkpoint_sha256']))
    parts=[None]*world if rank==0 else None;dist.gather_object(records,parts,dst=0)
    result=[None]
    if rank==0:
        all_records=[r for part in parts for r in part]
        if len(all_records)!=len(rows) or {r['trial_key'] for r in all_records}!={r['trial_key'] for r in rows}:
            raise ValueError('incomplete evaluation cohort')
        artifacts=[out/'protocol.json']
        for r in range(world):
            part=out/f'rank{r}_trials.jsonl';receipt=out/f'rank{r}_complete.json'
            done=json.loads(receipt.read_text())
            if done['sha256']!=sha(part) or done['checkpoint_sha256']!=binding[0]['checkpoint_sha256']:
                raise ValueError('evaluation statistics checksum/binding mismatch')
            artifacts.extend((part,receipt))
        six=summarize_six(all_records);six['binding']=binding[0]
        atomic_json(out/'six_aggregations.json',six);artifacts.append(out/'six_aggregations.json')
        with (out/'six_aggregations.csv').open('w') as f:
            import csv
            writer=csv.writer(f);writer.writerow(['method','name','corr','mape_percent','mae_n','valid_corr_units','valid_mape_units','valid_mae_units'])
            for method,x in six['pooled_five_fingers']['methods'].items():
                writer.writerow([method,x['name'],x['corr'],x['mape_percent'],x['mae_n'],
                    x['valid_units']['corr'],x['valid_units']['mape_percent'],x['valid_units']['mae_n']])
        artifacts.append(out/'six_aggregations.csv')
        result[0]=summarize(all_records);result[0].update(split=split,label=label)
        result[0]['six_aggregations']=six['pooled_five_fingers']['methods']
        result[0]['six_aggregation_finger_rule']='five fingers pooled; per-finger alternative also saved'
        atomic_json(out/'summary.json',result[0]);print(json.dumps(dict(phase='evaluation_complete',label=label,**{k:result[0][k] for k in ('loss','mae_n','mape_percent','corr','tau_b')})),flush=True)
        artifacts.append(out/'summary.json')
        atomic_json(out/'COMPLETE.json',dict(schema='ap304-six-aggregation-v1',binding=binding[0],
            files={p.name:sha(p) for p in artifacts},trials=len(all_records),subjects=six['subjects']))
        if c.get('filtered_evaluation', {}).get(split):
            from .filtered_cohort_report import report_filtered
            filtered = report_filtered(c, split, label, out, binding[0], all_records, summarize)
            result[0]['filtered_summary'] = filtered
            if split == 'val':
                result[0]['selection_loss'] = filtered['loss']
    dist.broadcast_object_list(result,src=0);net.train();return result[0]


def save(net,optimizer,state,c,out,rank,world,extra=()):
    rng=rng_state();all_rng=[None]*world if rank==0 else None
    dist.gather_object(rng,all_rng,dst=0)
    if rank==0:
        folder=out/'checkpoints';folder.mkdir(exist_ok=True)
        if shutil.disk_usage(folder).free<2*1024**3:raise RuntimeError('insufficient checkpoint disk space')
        payload=dict(model=net.state_dict(),optimizer=optimizer.state_dict(),state=state,config=c,
                     rng_per_rank=all_rng,output_units='normalized_calibrated_force',
                     norm_sha256=sha(out/'train_norm.npz'))
        tmp=folder/'last.pt.tmp';torch.save(payload,tmp);os.replace(tmp,folder/'last.pt')
        for name in extra:
            tmp=folder/(name+'.tmp');shutil.copyfile(folder/'last.pt',tmp);os.replace(tmp,folder/name)
        if state['epoch']>0 and state['next_update']==0:
            destination=folder/f"epoch{state['epoch']:03d}.pt"
            if destination.exists():raise ValueError('refusing to overwrite completed epoch checkpoint')
            os.link(folder/'last.pt',destination)
            atomic_json(folder/f"epoch{state['epoch']:03d}.json",dict(checkpoint=str(destination),
                sha256=sha(destination),state=state,config_sha256=sha(out/'resolved_config.yaml'),
                evaluation_receipt_sha256=sha(out/'evaluations'/f"epoch{state['epoch']:03d}_val"/'COMPLETE.json')))
    dist.barrier()


def train(c):
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=c['devices']:raise ValueError('wrong world size')
    torch.set_num_threads(2);torch.cuda.set_device(local)
    torch.cuda.set_per_process_memory_fraction(c['gpu_memory_fraction'],local)
    dist.init_process_group('nccl',timeout=datetime.timedelta(hours=12));device=torch.device('cuda',local)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed']);torch.cuda.manual_seed(c['seed'])
    out=Path(c['run_root']);ready=json.loads((out/'PREPARED.json').read_text())
    if ready['binding']!=preparation_binding(c) or ready['normalization_sha256']!=sha(out/'train_norm.npz'):
        raise ValueError('prepared normalization mismatch')
    with np.load(out/'train_norm.npz',allow_pickle=False) as z:
        mean=torch.from_numpy(z['force_mean'].copy()).to(device)[None,:,None]
        std=torch.from_numpy(z['force_std'].copy()).to(device)[None,:,None]
    rows=indices(c);count=int(window_counts(rows['train']).sum())
    if count!=c['expected_train_windows']:raise ValueError('window count mismatch')
    updates=math.ceil(count/c['global_batch']);total=updates*c.get('scheduler_epochs', c['max_epochs'])
    net=network(c);receipt=initialize(net,c);net=net.to(device)
    model=DDP(net,device_ids=[local],broadcast_buffers=False);optimizer=optimizer_for(net,c)
    state=dict(epoch=0,next_update=0,global_update=0,best_loss=None,early_stop_reference=None,bad_epochs=0,stop=False)
    torch.manual_seed(c['seed']+rank);torch.cuda.manual_seed(c['seed']+rank)
    last=out/'checkpoints/last.pt'
    if last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=True)
        if ck['config']!=c or ck['norm_sha256']!=sha(out/'train_norm.npz'):raise ValueError('resume config mismatch')
        net.load_state_dict(ck['model'],strict=True);optimizer.load_state_dict(ck['optimizer']);state=ck['state'];restore_rng(ck['rng_per_rank'][rank]);del ck
    elif rank==0:
        atomic_json(out/'initialization_receipt.json',receipt)
    if rank==0:
        atomic_json(out/'environment.json',dict(torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(local),
            world_size=world,parameters=sum(p.numel() for p in net.parameters()),all_parameters_trainable=True,
            force_norm_sha256=sha(out/'train_norm.npz'),updates_per_epoch=updates,
            gpu_memory_fraction=c['gpu_memory_fraction'],source_checkpoint_sha256=c['source_checkpoint_sha256']))
    while state['epoch']<c['max_epochs'] and not state['stop']:
        epoch=state['epoch'];plan=out/'plans'/f'epoch{epoch:03d}.npy'
        if rank==0 and not plan.exists():
            plan.parent.mkdir(exist_ok=True)
            order=np.random.default_rng(c['seed']+epoch).permutation(count)
            order=np.pad(order,(0,updates*c['global_batch']-count),constant_values=-1)
            tmp=plan.with_suffix('.tmp.npy');np.save(tmp,order);os.replace(tmp,plan)
        dist.barrier();ds=TrainWindows(c,rows['train'],plan,rank,world,state['next_update'])
        generator=torch.Generator().manual_seed(c['seed']+epoch*world+rank)
        workers=c['train_workers']
        loader=DataLoader(ds,batch_size=c['microbatch'],shuffle=False,num_workers=workers,
                          pin_memory=True,persistent_workers=workers>0,generator=generator,
                          multiprocessing_context='spawn' if workers else None)
        iterator=iter(loader);model.train()
        for update in range(state['next_update'],updates):
            tick=time.time();optimizer.zero_grad(set_to_none=True);stats=torch.zeros(5,dtype=torch.float64,device=device)
            factor=lr_factor(c,state['global_update'],total)
            for group in optimizer.param_groups:group['lr']=group['base_lr']*factor
            for micro in range(c['accumulate']):
                batch=next(iterator);x=batch['x'].to(device,non_blocking=True);y=batch['y'].to(device,non_blocking=True);valid=batch['valid'].to(device)
                ctx=model.no_sync() if micro<c['accumulate']-1 else contextlib.nullcontext()
                with ctx:
                    with torch.autocast('cuda',dtype=torch.bfloat16):z=model(x)
                    pred=z.float()*std+mean;loss,values=regression(pred,y,valid,c)
                    if not torch.isfinite(loss):raise ValueError('nonfinite regression loss')
                    loss.backward()
                stats+=values.double()
            dist.all_reduce(stats)
            if stats[4]>0:
                for p in net.parameters():
                    if p.grad is not None:p.grad.mul_(world/float(stats[4]))
                norm=torch.nn.utils.clip_grad_norm_(net.parameters(),c['gradient_clip'],error_if_nonfinite=True)
                optimizer.step()
            else:norm=torch.tensor(0.)
            state.update(next_update=update+1,global_update=state['global_update']+1)
            if rank==0 and (state['global_update']%c['log_every_updates']==0 or state['global_update']==1):
                v=(stats[:4]/stats[4].clamp_min(1)).cpu().tolist()
                row=dict(phase='training',epoch=epoch+1,update_in_epoch=update+1,updates_per_epoch=updates,
                    global_update=state['global_update'],loss=v[0],corr=v[1],mape_percent=v[2]*100,mae_n=v[3],
                    valid_windows=int(stats[4]),lr_backbone=c['lr_backbone']*factor,lr_output_head=c['lr_output_head']*factor,
                    grad_norm=float(norm),seconds=time.time()-tick,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
                row['loss_components'] = dict(corr=c['corr_weight']*(1-v[1]), mape=c['mape_weight']*v[2],
                    mae=v[0]-c['corr_weight']*(1-v[1])-c['mape_weight']*v[2])
                if c['mae_weight'] > 0:
                    row['mae_objective_n'] = row['loss_components']['mae']/c['mae_weight']
                if c.get('objective') == 'pure_huber':
                    row['loss_components'] = dict(huber=v[0])
                    row['huber_loss'] = v[0]
                    row['huber_delta_n'] = c['huber_delta_n']
                if c.get('objective') == 'corr_mae_huber':
                    corr_term = c['corr_weight'] * (1-v[1])
                    mae_term = c['mae_weight'] * v[3]
                    huber_term = v[0] - corr_term - mae_term
                    row['loss_components'] = dict(corr=corr_term, mae=mae_term, huber=huber_term)
                    row['mae_objective_n'] = v[3]
                    row['huber_loss'] = huber_term / c['huber_weight']
                    row['huber_weight'] = c['huber_weight']
                    row['huber_delta_n'] = c['huber_delta_n']
                if c.get('pairwise_force_weight', 0.0) and c.get('objective') == 'mixed_regression' and c.get('mae_high_force') is None:
                    mae_term = c['mae_weight'] * v[3]
                    pairwise_term = row['loss_components']['mae'] - mae_term
                    row['loss_components'].update(mae=mae_term, pairwise=pairwise_term)
                    row['mae_objective_n'] = v[3]
                    row['pairwise_force_loss_n'] = pairwise_term / c['pairwise_force_weight']
                with (out/'train_metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                atomic_json(out/'status.json',row);print(json.dumps(row),flush=True)
            if state['global_update']%c['save_every_updates']==0:save(net,optimizer,state,c,out,rank,world)
        del iterator,loader,ds
        # Preserve the exact trained epoch before validation, so evaluation can be
        # retried after interruption without losing the epoch's final updates.
        evaluation_checkpoint=out/'checkpoints'/f'epoch{epoch+1:03d}_training_end.pt'
        save(net,optimizer,state,c,out,rank,world,[evaluation_checkpoint.name])
        if rank==0:atomic_json(out/'status.json',dict(phase='validation',epoch=epoch+1,global_update=state['global_update']))
        rng=rng_state();result=evaluate(net,c,rows['val'],'val',f'epoch{epoch+1:03d}_val',rank,world,device,mean,std,evaluation_checkpoint);restore_rng(rng)
        loss=result.get('selection_loss',result['loss']);best=state['best_loss'] is None or loss<state['best_loss']
        if best:state['best_loss']=loss
        if state['early_stop_reference'] is None or loss<state['early_stop_reference']-c['early_stop_min_delta']:
            state['early_stop_reference']=loss;state['bad_epochs']=0
        else:state['bad_epochs']+=1
        state.update(epoch=epoch+1,next_update=0,stop=state['bad_epochs']>=c['patience'])
        if rank==0:
            with (out/'validation_history.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch+1,global_update=state['global_update'],**result))+'\n')
        save(net,optimizer,state,c,out,rank,world,['best_loss.pt'] if best else [])
    best=out/'checkpoints/best_loss.pt';ck=torch.load(best,map_location='cpu',weights_only=True);net.load_state_dict(ck['model']);selected=ck['state'];del ck
    if rank==0:
        atomic_json(out/'selected_checkpoint.json',dict(path=str(best),sha256=sha(best),state=selected,
            selection=c['validation_selection']+'; Test unused for selection'))
        atomic_json(out/'status.json',dict(phase='final_test',selected_state=selected))
    result=evaluate(net,c,rows['test'],'test','final_test',rank,world,device,mean,std,best)
    if rank==0:atomic_json(out/'status.json',dict(phase='complete',selected_checkpoint=str(best),state=state,final_test=result))
    dist.barrier();dist.destroy_process_group()


def smoke(c):
    """CPU synthetic integration test; checks actual raw Train sample separately below."""
    torch.set_num_threads(2);torch.manual_seed(c['seed']);net=network(c);net.train()
    x=torch.randn(1,16,4000);y=torch.rand(1,5,89)*2;valid=torch.ones(1)
    optimizer=optimizer_for(net,c);before=net.output_head.weight.detach().clone()
    z=net(x);assert z.shape==(1,5,89)
    pred=z.float()+torch.tensor([1.,.8,.6,.3,.2])[None,:,None]
    loss,stats=regression(pred,y,valid,c);loss.backward()
    blocks={'encoder':[],'decoder':[],'head':[]}
    for name,p in net.named_parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():raise ValueError('nonfinite smoke gradient')
            block='head' if name.startswith('output_head.') else 'encoder' if name.startswith(('patch_embedding.','encoder_blocks.','encoder_norm.')) else 'decoder'
            blocks[block].append(float(p.grad.abs().sum()))
    assert all(sum(v)>0 for v in blocks.values())
    torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    assert not torch.equal(before,net.output_head.weight)
    # Padding must not contribute to any of the regression statistics.
    out=torch.randn(2,5,89,requires_grad=True);target=torch.rand_like(out)
    a,_=regression(out,target,torch.tensor([1.,0.]),c);b,_=regression(out[:1],target[:1],torch.ones(1),c)
    assert torch.allclose(a,b);a.backward();assert torch.count_nonzero(out.grad[1])==0
    zero=torch.randn(2,5,89,requires_grad=True);lz,_=regression(zero,torch.zeros_like(zero),torch.zeros(2),c);lz.backward();assert float(lz.detach())==0
    result=dict(status='PASS_CPU_REGRESSION_SMOKE',loss=float(loss.detach()),shape=list(z.shape),
        gradient_l1_by_block={k:sum(v) for k,v in blocks.items()},optimizer_step_changes_head=True,
        zero_weight_padding_checked=True,lambda_ord=c['lambda_ord'],GPU_used=False,Test_signal_reads=False)
    atomic_json(Path(c['run_root'])/'cpu_smoke.json',result);print(json.dumps(result),flush=True)


def integration(c):
    torch.set_num_threads(2);torch.manual_seed(c['seed']);net=network(c);receipt=initialize(net,c)
    rows=indices(c);count=int(window_counts(rows['train']).sum())
    assert count==c['expected_train_windows']
    reader=WindowReader(c,rows['train'],'train');x,y,t=reader.window(0,0)
    assert x.shape==(16,4000) and y.shape==(5,89) and np.array_equal(t,470+40*np.arange(89))
    out=Path(c['run_root'])
    with np.load(out/'train_norm.npz',allow_pickle=False) as z:
        mean=torch.from_numpy(z['force_mean'].copy())[None,:,None]
        std=torch.from_numpy(z['force_std'].copy())[None,:,None]
    normalized=(torch.from_numpy(y[None])-mean)/std
    assert torch.allclose(normalized*std+mean,torch.from_numpy(y[None]),atol=1e-5)
    net.train();pred=net(torch.from_numpy(x[None])).float()*std+mean
    loss,stats=regression(pred,torch.from_numpy(y[None]),torch.ones(1),c);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
    assert torch.isfinite(loss) and float(norm)>0
    assert net.output_head.weight.grad is not None and net.patch_embedding.branch40[0].weight.grad is not None
    # Check a partial global batch and every-rank sharding without dropping or repeating real IDs.
    plan=out/'integration_plan.npy';np.save(plan,np.r_[np.arange(17),np.full(239,-1)].astype(np.int64))
    real=0;dummy=0
    for rank in range(8):
        ds=TrainWindows(c,rows['train'],plan,rank,8)
        assert len(ds)==32
        for j in range(len(ds)):
            item=ds[j];real+=int(item['valid']);dummy+=int(not item['valid'])
    assert real==17 and dummy==239
    result=dict(status='PASS_REAL_TRAIN_CHECKPOINT_INTEGRATION',source_checkpoint_sha256=c['source_checkpoint_sha256'],
        checkpoint_update=receipt['source_state']['global_update'],real_train_forward_backward_loss=float(loss.detach()),
        grad_norm=float(norm),train_windows=count,global_batch_real_dummy=[real,dummy],
        force_roundtrip_checked=True,time_alignment_checked=True,all_parameters_trainable=True,
        GPU_used=False,Test_signal_reads=False)
    atomic_json(out/'integration_check.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--mode',choices=['train','smoke','integration'],required=True)
    a=parser.parse_args();c=load_config(a.config)
    if a.mode=='smoke':smoke(c)
    elif a.mode=='integration':integration(c)
    else:train(c)
