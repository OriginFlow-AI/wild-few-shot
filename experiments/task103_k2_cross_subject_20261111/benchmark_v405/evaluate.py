"""Replay EMG models with the A Online W100 protocol on frozen v4.05 Test indices."""
import argparse
from collections import defaultdict
import csv
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import kendalltau
import torch

from prepare_test import HANDS, atomic, read, sha


METRICS=('corr','mae_N','mae_below_1N','mape_ge_0p5N_percent','mape_ge_1N_percent','kendall_tau_b')


def import_model_source(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def load_model(item,contact):
    sys.path.insert(0,str(contact))
    from adapters import RingOnline, normalize, precision

    precision()
    root=Path(item['root'])
    branch=item.get('branch')
    rank=item['format']=='ranking'
    if branch:
        export=root/'exported_models'/branch
        weights=export/'best_model_state_dict.pt'
        norm_path=export/'emg_train_norm.npz'
        params=read(export/'inference_parameters.json')
        assert sha(weights)==params['model_state_sha256']
        assert sha(norm_path)==params['emg_norm_sha256']
        done=read(root/'runs'/branch/'DONE.json')
        assert params['selected_training_checkpoint_sha256']==done['best_sha256']
        force=params['force_norm']
        epoch=params['selected_epoch']
        source=root/'code/vendor/model_source.py'
    else:
        weights, norm_path, source=(Path(item[k]) for k in ('weights','emg_norm','model_source'))
        epoch=item.get('epoch')
        if item.get('expected_weight_sha256'):
            assert sha(weights)==item['expected_weight_sha256']
        if item.get('expected_norm_sha256'):
            assert sha(norm_path)==item['expected_norm_sha256']
        force=None
    module=import_model_source(source,'benchmark_v405_model')
    net=module.K2RingNetwork(module.K2RingConfig())
    state=torch.load(weights,map_location='cpu',weights_only=True)
    net.load_state_dict(state,strict=True)
    assert all(torch.isfinite(p).all() for p in net.parameters())
    with np.load(norm_path,allow_pickle=False) as z:
        norm={k:np.asarray(z[k],np.float32) for k in ('emg_mean','emg_std')}
        if force:
            norm.update({f'force_{k}':np.asarray(force[k],np.float32) for k in ('mean','std')})
        elif rank:
            # Restore dimensionless ranking scores using the original ranking head scale.
            norm.update(force_mean=np.zeros(5,np.float32),force_std=np.full(5,1/net.config.output_scale,np.float32))
        else:
            norm.update({k:np.asarray(z[k],np.float32) for k in ('force_mean','force_std')})
    assert all(np.isfinite(v).all() for v in norm.values())
    assert (norm['emg_std']>0).all() and (norm['force_std']>0).all()
    net=net.cuda().eval()
    engine=RingOnline(net,norm)
    info=dict(model=item['name'],family=item['family'],output_unit='rank_score' if rank else 'N',epoch=epoch,
              weights_sha256=sha(weights),emg_norm_sha256=sha(norm_path),model_source_sha256=sha(source),
              parameter_count=sum(p.numel() for p in net.parameters()))
    return net,engine,norm,normalize,info


def emg_grid(emg,time_s):
    start=float(time_s[0])
    size=int(np.floor((time_s[-1]-start)*2000+1e-7))+1
    times=start+np.arange(size,dtype=np.float64)/2000
    source=np.searchsorted(time_s,times,side='right')-1
    assert source.min()>=0 and (times-time_s[source]).max()<=.002+1e-8
    return np.ascontiguousarray(emg[source],dtype=np.float32),times


@torch.inference_mode()
def qualify(net,engine,norm,normalize,emg):
    x=emg[:4000]
    direct=net(torch.from_numpy(normalize(x,norm)[None]).cuda())[0].T.cpu().numpy()
    expected=direct*norm['force_std']+norm['force_mean']
    replay=engine.predict(x)
    np.testing.assert_allclose(replay,expected,rtol=1e-5,atol=1e-5)
    a=engine.predict(emg[:16370])
    engine.predict(np.ascontiguousarray(emg[:8000]*.37+3.1,dtype=np.float32))
    b=engine.predict(emg[:16370])
    np.testing.assert_array_equal(a,b)
    return dict(window_4000_max_absolute_difference=float(np.max(abs(replay-expected))),reset_bitwise_equal=True)


def take_prediction(engine,emg,grid_t,query):
    values=engine.predict(emg)
    valid_times=grid_t[0]+(470+40*np.arange(len(values),dtype=np.int64))/2000
    nearest_past=np.searchsorted(valid_times,query,side='right')-1
    assert (nearest_past>=0).all()
    delays=query-valid_times[nearest_past]
    assert delays.min()>=0 and delays.max()<=.023
    out=values[nearest_past]
    assert np.isfinite(out).all() and out.shape==(len(query),5)
    return out, float(delays.max())


def recording_metrics(p,y,is_rank):
    corr=float(np.corrcoef(p.ravel(),y.ravel())[0,1]) if np.ptp(p)>0 and np.ptp(y)>0 else None
    if is_rank:
        tau=[kendalltau(x,t).statistic for x,t in zip(p.T,y.T) if np.ptp(x)>0 and np.ptp(t)>0]
        return dict(corr=corr,mae_N=None,mae_below_1N=None,mape_ge_0p5N_percent=None,
                    mape_ge_1N_percent=None,kendall_tau_b=float(np.mean(tau)) if tau else None)
    mag,error=abs(y),abs(p-y)
    low, half, one=mag<1, mag>=.5, mag>=1
    return dict(corr=corr,mae_N=float(error.mean()),
                mae_below_1N=float(error[low].mean()) if low.any() else None,
                mape_ge_0p5N_percent=float((error[half]/mag[half]).mean()*100) if half.any() else None,
                mape_ge_1N_percent=float((error[one]/mag[one]).mean()*100) if one.any() else None,kendall_tau_b=None)


def task103_rollup(rows):
    groups=defaultdict(lambda:defaultdict(list))
    for r in rows:
        for stage in ('all',r['stage']):
            groups[stage][r['subject_id']].append(r)
    output=[]
    for stage,people in sorted(groups.items()):
        item=dict(stage=stage,subjects=len(people),recordings=sum(map(len,people.values())),
                  timepoints=sum(r['timepoints'] for rr in people.values() for r in rr))
        for m in METRICS:
            values=[np.mean(v) for rr in people.values() if (v:=[r[m] for r in rr if r[m] is not None])]
            item[m]=float(np.mean(values)) if values else None
        output.append(item)
    return output


def run(item,root,contact,probe):
    from fusion2force.metrics import SubjectMetrics

    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(10*2**30/torch.cuda.get_device_properties(0).total_memory)
    net,engine,norm,normalize,info=load_model(item,contact)
    if probe:
        inputs=sorted((root/'signals').glob('*.rrd.npz'))
        assert inputs
        with np.load(inputs[0],allow_pickle=False) as z:
            emg,_=emg_grid(z['emg_left_values'],z['emg_left_time_s'])
        print(info,qualify(net,engine,norm,normalize,emg),flush=True)
        return
    source=root/'signal_receipt.json'
    receipt=read(source)
    assert receipt['trials']==len(receipt['recordings'])==109 and not receipt['errors']
    assert receipt['owned_points']==receipt['timepoints']==500040
    dest=root/'results'/item['name']
    dest.mkdir(parents=True,exist_ok=True)
    metrics={s:(SubjectMetrics(.5),SubjectMetrics(1.)) for s in ('all',*(r['stage'] for r in receipt['recordings']))}
    rows=[]
    timing_max=0
    checked=None
    start=time.monotonic()
    for number,trial in enumerate(receipt['recordings'],1):
        rid,stage,subject=(trial[k] for k in ('trial_id','stage','subject_id'))
        path=root/'signals'/f'{rid}.npz'
        assert sha(path)==trial['cache_sha256']
        prediction,truth=[],[]
        with np.load(path,allow_pickle=False) as z:
            for hand in HANDS:
                idx=z[f'{hand}_score_indices']
                assert len(idx)==trial['owned_points'][hand]
                if not len(idx):
                    continue
                emg,grid_t=emg_grid(z[f'emg_{hand}_values'],z[f'emg_{hand}_time_s'])
                if checked is None:
                    checked=qualify(net,engine,norm,normalize,emg)
                query=z[f'force_{hand}_time_s'][idx]
                y=z[f'force_{hand}_calibrated_N'][idx]
                p,lag=take_prediction(engine,emg,grid_t,query)
                timing_max=max(timing_max,lag)
                prediction.append(p)
                truth.append(y)
                pt,yt,mask=(torch.from_numpy(p[None]),torch.from_numpy(y[None]),
                            torch.ones((1,len(idx)),dtype=torch.bool))
                for scope in ('all',stage):
                    for metric in metrics[scope]:
                        metric.update(pt,yt,mask,[subject],[hand])
        p,y=(np.concatenate(arrays).astype(np.float64) for arrays in (prediction,truth))
        rows.append(dict(trial_id=rid,stage=stage,subject_id=subject,timepoints=len(y),
                         **recording_metrics(p,y,info['output_unit']=='rank_score')))
        if number%10==0 or number==109:
            print(item['name'],number,'/109',round(time.monotonic()-start,1),'s',flush=True)
    assert sum(r['timepoints'] for r in rows)==500040
    reports=[]
    for row in task103_rollup(rows):
        half,one=(metric.compute() for metric in metrics[row['stage']])
        row.update(benchmark_corr=half['r_Fz_weighted'])
        if info['output_unit']=='rank_score':
            row.update(benchmark_mae_N=None,benchmark_mape_gt_0p5N_percent=None,benchmark_mape_gt_1N_percent=None)
        else:
            row.update(benchmark_mae_N=half['mae_Fz_mean'],benchmark_mape_gt_0p5N_percent=half['mape_Fz_weighted'],
                       benchmark_mape_gt_1N_percent=one['mape_Fz_weighted'])
        reports.append(row)
    with (dest/'per_recording.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    atomic(dest/'report.json',dict(
        **info,signal_receipt_sha256=sha(source),mask_sha256=receipt['mask_sha256'],
        evaluation_token_identity_sha256=receipt['identity_sha256'],predictions='Online W100 FP32; causal ZOH at 2 kHz, latest 50 Hz prediction held to native force time',
        force_reference='CMCU first 5 channels, mN→N, official LUT applied once',
        qualifier=checked,max_hold_s=timing_max,metrics=reports))
    print(item['name'],next(r for r in reports if r['stage']=='all'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','registry','contact'):
        parser.add_argument(f'--{name}',type=Path,required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--probe',action='store_true')
    args=parser.parse_args()
    (item,)=tuple(r for r in read(args.registry) if r['name']==args.model)
    run(item,args.root,args.contact,args.probe)
