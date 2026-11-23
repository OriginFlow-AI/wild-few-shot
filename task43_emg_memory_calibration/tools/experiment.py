"""Portable preparation, training, qualification, inference and Val entry point."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
from plan import validate,read

PACKAGE=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(value,separators=(',',':'),allow_nan=False).encode()
    temp=path.with_name(path.name+'.tmp');temp.write_bytes(data);os.replace(temp,path)


def export_plan(args):
    plan=read(args.input);plan['roles']={k:plan['roles'][k] for k in ['train','val']};plan['summary']={k:plan['summary'][k] for k in ['train','val']}
    plan['training_export_source_sha256']=sha(args.input);counts=validate(plan)
    if Path(args.output).exists():raise FileExistsError('Do not overwrite a frozen plan')
    atomic(args.output,plan);print(json.dumps(dict(sha256=sha(args.output),counts=counts,test_signal_reads=0)))


def load(root):
    root=Path(root).resolve();sys.path.insert(0,str(root));return root


def binding(root):
    files=[*root.rglob('*.py'),root/'PLAN.json',root/'DATA.json',root/'PREPARED.json',root/'MODEL_CONFIG.json',root/'PARAMETER_SCOPE.json',root/'CONFIG_F.json',root/'CONFIG_B_L.json',root/'assets/A_ep1.pt',root/'assets/train_norm.npz',root/'calibration/LABEL_POLICY.json',root/'calibration/calibration_assets/coefficients.json',root/'calibration/calibration_assets/tables.npz']
    return {str(p.relative_to(root)):sha(p) for p in files if 'qualification/' not in str(p.relative_to(root))}


def prepare(args):
    cfg=read(args.config)
    synthetic=cfg['data_version']=='synthetic_fixture'
    if synthetic!=args.synthetic_smoke or (not synthetic and cfg['data_version']!='benchmark_v2.0'):
        raise ValueError('Use frozen Benchmark2.0 data, or explicitly opt into an isolated synthetic smoke fixture')
    plan=read(cfg['plan']);counts=validate(plan)
    expected=cfg.get('plan_sha256')
    if expected and sha(cfg['plan'])!=expected:raise ValueError('Input plan SHA mismatch')
    root=Path(args.workdir).resolve()
    if root.is_relative_to(PACKAGE.parent):raise ValueError('Keep runtime assets outside the Git checkout')
    root.mkdir(parents=True,exist_ok=False)
    for p in (PACKAGE/'runtime').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:
            q=root/p.relative_to(PACKAGE/'runtime');q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    for name in ['MODEL_CONFIG.json','PARAMETER_SCOPE.json']:shutil.copyfile(PACKAGE/'configs'/name,root/name)
    assets={'assets/A_ep1.pt':cfg['assets']['base'],'assets/train_norm.npz':cfg['assets']['norm'],
            'calibration/LABEL_POLICY.json':str(Path(cfg['assets']['calibration'])/'LABEL_POLICY.json'),
            'calibration/calibration_assets/coefficients.json':str(Path(cfg['assets']['calibration'])/'calibration_assets/coefficients.json'),
            'calibration/calibration_assets/tables.npz':str(Path(cfg['assets']['calibration'])/'calibration_assets/tables.npz')}
    references=read(PACKAGE/'PROVENANCE.json')['external_asset_hashes']
    for name,source in assets.items():
        if sha(source)!=references[name]:raise ValueError('This release requires its matched external asset: '+name)
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    shutil.copyfile(cfg['plan'],root/'PLAN.json')
    data=dict(cfg['data']);data['metadata_cache']=str(root/'source_headers');atomic(root/'DATA.json',data)
    for arm in ['F','B_L']:
        c=read(PACKAGE/'configs'/f'train_{arm}.json');c.update(cfg.get('training',{}))
        c.update(gpu_uuid=args.gpu,plan_sha256=sha(root/'PLAN.json'),train=dict(manifest=str(root/'PLAN.json')),dev=dict(manifest=str(root/'PLAN.json')))
        atomic(root/f'CONFIG_{arm}.json',c)
    inference={'checkpoints':{}}
    for arm,entry in cfg.get('checkpoints',{}).items():
        if arm not in ['F','B_L'] or sha(entry['path'])!=entry['sha256']:raise ValueError('Checkpoint key or SHA mismatch')
        dest=root/'checkpoints'/f'{arm}_val_best.pt';dest.parent.mkdir(exist_ok=True);shutil.copyfile(entry['path'],dest)
        inference['checkpoints'][arm]=dict(path=str(dest.relative_to(root)),sha256=entry['sha256'],epoch=entry.get('epoch'))
    atomic(root/'INFERENCE.json',inference)
    atomic(root/'PREPARED.json',dict(status='PASS',counts=counts,code_source='task43_emg_memory_calibration',data_version=cfg['data_version'],synthetic_fixture=synthetic,asset_hashes={name:sha(root/name) for name in assets}))
    print(json.dumps(dict(workdir=str(root),counts=counts)))


def targets(args):
    root=load(args.workdir);from source_reader import rows
    from calibration.calibration_tools import Calibration
    cal=Calibration(root/'calibration');plan=read(root/'PLAN.json');count=0
    for role,c in plan['roles'].items():
        for query in c['queries']:
            record=c['records'][query['query']];pieces=[]
            for start in range(0,query['windows'],32):
                end=min(start+32,query['windows']);raw=rows(record,'fz',start*4000,end*4000)
                indices=4000*np.arange(end-start)[:,None]+470+40*np.arange(89)
                pieces.append(cal.forward(raw[indices],layout=record['layout']))
            y=np.concatenate(pieces);high=(np.max(abs(y),axis=2)>=5).mean(axis=1)>=.1;low=np.max(abs(y),axis=2)<=.5
            rest=(low.mean(axis=1)>=.5)|((np.max(abs(y[:,:20]),axis=(1,2))>=5)&(low[:,-20:].mean(axis=1)>=.9))
            p=root/'targets'/role/(query['query']+'.npz');p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():
                with np.load(p,allow_pickle=False) as z:
                    if not all(np.array_equal(z[k],v) for k,v in dict(target=y,high=high,rest=rest).items()):raise ValueError('Existing targets differ')
            else:
                tmp=p.with_suffix('.tmp')
                with tmp.open('xb') as f:np.savez_compressed(f,target=y,high=high,rest=rest)
                os.replace(tmp,p)
            count+=1
    atomic(root/'TARGET_READY.json',dict(status='PASS',records=count,plan_sha256=sha(root/'PLAN.json')))
    print(json.dumps(dict(status='PASS',targets=count)))


def materialize(args):
    root=load(args.workdir);from source_reader import configured,relative_name
    source=configured();plan=read(root/'PLAN.json');destination=root/'waveform_cache';count=0
    for role in args.roles:
        for record in plan['roles'][role]['records'].values():
            for field in args.fields:
                path=destination/(relative_name(record['relative'])+'.'+field+'.npy');path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists():
                    a=np.load(path,mmap_mode='r',allow_pickle=False)
                    if a.shape!=(record['n'],16 if field=='emg' else 5) or a.dtype!=np.float32:raise ValueError('Existing raw cache schema mismatch')
                else:
                    tmp=path.with_name(path.name+'.partial')
                    a=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(record['n'],16 if field=='emg' else 5))
                    for start in range(0,record['n'],16384):a[start:start+16384]=source.rows(record,field,start,min(start+16384,record['n']))
                    a.flush();del a;os.replace(tmp,path)
                count+=1
    cfg=read(root/'DATA.json');cfg['local_roots']=list(dict.fromkeys([str(destination),*cfg.get('local_roots',[])]));atomic(root/'DATA.json',cfg)
    print(json.dumps(dict(materialized_arrays=count,path=str(destination),force_domain='raw N; calibration not yet applied')))


def banks(args):
    root=load(args.workdir);import torch
    import train_r6 as tr
    from arms import Arm,Banks
    from support_cache import keys,FIELDS,SIZES
    from support_cache_raw_r6 import path_for,unpack_raw,RAW_BYTES
    cfg=read(root/'CONFIG_F.json');os.environ['CUDA_VISIBLE_DEVICES']=cfg['gpu_uuid']
    if not (root/'TARGET_READY.json').exists():raise ValueError('Prepare targets first')
    with tr.hold_lock(root/'preparation.lock'),tr.gpu_lock(cfg):
        device=tr.setup(cfg);arm=Arm('F',device).eval();train,dev,_=tr.checked_inputs(cfg);bank=Banks(arm,train,dev,None,cfg['physical_chunk']);jobs=keys(train)+keys(dev)
        with torch.no_grad():
            for i,item in enumerate(jobs):
                path=path_for(*item);path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists():unpack_raw(path.read_bytes())
                else:
                    values=bank.one(*item)
                    for v,(shape,dtype) in zip(values,FIELDS):
                        if tuple(v.shape)!=shape or v.numpy().dtype!=dtype:raise ValueError('Bank layout mismatch')
                    raw=b''.join(v.contiguous().numpy().tobytes() for v in values)
                    if not all(torch.equal(a,b) for a,b in zip(values,unpack_raw(raw))):raise ValueError('Raw-bank roundtrip failed')
                    tmp=path.with_suffix('.tmp');tmp.write_bytes(raw);os.replace(tmp,path)
                if i%20==0:atomic(root/'BANK_PROGRESS.json',dict(completed=i+1,total=len(jobs)))
        done=dict(status='PASS',count=len(jobs),raw_bytes_per_bank=RAW_BYTES,plan_sha256=sha(root/'PLAN.json'))
        atomic(root/'RAW_BANK_READY_R6.json',done);atomic(root/'BANK_READY.json',done)
    print(json.dumps(done))


def plans(args):
    root=load(args.workdir);import train_r6 as tr
    from sampler import plan as make_plan
    cfg=read(root/'CONFIG_F.json');other=read(root/'CONFIG_B_L.json')
    if cfg['seed']!=other['seed']:raise ValueError('F and B_L must share sampling seed')
    train,_,_=tr.checked_inputs(cfg);folder=root/'sample_plans';folder.mkdir(exist_ok=True);digests={}
    with tr.hold_lock(root/'sampling.lock'):
        for epoch in range(24):
            file=folder/f'epoch{epoch+1:02}.npy'
            meta=file.with_suffix('.json');expected=dict(seed=cfg['seed'],epoch=epoch,plan_sha256=sha(root/'PLAN.json'),sampler_sha256=sha(root/'sampler.py'))
            if file.exists() and meta.exists() and read(meta).get('binding')==expected and read(meta).get('file_sha256')==sha(file):values=np.load(file,allow_pickle=False)
            else:
                values,stats=make_plan(train,epoch,752*1024,cfg['seed'])
                if file.exists():
                    if not np.array_equal(values,np.load(file,allow_pickle=False)):raise ValueError('Existing sampling plan differs from the canonical shared plan')
                else:
                    tmp=file.with_suffix('.tmp')
                    with tmp.open('xb') as f:np.save(f,values,allow_pickle=False)
                    os.replace(tmp,file)
                atomic(meta,dict(stats=stats,binding=expected,file_sha256=sha(file)))
            if values.shape!=(752*1024,) or values.min()<0 or values.max()>=len(train):raise ValueError('Sampling plan bounds/shape')
            digests[file.name]=sha(file)
    atomic(root/'SAMPLE_PLANS_READY.json',dict(status='PASS',files=digests,seed=cfg['seed'],plan_sha256=sha(root/'PLAN.json')))
    print(json.dumps(dict(status='PASS',epochs=24,shared_between=['F','B_L'])))


def qual_step(args):
    root=load(args.workdir);cfg=read(root/'qualification'/f'{args.arm}.json');os.environ['CUDA_VISIBLE_DEVICES']=cfg['gpu_uuid']
    import train_r6 as tr
    # Timing is non-scientific and excluded from checkpoint parity. Formal training
    # never receives this qualification-only clock facade.
    tr.time=SimpleNamespace(time=lambda:0.,monotonic=time.monotonic)
    out=root/'qualification'/args.arm/args.run;out.mkdir(parents=True,exist_ok=True)
    with tr.hold_lock(out/'run.lock'),tr.gpu_lock(cfg):tr.run(cfg,args.arm,out,(out/'last.pt').exists(),args.stop_after)


def equal(a,b):
    import torch
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert type(a)==type(b) and len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def qualify(args):
    root=load(args.workdir);import torch
    import train_r6 as tr
    if tr.manual_stop(root):raise ValueError('Manual stop marker present')
    q=root/'qualification';q.mkdir(exist_ok=False)
    plan=read(root/'PLAN.json');report={}
    for arm in ['F','B_L']:
        cfg=read(root/f'CONFIG_{arm}.json');cfg.update(mode='smoke',epochs=1,smoke_windows=2048,dev_windows_per_hand=2)
        for field,role in [('train','train'),('dev','val')]:cfg[field]=dict(cfg[field],smoke_query_keys=[r['query'] for r in plan['roles'][role]['queries'][:2]])
        atomic(q/f'{arm}.json',cfg)
        for run,step in [('continuous',2),('split',1),('split',2)]:
            with (q/f'{arm}_{run}_{step}.log').open('xb') as log:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'_qual_step','--workdir',str(root),'--arm',arm,'--run',run,'--stop-after',str(step)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=cfg['gpu_uuid']),stdout=log,stderr=log,check=True,timeout=1200)
        a=torch.load(q/arm/'continuous/last.pt',map_location='cpu',weights_only=False);b=torch.load(q/arm/'split/last.pt',map_location='cpu',weights_only=False);equal(a,b)
        assert a['step']==2 and a['cursor']==0 and len(a['history'])==1 and len(a['rng']['cuda'])==1
        assert all(int(v['step'])==2 for v in a['adam']['state'].values())
        report[arm]=dict(continuous2_equals_restart1plus1_bitwise=True,global_batch=1024,adam_states=len(a['adam']['state']),history_epochs=1,best_epoch=a['best']['epoch'])
        del a,b
    atomic(root/'QUALIFIED.json',dict(status='PASS',arms=report,binding=binding(root),timing_fields_zeroed_for_parity=True))
    print(json.dumps(report))


def train(args):
    root=load(args.workdir);qualified=read(root/'QUALIFIED.json')
    if read(root/'PREPARED.json')['synthetic_fixture']:raise ValueError('Synthetic fixtures cannot launch formal training')
    if qualified['status']!='PASS' or qualified['binding']!=binding(root):raise ValueError('Qualification binding changed; use a new execution revision')
    sampling=read(root/'SAMPLE_PLANS_READY.json')
    if sampling['status']!='PASS' or sampling['plan_sha256']!=sha(root/'PLAN.json') or sampling['seed']!=read(root/f'CONFIG_{args.arm}.json')['seed'] or len(sampling['files'])!=24:raise ValueError('Prepare all24 shared sampling plans before parallel training')
    for name,digest in sampling['files'].items():
        if sha(root/'sample_plans'/name)!=digest:raise ValueError('Shared sampling plan changed')
    cfg=root/f'CONFIG_{args.arm}.json';os.environ['CUDA_VISIBLE_DEVICES']=read(cfg)['gpu_uuid']
    import train_r6 as tr
    sys.argv=['train_r6.py',str(cfg),'--arm',args.arm,'--run',args.run]+(['--resume'] if args.resume else [])
    tr.main()


def use_best(args):
    root=load(args.workdir);import torch
    if not args.run.replace('_','').isalnum():raise ValueError('Safe run name required')
    path=root/'runs'/args.run/args.arm/'best.pt';raw=path.read_bytes();saved=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    epoch=saved['epoch']
    if saved['arm']!=args.arm or saved['config']['mode']!='formal' or saved['best']['epoch']!=epoch or saved['step']!=epoch*752 or saved['config']['plan_sha256']!=sha(root/'PLAN.json'):
        raise ValueError('Use the actual same-cohort formal Val-best checkpoint')
    digest=hashlib.sha256(raw).hexdigest();dest=root/'checkpoints'/f'{args.arm}_epoch{epoch:02}_{digest[:12]}.pt';dest.parent.mkdir(exist_ok=True)
    if dest.exists() and sha(dest)!=digest:raise ValueError('Checkpoint destination conflict')
    if not dest.exists():
        tmp=dest.with_suffix('.tmp');tmp.write_bytes(raw);os.replace(tmp,dest)
    config=read(root/'INFERENCE.json');config['checkpoints'][args.arm]=dict(path=str(dest.relative_to(root)),sha256=digest,epoch=epoch)
    atomic(root/'INFERENCE.json',config);print(json.dumps(config['checkpoints'][args.arm]))


def predict(args):
    root=load(args.workdir);from infer import Predictor,graph_predictor,check_windows,build_memory
    raw=check_windows(np.load(args.input,mmap_mode='r',allow_pickle=False));support=read(args.support) if args.support else None;admission='explicit_caller_binding' if support else 'cold'
    if args.support_catalog or args.query_metadata:
        if args.support or not args.support_catalog or not args.query_metadata:raise ValueError('Use either explicit --support, or both --query-metadata and --support-catalog')
        from admission import select_support
        catalog=read(args.support_catalog);catalog=catalog if isinstance(catalog,list) else catalog['records']
        if isinstance(catalog,dict):catalog=list(catalog.values())
        query=read(args.query_metadata)
        if query['n']//4000!=len(raw):raise ValueError('Query metadata/window population mismatch')
        support,admission=select_support(query,catalog)
    if args.backend=='cuda-graph':
        if set(args.arms)!=set(['A','B_L','F']):raise ValueError('Shared graph emits all three arms')
        engine=graph_predictor(args.device,args.batch)
        memory=build_memory(engine.f,support) if support else None;engine.bind(support['key'] if support else None,memory)
    else:
        engine=Predictor(args.device,args.arms,args.batch);engine.bind(support)
    result=engine.predict(raw);output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as f:np.savez_compressed(f,**result)
    print(json.dumps(dict(output=str(output),shape=[len(raw),5,89],support_used=support is not None,support_admission=admission,backend=args.backend)))


def evaluate(args):
    root=load(args.workdir);cfg=read(root/f'CONFIG_{args.arm if args.arm=="B_L" else "F"}.json');os.environ['CUDA_VISIBLE_DEVICES']=cfg['gpu_uuid']
    import torch,train_r6 as tr
    from infer import Predictor
    from train_data import Dataset
    from support_cache_raw_r6 import CachedBanks
    tr.setup(cfg)
    arm=args.arm;engine=Predictor('cuda:0',(arm,),cfg['val_chunk']);model=engine.bl if arm=='B_L' else engine.f
    data=Dataset(cfg['dev'],'dev',True)
    if arm=='F':bank=CachedBanks(model,None,data,None,cfg['physical_chunk'])
    else:bank=SimpleNamespace(batch=lambda role,samples:(None,None))
    report=tr.evaluate(model,bank,data,cfg,torch.device('cuda:0'));atomic(args.output,report)
    print(json.dumps({k:report[k] for k in ['L_A','MAE','MAPE','Corr','R2','windows','people']}))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    export=sub.add_parser('export-plan');export.add_argument('--input',required=True);export.add_argument('--output',required=True)
    for action in ['prepare','materialize','targets','banks','plans','qualify','train','use-best','predict','evaluate','_qual_step']:
        a=sub.add_parser(action);a.add_argument('--workdir',required=True)
        if action=='prepare':a.add_argument('--config',required=True);a.add_argument('--gpu',required=True);a.add_argument('--synthetic-smoke',action='store_true')
        if action=='materialize':a.add_argument('--roles',nargs='+',default=['train','val'],choices=['train','val']);a.add_argument('--fields',nargs='+',default=['emg','fz'],choices=['emg','fz'])
        if action in ['train','use-best','evaluate','_qual_step']:a.add_argument('--arm',required=True,choices=['F','B_L'] if action!='evaluate' else ['A','B_L','F'])
        if action in ['train','use-best','_qual_step']:a.add_argument('--run',required=True)
        if action=='train':a.add_argument('--resume',action='store_true')
        if action=='_qual_step':a.add_argument('--stop-after',type=int,required=True)
        if action=='evaluate':a.add_argument('--output',required=True)
        if action=='predict':
            a.add_argument('--input',required=True);a.add_argument('--support');a.add_argument('--output',required=True)
            a.add_argument('--query-metadata');a.add_argument('--support-catalog')
            a.add_argument('--arms',nargs='+',default=['A','B_L','F'],choices=['A','B_L','F']);a.add_argument('--backend',default='native',choices=['native','cuda-graph']);a.add_argument('--device',default='cpu');a.add_argument('--batch',type=int,default=64)
    args=p.parse_args();globals()[args.action.replace('-','_').lstrip('_')](args)


if __name__=='__main__':main()
