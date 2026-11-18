"""Train-only calibrated-force statistics and independent AP304 regression windows."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
import yaml
from torch.utils.data import Dataset

from .ap304_rank_data import atomic_json, atomic_npz, open_arrays, paths_for, sha, window_starts


def load_config(path):
    c=yaml.safe_load(Path(path).read_text())
    if c['schema']!='ap304-calibrated-stage2-v1':raise ValueError('wrong Stage-2 schema')
    if c['lambda_ord']!=0 or c['score_l2_weight']!=0:raise ValueError('this experiment requires pure regression')
    if c['global_batch']!=c['devices']*c['microbatch']*c['accumulate']:raise ValueError('global batch mismatch')
    if sha(c['index'])!=c['index_sha256']:raise ValueError('index changed')
    if sha(c['model_source'])!=c['model_source_sha256']:raise ValueError('model source changed')
    if sha(c['emg_norm'])!=c['emg_norm_sha256']:raise ValueError('EMG normalization changed')
    return c


def calibration(c):
    p=Path(c['calibration_package'])/'calibration_tools.py'
    if sha(p)!=c['calibration_tools_sha256']:raise ValueError('calibration code changed')
    spec=importlib.util.spec_from_file_location('stage2_frozen_calibration',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.Calibration(c['calibration_package'])


def indices(c):
    raw=json.loads(Path(c['index']).read_text())
    rows={s:raw[s] for s in ('train','val','test')}
    for role,rs in rows.items():
        if len(rs)!=c['expected_trials'][role]:raise ValueError('trial count changed')
        if len({r['trial_key'] for r in rs})!=len(rs):raise ValueError('duplicate trial')
        for r in rs:
            if r['ftype'] not in ('1x5x1','1x5x3'):raise ValueError('invalid authoritative ftype')
            if 'layout' in r and r['layout']!=r['ftype']:raise ValueError('conflicting layout')
    for a,b in (('train','val'),('train','test'),('val','test')):
        for key in ('subject','trial_key','content_hash'):
            if {r[key] for r in rows[a]} & {r[key] for r in rows[b]}:raise ValueError('split leakage: '+key)
    return rows


_CAL=None
def _worker_init(c):
    global _CAL
    _CAL=calibration(c)


def _scan(job):
    c,row,fit=job
    emg,force=open_arrays(row,c['cache_root'])
    total=np.zeros(5,np.float64);squares=np.zeros(5,np.float64)
    # The prior raw preparation checked EMG finite. Bind its file identity below;
    # do not rescan terabytes of unchanged EMG while another model is training.
    for begin in range(0,len(force),262144):
        y=_CAL.forward(np.asarray(force[begin:begin+262144]),layout=row['ftype'])
        if fit:
            q=y.astype(np.float64);total+=q.sum(0);squares+=(q*q).sum(0)
    ep,fp=paths_for(row,c['cache_root'])
    identity=[ep.stat().st_size,ep.stat().st_mtime_ns,fp.stat().st_size,fp.stat().st_mtime_ns]
    return total,squares,len(force),identity


def preparation_binding(c):
    return dict(index_sha256=c['index_sha256'],calibration_tools_sha256=c['calibration_tools_sha256'],
                lut_sha256=sha(Path(c['calibration_package'])/'calibration_assets/tables.npz'),
                emg_norm_sha256=c['emg_norm_sha256'])


def prepare(c,workers=16):
    import fcntl
    out=Path(c['run_root']);out.mkdir(parents=True,exist_ok=True)
    with (out/'.prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        binding=preparation_binding(c);ready=out/'PREPARED.json'
        if ready.exists():
            result=json.loads(ready.read_text())
            if result['binding']!=binding or sha(out/'train_norm.npz')!=result['normalization_sha256']:
                raise ValueError('prepared data binding changed')
            print('STAGE2_PREPARATION_ALREADY_COMPLETE',flush=True);return
        rows=indices(c);sums=np.zeros(5,np.float64);squares=np.zeros(5,np.float64);count=0
        for role in ('train','val'):
            rs=rows[role];prepared=out/'prepared';prepared.mkdir(exist_ok=True)
            checkpoint=prepared/f'{role}_scan.npz';done=0
            identities=np.zeros((len(rs),4),np.int64)
            if checkpoint.exists():
                with np.load(checkpoint,allow_pickle=False) as z:
                    if str(z['binding'])!=json.dumps(binding,sort_keys=True):raise ValueError('scan resume binding mismatch')
                    done=int(z['done']);identities[:]=z['file_stats']
                    if role=='train':sums=z['sum'].copy();squares=z['squares'].copy();count=int(z['count'])
            prior_path=Path(c['ranking_prepared'])/f'{role}_stats.npz'
            with np.load(prior_path,allow_pickle=False) as z:
                if str(z['index_sha256'])!=c['index_sha256']:raise ValueError('old raw preparation uses different index')
                prior_identity=z['file_stats'].copy()
            if done and not np.array_equal(identities[:done],prior_identity[:done]):raise ValueError('resumed raw provenance mismatch')
            tick=time.time()
            with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork'),initializer=_worker_init,initargs=(c,)) as pool:
                jobs=((c,row,role=='train') for row in rs[done:])
                for j,(total,sq,n,identity) in enumerate(pool.map(_scan,jobs,chunksize=8),start=done):
                    if not np.array_equal(identity,prior_identity[j]):raise ValueError('raw data changed since EMG finite scan: '+rs[j]['trial_key'])
                    identities[j]=identity
                    if role=='train':sums+=total;squares+=sq;count+=n
                    if (j+1)%1000==0 or j+1==len(rs):
                        atomic_npz(checkpoint,done=j+1,file_stats=identities,sum=sums,squares=squares,count=count,
                                   binding=json.dumps(binding,sort_keys=True))
                        status=dict(phase='preparing_calibrated_force',split=role,done=j+1,total=len(rs),
                                    seconds=time.time()-tick,train_samples_scanned=count)
                        atomic_json(out/'preparation_status.json',status);print(json.dumps(status),flush=True)
        mean=sums/count;variance=squares/count-mean*mean
        if np.any(variance<=0) or not np.isfinite(variance).all():raise ValueError('invalid calibrated force variance')
        with np.load(c['emg_norm'],allow_pickle=False) as z:
            emg_mean=z['emg_mean'].copy();emg_std=z['emg_std'].copy()
        atomic_npz(out/'train_norm.npz',emg_mean=emg_mean,emg_std=emg_std,
                   force_mean=mean.astype(np.float32),force_std=np.sqrt(variance).astype(np.float32),
                   source_split_sha256=c['index_sha256'],force_domain='calibrated_N',force_count=count,
                   calibration_lut_sha256=binding['lut_sha256'])
        result=dict(phase='prepared',binding=binding,normalization_sha256=sha(out/'train_norm.npz'),
                    force_mean=mean.tolist(),force_std=np.sqrt(variance).tolist(),force_sample_count=count,
                    raw_EMG_finite_check='inherited only after exact file size/mtime match to prior full raw preparation',
                    force_finite_and_calibration_validation='complete train and val arrays',Test_signal_reads=False)
        atomic_json(ready,result);atomic_json(out/'preparation_status.json',result);print(json.dumps(result),flush=True)


def window_counts(rows):
    n=np.asarray([r['n_samples'] for r in rows],np.int64)
    if np.any(n<4000):raise ValueError('short recording')
    return n//4000+(n%4000!=0)


class WindowReader:
    def __init__(self,c,rows,split):
        self.c,self.rows,self.split=c,rows,split
        self.cal=calibration(c);self.arrays=OrderedDict()
        with np.load(Path(c['run_root'])/'train_norm.npz',allow_pickle=False) as z:
            self.mean=z['emg_mean'].copy();self.std=z['emg_std'].copy()
        self.identities=None
        p=Path(c['run_root'])/f'prepared/{split}_scan.npz'
        if not p.exists() and split=='test':p=Path(c['ranking_prepared'])/'test_stats.npz'
        if p.exists():
            with np.load(p,allow_pickle=False) as z:self.identities=z['file_stats'].copy()

    def open(self,trial):
        if trial in self.arrays:
            self.arrays.move_to_end(trial);return self.arrays[trial]
        row=self.rows[trial];e,f=open_arrays(row,self.c['cache_root'])
        if self.identities is not None:
            ep,fp=paths_for(row,self.c['cache_root'])
            identity=[ep.stat().st_size,ep.stat().st_mtime_ns,fp.stat().st_size,fp.stat().st_mtime_ns]
            if not np.array_equal(identity,self.identities[trial]):raise ValueError('prepared raw data changed')
        self.arrays[trial]=(e,f)
        while len(self.arrays)>self.c.get('max_open_trials_per_worker',8):self.arrays.popitem(last=False)
        return e,f

    def window(self,trial,wid):
        e,f=self.open(trial);row=self.rows[trial]
        start=int(window_starts(len(e))[wid]);t=start+470+40*np.arange(89)
        x=((e[start:start+4000]-self.mean)/np.maximum(self.std,1e-6)).T.copy()
        y=self.cal.forward(np.asarray(f[t]),layout=row['ftype']).T.copy()
        if not np.isfinite(x).all():raise ValueError('nonfinite normalized EMG')
        return x,y,t


class TrainWindows(Dataset):
    def __init__(self,c,rows,plan,rank,world,start_update=0):
        self.c,self.rows,self.plan_path=c,rows,str(plan)
        self.rank,self.world=rank,world;self.offset=start_update*c['microbatch']*c['accumulate']
        self.cumulative=np.cumsum(window_counts(rows));self.reader=None;self.plan=None
        self.size=len(np.load(plan,mmap_mode='r'))//world-self.offset
    def __len__(self):return self.size
    def __getitem__(self,i):
        if self.reader is None:
            self.reader=WindowReader(self.c,self.rows,'train');self.plan=np.load(self.plan_path,mmap_mode='r')
        gid=int(self.plan[(i+self.offset)*self.world+self.rank])
        if gid<0:return dict(x=np.zeros((16,4000),np.float32),y=np.zeros((5,89),np.float32),valid=np.float32(0))
        trial=int(np.searchsorted(self.cumulative,gid,side='right'));base=int(self.cumulative[trial-1]) if trial else 0
        x,y,_=self.reader.window(trial,gid-base)
        return dict(x=x,y=y,valid=np.float32(1))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--workers',type=int,default=16)
    a=parser.parse_args();prepare(load_config(a.config),a.workers)
