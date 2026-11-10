"""Resumable preparation -> real-data GPU smoke -> three training branches."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from common import BRANCHES, EXPERIMENT, atomic, load, settings, sha, source_binding
from extract import extract_many
from prepare import freeze_plan, prepare_routes, build_dataset
from gpu_selector import choose

def train_shared(root,smoke=False):
    c=settings();scripts=Path(__file__).resolve().parent
    tuning=root/'checks/gpu_tuning.json'
    if not tuning.exists():
        while True:
            gpu=choose(c['gpu_memory_limit_GiB'],c['gpu_free_reserve_GiB'],minimum_limit_GiB=16.)
            if gpu:break
            atomic(root/'STATUS.json',dict(phase='waiting_for_shared_GPU_headroom_for_benchmark'))
            time.sleep(20)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TMPDIR=os.environ.get('TMPDIR',str(root/'tmp')))
        atomic(root/'checks/GPU_benchmark.json',gpu)
        with (root/'logs/GPU_benchmark.log').open('a') as f:
            subprocess.run([sys.executable,'-B',str(scripts/'probe.py'),'--root',str(root),'--limit',str(gpu['process_memory_limit_GiB'])],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    record=load(tuning);tune=record['chosen']
    requested=c.get('common_microbatch')
    if requested and tune['microbatch']!=requested:
        attempts=[r for r in record['attempts'] if r['microbatch']==requested and 'windows_per_second' in r]
        if {r['branch'] for r in attempts}!=set(BRANCHES):raise ValueError('Requested common microbatch was not verified for all objectives')
        record['fastest_single_GPU_candidate']=tune
        tune=dict(microbatch=requested,windows_per_second=min(r['windows_per_second'] for r in attempts),
                  peak_allocated_GiB=max(r['peak_allocated_GiB'] for r in attempts))
        record['chosen']=tune;record['selection_reason']='Three independent GPUs and common microbatch prioritize experiment makespan before requested deadline'
        atomic(tuning,record)
    minimum_limit=max(8.,math.ceil(tune['peak_allocated_GiB']+.5))
    if not smoke and c.get('concurrent_training_branches',1)>1:
        return parallel_training(root,c,tune,minimum_limit)
    for branch in BRANCHES:
        done=root/('checks/smoke_'+branch+'/PASS.json' if smoke else 'runs/'+branch+'/DONE.json')
        if done.exists() and (not smoke or load(done).get('microbatch')==tune['microbatch']):continue
        while True:
            gpu=choose(c['gpu_memory_limit_GiB'],c['gpu_free_reserve_GiB'],minimum_limit_GiB=minimum_limit)
            if gpu:break
            atomic(root/'STATUS.json',dict(phase='waiting_for_shared_GPU_headroom',branch=branch,memory_limit_GiB=c['gpu_memory_limit_GiB']))
            time.sleep(20)
        log=root/'logs'/(("smoke_" if smoke else "train_")+branch+'.log')
        atomic(root/'checks'/('GPU_'+('smoke_' if smoke else '')+branch+'.json'),gpu)
        atomic(root/'STATUS.json',dict(phase='real_data_GPU_smoke' if smoke else 'training',branch=branch,GPU=gpu,
                                     process_memory_limit_GiB=gpu['process_memory_limit_GiB']))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],PYTHONDONTWRITEBYTECODE='1',
                 OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TMPDIR=os.environ.get('TMPDIR',str(root/'tmp')),
                 TASK103_MICROBATCH=str(tune['microbatch']),TASK103_GPU_MEMORY_GIB=str(gpu['process_memory_limit_GiB']))
        command=[sys.executable,'-B',str(scripts/'train.py'),'--root',str(root),'--branch',branch]
        if smoke:
            command.append('--smoke');env['TASK103_SMOKE_WINDOWS']=str(c['global_batch'])
        with log.open('a') as f:
            p=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT)
            atomic(root/'checks'/('process_'+branch+'.json'),dict(pid=p.pid,command=command,GPU=gpu,smoke=smoke))
            code=p.wait()
        if code:raise RuntimeError(f'{branch} exited {code}; inspect {log}')
        if not done.exists():raise RuntimeError('Trainer exited without completion receipt')

def parallel_training(root,c,tune,minimum_limit):
    """Different experiments use separate GPUs; never select another user's PID."""
    scripts=Path(__file__).resolve().parent
    pending=[b for b in BRANCHES if not (root/'runs'/b/'DONE.json').exists()]
    active={};retries={b:0 for b in pending}
    class Adopted:
        def __init__(self,pid,branch):self.pid,self.branch=pid,branch
        def poll(self):
            p=Path('/proc')/str(self.pid)/'stat'
            if p.exists() and p.read_text().split()[2]!='Z':return None
            return 0 if (root/'runs'/self.branch/'DONE.json').exists() else 1
    for branch in list(pending):
        receipt=root/'checks'/('process_'+branch+'.json')
        if not receipt.exists():continue
        previous=load(receipt);path=Path('/proc')/str(previous['pid'])/'cmdline'
        if previous.get('smoke') or not path.exists():continue
        argv=[x.decode() for x in path.read_bytes().split(b'\0') if x]
        if argv!=previous['command']:continue
        active[branch]=dict(process=Adopted(previous['pid'],branch),gpu=previous['GPU'],log_offset=previous.get('log_offset',0))
        pending.remove(branch)
    while pending or active:
        for branch,job in list(active.items()):
            code=job['process'].poll()
            if code is None:continue
            del active[branch]
            if code==0 and (root/'runs'/branch/'DONE.json').exists():continue
            log=(root/'logs'/('train_'+branch+'.log')).read_text()[job.get('log_offset',0):][-12000:]
            if 'out of memory' in log.lower() and retries[branch]<3:
                retries[branch]+=1;pending.append(branch)
            else:raise RuntimeError(f'{branch} exited {code}; inspect train log')
        while pending and len(active)<c['concurrent_training_branches']:
            gpu=choose(c['gpu_memory_limit_GiB'],c['gpu_free_reserve_GiB'],
                       exclude=[j['gpu']['index'] for j in active.values()],minimum_limit_GiB=minimum_limit)
            if gpu is None:break
            branch=pending.pop(0)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],PYTHONDONTWRITEBYTECODE='1',
                OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                TMPDIR=os.environ.get('TMPDIR',str(root/'tmp')),TASK103_MICROBATCH=str(tune['microbatch']),
                TASK103_GPU_MEMORY_GIB=str(gpu['process_memory_limit_GiB']))
            command=[sys.executable,'-B',str(scripts/'train.py'),'--root',str(root),'--branch',branch]
            log_path=root/'logs'/('train_'+branch+'.log');offset=log_path.stat().st_size if log_path.exists() else 0
            with log_path.open('a') as f:
                process=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT)
            active[branch]=dict(process=process,gpu=gpu,log_offset=offset)
            atomic(root/'checks'/('process_'+branch+'.json'),dict(pid=process.pid,command=command,GPU=gpu,smoke=False,log_offset=offset))
            atomic(root/'checks'/('GPU_'+branch+'.json'),gpu)
        statuses={}
        for b,job in active.items():
            path=root/'runs'/b/'STATUS.json'
            statuses[b]=dict(pid=job['process'].pid,GPU=job['gpu'],training=load(path) if path.exists() else {'phase':'starting'})
        atomic(root/'STATUS.json',dict(phase='parallel_training_or_waiting_for_capacity',active=statuses,pending=pending,
            configured_parallel_branches=c['concurrent_training_branches'],common_microbatch=tune['microbatch'],
            target_deadline_Asia_Shanghai=c.get('target_deadline_Asia_Shanghai')))
        if pending or active:time.sleep(10)

def main(root,workers=8,smoke_ids=None,prepare_only=False):
    root=Path(root).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    for name in ['logs','tmp','checks','cache','assets']:(root/name).mkdir(exist_ok=True)
    if os.environ.get('TASK103_REQUIRE_PUB_DATA')=='1' and not str(root).startswith('/pub_data/zjl/'):
        raise ValueError('This server execution must write under /pub_data/zjl')
    with (root/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic(root/'PIPELINE_PROCESS.json',dict(pid=os.getpid(),root=str(root),source=str(EXPERIMENT),config_sha256=sha(EXPERIMENT/'config.json')))
        try:
            plan=freeze_plan(root);rows=plan['rows']
            if smoke_ids:rows=[r for r in rows if r['id'] in set(smoke_ids)]
            cal=[r for r in rows if r['stage'].startswith('mix-test-') and r['active']]
            atomic(root/'STATUS.json',dict(phase='extracting_calibration_and_holdout_streams',records=len(cal),smoke=bool(smoke_ids)))
            extract_many(root,cal,workers,'calibration_smoke' if smoke_ids else 'calibration')
            quality=prepare_routes(root,cal)
            ordinary=[r for r in rows if r['split']=='train' and r['active'] and quality.get(r['source_id'],{}).get('channels')]
            atomic(root/'STATUS.json',dict(phase='extracting_ordinary_streams',records=len(ordinary),smoke=bool(smoke_ids)))
            extract_many(root,ordinary,workers,'ordinary_smoke' if smoke_ids else 'ordinary')
            atomic(root/'STATUS.json',dict(phase='building_common_masks_windows_and_train_force_statistics'))
            existing=root/'prepared/dataset.json'
            reuse=False
            if existing.exists() and not smoke_ids:
                prepared=load(existing)
                reuse=prepared.get('plan_sha256')==sha(root/'plan.json') and prepared.get('config_sha256')==sha(EXPERIMENT/'config.json') and prepared.get('source_code')==source_binding()
            if not reuse:build_dataset(root,rows)
            if not prepare_only:
                train_shared(root,smoke=True)
                if not smoke_ids:
                    train_shared(root)
                    subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('report.py')),'--root',str(root)],check=True)
            atomic(root/'STATUS.json',dict(phase='smoke_complete' if smoke_ids else 'prepared' if prepare_only else 'complete',
                training_branches=BRANCHES,experiments=[1,2,3,4],smoke=bool(smoke_ids)))
        except Exception as exc:
            atomic(root/'STATUS.json',dict(phase='failed',error=str(exc)));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--smoke-id',action='append');p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();main(a.root,a.workers,a.smoke_id,a.prepare_only)
