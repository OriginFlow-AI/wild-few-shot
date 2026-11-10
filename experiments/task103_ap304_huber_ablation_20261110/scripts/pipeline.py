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
from common import BRANCHES, EXPERIMENT, atomic, load, settings, sha
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
    tune=load(tuning)['chosen'];minimum_limit=max(8.,math.ceil(tune['peak_allocated_GiB']+2.))
    for branch in BRANCHES:
        done=root/('checks/smoke_'+branch+'/PASS.json' if smoke else 'runs/'+branch+'/DONE.json')
        if done.exists():continue
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
        if smoke:command.append('--smoke')
        with log.open('a') as f:
            p=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT)
            atomic(root/'checks'/('process_'+branch+'.json'),dict(pid=p.pid,command=command,GPU=gpu,smoke=smoke))
            code=p.wait()
        if code:raise RuntimeError(f'{branch} exited {code}; inspect {log}')
        if not done.exists():raise RuntimeError('Trainer exited without completion receipt')

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
            build_dataset(root,rows)
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
