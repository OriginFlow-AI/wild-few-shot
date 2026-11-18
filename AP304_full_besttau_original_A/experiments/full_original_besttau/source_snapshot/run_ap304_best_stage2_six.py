"""Preserve the replaced run at an automatic checkpoint, then launch best-tau S2."""
from pathlib import Path
import fcntl
import json
import os
import signal
import subprocess
import sys
import time

import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/ap304_full_calibrated_stage2_besttau_lambda0_six_seed43_20261029'
OLD=ROOT/'runs/ap304_full_calibrated_stage2_lambda0_seed43_20261028'
sys.path.insert(0,str(OUT/'source_snapshot'))
from emg2force.ap304_rank_data import atomic_json,sha


def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False


def arguments(pid):return [x.decode() for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if x]


def descendants(pid):
    result=[pid]
    try:children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
    except FileNotFoundError:children=[]
    for child in children:result.extend(descendants(int(child)))
    return result


def stop_old_at_checkpoint():
    previous=OUT/'transition_receipt.json'
    if previous.exists():return json.loads(previous.read_text())
    status=json.loads((OLD/'status.json').read_text())
    original_pid=json.loads((OLD/'training_process.json').read_text())['pid']
    if status['phase']=='complete' and not alive(original_pid):
        receipt=dict(phase='old_run_already_complete',old_run=str(OLD))
        atomic_json(previous,receipt);return receipt
    args=arguments(original_pid)
    if str(OLD/'resolved_config.yaml') not in args or 'emg2force.ap304_stage2_train' not in args:
        raise ValueError('old training PID does not match the intended experiment')
    update=int(status.get('global_update',0));target=(update//1000+1)*1000
    source=OLD/'checkpoints/last.pt';temporary=OLD/'checkpoints/best_restart_capture.pt';seen=None
    while True:
        if not alive(original_pid):raise RuntimeError('old trainer exited while waiting for checkpoint')
        stat=source.stat();identity=(stat.st_ino,stat.st_mtime_ns,stat.st_size)
        if identity!=seen:
            temporary.unlink(missing_ok=True);os.link(source,temporary)
            ck=torch.load(temporary,map_location='cpu',weights_only=True,mmap=True);state=dict(ck['state']);del ck
            if state['global_update']>=target or (state['epoch']>0 and state['next_update']==0):break
            temporary.unlink();seen=identity
        atomic_json(OUT/'status.json',dict(phase='waiting_for_old_checkpoint_before_switch',
            old_target_update=target,old_status=json.loads((OLD/'status.json').read_text()),
            new_source_global_update=67982,reason='preserve old model+optimizer+RNG before replacement'))
        time.sleep(2)
    archive=OLD/'checkpoints'/f"superseded_update{state['global_update']:06d}.pt"
    if archive.exists():
        if sha(archive)!=sha(temporary):raise ValueError('old checkpoint archive collision')
        temporary.unlink()
    else:os.replace(temporary,archive)
    digest=sha(archive)
    # Only this exact experiment's launcher/process tree is stopped. Other GPU
    # jobs are not selected by GPU id, process name, or a broad kill command.
    supervisor=json.loads((OLD/'pipeline_process.json').read_text())['pid']
    if alive(supervisor) and not any(x.endswith('/run_ap304_stage2_pipeline.py') for x in arguments(supervisor)):
        raise ValueError('old supervisor PID does not match')
    targets=([supervisor] if alive(supervisor) else [])+descendants(original_pid)
    targets=list(dict.fromkeys(targets));paused=[]
    try:
        for pid in targets:
            if alive(pid):os.kill(pid,signal.SIGSTOP);paused.append(pid)
        atomic_json(OUT/'old_run_preserved_before_stop.json',dict(checkpoint=str(archive),sha256=digest,
            state=state,exact_target_pids=targets,training_status_before_stop=json.loads((OLD/'status.json').read_text())))
        for pid in paused:
            if alive(pid):os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
        deadline=time.monotonic()+20
        while any(alive(pid) for pid in targets) and time.monotonic()<deadline:time.sleep(.2)
        for pid in targets:
            if alive(pid):os.kill(pid,signal.SIGKILL)
    finally:
        for pid in paused:
            if alive(pid):os.kill(pid,signal.SIGCONT)
    if any(alive(pid) for pid in targets):raise RuntimeError('old experiment process tree still alive')
    receipt=dict(phase='stopped_by_request',reason='User requested new stage2 from stage1 best tau checkpoint',
        checkpoint=str(archive),sha256=digest,state=state,old_run=str(OLD),
        old_processes_stopped=targets,new_run=str(OUT),new_ranking_source_global_update=67982)
    atomic_json(OLD/'superseded_by_best_restart.json',receipt)
    atomic_json(OLD/'status.json',receipt);atomic_json(previous,receipt)
    return receipt


def main():
    torch.set_num_threads(1)
    with (OUT/'.pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(OUT/'pipeline_process.json',dict(pid=os.getpid(),script=str(Path(__file__).resolve())))
        c=yaml.safe_load((OUT/'resolved_config.yaml').read_text());stage='verifying_preflight'
        try:
            for name,digest in json.loads((OUT/'source_hashes.json').read_text()).items():
                if sha(OUT/name)!=digest:raise ValueError('frozen input/source checksum mismatch: '+name)
            for name,expected in [('integration_check.json','PASS_REAL_TRAIN_CHECKPOINT_INTEGRATION'),
                                  ('cpu_smoke.json','PASS_CPU_REGRESSION_SMOKE')]:
                if json.loads((OUT/name).read_text())['status']!=expected:raise ValueError('preflight not passed')
            if not json.loads((OUT/'six_metric_checks.json').read_text())['passed']:raise ValueError('six metric checks not passed')
            stage='preserving_and_stopping_old_run';stop_old_at_checkpoint()
            stage='waiting_for_gpu_capacity'
            while True:
                result=subprocess.run(['nvidia-smi','--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],
                    capture_output=True,text=True,check=True)
                memory=[tuple(map(int,line.split(','))) for line in result.stdout.splitlines()]
                if all(memory[i][0]>memory[i][1]*c['gpu_memory_fraction']+1024 for i in c['gpu_ids']):break
                atomic_json(OUT/'status.json',dict(phase=stage,memory_free_total_mib=memory));time.sleep(10)
            env=os.environ.copy();env.update(PYTHONPATH=str(OUT/'source_snapshot'),OMP_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
                CUDA_VISIBLE_DEVICES=','.join(map(str,c['gpu_ids'])))
            command=[sys.executable,'-B','-u','-m','torch.distributed.run','--standalone',
                '--nproc-per-node='+str(c['devices']),'--module','emg2force.ap304_stage2_six_train',
                '--config',str(OUT/'resolved_config.yaml'),'--mode','train']
            stage='training_best_checkpoint_stage2'
            atomic_json(OUT/'status.json',dict(phase='starting_stage2',source_global_update=67982,lambda_ord=0))
            with (OUT/'train.log').open('a') as log:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                atomic_json(OUT/'training_process.json',dict(pid=process.pid,command=command));code=process.wait()
            if code:raise RuntimeError('new trainer exited with code '+str(code))
            if json.loads((OUT/'status.json').read_text())['phase'] not in ('complete','stopped_by_request'):
                raise RuntimeError('new trainer exited without a completion receipt')
        except Exception as exc:
            atomic_json(OUT/'status.json',dict(phase='failed',stage=stage,error=str(exc)));raise


if __name__=='__main__':main()
