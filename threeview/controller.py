"""Explicit candidate-only deployment; no server assignment or baseline launch."""
from pathlib import Path
import argparse,os,subprocess,sys,time,shutil
from io_local import ROOT,read,write,sha,closed_rows
from release import code_digest
def call(mode,phase,extra=(),label=None):
 cmd=[sys.executable,'-B']
 if mode in ('formal','qualify'):cmd+=['-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=8']
 cmd+=[str(ROOT/'train.py'),mode,'--phase',str(phase)]+list(extra)
 log=ROOT/f'{label or mode}_p{phase}.log'
 env=dict(os.environ,TRITON_F32_DEFAULT='ieee',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 with log.open('ab',buffering=0) as f:
  child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
  write(ROOT/'ACTIVE_CHILD.json',dict(pid=child.pid,args=cmd,time=time.time()));result=child.wait()
 if result:raise RuntimeError(f'{mode} phase{phase} exit{result}; inspect {log}')
def check_assets():
 rows=closed_rows();roles=list(rows)
 for i,a in enumerate(roles):
  assert all(r['split']==a for r in rows[a]);assert len({r['id'] for r in rows[a]})==len(rows[a])
  for b in roles[i+1:]:
   assert not {r['subject'] for r in rows[a]} & {r['subject'] for r in rows[b]},('Subject leakage',a,b)
   assert not {(r['subject'],r['session']) for r in rows[a]} & {(r['subject'],r['session']) for r in rows[b]},('Session leakage',a,b)
   assert not {r['id'] for r in rows[a]} & {r['id'] for r in rows[b]},('Recording leakage',a,b)
 assert rows['train'] and rows['val']
 assert all(r['stage'] and r['stage'] not in ('legacy/4','legacy/5','task83/mix-9','task83/mix-10') for r in rows['train'])
 for n in ('data/INDEX.json.gz','data/CLOCK_BINDINGS.json','data/LABEL_BINDINGS.json.gz','data/norm.npz','tokenizer/TOKENIZER_FROZEN.pt','tokenizer/TOKENIZER_FROZEN.json','LABEL_POLICY.json'):assert (ROOT/n).is_file(),n
def calibrate_and_qualify(phase):
 if phase==2:assert read(ROOT/'runs/phase1/RUN_DONE.json')['status']=='PASS_COMPLETE_VAL_BEST_RELOADED'
 if not (ROOT/f'calibration/phase{phase}.json').exists():call('calibrate',phase)
 cal=read(ROOT/f'calibration/phase{phase}.json');assert cal['status']=='PASS' and abs(cal['measured_panel_median']-.075)<1e-10 and not cal['Test_read']
 qc=ROOT/'qualification/calibration';qc.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/f'calibration/phase{phase}.json',qc/f'phase{phase}.json')
 if phase==2:
  parent=ROOT/'runs/phase1/best.pt';assert cal['parent_sha256']==sha(parent);shutil.copyfile(parent,ROOT/'qualification/phase1_parent.pt')
 admit=ROOT/f'ADMISSION_phase{phase}.json'
 if admit.exists():
  a=read(admit);assert a['code_sha256']==code_digest() and a['calibration']==cal
  for n,digest in a['assets'].items():assert sha(ROOT/n)==digest,n
  return
 for name,steps,resume in [(f'p{phase}_continuous',2,False),(f'p{phase}_split',1,False),(f'p{phase}_split',1,True)]:
  if not resume and (ROOT/'qualification'/name).exists():raise FileExistsError('Inspect existing qualification output before retrying: '+name)
  call('qualify',phase,['--name',name,'--steps',str(steps)]+(['--resume'] if resume else []),name+('_restore' if resume else ''))
 a=read(ROOT/f'qualification/p{phase}_continuous/EXIT_STATE.json');b=read(ROOT/f'qualification/p{phase}_split/EXIT_STATE.json');assert a['global_step']==b['global_step']==2 and a['fingerprints']==b['fingerprints']
 for rank in range(8):
  for prefix in ('INPUT','GRAD'):
   assert (ROOT/f'qualification/p{phase}_continuous/{prefix}_rank{rank}.jsonl').read_bytes()==(ROOT/f'qualification/p{phase}_split/{prefix}_rank{rank}.jsonl').read_bytes()
 assets={n:sha(ROOT/n) for n in ('PROTOCOL.json','data/FULL_SPLIT.json.gz','data/INDEX.json.gz','data/CLOCK_BINDINGS.json','data/LABEL_BINDINGS.json.gz','data/TASK43_LOCAL_INVENTORY.json.gz','LABEL_POLICY.json','data/norm.npz','tokenizer/TOKENIZER_FROZEN.pt','tokenizer/TOKENIZER_FROZEN.json')}
 write(admit,dict(status='PASS_EXACT_RESTORE_AND_COMMON55',phase=phase,code_sha256=code_digest(),assets=assets,calibration=cal,continuous=a,restored=b,Test_read=False))
def main():
 import fcntl
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('all','qualify','formal'));p.add_argument('--phase',type=int,choices=(1,2),default=1);args=p.parse_args()
 lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_assets()
 if args.mode=='qualify':calibrate_and_qualify(args.phase);return
 for phase in (1,2):
  run=ROOT/f'runs/phase{phase}'
  if (run/'RUN_DONE.json').exists():continue
  if args.mode=='formal' and phase==1:assert (ROOT/'ADMISSION_phase1.json').exists(),'Run qualification first'
  calibrate_and_qualify(phase)
  write(ROOT/'TRAINING_STARTED.json',dict(time=time.time(),controller_pid=os.getpid(),phase=phase,code_sha256=code_digest(),Test_read=False))
  call('formal',phase,['--resume'] if (run/'last.pt').exists() else [])
  assert read(run/'RUN_DONE.json')['status']=='PASS_COMPLETE_VAL_BEST_RELOADED'
 write(ROOT/'RUN_DONE.json',dict(status='PASS_BOTH_STAGES_COMPLETE',Test_read=False,time=time.time()))
if __name__=='__main__':main()
