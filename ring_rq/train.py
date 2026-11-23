"""Eight-rank, finite two-stage Kendall supervision with complete restoration."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,contextlib,fcntl,gzip,hashlib,json,math,random,signal,sys,time,traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
sys.path.insert(0,str(Path(__file__).resolve().parent))
from io_local import ROOT,SOURCE,read,write,sha
EXECUTION_ROOT=Path(__file__).resolve().parent
QUAL_ROOT=ROOT/'qualification'
EXECUTION=read(EXECUTION_ROOT/'EXECUTION.json')
DEV_DIRECTORY=None
from fast_batches import Batcher
from model import build
from source.stable_interpolation import OrderedLinearBackward
from source.ordinal import differences
from losses import total,a_loss
from dev_metrics import Accumulator,window_stats
from ordered_reducer import Reducer
RANK=int(os.environ.get('RANK','0'));WORLD=int(os.environ.get('WORLD_SIZE','1'));LOCAL=int(os.environ.get('LOCAL_RANK','0'))
STOP=False
def stop(*_):
    global STOP;STOP=True
def setup():
    torch.set_num_threads(4);torch.cuda.set_device(LOCAL)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    if WORLD>1:dist.init_process_group('nccl',timeout=timedelta(hours=4))
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed(42)
def barrier():
    if WORLD>1:dist.barrier()
def broadcast(v):
    if WORLD>1:
        data=[v];dist.broadcast_object_list(data,src=0);return data[0]
    return v
def gather(v):
    if WORLD==1:return [v]
    data=[None]*WORLD;dist.all_gather_object(data,v);return data
def raw(m):return m.module if isinstance(m,DDP) else m
def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
def restore_rng(v):random.setstate(v['python']);np.random.set_state(v['numpy']);torch.set_rng_state(v['cpu']);torch.cuda.set_rng_state(v['cuda'])
def atomic_torch(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');torch.save(value,tmp)
    with open(tmp,'rb') as f:os.fsync(f.fileno())
    tmp.replace(p)
def digest_state(value):
    h=hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):h.update(str((v.dtype,tuple(v.shape))).encode());h.update(v.detach().contiguous().cpu().numpy().tobytes())
        elif isinstance(v,np.ndarray):h.update(str((v.dtype,v.shape)).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(str(k).encode());add(v[k])
        elif isinstance(v,(list,tuple)):
            for a in v:add(a)
        else:h.update(repr(v).encode())
    add(value);return h.hexdigest()
def optimizer(m):
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=3e-4,weight_decay=1e-4,betas=(.9,.999),eps=1e-8)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=2,cooldown=1,min_lr=1e-5)
    return opt,sch
def initial(phase,smoke=False):
    m=build().cuda();parent=None
    if phase==2:
        p=QUAL_ROOT/'phase1_parent.pt' if smoke else ROOT/'runs/phase1/best.pt'
        data=torch.load(p,map_location='cpu',weights_only=False);m.load_state_dict(data['model'],strict=True);parent=sha(p)
    return m,parent
def batch_tensors(b):return torch.from_numpy(b['emg']).cuda(non_blocking=True),torch.from_numpy(b['raw'][...,428:]).cuda(non_blocking=True)
def forward_loss(m,b,norm,cal,phase,epoch,logical):
    seed=(42*1000003+epoch*2000033+logical*97409)%(2**63-1);torch.manual_seed(seed);torch.cuda.manual_seed(seed)
    x,y=batch_tensors(b);native=m(x);pred=OrderedLinearBackward.apply(native)
    result=total(pred,y,norm,cal['temperature_N'],seed+phase*10000019,phase,cal['coefficient'])
    assert all(torch.isfinite(z).all() for z in result[:3]);return result
def calibrate(phase,smoke=False):
    path=(QUAL_ROOT/'calibration' if smoke else ROOT/'calibration')/f'phase{phase}.json'
    cached=broadcast(read(path) if RANK==0 and path.exists() else None)
    if cached is not None:return cached
    value=None
    if RANK==0:
        norm=dict(np.load(ROOT/'data/norm.npz'));batcher=Batcher('train',norm);m,parent=initial(phase,smoke)
        from calibration_memory import install
        install(m);m.train()
        panel=[0,348*256,696*256,1044*256];batches=[batcher.train(0,k,256) for k in panel]
        if phase==1:
            columns=[[] for _ in range(5)]
            for i,b in zip(panel,batches):
                delta,_,_=differences(torch.from_numpy(b['raw'][...,428:]),42+i)
                for f in range(5):v=delta[:,f].abs().numpy().ravel();columns[f].append(v[v>0])
            temp=[float(np.median(np.concatenate(v))) for v in columns];assert all(x>0 and np.isfinite(x) for x in temp)
        else:temp=read(path.parent/'phase1.json')['temperature_N']
        parameters=[p for p in m.parameters() if p.requires_grad];records=[]
        for logical,b in zip(panel,batches):
            _,a,rank,parts,cov=forward_loss(m,b,norm,dict(temperature_N=temp,coefficient=1),phase,1,logical)
            norms={}
            for key,loss in [('A_loss',a),('ordinal',rank),('Corr_term',.5*(1-parts['corr'])),('MAPE_term',.1*parts['mape_ratio']),('MAE_term',.15*parts['mae'])]:
                write(path.parent/f'phase{phase}_PROGRESS.json',dict(phase=phase,qualification=smoke,logical=logical,gradient_component=key,time=time.time(),peak_cuda_bytes=torch.cuda.max_memory_allocated(),checkpoint_strategy='nested calibration only; regular training inner only'))
                gs=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
                norms[key]=float(torch.sqrt(sum((g.detach().double().square().sum() for g in gs if g is not None),torch.tensor(0.,device='cuda',dtype=torch.float64))))
                del gs
            assert norms['A_loss']>0 and norms['ordinal']>0 and all(np.isfinite(v) for v in norms.values())
            records.append(dict(logical=logical,gradient_norms=norms,coverage=cov,A_loss=float(a.detach()),ordinal=float(rank.detach())))
        main,aux=('ordinal','A_loss') if phase==1 else ('A_loss','ordinal')
        ratio=float(np.median([v['gradient_norms'][aux]/v['gradient_norms'][main] for v in records]));coef=.075/ratio
        measured=float(np.median([coef*v['gradient_norms'][aux]/v['gradient_norms'][main] for v in records]));assert abs(measured-.075)<1e-10
        value=dict(status='PASS',phase=phase,temperature_N=temp,coefficient=coef,measured_panel_median=measured,panel_indices=panel,records=records,parent_sha256=parent,definition='median weighted auxiliary/main full-trainable-parameter L2 gradient norm ratio before clipping; fixed coefficient per phase',Train_only=True,Test_read=False)
        write(path,value);atomic_torch(path.with_suffix('.pt'),dict(model=m.state_dict(),parent_sha256=parent));batcher.close();del m,batches
        torch.cuda.empty_cache()
    return broadcast(value)
def save(path,m,opt,sch,state,binding):
    states=gather(rng())
    if RANK==0:
        p=Path(path)
        if p.name=='last.pt' and p.exists():os.replace(p,p.with_name('previous.pt'))
        payload=dict(model=raw(m).state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),state=state,rng_by_rank=states,binding=binding)
        atomic_torch(p,payload);write(p.with_suffix('.json'),dict(sha256=sha(p),epoch=state['epoch'],step=state['step'],global_step=state['global_step'],all_optimizer_steps=sorted({int(v['step']) for v in opt.state.values() if 'step' in v})))
        if p.name=='last.pt' and state['step']==0 and state['history']:
            completed=state['history'][-1]['epoch'];dest=ROOT/'snapshots'/f'phase{binding["phase"]}'/f'epoch{completed:03d}.pt'
            if not dest.exists():atomic_torch(dest,dict(model=payload['model'],binding=binding,history_record=state['history'][-1],epoch=completed,phase=binding['phase'],global_step=state['global_step'],weights_only=True))
            write(dest.with_suffix('.json'),dict(sha256=sha(dest),epoch=completed,phase=binding['phase'],global_step=state['global_step'],weights_only=True,Test_read=False))
    barrier()
def restore(path,m,opt,sch,binding):
    receipt=read(Path(path).with_suffix('.json'));assert sha(path)==receipt['sha256']
    p=torch.load(path,map_location='cpu',weights_only=False)
    assert p['binding']==binding,'Independent calibrated run cannot resume raw-label state'
    raw(m).load_state_dict(p['model'],strict=True);opt.load_state_dict(p['optimizer']);sch.load_state_dict(p['scheduler']);restore_rng(p['rng_by_rank'][RANK]);state=p['state']
    assert digest_state(raw(m).state_dict())==digest_state(p['model'])
    assert digest_state(opt.state_dict())==digest_state(p['optimizer']) and digest_state(sch.state_dict())==digest_state(p['scheduler'])
    assert digest_state(rng())==digest_state(p['rng_by_rank'][RANK])
    return state
def dev(m,norm,phase,epoch,limit=None):
    from val_pipeline import evaluate_batches
    before=rng();model=raw(m);model.eval();batcher=Batcher('val',norm)
    specs=[(i,start,k) for k,(i,start) in enumerate(batcher.index.fixed_windows()) if k%WORLD==RANK]
    batcher.close()
    if limit is not None:specs=specs[:limit*64]
    start_time=time.time()
    def progress(count):
        if count%640==0 or count==len(specs):
            write(DEV_DIRECTORY/'execution'/f'VAL_rank{RANK}.json',dict(epoch=epoch,windows=count,total=len(specs),elapsed=time.time()-start_time,time=time.time()))
    acc,_,_=evaluate_batches(model,norm,specs,progress=progress)
    parts=gather(acc.payload());out=None
    if RANK==0:
        merged=Accumulator()
        for part in parts:merged.merge(part)
        regular=set(BatcherStageCache.train_stages());selection=merged.summarize(regular);full=merged.summarize()
        macro=selection['macro'];admissible=macro['tau'] is not None and macro['tau_windows']==macro['tau_eligible_windows'] and macro['people']>0
        out=dict(phase=phase,epoch=epoch,selection=selection,all_Val=full,selection_stage_exclusions=sorted(set(s for p,s in merged.cells)-regular),tau_admissible=admissible,GT_constant_cells=merged.constant,prediction_constant_cells=merged.bad,complete=limit is None,Test_read=False,seconds=time.time()-start_time)
    model.train();restore_rng(before);return broadcast(out)
class BatcherStageCache:
    @staticmethod
    def train_stages():
        from io_local import closed_rows
        return sorted({r['stage'] for r in closed_rows()['train']})
def run(args):
    global DEV_DIRECTORY
    phase=args.phase;smoke=args.mode!='formal';directory=QUAL_ROOT/args.name if smoke else ROOT/f'runs/phase{phase}';directory.mkdir(parents=True,exist_ok=True);DEV_DIRECTORY=directory
    lock=open(directory/'run.lock','a') if RANK==0 else None
    if RANK==0:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    barrier();cal=calibrate(phase,smoke);norm=dict(np.load(ROOT/'data/norm.npz'));batcher=Batcher('train',norm)
    protocol=read(ROOT/'PROTOCOL.json');micro=protocol['micro_per_rank'];global_batch=micro*WORLD
    assert WORLD==8 and global_batch==protocol['global_batch']
    windows=sum((b-a) for seg in batcher.index.segments for a,b in seg)//4000;updates=windows//global_batch
    m,parent=initial(phase,smoke)
    for v in m.state_dict().values():dist.broadcast(v,src=0)
    opt,sch=optimizer(m);binding=dict(protocol_sha256=sha(ROOT/'PROTOCOL.json'),split_sha256=sha(ROOT/'data/FULL_SPLIT.json.gz'),index_sha256=sha(ROOT/'data/INDEX.json.gz'),clocks_sha256=sha(ROOT/'data/CLOCK_BINDINGS.json'),model_code_sha256=sha(ROOT/'model.py'),training_code_sha256=sha(EXECUTION_ROOT/'train.py'),execution_config_sha256=sha(EXECUTION_ROOT/'EXECUTION.json'),label_policy_sha256=sha(ROOT/'LABEL_POLICY.json'),label_bindings_sha256=sha(ROOT/'data/LABEL_BINDINGS.json.gz'),norm_sha256=sha(ROOT/'data/norm.npz'),RQ_sha256=sha(ROOT/'data/RQ_FROZEN.pt'),phase=phase,micro=micro,world=WORLD,global_batch=global_batch,calibration=cal,parent_sha256=parent,smoke=smoke)
    state=dict(epoch=1,step=0,global_step=0,history=[],best_epoch_so_far=None,best_value_so_far=None,best_metrics_so_far=None,no_improve=0,loss_sums=[0.,0.,0.],consumed_windows=0,updates_per_epoch=updates)
    if args.resume:state=restore(directory/'last.pt',m,opt,sch,binding)
    if RANK==0:write(directory/'OWNER.json',dict(pid=os.getpid(),ranks=WORLD,host=__import__('socket').gethostname(),time=time.time(),proc_start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split(') ',1)[1].split()[19],root=str(ROOT),phase=phase))
    reader_pool=ThreadPoolExecutor(max_workers=1);max_epoch=6 if phase==1 else 50;minimum=5 if phase==1 else 1;patience=5 if phase==1 else 10
    target=state['global_step']+args.steps if smoke else None;start=time.time();count=0
    events=[torch.cuda.Event(enable_timing=True) for _ in range(5)]
    while state['epoch']<=max_epoch:
        epoch=state['epoch'];first=(state['step']*WORLD+RANK)*micro
        future=reader_pool.submit(batcher.train,epoch,first,micro) if state['step']<updates else None
        while state['step']<updates:
            wait_started=time.perf_counter();b=future.result();input_wait=time.perf_counter()-wait_started;step=state['step'];logical=step*WORLD+RANK
            if smoke:
                trace=dict(logical=logical,ids=b['ids'],starts=b['starts'],emg_sha256=hashlib.sha256(b['emg'].tobytes()).hexdigest(),raw_sha256=hashlib.sha256(b['raw'].tobytes()).hexdigest())
                with open(directory/f'INPUT_rank{RANK}.jsonl','a') as f:f.write(json.dumps(trace,separators=(',',':'))+'\n')
            if step+1<updates:future=reader_pool.submit(batcher.train,epoch,((step+1)*WORLD+RANK)*micro,micro)
            t=time.time();opt.zero_grad(set_to_none=True);events[0].record();loss,a,rank,parts,cov=forward_loss(m,b,norm,cal,phase,epoch,logical);events[1].record()
            reducer=Reducer(m,WORLD,RANK,directory/f'GRAD_rank{RANK}.jsonl' if smoke else None)
            (loss/WORLD).backward();events[2].record();reducer.add(logical);reducer.finish();events[3].record()
            grad=torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step();events[4].record();torch.cuda.synchronize()
            del reducer
            values=torch.tensor([float(loss.detach()),float(a.detach()),float(rank.detach())],device='cuda',dtype=torch.float64);dist.all_reduce(values);values/=WORLD
            state['step']+=1;state['global_step']+=1;state['consumed_windows']+=global_batch;state['loss_sums']=[x+y for x,y in zip(state['loss_sums'],values.cpu().tolist())];count+=1
            if RANK==0:
                progress_record=dict(phase=phase,epoch=epoch,step=state['step'],updates_per_epoch=updates,global_step=state['global_step'],objective=float(values[0]),A_loss=float(values[1]),ordinal=float(values[2]),gradient_norm_before_clip=float(grad),input_wait_seconds=input_wait,forward_loss_seconds=events[0].elapsed_time(events[1])/1000,backward_seconds=events[1].elapsed_time(events[2])/1000,gradient_gather_seconds=events[2].elapsed_time(events[3])/1000,clip_Adam_seconds=events[3].elapsed_time(events[4])/1000,gpu_compute_seconds=time.time()-t,wall_seconds=time.time()-start,average_seconds_per_update=(time.time()-start)/count,peak_cuda_bytes=torch.cuda.max_memory_allocated(),time=time.time(),best_epoch_so_far=state['best_epoch_so_far'],best_value_so_far=state['best_value_so_far'],stage_counts={s:b['stages'].count(s) for s in set(b['stages'])})
                write(directory/'PROGRESS.json',progress_record)
                with open(directory/'PERFORMANCE.jsonl','a') as log:log.write(json.dumps(progress_record,separators=(',',':'))+'\n')
            must_stop=torch.tensor(int(STOP or (target is not None and state['global_step']>=target)),device='cuda');dist.all_reduce(must_stop,op=dist.ReduceOp.MAX)
            if state['step']%16==0 or bool(must_stop):save(directory/'last.pt',m,opt,sch,state,binding)
            if must_stop:
                if smoke:
                    # Also exercise read-only Train->Val transition on a fixed small Val batch.
                    result=dev(m,norm,phase,epoch,limit=1)
                    if RANK==0:write(directory/'DEV_SMOKE.json',result)
                fingerprints=gather(dict(rank=RANK,model=digest_state(raw(m).state_dict()),optimizer=digest_state(opt.state_dict()),scheduler=digest_state(sch.state_dict()),state=digest_state(state),rng=digest_state(rng())))
                if RANK==0:
                    write(directory/'EXIT_STATE.json',dict(status='PASS_BOUNDED_STOP' if smoke else 'SAFE_STOP',fingerprints=fingerprints,global_step=state['global_step']))
                    if smoke and phase==1:atomic_torch(QUAL_ROOT/'phase1_parent.pt',dict(model=raw(m).state_dict()))
                reader_pool.shutdown();batcher.close();return
        # All fixed Val windows, all ranks, no model Test reads.
        metrics=dev(m,norm,phase,epoch);macro=metrics['selection']['macro'];value=macro['tau'] if phase==1 else macro['L_A']
        admissible=metrics['tau_admissible'] if phase==1 else value is not None
        improved=admissible and (state['best_value_so_far'] is None or (value>state['best_value_so_far'] if phase==1 else value<state['best_value_so_far']))
        if improved:
            state.update(best_epoch_so_far=epoch,best_value_so_far=value,best_metrics_so_far={k:macro[k] for k in ('L_A','MAE','MAPE_percent','Corr','R2','tau')},no_improve=0)
        else:state['no_improve']+=1
        snapshot=f'dev/epoch{epoch:03d}.json.gz'
        history=dict(epoch=epoch,phase=phase,Train_objective=state['loss_sums'][0]/updates,Train_A_loss=state['loss_sums'][1]/updates,Train_ordinal=state['loss_sums'][2]/updates,metrics={k:macro[k] for k in ('L_A','MAE','MAPE_percent','Corr','R2','tau')},bins=macro['bins'],dev_snapshot=snapshot,selection_metric='tau_b' if phase==1 else 'L_A',direction='max' if phase==1 else 'min',best_epoch_so_far=state['best_epoch_so_far'],best_value_so_far=state['best_value_so_far'],best_metrics_so_far=state['best_metrics_so_far'])
        state['history'].append(history);sch.step(-value if phase==1 and value is not None else value if value is not None else float('inf'))
        if RANK==0:
            q=directory/snapshot;q.parent.mkdir(parents=True,exist_ok=True)
            with gzip.open(q,'wt') as f:json.dump(metrics,f,separators=(',',':'),allow_nan=False)
            write(directory/'HISTORY.json',state['history'])
        if improved:save(directory/'best.pt',m,opt,sch,state,binding)
        done=epoch==max_epoch or (epoch>=minimum and state['no_improve']>=patience)
        state.update(epoch=epoch+1,step=0,loss_sums=[0.,0.,0.]);save(directory/'last.pt',m,opt,sch,state,binding)
        if done:
            assert state['best_epoch_so_far'] is not None,'No admissible complete Validation best'
            best=torch.load(directory/'best.pt',map_location='cpu',weights_only=False);raw(m).load_state_dict(best['model'],strict=True)
            if RANK==0:write(directory/'RUN_DONE.json',dict(status='PASS_COMPLETE_VAL_BEST_RELOADED',best_epoch=state['best_epoch_so_far'],best_value=state['best_value_so_far'],best_metrics=state['best_metrics_so_far'],best_sha256=sha(directory/'best.pt'),total_updates=state['global_step'],epochs=epoch,Test_read=False))
            reader_pool.shutdown();batcher.close();return
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['calibrate','calibrate_qualification','qualify','formal']);p.add_argument('--phase',type=int,choices=[1,2],required=True);p.add_argument('--name',default='reference');p.add_argument('--steps',type=int,default=2);p.add_argument('--resume',action='store_true');args=p.parse_args();setup();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        if args.mode in ('calibrate','calibrate_qualification'):calibrate(args.phase,args.mode=='calibrate_qualification')
        else:
            if args.mode=='formal':assert read(ROOT/'ADMISSION.json')['status']=='PASS_FULL_EIGHT_RANK_RESTORE_AND_BOTH_PHASES'
            run(args)
    except Exception:
        write(ROOT/f'FAILED_rank{RANK}.json',dict(traceback=traceback.format_exc(),time=time.time(),args=vars(args)));raise
    finally:
        if dist.is_initialized():dist.destroy_process_group()
if __name__=='__main__':main()
