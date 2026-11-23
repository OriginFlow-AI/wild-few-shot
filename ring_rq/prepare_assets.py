"""Finite local consumer qualification, Train-only normalization and new RQ fit."""
import concurrent.futures as cf,gzip,hashlib,json,os,sys,time,traceback
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import numpy as np
from io_local import ROOT,SOURCE,LocalReader,Index,closed_rows,read,write,sha
def build_index():
    rows=closed_rows();r=LocalReader();out={}
    for role in ('train','val'):
        segments=[]
        for j,row in enumerate(rows[role]):
            seg=[(int(a),int(b)) for a,b in r.segments(row) if b-a>=4000]
            assert seg,('NO_VALID_2S_WINDOW',row['id']);segments.append(seg)
            if j%2000==0:write(ROOT/'PREP_PROGRESS.json',dict(phase='index',role=role,done=j,total=len(rows[role])))
        out[role]=segments
    with gzip.open(ROOT/'data/INDEX.json.gz','wt') as f:json.dump(out,f,separators=(',',':'))
    summary={}
    for role,rr in rows.items():
        groups={}
        for i,row in enumerate(rr):groups.setdefault(row['stage'],[]).append(i)
        def summarize(ids):
            item=dict(people=len({rr[i]['subject'] for i in ids}),recordings=len(ids),effective_hand_hours=sum(rr[i]['effective_hand_ns'] for i in ids)/3.6e12)
            if role in out:
                samples=sum(b-a for i in ids for a,b in out[role][i]);windows=sum((b-a)//4000 for i in ids for a,b in out[role][i])
                item.update(eligible_resampled_grid_hours=samples/7200000,fixed_2s_windows=windows,fixed_input_hours=windows/1800,scored_hours=windows*3572/7200000)
            else:item.update(eligible_resampled_grid_hours=None,reason='Test signals closed; membership and trusted technical support frozen only')
            return item
        summary[role]=dict(total=summarize(list(range(len(rr)))),stages={s:summarize(ids) for s,ids in sorted(groups.items())})
    write(ROOT/'data/SPLIT_SUMMARY.json',summary)
    return rows,out
def fitted_assets(rows,segments):
    # The fixed normalization/tokenizer panel is declared before opening waves.
    # This is a sample-fit new norm, never claimed to be whole-population moments.
    groups={}
    for i,row in enumerate(rows['train']):groups.setdefault(row['stage'],[]).append(i)
    rng=np.random.default_rng(20261120);plan=[]
    for stage,ids in sorted(groups.items()):
        choose=rng.choice(ids,size=min(96,len(ids)),replace=False)
        for i in choose:
            seg=segments['train'][int(i)];a,b=seg[int(rng.integers(len(seg)))];start=int(rng.integers(a,b-3999))
            plan.append(dict(index=int(i),id=rows['train'][int(i)]['id'],stage=stage,start=start))
    write(ROOT/'data/FIT_PANEL.json',dict(Train_only=True,seed=20261120,stages=sorted(groups),windows=plan,window_samples=4000,force_points_per_window=256,definition='96 without-replacement recordings per stage; deterministic gap-contained random two-second windows; stage balanced sample norm'))
    import threading
    local=threading.local()
    def get(q):
        if not hasattr(local,'reader'):local.reader=LocalReader(max_bytes=256<<20)
        e,f=local.reader.window(rows['train'][q['index']],q['start'])
        # Float64 merge via raw moments, sufficient accuracy for finite panel.
        z=np.c_[e,f].astype(np.float64)
        k=np.linspace(0,3999,256,dtype=np.int64)
        return z.sum(0),np.square(z).sum(0),len(z),f[k].copy()
    sums=np.zeros(21);squares=np.zeros(21);count=0;points=[];start=time.time()
    with cf.ThreadPoolExecutor(max_workers=16) as pool:
        # executor.map preserves panel order regardless of I/O completion.
        for j,(a,b,n,p) in enumerate(pool.map(get,plan)):
            sums+=a;squares+=b;count+=n;points.append(p)
            if j%100==0:write(ROOT/'PREP_PROGRESS.json',dict(phase='Train_only_norm',done=j,total=len(plan),seconds=time.time()-start))
    mean=sums/count;std=np.sqrt(np.maximum(squares/count-mean**2,1e-12));assert np.isfinite(mean).all() and np.all(std>1e-6)
    np.savez(ROOT/'data/norm.npz',emg_mean=mean[:16].astype(np.float32),emg_std=std[:16].astype(np.float32),force_mean=mean[16:].astype(np.float32),force_std=std[16:].astype(np.float32))
    y=(np.concatenate(points)-mean[16:].astype(np.float32))/std[16:].astype(np.float32)
    np.save(ROOT/'data/force_fit_points.npy',y.astype(np.float32))
    write(ROOT/'data/NORM_DONE.json',dict(Train_only=True,Val_fit=False,Test_read=False,panel_windows=len(plan),sample_points=count,fit_points=len(y),norm_sha256=sha(ROOT/'data/norm.npz'),panel_sha256=sha(ROOT/'data/FIT_PANEL.json'),seconds=time.time()-start))
def fit_rq():
    import torch
    sys.path.insert(0,str(ROOT/'source'));from rq_model import ForceTokenizer
    torch.set_num_threads(8);torch.manual_seed(42);np.random.seed(42);torch.use_deterministic_algorithms(True)
    import scipy
    write(ROOT/'RUNTIME.json',dict(python=sys.version,python_executable=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__,scipy=scipy.__version__,CUDA_devices=torch.cuda.device_count(),GPU0=torch.cuda.get_device_name(0),private_dependencies=['See requirements.txt; versions recorded from the executing environment'],shared_environment_modified=False))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # Train-only points get a deterministic internal held-out partition.
    y=np.load(ROOT/'data/force_fit_points.npy',mmap_mode='r');rng=np.random.default_rng(42);order=rng.permutation(len(y));split=int(len(order)*.9)
    train=torch.from_numpy(np.array(y[order[:split]],copy=True)).cuda();dev=torch.from_numpy(np.array(y[order[split:]],copy=True)).cuda()
    model=ForceTokenizer().cuda();model.initialize(train[:200000]);opt=torch.optim.Adam(model.parameters(),lr=.001)
    history=[];best=float('inf');best_state=None;stale=0;generator=torch.Generator(device='cuda').manual_seed(42)
    for epoch in range(1,31):
        model.train();losses=[]
        for step in range(128):
            ids=torch.randint(len(train),(2048,),device='cuda',generator=generator);batch=train[ids]
            opt.zero_grad(set_to_none=True);pred,commit,_=model(batch,update=True);loss=(pred-batch).square().mean()+.25*commit
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            error=0.;n=0
            for block in dev.split(4096):pred,_,_=model(block);error+=float((pred-block).square().sum());n+=block.numel()
            rmse=(error/n)**.5
        if rmse<best-1e-6:
            best=rmse;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0;best_epoch=epoch
        else:stale+=1
        history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),internal_Train_holdout_normalized_RMSE=rmse,best_epoch_so_far=best_epoch,best_value_so_far=best))
        write(ROOT/'RQ_PROGRESS.json',dict(epoch=epoch,rmse=rmse,best=best,stale=stale))
        if epoch>=5 and stale>=5:break
    assert best_state is not None and best<=.05,('Train-only tokenizer quality gate',best)
    torch.save(dict(model=best_state,history=history,best_epoch=best_epoch,normalization_sha256=sha(ROOT/'data/norm.npz'),Train_only=True),ROOT/'data/RQ_FROZEN.pt')
    write(ROOT/'data/RQ_DONE.json',dict(status='PASS',best_epoch=best_epoch,best_normalized_RMSE=best,threshold=.05,history=history,sha256=sha(ROOT/'data/RQ_FROZEN.pt'),internal_holdout='point-disjoint within fixed Train-only panel; not subject-generalization evidence',Test_read=False))
def qualification(rows,segments):
    reader=LocalReader();seen={};results=[]
    for role in ('train','val'):
        groups={}
        for i,r in enumerate(rows[role]):groups.setdefault((r['component'],r['source'],r['stage'],r['hands'][0]['side']),i)
        for key,i in sorted(groups.items()):
            row=rows[role][i];a,b=segments[role][i][0];e,f=reader.window(row,a)
            again=reader.window(row,a)
            assert np.array_equal(e,again[0]) and np.array_equal(f,again[1])
            result=dict(role=role,id=row['id'],stage=row['stage'],source=row['source'],side=row['hands'][0]['side'],start=a,emg_shape=list(e.shape),force_shape=list(f.shape),emg_peak=float(abs(e).max()),force_range=[float(f.min()),float(f.max())],repeat_exact=True)
            if row['component']=='task83':
                full_e,full_f,et,ft=reader.load(row)
                t=et[0]+np.arange(a,a+4000,dtype=np.int64)*500000
                for c in range(5):
                    oracle=np.interp((t-ft[0]).astype(float),(ft-ft[0]).astype(float),full_f[:,c]).astype(np.float32)
                    assert np.array_equal(f[:,c],oracle)
                result['independent_time_alignment_exact']=True
                assert all(not (aa<=a and a+4000>bb) for aa,bb in segments[role][i] if aa<=a<bb)
            results.append(result)
    write(ROOT/'DATA_CONSUMER_QUALIFIED.json',dict(status='PASS_SOURCE_SPECIFIC_UNITS_CLOCKS_SHAPES_GAPS_LOCAL_ONLY',windows=results,Test_waveform_reads=0,no_additional_filter=True,force_unit='N',source_data_sha_recheck=False))
def main():
    import argparse
    parser=argparse.ArgumentParser(description='Train-only norm and RQ fit on an approved local manifest.')
    parser.add_argument('--build-index',action='store_true')
    args=parser.parse_args()
    assert not (ROOT/'TRAINING_STARTED.json').exists(),'Never refit active or previously started training'
    assert not (ROOT/'data/RQ_FROZEN.pt').exists() and not (ROOT/'data/norm.npz').exists(),'Refuse overwriting fitted assets'
    rows=closed_rows()
    if args.build_index:rows,segments=build_index()
    else:segments=read(ROOT/'data/INDEX.json.gz')
    qualification(rows,segments);fitted_assets(rows,segments);fit_rq()
    write(ROOT/'PREP_DONE.json',dict(status='PASS',norm_sha256=sha(ROOT/'data/norm.npz'),rq_sha256=sha(ROOT/'data/RQ_FROZEN.pt'),Train_only=True,Test_read=False))
if __name__=='__main__':main()
