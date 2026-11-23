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
                item.update(eligible_resampled_grid_hours=samples/7200000,fixed_2s_windows=windows,fixed_input_hours=windows/1800,scored_sample_count=windows*55)
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
    p=argparse.ArgumentParser(description='Prepare approved gap-contained index and Train-only norm.')
    p.add_argument('--index-only',action='store_true')
    args=p.parse_args()
    assert not (ROOT/'TRAINING_STARTED.json').exists(),'Never refit active training'
    rows,segments=build_index() if not (ROOT/'data/INDEX.json.gz').exists() else (closed_rows(),read(ROOT/'data/INDEX.json.gz'))
    if args.index_only:return
    assert not (ROOT/'data/norm.npz').exists(),'Refuse overwriting a fitted norm'
    qualification(rows,segments);fitted_assets(rows,segments)
    write(ROOT/'PREP_DONE.json',dict(status='PASS_TRAIN_ONLY_NORM',norm_sha256=sha(ROOT/'data/norm.npz'),Train_only=True,Test_read=False))
if __name__=='__main__':main()
