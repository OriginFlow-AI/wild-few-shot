"""Exact frozen local paths, gap-contained windows, source-specific time grids."""
from collections import OrderedDict,defaultdict
import gzip,hashlib,json,math,os,sys,zipfile
from pathlib import Path
import numpy as np
from functools import lru_cache
ROOT=Path(__file__).resolve().parent
SOURCE=Path(os.environ.get('RING_RQ_SOURCE',str(ROOT/'data/source')))
sys.path.insert(0,str(ROOT/'source'))
sys.path.insert(0,str(ROOT/'private_deps'))
def read(p):
    if str(p).endswith('.gz'):
        with gzip.open(p,'rt') as f:return json.load(f)
    return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(v,separators=(',',':'),allow_nan=False));tmp.replace(p)
def closed_rows():
    p=ROOT/'data/FULL_SPLIT.json.gz'
    d=read(p if p.exists() else SOURCE/'SPLIT.json.gz');assert d['revision']==2
    return d['splits']
@lru_cache(maxsize=1)
def task43_inventory():
    p=ROOT/'data/TASK43_LOCAL_INVENTORY.json.gz'
    return read(p if p.exists() else SOURCE/'inputs/TASK43_LOCAL_INVENTORY.json.gz')['rows']
def statcheck(row):
    if row['component']=='task83':items=[dict(row['local_stat'],path=row['local_files']['npz'])]
    else:
        source=task43_inventory()[row['local_inventory_index']]
        assert source['source_uri']==row['source_uri']
        items=list(source['files'].values())
    for b in items:
        s=Path(b['path']).stat()
        assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==tuple(b[k] for k in ('dev','ino','bytes','mtime_ns')),('Source cache changed',b['path'])
def zip_arrays(path):
    """NPZ ZIP_STORED members are mapped in place, never extracted to disk."""
    arrays={}
    with zipfile.ZipFile(path) as z:
        assert z.comment==b'old_sos_2khz_v1',('Unbound filter marker',z.comment)
        for key,width in [('emg',16),('force',5)]:
            info=z.getinfo(key+'.npy')
            if info.compress_type!=zipfile.ZIP_STORED:
                with z.open(info) as f:arrays[key]=np.load(f,allow_pickle=False)
                continue
            with open(path,'rb') as f:
                f.seek(info.header_offset);header=f.read(30)
                import struct
                nl,el=struct.unpack_from('<HH',header,26);f.seek(info.header_offset+30+nl+el)
                version=np.lib.format.read_magic(f)
                if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
                elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
                else:
                    with z.open(info) as member:arrays[key]=np.load(member,allow_pickle=False)
                    assert arrays[key].dtype==np.float32 and arrays[key].ndim==2 and arrays[key].shape[1]==width
                    continue
                offset=f.tell()
            assert dtype==np.float32 and len(shape)==2 and shape[1]==width and not fortran
            arrays[key]=np.memmap(path,dtype=dtype,shape=shape,mode='r',offset=offset)
    return arrays['emg'],arrays['force']
def restore_clock(z,kind):return int(z[kind+'_t0_ns'])+np.r_[np.int64(0),np.cumsum(z[kind+'_dt_ns'],dtype=np.int64)]
class LocalReader:
    def __init__(self,max_bytes=2<<30):
        self.cache=OrderedDict();self.bytes=0;self.max_bytes=max_bytes
        self.clocks=read(ROOT/'data/CLOCK_BINDINGS.json')['rows'] if (ROOT/'data/CLOCK_BINDINGS.json').exists() else {}
    def times(self,row):
        if row['id'] in self.clocks:
            bind=self.clocks[row['id']];p=ROOT/bind['path'];expected=bind['sha256']
        elif row.get('native_clock_cache'):
            p=SOURCE/row['native_clock_cache']['path'];expected=row['native_clock_cache']['cache_file_sha256']
        else:
            ident=hashlib.sha256(row['raw_authority']['key'].encode()).hexdigest()[:24]
            receipt=read(ROOT/'data/clocks'/ident[:2]/(ident+'.json'));bind=receipt['rows'][row['id']];p=ROOT/bind['path'];expected=bind['sha256']
        assert sha(p)==expected,'Local clock artifact changed'
        with np.load(p,allow_pickle=False) as z:return restore_clock(z,'emg'),restore_clock(z,'force')
    def load(self,row):
        if row['split'] not in ('train','val'):raise PermissionError('Model Test signals closed')
        key=row['id']
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key][0]
        statcheck(row)
        files=row['local_files']
        if row['component']=='task83':
            e,f=zip_arrays(files['npz']);et,ft=self.times(row)
            assert e.shape==(len(et),16) and f.shape==(len(ft),5)
        elif 'emg' in files:
            e=np.load(files['emg'],mmap_mode='r',allow_pickle=False);f=np.load(files['fz'],mmap_mode='r',allow_pickle=False);et=ft=None
            assert e.dtype==f.dtype==np.float32 and e.shape[1]==16 and f.shape==(len(e),5)
        else:
            assert read(ROOT/'MCAP_FAST_QUALIFIED.json')['status']=='PASS_FULL_ARRAY_AND_TIME_METADATA_EXACT'
            from processed_mcap_fast import decode_mcap
            p=Path(files['mcap']);assert p.is_file()
            # Task43 decoder checks embedded raw identity, not canonical alias.
            source=row['source_uri'].split('/')
            rawsubject=source[-3]
            (e,f),meta=decode_mcap(p.read_bytes(),dict(subject=rawsubject,session=row['session'],stage_raw='wzswl-'+row['stage'].split('/')[-1],n_samples=row.get('samples_2khz')))
            assert meta['hand']==row['hands'][0]['side'];et=ft=None
        from label_policy import apply_source_force
        f=apply_source_force(f,row)  # ONCE at native source grid, before interpolation or statistics.
        v=(e,f,et,ft);nb=sum(a.nbytes for a in v if a is not None)
        self.cache[key]=(v,nb);self.bytes+=nb
        while self.bytes>self.max_bytes and len(self.cache)>1:
            _,(_,old)=self.cache.popitem(last=False);self.bytes-=old
        return v
    def segments(self,row):
        intervals=row['hands'][0]['intervals']
        if row['component']=='task83':
            et,ft=self.times(row);origin=int(et[0]);out=[]
            for a,b in intervals:
                elo=int(np.searchsorted(et,a));ehi=int(np.searchsorted(et,b))-1
                flo=int(np.searchsorted(ft,a));fhi=int(np.searchsorted(ft,b))-1
                if elo>=len(et) or flo>=len(ft) or ehi<elo or fhi<flo:continue
                lo=max(int(et[elo]),int(ft[flo]));hi=min(int(et[ehi]),int(ft[fhi]))
                # Every interpolation neighbor lies inside THIS audited segment.
                out.append(((lo-origin+499999)//500000,(hi-origin)//500000+1))
            return out
        return [(int(a),int(b)) for a,b in intervals]
    def window(self,row,start):
        e,f,et,ft=self.load(row)
        if et is None:x=np.array(e[start:start+4000],dtype=np.float32,copy=True);y=np.array(f[start:start+4000],dtype=np.float32,copy=True)
        else:
            target=int(et[0])+np.arange(start,start+4000,dtype=np.int64)*500000
            assert target[0]>=max(et[0],ft[0]) and target[-1]<=min(et[-1],ft[-1]),'No extrapolation'
            # Subtract integer origins BEFORE float conversion to preserve precision.
            tt=(target-et[0]).astype(np.float64);te=(et-et[0]).astype(np.float64)
            tf=(ft-ft[0]).astype(np.float64);ttf=(target-ft[0]).astype(np.float64)
            lo=max(0,np.searchsorted(et,target[0],side='right')-1);hi=min(len(et),np.searchsorted(et,target[-1],side='left')+1)
            x=np.stack([np.interp(tt,te[lo:hi],e[lo:hi,c]) for c in range(16)],axis=1).astype(np.float32)
            y=np.stack([np.interp(ttf,tf,f[:,c]) for c in range(5)],axis=1).astype(np.float32)
        assert x.shape==(4000,16) and y.shape==(4000,5) and np.isfinite(x).all() and np.isfinite(y).all()
        return x,y
class Index:
    def __init__(self,rows,reader,prepared=None):
        self.rows=rows;self.segments=[];self.counts=[];self.groups=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
        for i,row in enumerate(rows):
            seg=[(int(a),int(b)) for a,b in (prepared[i] if prepared is not None else reader.segments(row)) if b-a>=4000]
            assert seg,('No complete two-second window',row['id'])
            self.segments.append(seg);count=sum(b-a-3999 for a,b in seg);self.counts.append(count)
            self.groups[row['stage']][row['subject']][row['hands'][0]['side']].append(i)
        self.stages=sorted(self.groups)
    def sample(self,epoch,global_index,seed=42):
        # Blocks amortize local MCAP decode. Window draws remain independent.
        block=global_index//16;rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,block,0xE90C]))
        stage=self.stages[int(rng.integers(len(self.stages)))];people=self.groups[stage];person=sorted(people)[int(rng.integers(len(people)))];hands=people[person];hand=sorted(hands)[int(rng.integers(len(hands)))];ids=hands[hand]
        weights=np.array([self.counts[i] for i in ids],np.float64);i=ids[int(rng.choice(len(ids),p=weights/weights.sum()))]
        rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,global_index,0xB47C]));k=int(rng.integers(self.counts[i]))
        for a,b in self.segments[i]:
            n=b-a-3999
            if k<n:return i,a+k
            k-=n
        raise AssertionError('Window accounting')
    def fixed_windows(self):
        for i,segments in enumerate(self.segments):
            for a,b in segments:
                for start in range(a,b-3999,4000):yield i,start
