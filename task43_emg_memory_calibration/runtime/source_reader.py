"""Read-only local mmap or ZIP_STORED S3 Range input. No implicit SSH fallback."""
import io
import json
import os
import struct
import threading
import zipfile
from pathlib import Path
import hashlib
import numpy as np
import s3_readonly

ROOT=Path(__file__).resolve().parent


def relative_name(value):
    value=str(value).replace('\\','/')
    if not value or value.startswith('/') or ':' in value or any(p in ('','..','.') for p in value.split('/')):
        raise ValueError('Expected a safe relative recording name')
    return value


def atomic(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.{threading.get_ident()}.tmp')
    tmp.write_bytes(payload);os.replace(tmp,path)


class RemoteFile:
    def __init__(self,source,key):
        self.source,self.key,self.pos,self.block=source,key,0,None
        response=source.client().request('HEAD',source.bucket,key)
        h={k.lower():v for k,v in response['headers'].items()}
        self.size=int(h['content-length']);self.etag=h['etag'];self.modified=h['last-modified']

    def seek(self,n,whence=0):
        self.pos=n if whence==0 else self.pos+n if whence==1 else self.size+n
        return self.pos

    def tell(self):return self.pos
    def seekable(self):return True

    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        n=min(n,self.size-self.pos)
        if not n:return b''
        start=self.pos;end=start+n;self.pos=end
        if self.block is not None:
            offset,b=self.block
            if offset<=start and end<=offset+len(b):return b[start-offset:end-offset]
        stop=min(self.size,max(end,start+4096))
        response=self.source.client().request('GET',self.source.bucket,self.key,
            headers={'Range':f'bytes={start}-{stop-1}','If-Match':self.etag,'If-Unmodified-Since':self.modified},limit=stop-start)
        h={k.lower():v for k,v in response['headers'].items()}
        if response['status']!=206 or len(response['body'])!=stop-start or h.get('content-range')!=f'bytes {start}-{stop-1}/{self.size}':
            raise ValueError('S3 Range status, length or Content-Range mismatch')
        self.block=start,response['body'];return response['body'][:n]


class Source:
    def __init__(self,config,client_factory=None):
        self.config=config
        self.local_roots=[Path(p).expanduser().resolve() for p in config.get('local_roots',[])]
        self.metadata_root=Path(config['metadata_cache']).expanduser().resolve()
        self.bucket=config.get('s3_bucket','')
        self.prefix=config.get('s3_prefix','').strip('/')
        self.local=threading.local();self.client_factory=client_factory
        for name in ['access_key_id','secret_access_key','session_token']:
            if name in config:raise ValueError('Secrets belong in environment or an external private credential file')

    def client(self):
        if not hasattr(self.local,'client'):
            self.local.client=self.client_factory() if self.client_factory else s3_readonly.Client(
                self.config.get('endpoint_url'),self.config.get('region'),self.config.get('credentials_file'))
        return self.local.client

    def _metadata(self,record,field):
        if not self.bucket:raise FileNotFoundError('No local input and no configured S3 bucket')
        relative=relative_name(record['relative'])
        key='/'.join(p for p in [self.prefix,relative+'.npz'] if p)
        binding=json.dumps([self.config.get('endpoint_url') or os.environ.get('S3_ENDPOINT_URL'),self.bucket,key,record['n'],record['hand'],field],separators=(',',':'))
        path=self.metadata_root/(hashlib.sha256(binding.encode()).hexdigest()+'.json')
        if path.exists():
            result=json.loads(path.read_bytes())
            if result['binding']!=binding:raise ValueError('Header cache identity mismatch')
            if record.get('etag') and result['etag']!=record['etag']:raise ValueError('Frozen source ETag mismatch')
            return result
        f=RemoteFile(self,key)
        if record.get('etag') and f.etag!=record['etag']:raise ValueError('Frozen source ETag mismatch')
        member='emg' if field=='emg' else record.get('force_member','cmcu05b_'+record['hand']+'_force_mn')
        if field=='fz' and record['layout']!='1x5x1':
            raise ValueError('S3 force member is qualified for CMCU 1x5x1 only; use an explicit raw-N local cache for other layouts')
        result=dict(binding=binding,key=key,size=f.size,etag=f.etag,modified=f.modified)
        with zipfile.ZipFile(f) as z:
            if len(z.namelist())!=len(set(z.namelist())):raise ValueError('Duplicate NPZ fields')
            info=z.getinfo(member+'.npy')
            if info.compress_type!=zipfile.ZIP_STORED or info.flag_bits&1:raise ValueError('Range input requires an uncompressed, unencrypted ZIP_STORED member')
            f.seek(info.header_offset);header=f.read(30);a,b=struct.unpack_from('<HH',header,26);begin=info.header_offset+30+a+b
            f.seek(begin);version=np.lib.format.read_magic(f)
            readers={(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}
            if version not in readers:raise ValueError('Only NPY1.0/2.0 numeric headers are qualified')
            shape,order,dtype=readers[version](f)
            width=16 if field=='emg' else 5
            if dtype!=np.dtype('float32') or len(shape)!=2 or shape[0]!=record['n'] or shape[1]<width or (field=='emg' and shape[1]!=16):
                raise ValueError('NPY shape/dtype mismatch')
            if f.tell()-begin+int(np.prod(shape))*4!=info.file_size:raise ValueError('NPY payload length mismatch')
            result.update(offset=f.tell(),shape=list(shape),fortran=bool(order))
        atomic(path,json.dumps(result,separators=(',',':')).encode());return result

    def _local_path(self,record,field):
        relative=relative_name(record['relative'])
        explicit=record.get(field+'_path')
        if explicit:
            path=Path(explicit).expanduser().resolve()
            if not any(path.is_relative_to(root) for root in self.local_roots):raise PermissionError('Explicit input path is outside configured read-only local roots')
            return path if path.is_file() else None
        for root in self.local_roots:
            path=(root/(relative+'.'+field+'.npy')).resolve()
            if not path.is_relative_to(root):raise PermissionError('Input path escapes local root')
            if path.is_file():return path
        return None

    def _rows(self,record,field,start,stop):
        if field not in ('emg','fz') or not 0<=start<stop<=record['n']:raise ValueError('Field or row bounds')
        path=self._local_path(record,field)
        channels=16 if field=='emg' else 5
        if path:
            value=np.load(path,mmap_mode='r',allow_pickle=False)
            if value.dtype!=np.float32 or value.shape!=(record['n'],channels):raise ValueError('Local raw cache schema mismatch')
            value=value[start:stop].copy()
            if field=='fz' and record.get('local_force_unit','N')!='N':raise ValueError('Local fz cache must already be raw N, not mN or calibrated N')
        else:
            meta=self._metadata(record,field);width=meta['shape'][1]*4
            f=RemoteFile.__new__(RemoteFile);f.source=self;f.key=meta['key'];f.size=meta['size'];f.etag=meta['etag'];f.modified=meta['modified'];f.block=None;f.pos=meta['offset']+start*width
            if meta['fortran']:
                columns=[]
                for channel in range(channels):
                    f.block=None;f.pos=meta['offset']+(channel*record['n']+start)*4
                    columns.append(np.frombuffer(f.read((stop-start)*4),np.float32).copy())
                value=np.ascontiguousarray(np.stack(columns,axis=1))
            else:value=np.frombuffer(f.read((stop-start)*width),np.float32).reshape(stop-start,-1)[:,:channels].copy()
            if field=='fz':
                if record.get('s3_force_unit','mN')!='mN':raise ValueError('S3 CMCU force member must be explicitly raw mN')
                value*=np.float32(.001)
        if not np.isfinite(value).all():raise ValueError('Nonfinite input window')
        return value

    def rows(self,record,field,start,stop):
        for attempt in range(2):
            try:return self._rows(record,field,start,stop)
            except s3_readonly.ReadError:
                if attempt:raise


_source=None


def configured():
    global _source
    if _source is None:_source=Source(json.loads((ROOT/'DATA.json').read_bytes()))
    return _source


def rows(record,field,start,stop):return configured().rows(record,field,start,stop)


def ensure_prefix(record):
    """Supports are read on demand; no implicit persistent waveform copy."""
    if record['n']<120000:raise ValueError('Support must contain a complete60-second prefix')
