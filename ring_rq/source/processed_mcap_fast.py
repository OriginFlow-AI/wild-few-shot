"""CPU MCAP label validator; compiled framing/protobuf loop, original array semantics.

Not a training reader. Unique module name avoids the original source's fast_mcap.
Must pass full-array/time equality against the frozen reader.
"""
import hashlib
import io
import re
import numpy as np
from numba import njit

@njit(cache=True,nogil=True)
def uint(data,p,n):
    assert p+n<=len(data)
    x=np.uint64(0)
    for i in range(n):x|=np.uint64(data[p+i])<<np.uint64(8*i)
    return x

@njit(cache=True,nogil=True)
def varint(data,p,end):
    value=np.uint64(0);shift=0
    while True:
        assert p<end and shift<=63
        b=data[p];p+=1
        if shift==63:assert b<=1
        value|=np.uint64(b&127)<<np.uint64(shift)
        if b<128:return value,p
        shift+=7

@njit(cache=True,nogil=True)
def chunk_values(data,emg_id,force_id,emg,force,et,ft,counters):
    p=0;end=len(data)
    while p<end:
        assert p+9<=end
        op=data[p];size=int(uint(data,p+1,8));p+=9;stop=p+size;assert stop<=end
        if op!=5:
            assert op==3 or op==4
            p=stop;continue
        assert size>=22
        channel=int(uint(data,p,2));log=uint(data,p+6,8);publish=uint(data,p+14,8);p+=22
        if channel!=emg_id and channel!=force_id:p=stop;continue
        assert log==publish and log<=np.uint64(9223372036854775807)
        k=0 if channel==emg_id else 1;i=counters[k];assert i<emg.shape[0]
        sync=np.uint64(0);pressure_count=0
        while p<stop:
            tag,p=varint(data,p,stop);field=int(tag>>np.uint64(3));wire=int(tag&np.uint64(7));assert field>0
            if wire==0:
                value,p=varint(data,p,stop)
                if field==1:sync=value
                if k==1 and field==4:
                    assert value<=np.uint64(4294967295) and pressure_count<9
                    if pressure_count<5:force[i,pressure_count]=np.float32(value)
                    pressure_count+=1
            elif wire==5:
                value=uint(data,p,4);p+=4;assert p<=stop
                if k==0 and 5<=field<=20:emg[i,field-5]=np.uint32(value).view(np.float32)
            elif wire==1:p+=8;assert p<=stop
            elif wire==2:
                length,p=varint(data,p,stop);limit=p+int(length);assert limit<=stop
                if k==1 and field==4:
                    while p<limit:
                        value,p=varint(data,p,limit);assert value<=np.uint64(4294967295) and pressure_count<9
                        if pressure_count<5:force[i,pressure_count]=np.float32(value)
                        pressure_count+=1
                    assert p==limit
                else:p=limit
            else:assert False
        assert p==stop and sync==log
        if k==0:et[i]=np.int64(log)
        else:assert pressure_count==9;ft[i]=np.int64(log)
        counters[k]+=1
    assert p==end

def decode_mcap(raw,row):
    from mcap.reader import make_reader
    from mcap.stream_reader import StreamReader,get_chunk_data_stream
    from mcap.records import Chunk,Message
    reader=make_reader(io.BytesIO(raw),validate_crcs=True);summary=reader.get_summary();assert summary and summary.statistics
    metadata={m.name:dict(m.metadata) for m in reader.iter_metadata()};task=metadata.get('/session/task',{})
    assert task.get('subject_id')==row['subject'] and str(task.get('task_id'))=='43'
    assert str(task.get('stage'))==row['stage_raw'] and str(task.get('session_id'))==row['session']
    channels=list(summary.channels.values());fs=[c for c in channels if re.fullmatch('/cmcu05b/(left|right)/cmcu05b',c.topic)]
    assert len(fs)==1;f=fs[0];hand=f.topic.split('/')[2]
    es=[c for c in channels if c.topic=='/band/'+hand+'/sEMG'];assert len(es)==1 and sum(c.topic.endswith('/sEMG') for c in channels)==1;e=es[0]
    assert hashlib.sha256(summary.schemas[e.schema_id].data).hexdigest()=='246024803dad70cd14078ae63e6b5786f0839f12777022a544f855b61eefef9e'
    assert hashlib.sha256(summary.schemas[f.schema_id].data).hexdigest()=='bc7bb803a317b62d54c194028829a3c2dbffba74914048b76c87a5ddd4916984'
    for c,unit,pre in ((e,'uV','toolbox_semg_filter_v1'),(f,'mN','toolbox_force_native_v1')):
        assert c.message_encoding=='protobuf' and c.metadata.get('unit')==unit and c.metadata.get('preprocess')==pre
        assert c.metadata.get('grid')=='pulse_uniform_2khz_common_overlap' and c.metadata.get('rate')=='2000'
    assert e.metadata['channel_count']=='16' and f.metadata['channel_count']=='9'
    n=summary.statistics.channel_message_counts[e.id]
    assert n==row['n_samples']==summary.statistics.channel_message_counts[f.id]==int(e.metadata['samples'])==int(f.metadata['samples']) and n>=4000
    emg=np.zeros((n,16),np.float32);force=np.zeros((n,5),np.float32);et=np.empty(n,np.int64);ft=np.empty(n,np.int64);counts=np.zeros(2,np.int64)
    chunks=0
    for record in StreamReader(io.BytesIO(raw),emit_chunks=True,validate_crcs=True).records:
        if isinstance(record,Chunk):
            stream,length=get_chunk_data_stream(record,validate_crc=True);body=stream.read(length);assert len(body)==length
            chunk_values(np.frombuffer(body,np.uint8),e.id,f.id,emg,force,et,ft,counts);chunks+=1
        elif isinstance(record,Message):raise AssertionError('Unchunked messages require original fallback review')
    assert chunks==summary.statistics.chunk_count and np.all(counts==n)
    force*=np.float32(.001)
    for values,times in ((emg,et),(force,ft)):
        assert np.isfinite(values).all();delta=np.diff(times)
        assert np.all((delta>=499950)&(delta<=500050)) and np.ptp(delta)<=2
    assert np.array_equal(et,ft)
    return (emg,force),{'n_samples':n,'hand':hand,'force_unit':'N','source_force_unit':'mN','force_channels':[0,1,2,3,4],
        'emg_unit':'uV','already_processed':True,'first_timestamp_ns':int(et[0]),'last_timestamp_ns':int(et[-1]),
        'grid_step_ns_min':int(np.diff(et).min()),'grid_step_ns_max':int(np.diff(et).max()),'no_additional_filtering_or_resampling':True}
