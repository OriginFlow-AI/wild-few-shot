"""Choose a shared GPU by observed free memory and utilization, with a reserve."""
import subprocess
import time

def snapshot():
    r=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu',
                      '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    rows=[]
    for line in r.stdout.splitlines():
        index,uuid,total,free,util=[x.strip() for x in line.split(',')]
        rows.append(dict(index=int(index),uuid=uuid,total_MiB=int(total),free_MiB=int(free),utilization=int(util)))
    return rows

def choose(limit_GiB=32.,reserve_GiB=8.,exclude=(),minimum_limit_GiB=8.):
    observations=[snapshot()]
    for _ in range(2):time.sleep(1);observations.append(snapshot())
    candidates=[]
    for latest in observations[-1]:
        index=latest['index'];samples=[next(r for r in s if r['index']==index) for s in observations]
        minimum=min(r['free_MiB'] for r in samples)
        # Reserve 2 GiB for CUDA/NCCL outside the PyTorch allocator, in addition
        # to the external workload reserve. Do not double-reserve by a fraction.
        budget=min(limit_GiB, int(max(0., minimum/1024-reserve_GiB-2.))//4*4)
        if budget<minimum_limit_GiB or index in exclude:continue
        row=dict(latest,minimum_free_MiB=minimum,mean_utilization=sum(r['utilization'] for r in samples)/len(samples))
        row['process_memory_limit_GiB']=budget
        # Sharing is authorized. Free capacity dominates, utilization breaks close ties.
        row['score']=minimum-row['mean_utilization']*64
        candidates.append(row)
    return max(candidates,key=lambda r:r['score']) if candidates else None
