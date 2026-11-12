"""Recover EMG and native force for the frozen v4.05 Test without opening vision features."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys

import numpy as np


HANDS = ('left', 'right')
OLD_MOUNT = Path('/mnt/cpfs/processed/gongji/53_rerun')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def atomic(path, document):
    tmp = path.with_suffix('.partial.json')
    tmp.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def frozen_test(benchmark, oss):
    root = benchmark/'input'
    ready = read(root/'READY.json')
    test = ready['datasets']['test']
    split = read(root/'split_v4_05.json')
    path = root/'test_loss_masks.npy'
    assert sha(path)==test['loss_masks_sha256']
    masks = np.load(path,allow_pickle=False)
    with (root/'test_dataset.pkl').open('rb') as f:
        index = pickle.load(f)
    assert index.tokens==80 and index.burn_in==40 and index.partition=='test'
    assert masks.shape==(len(index.eval_index),index.tokens) and masks.dtype==bool
    assert len(index.rows)==len(split['test'])==test['trial_count']==109
    assert {row['trial_id'] for row in index.rows}==set(split['test'])
    assert len({row['subject_id'] for row in index.rows})==test['subjects']==6
    inputs = read(root/'cache_bindings.json')
    assert sha(root/'cache_bindings.json')==ready['cache_bindings_sha256']
    by_trial = defaultdict(list)
    h = hashlib.sha256()
    for (trial,hand,start),mask in zip(index.eval_index,masks):
        rid=index.rows[trial]['trial_id']
        assert hand in HANDS and start>=0 and mask.any() and not mask[:index.burn_in].any()
        by_trial[trial].append((hand,int(start),mask))
        for offset in np.flatnonzero(mask):
            h.update(f'{rid}\t{hand}\t{start+int(offset)}\n'.encode())
    assert masks.sum()==test['evaluation_token_count']==500040
    assert h.hexdigest()==test['evaluation_token_identity_sha256']
    tasks=[]
    for i,row in enumerate(index.rows):
        rid=row['trial_id']
        assert inputs['caches'][row['cache_path']]['trial_id']==rid
        assert row['emg_preprocessing_contract_id']=='task53_rrd_as_delivered_no_extra_filter_v1'
        rrd,checker = (oss/Path(row[key]).relative_to(OLD_MOUNT) for key in ('rrd_path','checker_path'))
        assert rrd.name==rid and rrd.stat().st_size==row['rrd_file_identity']['size_bytes']
        assert sha(checker)==row['checker_sha256']
        tasks.append((row,rrd,checker,by_trial[i]))
    assets=benchmark/'assets/new_calibration'
    assert sha(assets/'coefficients.json')==ready['calibration_coefficients_sha256']
    assert sha(assets/'tables.npz')==ready['calibration_tables_sha256']
    receipt=dict(trials=len(tasks),subjects=6,windows=len(index.eval_index),timepoints=int(masks.sum()),
                 mask_sha256=sha(path),identity_sha256=h.hexdigest(),
                 pickle_sha256=sha(root/'test_dataset.pkl'),split_sha256=sha(root/'split_v4_05.json'),
                 calibration_tables_sha256=ready['calibration_tables_sha256'])
    return tasks,receipt


def read_signals(path):
    from fusion2force.data import rrd
    from rerun.experimental import RrdReader

    groups=defaultdict(list)
    desired={rrd.MCAP_META_ENTITY}
    for hand in HANDS:
        desired.update(rrd._signal_entity('emg',hand,c) for c in rrd.EMG_CHANNEL_ORDER)
        desired.update(rrd._signal_entity('force',hand,c) for c in rrd.FORCE_CHANNEL_ORDER)
    stream=RrdReader(path).stream().filter(content=['/signals/emg/**','/signals/force/**',rrd.MCAP_META_ENTITY],
                                         components=[rrd.SCALAR_COMPONENT,*rrd._MCAP_META_COMPONENTS])
    for chunk in stream:
        entity=str(chunk.entity_path)
        if entity in desired:
            groups[entity].append(chunk.to_record_batch())
    assert set(groups)==desired, (path,desired-set(groups))
    meta=rrd._static_components(groups[rrd.MCAP_META_ENTITY],entity_path=rrd.MCAP_META_ENTITY)
    units,channel_order,schema=rrd._verify_force_metadata(meta)
    assert units=='mN' and tuple(channel_order[:5])==rrd.FORCE_CHANNEL_ORDER
    arrays={}
    for hand in HANDS:
        emg_t,emg=rrd._read_signal_group(groups,modality='emg',hand=hand,channel_order=rrd.EMG_CHANNEL_ORDER)
        force_t,force=rrd._read_signal_group(groups,modality='force',hand=hand,channel_order=rrd.FORCE_CHANNEL_ORDER)
        arrays.update({
            f'emg_{hand}_values':np.ascontiguousarray(emg,dtype=np.float32),
            f'emg_{hand}_time_s':emg_t.astype(np.float64)*1e-9,
            f'force_{hand}_values_N':np.ascontiguousarray(force.astype(np.float32)*np.float32(1e-3)),
            f'force_{hand}_time_s':force_t.astype(np.float64)*1e-9,
        })
    return arrays,schema


def verify_support(emg_time, query):
    from fusion2force.data.dataset import receptive_valid
    origin=query[0]-428/2000
    index=np.floor((query-origin)*2000+1e-7).astype(np.int64)
    grid=origin+np.arange(index[-1]+1,dtype=np.float64)/2000
    source=np.searchsorted(emg_time,grid,side='right')-1
    delta=grid-emg_time[np.maximum(source,0)]
    supported=(source>=0)&(delta>=-1e-12)&(delta<=.002+1e-12)
    return receptive_valid(supported,index,428)


def extract(job):
    output,benchmark,stage_rrd,(row,rrd,checker,windows)=job
    rid=row['trial_id']
    cache=output/'signals'/f'{rid}.npz'
    provenance=output/'signals'/f'{rid}.json'
    if cache.is_file() and provenance.is_file():
        prior=read(provenance)
        assert prior['cache_sha256']==sha(cache) and prior['rrd_size_bytes']==rrd.stat().st_size
        assert prior['checker_sha256']==row['checker_sha256']
        return prior
    from data import ForceCalibration

    if stage_rrd:
        spool=output/'rrd_spool'/rid
        if not spool.is_file():
            tmp=spool.with_suffix('.partial.rrd')
            with rrd.open('rb') as source, tmp.open('wb') as dest:
                shutil.copyfileobj(source,dest,8<<20)
            assert tmp.stat().st_size==rrd.stat().st_size
            tmp.replace(spool)
        assert spool.stat().st_size==rrd.stat().st_size
        print('Staged RRD',rid,flush=True)
    else:
        spool=rrd
    arrays,schema=read_signals(spool)
    lut=ForceCalibration(benchmark/'assets/new_calibration')
    owned=defaultdict(list)
    for hand,start,mask in windows:
        emg=arrays[f'emg_{hand}_time_s']
        query=arrays[f'force_{hand}_time_s'][start:start+80]
        assert len(query)==80 and np.all(~mask|verify_support(emg,query)), (rid,hand,start)
        owned[hand].extend(start+np.flatnonzero(mask))
    for hand in HANDS:
        indices=np.asarray(owned[hand],dtype=np.int64)
        assert np.array_equal(indices,np.unique(indices))
        arrays[f'{hand}_score_indices']=indices
        force=arrays[f'force_{hand}_values_N']
        arrays[f'force_{hand}_calibrated_N']=lut(force)
    tmp=cache.with_suffix('.partial.npz')
    np.savez(tmp,**arrays)
    tmp.replace(cache)
    report=dict(trial_id=rid,subject_id=row['subject_id'],stage=row['stage'],rrd=str(rrd),checker=str(checker),
                checker_sha256=row['checker_sha256'],rrd_size_bytes=rrd.stat().st_size,force_schema=schema,
                emg_samples={h:len(arrays[f'emg_{h}_time_s']) for h in HANDS},
                force_samples={h:len(arrays[f'force_{h}_time_s']) for h in HANDS},
                owned_points={h:len(owned[h]) for h in HANDS},cache_sha256=sha(cache))
    atomic(provenance,report)
    if stage_rrd:
        spool.unlink()
    return report


def main(benchmark,oss,output,workers,stage_rrd):
    sys.path.insert(0,str(benchmark/'source'))
    tasks,receipt=frozen_test(benchmark,oss)
    (output/'signals').mkdir(parents=True,exist_ok=True)
    if stage_rrd:
        (output/'rrd_spool').mkdir(exist_ok=True)
        tasks.sort(key=lambda task: (output/'rrd_spool'/f"{task[0]['trial_id']}.partial").exists())
    atomic(output/'frozen_inputs.json',receipt)
    print('Verified all RRD sizes and checker hashes; extracting',len(tasks),'trials',flush=True)
    successes,errors=[],{}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(extract,(output,benchmark,stage_rrd,task)):task[0]['trial_id'] for task in tasks}
        for i,future in enumerate(as_completed(pending),1):
            rid=pending[future]
            try:
                successes.append(future.result())
            except Exception as error:
                errors[rid]=f'{type(error).__name__}: {error}'
                print(rid,errors[rid],flush=True)
            if i%10==0 or i==len(tasks):
                print(f'Signals {i}/{len(tasks)}; errors {len(errors)}',flush=True)
    atomic(output/'signal_receipt.json',dict(
        **receipt,errors=errors,recordings=sorted(successes,key=lambda r:r['trial_id']),
        owned_points=sum(sum(r['owned_points'].values()) for r in successes)))
    assert not errors and len(successes)==109
    assert sum(sum(r['owned_points'].values()) for r in successes)==receipt['timepoints']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('benchmark','oss','output'):
        parser.add_argument(f'--{name}',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--stage-rrd',action='store_true')
    args=parser.parse_args()
    main(args.benchmark,args.oss,args.output,args.workers,args.stage_rrd)
