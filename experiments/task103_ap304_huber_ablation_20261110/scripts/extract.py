"""Bounded streaming extraction. Does not save videos or whole MCAP objects."""
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from mcap.reader import make_reader
from common import VERSION, atomic, client, load, sha
from mcap_io import RangeReader, decoder, unique
from preprocessing import regularize

TOPICS = ['/bsq/middle/force', '/cmcu05b/left/cmcu05b', '/cmcu05b/right/cmcu05b',
          '/band/left/sEMG', '/band/right/sEMG']

def one(root, row):
    root = Path(root); out = root/'cache/emg'/row['id']; out.mkdir(parents=True, exist_ok=True)
    native_file = root/'cache/native'/(row['id']+'.npz'); native_file.parent.mkdir(parents=True, exist_ok=True)
    done = out/'done.json'
    if done.exists() and native_file.exists():
        d = load(done)
        if d['version'] != VERSION or d['etag'] != row['etag']: raise ValueError('Cache identity mismatch')
        if 'processed_files' not in d:
            d['processed_files']={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.glob('*.npy')}
            atomic(done,d)
        return d
    if shutil.disk_usage(root).free < 100*2**30: raise RuntimeError('Disk reserve below 100 GiB')
    source = RangeReader(client(), row, os.environ.get('TASK103_S3_BUCKET', 'tes-bucket'))
    classes, units, clocks = {}, {}, {}
    force = {t: [] for t in TOPICS[:3]}
    emg = {s: dict(t=array('Q'), recv=array('Q'), x=array('f')) for s in ['left', 'right']}
    try:
        for schema, ch, message in make_reader(source).iter_messages(topics=TOPICS, log_time_order=False):
            if schema.id not in classes: classes[schema.id] = decoder(schema)
            m = classes[schema.id].FromString(message.data)
            t = int(m.sync_ts_ns or message.log_time); recv = int(m.recv_ts_ns)
            if not m.sync_ts_ns: raise ValueError('Missing synchronized timestamp')
            units[ch.topic] = ch.metadata
            if '/sEMG' in ch.topic:
                a = emg[ch.topic.split('/')[2]]; a['t'].append(t); a['recv'].append(recv)
                a['x'].extend(getattr(m, f'ch{j:02d}') for j in range(16))
            elif ch.topic == TOPICS[0]: force[ch.topic].append((t, recv, float(m.force_n)))
            else: force[ch.topic].append((t, recv, list(m.ad_values[:5]), list(m.pressures[:5])))
    finally: source.close()
    native, offsets, temporal = {}, {}, {}
    for name, topic in zip(['bsq', 'left', 'right'], TOPICS[:3]):
        rows = force[topic]
        if not rows: raise ValueError('Missing force stream: '+name)
        raw_t = np.array([r[0] for r in rows], np.int64)
        temporal[name] = dict(backward_steps=int(np.sum(np.diff(raw_t)<0)), duplicates=int(len(raw_t)-len(np.unique(raw_t))))
        if temporal[name]['backward_steps']: raise ValueError('Nonmonotone force sync clock: '+name)
        arrays = [[r[j] for r in rows] for j in range(1, len(rows[0]))]
        ts, received, *values = unique(raw_t, *arrays)
        native[name+'_t'] = ts; native[name+'_recv'] = received
        offsets[name] = float(np.median(received-ts))
        if name == 'bsq':
            if units[topic].get('unit') != 'N': raise ValueError('Unexpected BSQ units')
            native['bsq_force'] = np.asarray(values[0], float)
        else:
            if units[topic].get('unit') != 'mN': raise ValueError('Unexpected CMCU units')
            native[name+'_ad'] = np.asarray(values[0], float)
            native[name+'_force'] = np.asarray(values[1], float)*.001
            if native[name+'_ad'].shape[1] != 5: raise ValueError('CMCU finger count')
    hands = {}
    for side, stream in emg.items():
        u = units.get('/band/'+side+'/sEMG', {})
        if u.get('unit') != 'uV' or int(u.get('channel_count', 0)) != 16: raise ValueError('EMG unit/channel contract')
        ts = np.asarray(stream['t'], np.int64)
        temporal[side+'_emg'] = dict(backward_steps=int(np.sum(np.diff(ts)<0)), duplicates=int(len(ts)-len(np.unique(ts))))
        if temporal[side+'_emg']['backward_steps']: raise ValueError('Nonmonotone EMG sync clock')
        ts, recv, x = unique(ts, stream['recv'], np.asarray(stream['x'], np.float32).reshape(-1, 16))
        start, filtered, valid = regularize(ts, x)
        for name, a in [(side, filtered), (side+'_valid', valid)]:
            tmp = out/(name+'.partial.npy'); np.save(tmp, a); os.replace(tmp, out/(name+'.npy'))
        hands[side] = dict(start_ns=start, samples=len(filtered), invalid_samples=int((~valid).sum()),
                           clock_offset_ns=float(np.median(recv-ts)), units=u)
    tmp = native_file.with_name(native_file.stem+'.partial.npz')
    np.savez_compressed(tmp, **native); os.replace(tmp, native_file)
    d = dict(id=row['id'], etag=row['etag'], version=VERSION, hands=hands, temporal=temporal,
             clock_ok=max(offsets.values())-min(offsets.values()) <= 50e6, clock_offsets=offsets,
             native_sha256=sha(native_file), bytes_read=source.bytes_read,
             preprocessing='2000Hz; 50Hz harmonic notches Q30; order4 40-850Hz bandpass; causal per continuous segment; .5s warmup masked')
    d['processed_files']={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.glob('*.npy')}
    atomic(done, d); return d

def extract_many(root, rows, workers=8, label='all'):
    root = Path(root); results, errors = [], []; start = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(one, str(root), r): r for r in rows}
        for future in as_completed(jobs):
            r = jobs[future]
            try: results.append(future.result())
            except Exception as exc: errors.append(dict(id=r['id'], key=r['key'], error=str(exc)))
            d = dict(stage=label, completed=len(results), failed=len(errors), total=len(rows), seconds=time.monotonic()-start)
            atomic(root/'checks/extraction_progress.json', d)
            if (len(results)+len(errors))%10 == 0: print(json.dumps(d), flush=True)
    result = dict(results=results, errors=errors, total=len(rows), seconds=time.monotonic()-start)
    atomic(root/'checks'/('extraction_'+label+'.json'), result)
    if errors and len(errors) > max(3, .1*len(rows)): raise RuntimeError('More than 10% extraction failures; see extraction audit')
    return result
