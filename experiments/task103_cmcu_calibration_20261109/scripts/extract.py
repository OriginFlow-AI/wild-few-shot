"""Extract the frozen five streams from local MCAPs or authorized S3 access."""
import argparse
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
import boto3
from botocore.config import Config
from mcap.reader import make_reader
import numpy as np
from common import ROOT, EXPERIMENT
from mcap_io import RangeReader, decoder, unique
from preprocessing import regularize

TOPICS = ['/bsq/middle/force', '/cmcu05b/left/cmcu05b', '/cmcu05b/right/cmcu05b',
          '/band/left/sEMG', '/band/right/sEMG']
VERSION = 'task103-training-input-v1'


def one(row, options):
    out = ROOT / 'emg' / row['id']; out.mkdir(parents=True, exist_ok=True)
    done = out / 'done.json'
    native_file = ROOT / 'native' / (row['id'] + '.npz')
    if done.exists() and native_file.exists():
        cached = json.loads(done.read_text())
        if cached.get('version') != VERSION or cached['etag'] != row['etag']:
            raise ValueError('Incompatible cache; use a fresh run directory')
        return
    if options['mcap_dir']:
        path = Path(options['mcap_dir']) / row['key']
        if path.stat().st_size != row['size']: raise ValueError('Local MCAP size differs from frozen manifest')
        reader = path.open('rb')
    else:
        session = boto3.Session(profile_name=options['profile'])
        client = session.client('s3', endpoint_url=options['endpoint_url'],
                                config=Config(connect_timeout=8, read_timeout=120,
                                              retries={'max_attempts': 3}, s3={'addressing_style': 'path'}))
        reader = RangeReader(client, row, options['bucket'])
    classes, units = {}, {}
    force = {topic: [] for topic in TOPICS[:3]}
    emg = {s: dict(t=array('Q'), recv=array('Q'), x=array('f')) for s in ['left', 'right']}
    try:
        for schema, channel, message in make_reader(reader).iter_messages(topics=TOPICS, log_time_order=False):
            if schema.id not in classes: classes[schema.id] = decoder(schema)
            m = classes[schema.id].FromString(message.data)
            timestamp = int(m.sync_ts_ns or message.log_time); received = int(m.recv_ts_ns)
            units[channel.topic] = channel.metadata
            if '/sEMG' in channel.topic:
                a = emg[channel.topic.split('/')[2]]
                a['t'].append(timestamp); a['recv'].append(received)
                a['x'].extend(getattr(m, f'ch{j:02d}') for j in range(16))
            elif channel.topic == TOPICS[0]:
                force[channel.topic].append((timestamp, received, float(m.force_n)))
            else:
                force[channel.topic].append((timestamp, received, list(m.ad_values[:5]), list(m.pressures[:5])))
    finally:
        reader.close()
    native, offsets = {}, {}
    for name, topic in zip(['bsq', 'left', 'right'], TOPICS[:3]):
        rows = force[topic]
        if not rows: raise ValueError('Missing force stream: ' + name)
        if name == 'bsq':
            t, recv, values = unique([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows])
            native['bsq_force'] = values
        else:
            t, recv, ad, values = unique([r[0] for r in rows], [r[1] for r in rows],
                                         [r[2] for r in rows], [r[3] for r in rows])
            native[name + '_ad'] = np.asarray(ad, float)
            native[name + '_force'] = np.asarray(values, float) * .001  # mN -> N
        native[name + '_t'] = t; native[name + '_recv'] = recv
        offsets[name] = float(np.median(recv - t))
    hands = {}
    for side, stream in emg.items():
        metadata = units['/band/' + side + '/sEMG']
        if metadata.get('unit') != 'uV' or int(metadata.get('channel_count', 0)) != 16:
            raise ValueError('Unexpected EMG unit/channel contract')
        t, recv, raw = unique(stream['t'], stream['recv'], np.asarray(stream['x'], np.float32).reshape(-1, 16))
        if len(t) < 4000 or not np.isfinite(raw).all(): raise ValueError('Invalid or short EMG')
        start, filtered, valid = regularize(t, raw)
        np.save(out / (side + '.npy'), filtered); np.save(out / (side + '_valid.npy'), valid)
        hands[side] = dict(start_ns=start, samples=len(filtered), invalid_samples=int((~valid).sum()),
                           clock_offset_ns=float(np.median(recv-t)), units=metadata)
    np.savez_compressed(native_file, **native)
    # recv-sync includes transport/batching latency; do not shift EMG by that latency.
    force_clock_ok = max(offsets.values()) - min(offsets.values()) <= 50e6
    if force_clock_ok != bool(row['clock_ok']): raise ValueError('Force clock audit differs from frozen source')
    result = dict(id=row['id'], etag=row['etag'], version=VERSION, hands=hands,
                  clock_ok=force_clock_ok, clock_offsets=offsets,
                  preprocessing='2000Hz timestamp interpolation; one canonical causal SOS filter')
    temporary = done.with_suffix('.partial.json'); temporary.write_text(json.dumps(result)); temporary.replace(done)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--endpoint-url', default=os.environ.get('TASK103_S3_ENDPOINT'))
    parser.add_argument('--bucket', default=os.environ.get('TASK103_S3_BUCKET', 'tes-bucket'))
    parser.add_argument('--profile', default=None, help='Optional standard AWS profile')
    parser.add_argument('--mcap-dir', default=None, help='Local root containing raw/103/YYYY-MM-DD/... MCAPs')
    parser.add_argument('--id', action='append', help='Optional trial IDs for extraction/parity smoke checks')
    args = parser.parse_args()
    if not args.mcap_dir and not args.endpoint_url: parser.error('Set TASK103_S3_ENDPOINT or pass --mcap-dir')
    ROOT.mkdir(parents=True, exist_ok=True); (ROOT/'native').mkdir(exist_ok=True)
    source = ROOT/'sources.json'
    if not source.exists(): source.write_bytes((EXPERIMENT/'manifests/sources.json').read_bytes())
    rows = json.loads(source.read_text())
    if args.id: rows = [r for r in rows if r['id'] in args.id]
    if not rows: parser.error('No matching source trials')
    errors, done = [], 0; started = time.monotonic()
    options = {k: getattr(args, k) for k in ['mcap_dir', 'endpoint_url', 'bucket', 'profile']}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(one, r, options): r for r in rows}
        for job in as_completed(jobs):
            try: job.result(); done += 1
            except Exception as exc: errors.append(dict(id=jobs[job]['id'], error=str(exc)))
            if (done+len(errors)) % 20 == 0: print(f'EXTRACT {done}/{len(rows)}, errors={len(errors)}', flush=True)
    result = dict(done=done, total=len(rows), errors=errors, seconds=time.monotonic()-started)
    name = 'extraction_smoke_result.json' if args.id else 'extraction_result.json'
    (ROOT/name).write_text(json.dumps(result, indent=2))
    if not args.id: (ROOT/'filter_result.json').write_text(json.dumps(result, indent=2))
    if errors: raise SystemExit(f'{len(errors)} extraction failures; see {name}')
    print(f'Extraction complete: {done}', flush=True)
