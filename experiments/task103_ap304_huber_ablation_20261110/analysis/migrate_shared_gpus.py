"""Move selected *own* jobs to less busy GPUs without changing the protocol.

Requires the exact live launcher/trainer receipts, three fresh GPU samples,
and compatible checkpoint bindings. Saves the exact resume files and an audit
under ROOT/archive/gpu_migrations. Other users' processes are never signalled.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    tmp = path.with_suffix('.partial.json')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(pid):
    p = Path('/proc') / str(pid)
    if not p.exists():
        return None
    fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, state=fields[0], ppid=int(fields[1]), start=fields[19],
                uid=p.stat().st_uid,
                command=[x.decode() for x in (p / 'cmdline').read_bytes().split(b'\0') if x])


def same(handle):
    now = identity(handle['pid'])
    return now and now['state'] != 'Z' and all(now[k] == handle[k] for k in ('pid', 'start', 'uid', 'command'))


def checked(receipt):
    handle = identity(receipt['pid'])
    assert handle and handle['state'] != 'Z'
    assert handle['command'] == receipt['command'] and handle['uid'] == os.getuid()
    return handle


def send(handle, sig):
    if same(handle):
        os.kill(handle['pid'], sig)


def environ(pid):
    return {k.decode(): v.decode() for x in (Path('/proc') / str(pid) / 'environ').read_bytes().split(b'\0')
            if b'=' in x for k, v in [x.split(b'=', 1)]}


def migrate(root, mapping):
    root = Path(root).resolve()
    assert str(root).startswith('/pub_data/zjl/')
    sys.path.insert(0, str(root / 'code/scripts'))
    from common import source_binding
    launcher_path = root / 'checks/background_launcher.json'
    launcher = read(launcher_path)
    controller = checked(launcher)
    assert str(root / 'code/scripts/pipeline.py') in controller['command']
    controller_env = environ(controller['pid'])
    jobs = {}
    for name, target in mapping.items():
        receipt = read(root / 'checks' / f'process_{name}.json')
        handle = checked(receipt)
        assert not receipt['smoke'] and str(root / 'code/scripts/train.py') in handle['command']
        assert handle['command'][-2:] == ['--branch', name]
        contract = read(root / 'runs' / name / 'CONTRACT.json')
        assert contract['source_code'] == source_binding()
        assert contract['dataset_sha256'] == sha(root / 'prepared/dataset.json')
        assert contract['config_sha256'] == sha(root / 'code/config.json')
        jobs[name] = dict(receipt=receipt, handle=handle, env=environ(handle['pid']), contract=contract, target=target)
    active = [read(root / 'checks' / f'process_{b}.json')['GPU']['index']
              for b in ('calibrated_huber', 'raw_huber', 'raw_original')]
    assert len(set(mapping.values())) == len(mapping) and not set(mapping.values()).intersection(active)
    observations = []
    for sample in range(3):
        data = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu',
                                        '--format=csv,noheader,nounits'], text=True)
        rows = {}
        for line in data.strip().splitlines():
            a = [v.strip() for v in line.split(',')]
            rows[int(a[0])] = dict(index=int(a[0]), uuid=a[1], total_MiB=int(a[2]), free_MiB=int(a[3]), utilization=int(a[4]))
        observations.append(rows)
        if sample < 2:
            time.sleep(3)
    for job in jobs.values():
        rows = [obs[job['target']] for obs in observations]
        assert min(v['free_MiB'] for v in rows) >= 46 * 1024
        assert max(v['utilization'] for v in rows) <= 60
        job['gpu'] = dict(**rows[-1], minimum_free_MiB=min(v['free_MiB'] for v in rows),
                          mean_utilization=sum(v['utilization'] for v in rows) / len(rows),
                          process_memory_limit_GiB=32, selection='three fresh low-utilization/headroom samples')
    tag = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    archive = root / 'archive/gpu_migrations' / tag
    archive.mkdir(parents=True, exist_ok=False)
    write(archive / 'old_launcher.json', launcher)
    stopped = []
    try:
        send(controller, signal.SIGSTOP)
        stopped.append(controller)
        for name, job in jobs.items():
            send(job['handle'], signal.SIGSTOP)
            stopped.append(job['handle'])
        children = []
        pending = {job['handle']['pid'] for job in jobs.values()}
        for _ in range(8):
            found = []
            for p in Path('/proc').iterdir():
                if not p.name.isdigit():
                    continue
                try:
                    h = identity(int(p.name))
                except (FileNotFoundError, ProcessLookupError):
                    continue
                if h and h['ppid'] in pending and h['uid'] == os.getuid():
                    found.append(h)
            if not found:
                break
            children.extend(found)
            pending = {h['pid'] for h in found}
        audit_jobs = {}
        for name, job in jobs.items():
            path = root / 'runs' / name / 'last.pt'
            ck = torch.load(path, map_location='cpu', weights_only=True)
            assert ck['binding'] == {k: v for k, v in job['contract'].items() if k != 'configuration'}
            assert all(torch.isfinite(v).all().item() for v in ck['model'].values())
            status = read(root / 'runs' / name / 'STATUS.json')
            assert status['phase'] == 'training'
            replay = status['global_update'] - ck['state']['global_update']
            assert 0 <= replay <= 25
            destination = archive / f'{name}_resume.pt'
            os.link(path, destination)  # trainer replaces last.pt atomically; preserved inode stays immutable
            audit_jobs[name] = dict(old_receipt=job['receipt'], before_status=status,
                                   resume_sha256=sha(destination), resume_state=ck['state'],
                                   unsaved_updates_replayed=replay, target_GPU=job['gpu'])
            del ck
        write(archive / 'before.json', dict(status='READY_EXACT_RESUME', jobs=audit_jobs))
        for h in [controller] + [j['handle'] for j in jobs.values()] + children:
            send(h, signal.SIGTERM)
            send(h, signal.SIGCONT)
        handles = [controller] + [j['handle'] for j in jobs.values()] + children
        for _ in range(100):
            if not any(same(h) for h in handles):
                break
            time.sleep(.1)
        for h in handles:
            send(h, signal.SIGKILL)
        for _ in range(30):
            if not any(same(h) for h in handles):
                break
            time.sleep(.1)
        assert not any(same(h) for h in handles)
        for name, job in jobs.items():
            log = root / 'logs' / f'train_{name}.log'
            offset = log.stat().st_size
            env = dict(job['env'], CUDA_VISIBLE_DEVICES=job['gpu']['uuid'], TASK103_GPU_MEMORY_GIB='32')
            with log.open('a') as f:
                process = subprocess.Popen(job['receipt']['command'], env=env, cwd=root,
                                           stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            receipt = dict(pid=process.pid, command=job['receipt']['command'], GPU=job['gpu'], smoke=False,
                           log_offset=offset, migration_archive=str(archive))
            write(root / 'checks' / f'process_{name}.json', receipt)
            write(root / 'checks' / f'GPU_{name}.json', job['gpu'])
            audit_jobs[name]['new_receipt'] = receipt
        with (root / 'logs/pipeline.log').open('a') as f:
            process = subprocess.Popen(launcher['command'], env=controller_env, cwd=root,
                                       stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        updated_launcher = dict(launcher, pid=process.pid, migration_archive=str(archive))
        write(launcher_path, updated_launcher)
        write(archive / 'MIGRATION.json', dict(status='LAUNCHED_FROM_EXACT_RESUME_AWAITING_PROGRESS', jobs=audit_jobs,
                                             launcher=updated_launcher, data_or_training_algorithm_changed=False,
                                             unselected_trainers_restarted=False, epoch_archiver_restarted=False))
        print(json.dumps(dict(archive=str(archive), launcher_pid=process.pid,
                              jobs={k: dict(pid=v['new_receipt']['pid'], GPU=v['target_GPU']['index'],
                                            resume_global_update=v['resume_state']['global_update'],
                                            replay_updates=v['unsaved_updates_replayed']) for k, v in audit_jobs.items()})))
    finally:
        # Never leave a still-live own job suspended after a failed preflight.
        for h in stopped:
            send(h, signal.SIGCONT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--mapping', required=True, help='JSON branch-to-GPU-index mapping')
    args = parser.parse_args()
    migrate(args.root, json.loads(args.mapping))
