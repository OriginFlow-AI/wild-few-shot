"""Audit real resumed updates and GPU placement after migrate_shared_gpus.py."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def audit(root, archive):
    root, archive = Path(root).resolve(), Path(archive).resolve()
    assert archive.is_relative_to(root / 'archive/gpu_migrations')
    sys.path.insert(0, str(root / 'code/scripts'))
    from common import source_binding
    migration = read(archive / 'MIGRATION.json')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                                    '--format=csv,noheader,nounits'], text=True)
    actual = {}
    for line in apps.splitlines():
        v = line.split(',')
        if len(v) == 3 and v[0].strip().isdigit():
            actual[int(v[0].strip())] = v[1].strip()
    rows = {}
    for branch, job in migration['jobs'].items():
        original = archive / f'{branch}_resume.pt'
        assert sha(original) == job['resume_sha256']
        before = torch.load(original, map_location='cpu', weights_only=True)
        after = torch.load(root / 'runs' / branch / 'last.pt', map_location='cpu', weights_only=True)
        assert before['binding'] == after['binding']
        assert after['binding']['source_code'] == source_binding()
        assert after['binding']['dataset_sha256'] == sha(root / 'prepared/dataset.json')
        assert after['binding']['config_sha256'] == sha(root / 'code/config.json')
        assert after['state']['global_update'] > job['before_status']['global_update']
        assert after['state']['epoch'] == before['state']['epoch']
        assert all(torch.isfinite(t).all().item() for t in after['model'].values())
        assert all(int(v['step']) == after['state']['global_update'] for v in after['optimizer']['state'].values())
        pid = job['new_receipt']['pid']
        argv = [x.decode() for x in (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0') if x]
        assert argv == job['new_receipt']['command']
        assert actual[pid] == job['target_GPU']['uuid']
        metrics = [json.loads(line) for line in (root / 'runs' / branch / 'metrics.jsonl').read_text().splitlines()]
        index = max(i for i, m in enumerate(metrics) if m['global_update'] == before['state']['global_update'] + 1)
        pre, post = metrics[max(0, index - 20):index], metrics[index:]
        assert len(post) >= 25
        assert all(m['epoch'] == before['state']['epoch'] + 1 for m in post)
        assert all(math.isfinite(m['loss']) and math.isfinite(m['gradient_norm']) and m['valid_points'] > 0 for m in post)

        def rate(values):
            elapsed = values[-1]['elapsed_epoch_s'] - values[0]['elapsed_epoch_s']
            assert elapsed > 0
            return 256 * (values[-1]['global_update'] - values[0]['global_update']) / elapsed

        rows[branch] = dict(new_PID=pid, actual_GPU_uuid=actual[pid], new_GPU_index=job['target_GPU']['index'],
                            resume_global_update=before['state']['global_update'],
                            first_resumed_global_update=post[0]['global_update'],
                            verified_latest_saved_global_update=after['state']['global_update'],
                            observed_new_updates=len(post), unsaved_updates_replayed=job['unsaved_updates_replayed'],
                            before_windows_per_second=rate(pre), after_last20_windows_per_second=rate(post[-20:]),
                            exact_archived_resume_sha256=job['resume_sha256'], binding_unchanged=True,
                            all_parameters_finite=True, all_Adam_steps_equal_global_update=True)
        del before, after
    status = read(root / 'STATUS.json')
    assert status['phase'] == 'parallel_training_or_waiting_for_capacity'
    receipts = {b: read(root / 'checks' / f'process_{b}.json')
                for b in ('calibrated_huber', 'raw_huber', 'raw_original')}
    assert {b: v['pid'] for b, v in status['active'].items()} == {b: v['pid'] for b, v in receipts.items()}
    for receipt in receipts.values():
        argv = [x.decode() for x in (Path('/proc') / str(receipt['pid']) / 'cmdline').read_bytes().split(b'\0') if x]
        assert argv == receipt['command']
    result = dict(status='PASS_REAL_SAME_PROTOCOL_GPU_MIGRATION_AND_PROGRESS', observed_unix=time.time(),
                  migration_archive=str(archive), branches=rows, controller_adopted_all_three_verified_PIDs=True,
                  current_PIDs={b: v['pid'] for b, v in receipts.items()},
                  dataset_sha256=sha(root / 'prepared/dataset.json'),
                  training_algorithm_or_scoring_changed=False, test_metrics_read=False)
    output = root / 'checks' / f'GPU_migration_progress_{archive.name}.json'
    temporary = output.with_suffix('.partial.json')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(output)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--archive', required=True)
    args = parser.parse_args()
    audit(args.root, args.archive)
