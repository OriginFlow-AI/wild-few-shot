"""Run the two historical AP304 objectives on the frozen all-stage Task103 data."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from launch_parallel import CODE, AdoptedTrainer, atomic, gpu_candidates, invoke, load, settings


def select_cards(dataset_root, reserved):
    k2 = dataset_root / 'checks/train_process_raw_original.json'
    done = dataset_root / 'runs/raw_original/DONE.json'
    if k2.is_file() and not done.exists():
        reserved = reserved | {load(k2)['gpu']['uuid']}
    cards = [c for c in gpu_candidates() if c['uuid'] not in reserved and c['process_memory_limit_GiB'] >= 32]
    devices = settings().get('ddp_devices', 2)
    if len(cards) < devices:
        raise RuntimeError(f'{devices} GPUs with a 32 GiB shared allowance are required for AP304')
    return cards[:devices]


def start(root, branch, cards, attempt, smoke):
    kind = 'smoke' if smoke else 'train'
    path = root / 'logs' / f'{kind}_{branch}_{attempt}.log'
    command = [sys.executable, '-B', '-m', 'torch.distributed.run', '--standalone',
               '--nproc-per-node', str(len(cards)), str(CODE / 'scripts/train_ap304_ddp.py'),
               '--root', str(root), '--branch', branch]
    if smoke:
        command.append('--smoke')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(c['uuid'] for c in cards),
               TASK103_GPU_MEMORY_GIB=str(min(c['process_memory_limit_GiB'] for c in cards)))
    with path.open('ab') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    receipt = dict(pid=process.pid, command=command, gpus=cards, attempt=attempt, log=str(path))
    atomic(root / 'checks' / f'{kind}_process_{branch}.json', receipt)
    print(f'{kind} {branch}: PID {process.pid}, GPUs {[c["index"] for c in cards]}', flush=True)
    return process, receipt


def run_branches(root, dataset_root, branches):
    active, errors = {}, {}
    for branch in branches:
        done = root / 'runs' / branch / 'DONE.json'
        receipt_path = root / 'checks' / f'train_process_{branch}.json'
        if not done.is_file() and receipt_path.is_file():
            receipt = load(receipt_path)
            adopted = AdoptedTrainer(receipt, done)
            if adopted.poll() is None:
                active[branch] = (adopted, receipt)
    attempts = dict.fromkeys(branches, 0)
    while True:
        for branch, (process, receipt) in list(active.items()):
            result = process.poll()
            if result is None:
                continue
            del active[branch]
            if result and attempts[branch] >= 3:
                errors[branch] = receipt['log']
        for branch in branches:
            if branch in active or branch in errors or (root / 'runs' / branch / 'DONE.json').is_file():
                continue
            used = {card['uuid'] for _, receipt in active.values() for card in receipt['gpus']}
            try:
                cards = select_cards(dataset_root, used)
                attempts[branch] += 1
                smoke, receipt = start(root, branch, cards, attempts[branch], True)
                if smoke.wait() != 0:
                    if attempts[branch] >= 3:
                        errors[branch] = receipt['log']
                    continue
                active[branch] = start(root, branch, cards, attempts[branch], False)
            except RuntimeError as error:
                print(error, flush=True)
        atomic(root / 'STATUS.json', dict(phase='training_or_evaluating', active=list(active), failed=errors))
        for branch in branches:
            done = root / 'runs' / branch / 'DONE.json'
            report = root / 'results' / branch
            if done.is_file() and not (report / 'REPORT_COMPLETE.json').is_file():
                invoke('analysis/report_all_stages.py', '--root', root, '--branch', branch, '--output', report)
        if all((root / 'results' / b / 'REPORT_COMPLETE.json').is_file() or b in errors for b in branches):
            if errors:
                raise RuntimeError(f'AP304 branches failed: {errors}')
            return
        time.sleep(15)


def main(root, dataset_root, previous_ap304, calibration_refit=None):
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    with (root / 'launcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if calibration_refit:
                if not (root / 'prepared/dataset.json').is_file():
                    invoke('scripts/prepare_calibrated_subject.py', '--root', root,
                           '--dataset-root', dataset_root, '--previous-ap304', previous_ap304,
                           '--calibration-refit', calibration_refit)
            else:
                invoke('scripts/bind_ap304_all_stages.py', '--root', root, '--dataset-root', dataset_root,
                       '--previous-ap304', previous_ap304)
            branches = settings()['branches']
            run_branches(root, dataset_root, branches)
            atomic(root / 'STATUS.json', dict(phase='complete', branches=branches))
        except Exception as error:
            atomic(root / 'STATUS.json', dict(phase='failed', error=str(error)))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--previous-ap304', type=Path, required=True)
    parser.add_argument('--calibration-refit', type=Path)
    args = parser.parse_args()
    main(args.root, args.dataset_root, args.previous_ap304, args.calibration_refit)
