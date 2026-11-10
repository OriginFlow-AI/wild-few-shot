"""Install a common future compute schedule, preserving every saved prefix.

Scientific inputs, logical losses, optimizer, LR, budget and evaluation remain
fixed. Physical forward shape/dropout trajectories intentionally change from
the same future update in every branch. Original artifacts support rollback.
"""
import argparse
import ast
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import torch

from audit_first_validation import equal
from migrate_shared_gpus import checked, environ, identity, same, send
from migrate_validation_only_binding import train_identity


def copy_atomic(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.compute_migration.partial')
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def functions(path):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def main(root, candidate):
    root, candidate = Path(root).resolve(), Path(candidate).resolve()
    assert str(root).startswith('/pub_data/zjl/') and candidate.is_relative_to(root)
    sys.path.insert(0, str(root/'code/scripts'))
    from common import BRANCHES, atomic, load, sha, source_binding
    torch.set_num_threads(2)
    old_config = load(root/'code/config.json')
    new_config = load(candidate/'formal_config.json')
    allowed = {'physical_forward_schedule', 'forward_grouping_note', 'microbatch_selection'}
    assert {k:v for k,v in old_config.items() if k not in allowed} == {k:v for k,v in new_config.items() if k not in allowed}
    assert 'physical_forward_schedule' not in old_config, 'One-time migration only'
    original_data = load(root/'prepared/dataset.json')
    updates = math.ceil(original_data['windows']['train']/old_config['global_batch'])
    schedule = [dict(start_global_update=0, windows=16), dict(start_global_update=3*updates, windows=64)]
    assert new_config['physical_forward_schedule'] == schedule and updates == 2660
    old_source = source_binding()
    passed = load(root/'checks/compute64_candidate_smoke_summary.json')
    assert passed['status'] == 'PASS_ALL_THREE_TRAIN_ONLY_COMPUTE64_SMOKES'
    assert passed['original_dataset_sha256'] == sha(root/'prepared/dataset.json')
    assert passed['original_train_source_sha256'] == sha(root/'code/scripts/train.py')
    assert passed['original_config_sha256'] == sha(root/'code/config.json')
    expected_source = passed['candidate_source_code']
    assert all(v == expected_source[k] for k,v in old_source.items() if k != 'scripts/train.py')
    assert sha(candidate/'code/scripts/train.py') == expected_source['scripts/train.py']
    old_functions = functions(root/'code/scripts/train.py')
    new_functions = functions(candidate/'code/scripts/train.py')
    protected = ['network', 'Windows', 'objective', 'measures', 'evaluate', 'optimizer', 'save']
    assert all(old_functions[k] == new_functions[k] for k in protected)
    launcher = load(root/'checks/background_launcher.json')
    controller = checked(launcher)
    controller_env = environ(controller['pid'])
    archiver_receipt = load(root/'checks/epoch_archive_process.json')
    archiver = checked(archiver_receipt)
    jobs = {}
    for branch in BRANCHES:
        receipt = load(root/'checks'/f'process_{branch}.json')
        handle = checked(receipt)
        assert receipt['command'][-2:] == ['--branch', branch] and not receipt['smoke']
        env = environ(handle['pid'])
        assert float(env['TASK103_GPU_MEMORY_GIB']) >= 28 and env['TASK103_MICROBATCH'] == '16'
        status = load(root/'runs'/branch/'STATUS.json')
        assert status['phase'] == 'training' and status['global_update'] < 3*updates
        jobs[branch] = dict(receipt=receipt, handle=handle, env=env)
    GPU_samples = []
    for sample in range(3):
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu',
                                          '--format=csv,noheader,nounits'], text=True)
        observed = {}
        for line in output.strip().splitlines():
            index, free, utilization = [int(x.strip()) for x in line.split(',')]
            observed[index] = dict(free_MiB=free, utilization=utilization)
        for job in jobs.values():
            assert observed[job['receipt']['GPU']['index']]['free_MiB'] > 40*1024
        GPU_samples.append(observed)
        if sample < 2:
            time.sleep(2)
    tag = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    archive = root/'archive/forward_schedule64'/tag
    archive.mkdir(parents=True, exist_ok=False)
    stopped, launched = [], []
    terminated = mutated = False

    def launch_all():
        new_receipts = {}
        for branch, job in jobs.items():
            log = root/'logs'/f'train_{branch}.log'
            offset = log.stat().st_size
            with log.open('a') as f:
                p = subprocess.Popen(job['receipt']['command'], cwd=root, env=job['env'], stdin=subprocess.DEVNULL,
                                     stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            launched.append(identity(p.pid))
            receipt = dict(job['receipt'], pid=p.pid, log_offset=offset, compute_schedule_archive=str(archive))
            atomic(root/'checks'/f'process_{branch}.json', receipt)
            new_receipts[branch] = receipt
        with (root/'logs/pipeline.log').open('a') as f:
            p = subprocess.Popen(launcher['command'], cwd=root, env=controller_env, stdin=subprocess.DEVNULL,
                                 stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        launched.append(identity(p.pid))
        atomic(root/'checks/background_launcher.json', dict(launcher, pid=p.pid, compute_schedule_archive=str(archive)))
        return new_receipts

    try:
        for handle in [controller] + [j['handle'] for j in jobs.values()] + [archiver]:
            assert same(handle)
            send(handle, signal.SIGSTOP)
            stopped.append(handle)
        shutil.copytree(root/'code', archive/'code', ignore=shutil.ignore_patterns('__pycache__'))
        (archive/'prepared').mkdir()
        shutil.copy2(root/'prepared/dataset.json', archive/'prepared/dataset.json')
        for split in ('train', 'val', 'test'):
            os.link(root/'prepared'/f'{split}_windows.npy', archive/'prepared'/f'{split}_windows.npy')
        shutil.copy2(root/'checks/gpu_tuning.json', archive/'gpu_tuning.json')
        shutil.copy2(root/'plan.json', archive/'plan.json')
        before_identity = train_identity(root, original_data)
        before_rows = []
        children, pending = [], {j['handle']['pid'] for j in jobs.values()}
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
        for branch, job in jobs.items():
            folder = archive/'runs'/branch
            folder.mkdir(parents=True)
            contract = load(root/'runs'/branch/'CONTRACT.json')
            shutil.copy2(root/'runs'/branch/'CONTRACT.json', folder/'CONTRACT.json')
            for name in ('last.pt', 'best.pt'):
                os.link(root/'runs'/branch/name, folder/name)
                ck = torch.load(folder/name, map_location='cpu', weights_only=True)
                assert ck['binding'] == {k:v for k,v in contract.items() if k != 'configuration'}
                assert ck['state']['global_update'] < 3*updates
                assert all(torch.isfinite(v).all().item() for v in ck['model'].values())
                before_rows.append(dict(branch=branch, checkpoint=name, global_update=ck['state']['global_update'],
                                        original_sha256=sha(folder/name)))
                del ck
        atomic(archive/'BEFORE.json', dict(jobs={b:j['receipt'] for b,j in jobs.items()}, launcher=launcher,
                                         archiver=archiver_receipt, checkpoints=before_rows, train_identity=before_identity))
        handles = [controller] + [j['handle'] for j in jobs.values()] + children
        for h in handles:
            send(h, signal.SIGTERM)
            send(h, signal.SIGCONT)
        terminated = True
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
        mutated = True
        copy_atomic(candidate/'code/scripts/train.py', root/'code/scripts/train.py')
        copy_atomic(candidate/'formal_config.json', root/'code/config.json')
        assert source_binding() == expected_source
        new_data = dict(original_data, config_sha256=sha(root/'code/config.json'), source_code=source_binding())
        atomic(root/'prepared/dataset.json', new_data)
        assert all(original_data[k] == new_data[k] for k in original_data if k not in ('config_sha256', 'source_code'))
        assert train_identity(root, new_data) == before_identity
        assert sha(root/'plan.json') == sha(archive/'plan.json')
        for split in ('train', 'val', 'test'):
            assert sha(root/'prepared'/f'{split}_windows.npy') == sha(archive/'prepared'/f'{split}_windows.npy')
        migrated = []
        for branch in BRANCHES:
            for name in ('last.pt', 'best.pt'):
                origin = archive/'runs'/branch/name
                ck = torch.load(origin, map_location='cpu', weights_only=True)
                binding = dict(ck['binding'], config_sha256=sha(root/'code/config.json'),
                               dataset_sha256=sha(root/'prepared/dataset.json'), source_code=source_binding(),
                               physical_forward_schedule=schedule)
                ck['binding'] = binding
                target = root/'runs'/branch/name
                tmp = target.with_name(name+'.rebind.partial')
                torch.save(ck, tmp)
                tmp.replace(target)
                check = torch.load(target, map_location='cpu', weights_only=True)
                assert check['binding'] == binding
                for key in ('model', 'optimizer', 'state', 'torch_rng', 'cuda_rng'):
                    assert equal(ck[key], check[key]), key
                migrated.append(dict(branch=branch, checkpoint=name, global_update=ck['state']['global_update'],
                                     previous_sha256=sha(origin), rebound_sha256=sha(target),
                                     exact_model_Adam_cursor_history_CPU_CUDA_RNG_preserved=True))
                del check, ck
            atomic(root/'runs'/branch/'CONTRACT.json', dict(**binding, configuration=new_config))
        tuning = load(archive/'gpu_tuning.json')
        tuning['historical_logical16_chosen'] = dict(tuning['chosen'])
        tuning['chosen']['peak_allocated_GiB'] = passed['max_peak_allocated_GiB']
        tuning['chosen']['physical_forward_schedule'] = schedule
        tuning['selection_reason'] = 'Reserve measured peak for scheduled compute64; logical objective groups16 and global256 fixed'
        atomic(root/'checks/gpu_tuning.json', tuning)
        receipts = launch_all()
        send(archiver, signal.SIGCONT)
        result = dict(status='PASS_EXACT_PREFIX_PRESERVATION_COMMON_FUTURE_COMPUTE_SCHEDULE_LAUNCHED',
                      archive=str(archive), checkpoints=migrated, new_processes=receipts,
                      physical_forward_schedule=schedule, first_compute64_optimizer_update=3*updates+1,
                      all_three_use_identical_future_schedule=True, logical_loss_group=16, global_batch=256,
                      max_epochs=10, patience=3, dataset_sha256=sha(root/'prepared/dataset.json'),
                      train_identity_before=before_identity, train_identity_after=train_identity(root, new_data),
                      all_splits_entries_labels_norm_and_window_indices_unchanged=True,
                      protected_AST_functions_unchanged=protected, common_stage1_and_initial_weights_unchanged=True,
                      three_preflight_GPU_samples=GPU_samples,
                      exact_saved_prefix_state_preserved=True, training_restarted_from_initial_weights=False,
                      future_dropout_and_BF16_trajectory_changes_from_epoch4=True, evaluation_forward_group_unchanged=16,
                      test_metrics_read=False, awaiting_actual_resume_progress=True)
        atomic(archive/'MIGRATION.json', result)
        atomic(root/'checks/forward_schedule_migration.json', result)
        print(json.dumps(dict(status=result['status'], archive=str(archive),
                             new_PIDs={b:v['pid'] for b,v in receipts.items()},
                             dataset_sha256=result['dataset_sha256'], first_compute64_optimizer_update=3*updates+1)))
    except Exception as error:
        for h in launched:
            if h:
                send(h, signal.SIGTERM)
        for _ in range(100):
            if not any(h and same(h) for h in launched):
                break
            time.sleep(.1)
        for h in launched:
            if h:
                send(h, signal.SIGKILL)
        time.sleep(.2)
        if terminated:
            if mutated:
                for relative in ('scripts/train.py', 'config.json'):
                    copy_atomic(archive/'code'/relative, root/'code'/relative)
                copy_atomic(archive/'prepared/dataset.json', root/'prepared/dataset.json')
                copy_atomic(archive/'gpu_tuning.json', root/'checks/gpu_tuning.json')
                for branch in BRANCHES:
                    for name in ('last.pt', 'best.pt', 'CONTRACT.json'):
                        copy_atomic(archive/'runs'/branch/name, root/'runs'/branch/name)
            launch_all()
        send(archiver, signal.SIGCONT)
        atomic(root/'checks/forward_schedule_migration_failure.json', dict(error=str(error), archive=str(archive),
               original_protocol_restored=terminated, source_mutation_had_started=mutated))
        raise
    finally:
        for h in stopped:
            send(h, signal.SIGCONT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--candidate', required=True)
    args = parser.parse_args()
    main(args.root, args.candidate)
