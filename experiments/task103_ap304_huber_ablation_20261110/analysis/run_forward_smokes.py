"""Validate all three objectives in a separate Train-only compute64 runtime."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main(root, candidate):
    root, candidate = Path(root).resolve(), Path(candidate).resolve()
    assert candidate.is_relative_to(root)
    sys.path.insert(0, str(candidate / 'code/scripts'))
    from common import BRANCHES, atomic, load, sha, source_binding
    runtime = candidate / 'runtime'
    configuration = load(candidate / 'code/config.json')
    assert configuration['physical_forward_schedule'] == [dict(start_global_update=0, windows=64)]
    assert configuration['common_microbatch'] == 16 and configuration['global_batch'] == 256
    original_data_sha = sha(root / 'prepared/dataset.json')
    original_source_sha = sha(root / 'code/scripts/train.py')
    original_configuration_sha = sha(root / 'code/config.json')
    samples = []
    for i in range(3):
        text = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=uuid,memory.free,utilization.gpu',
                                       '--format=csv,noheader,nounits'], text=True)
        uuid, free, util = [v.strip() for v in text.strip().split(',')]
        samples.append(dict(uuid=uuid, free_MiB=int(free), utilization=int(util)))
        if i < 2:
            time.sleep(2)
    assert min(s['free_MiB'] for s in samples) > 46*1024
    results = []
    for branch in BRANCHES:
        command = [sys.executable, '-B', str(candidate/'code/scripts/train.py'),
                   '--root', str(runtime), '--branch', branch, '--smoke']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=samples[-1]['uuid'], TASK103_MICROBATCH='16',
                   TASK103_SMOKE_WINDOWS='256', TASK103_GPU_MEMORY_GIB='32', PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   TMPDIR='/pub_data/zjl/.t103h_4e3bd68a', CUDA_CACHE_PATH=str(root/'cache/cuda_kernels'),
                   TORCH_HOME=str(root/'cache/torch'), XDG_CACHE_HOME=str(root/'cache/xdg'))
        with (runtime/'logs'/f'smoke_{branch}.log').open('a') as log:
            process = subprocess.Popen(command, cwd=runtime, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic(root/'checks'/f'compute64_candidate_process_{branch}.json', dict(pid=process.pid, command=command,
               formal_training_changed=False, test_metrics_read=False))
        code = process.wait()
        assert code == 0, (branch, code)
        passed = load(runtime/'checks'/f'smoke_{branch}'/'PASS.json')
        contract = load(runtime/'checks'/f'smoke_{branch}'/'CONTRACT.json')
        assert passed['microbatch'] == 16 and passed['physical_forward_windows'] == 64
        assert passed['valid_points'] > 0 and passed['peak_allocated_GiB'] < 32
        assert contract['source_code'] == source_binding()
        assert contract['common_initial_sha256'] == sha(root/'assets/common_stage2_initial_state_dict.pt')
        assert contract['effective_microbatch'] == 16
        results.append(dict(branch=branch, passed=passed, contract_sha256=sha(runtime/'checks'/f'smoke_{branch}'/'CONTRACT.json')))
    original = load(root/'prepared/dataset.json')
    probe_data = load(runtime/'prepared/dataset.json')
    assert all(original[k] == probe_data[k] for k in original if k not in ('source_code', 'config_sha256'))
    assert original_data_sha == sha(root/'prepared/dataset.json')
    assert original_source_sha == sha(root/'code/scripts/train.py')
    assert original_configuration_sha == sha(root/'code/config.json')
    atomic(root/'checks/compute64_candidate_smoke_summary.json', dict(status='PASS_ALL_THREE_TRAIN_ONLY_COMPUTE64_SMOKES',
           cases=results, candidate_code=str(candidate/'code'), GPU_samples=samples,
           candidate_source_code=source_binding(), max_peak_allocated_GiB=max(x['passed']['peak_allocated_GiB'] for x in results),
           original_dataset_sha256=original_data_sha, original_train_source_sha256=original_source_sha,
           original_config_sha256=original_configuration_sha, data_entries_labels_norm_and_windows_unchanged=True,
           formal_training_source_and_configuration_unchanged=True, test_metrics_read=False))
    print(json.dumps(results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--candidate', required=True)
    args = parser.parse_args()
    main(args.root, args.candidate)
