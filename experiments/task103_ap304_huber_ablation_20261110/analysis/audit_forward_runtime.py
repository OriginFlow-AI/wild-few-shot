"""Audit one real production compute64 branch and its uninterrupted RNG span."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch


def main(root, branch):
    root = Path(root)
    torch.set_num_threads(2)
    sys.path.insert(0, str(root/'code/scripts'))
    from common import atomic, load, settings, sha, source_binding
    c = settings()
    assert c['physical_forward_schedule'] == [dict(start_global_update=0,windows=16),dict(start_global_update=7980,windows=64)]
    assert (c['common_microbatch'],c['global_batch'],c['max_epochs'],c['patience']) == (16,256,10,3)
    folder = root/'runs'/branch
    data = load(root/'prepared/dataset.json')
    updates = math.ceil(data['windows']['train']/c['global_batch'])
    assert updates == 2660
    previous = root/'archive/epoch_candidates'/branch/sha(root/'prepared/dataset.json')[:16]/'epoch003.pt'
    before = torch.load(previous,map_location='cpu',weights_only=True)
    checkpoint = torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
    state = checkpoint['state']
    assert before['state']['global_update'] == 7980 and state['global_update'] >= 8005
    assert checkpoint['binding'] == {k:v for k,v in load(folder/'CONTRACT.json').items() if k!='configuration'}
    assert checkpoint['binding']['source_code'] == source_binding()
    assert checkpoint['binding']['dataset_sha256'] == sha(root/'prepared/dataset.json')
    assert checkpoint['binding']['config_sha256'] == sha(root/'code/config.json')
    assert checkpoint['binding']['physical_forward_schedule'] == c['physical_forward_schedule']
    assert checkpoint['binding']['common_initial_sha256'] == sha(root/'assets/common_stage2_initial_state_dict.pt')
    assert all(torch.isfinite(t).all().item() for t in checkpoint['model'].values())
    assert all(int(v['step']) == state['global_update'] for v in checkpoint['optimizer']['state'].values())
    probe = next(p for p in load(root/'analysis_output/forward_grouping_probe.json')['attempts'] if p['physical_forward']==64)

    def rng(tensor):
        raw = tensor.cpu().numpy().tobytes()
        assert len(raw) == 16
        return int.from_bytes(raw[:8],'little'),int.from_bytes(raw[8:],'little')

    full_delta = int.from_bytes(bytes.fromhex(probe['full256_after_first_update_cuda_rng'])[8:],'little')
    tail_delta = int.from_bytes(bytes.fromhex(probe['tail111_cuda_rng'])[8:],'little')
    seed0,offset0 = rng(before['cuda_rng'])
    seed1,offset1 = rng(checkpoint['cuda_rng'])
    tails = state['global_update']//updates - 3
    full = state['global_update'] - 7980 - tails
    assert seed0 == seed1 and offset1-offset0 == full*full_delta+tails*tail_delta
    metrics = [json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
    new = [m for m in metrics if m.get('physical_forward_windows') == 64]
    assert new[0]['global_update'] == 7981
    assert any(m['global_update']==7980 and m.get('physical_forward_windows')==16 for m in metrics)
    assert all(m['microbatch']==16 and m['global_update']>=7981 and math.isfinite(m['loss']) and math.isfinite(m['gradient_norm']) for m in new)
    recent = [m for m in new if m['epoch']==new[-1]['epoch']][-20:]
    elapsed = recent[-1]['elapsed_epoch_s']-recent[0]['elapsed_epoch_s']
    windows = sum(c['global_batch'] if m['epoch_update']<updates else data['windows']['train']-(updates-1)*c['global_batch'] for m in recent[1:])
    assert len(recent)>=20 and elapsed>0
    receipt = load(root/'checks'/f'process_{branch}.json')
    p = Path('/proc')/str(receipt['pid'])
    assert (p/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    assert [x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x] == receipt['command']
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
    actual = {int(a[0].strip()):a[1].strip() for line in apps.splitlines() if len(a:=line.split(','))==2 and a[0].strip().isdigit()}
    assert actual[receipt['pid']] == receipt['GPU']['uuid']
    result = dict(status='PASS_ONE_BRANCH_PRODUCTION_COMPUTE64_AND_CONTINUITY',branch=branch,observed_unix=time.time(),
                  PID=receipt['pid'],actual_GPU_uuid=actual[receipt['pid']],GPU_index=receipt['GPU']['index'],
                  first_actual_compute64_optimizer_update=7981,logical_loss_group=16,physical_forward_windows=64,
                  verified_saved_global_update=state['global_update'],observed_compute64_metric_rows=len(new),
                  recent_windows_per_second=windows/elapsed,peak_allocated_GiB=max(m['peak_allocated_GiB'] for m in new),
                  model_parameters_and_logged_loss_gradient_finite=True,all_Adam_steps_equal_global_update=True,
                  CUDA_RNG_seed_preserved=True,CUDA_RNG_advance_matches_compute64_from_epoch3_boundary_exactly=True,
                  epoch3_CUDA_rng_sha256=hashlib.sha256(before['cuda_rng'].numpy().tobytes()).hexdigest(),
                  dataset_sha256=sha(root/'prepared/dataset.json'),source_and_config_binding_verified=True,
                  original_common_initial_weight_verified=True,budget_max_epochs=10,budget_patience=3,test_metrics_read=False)
    atomic(root/'checks'/f'compute64_production_{branch}.json',result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--branch',choices=['calibrated_huber','raw_huber','raw_original'],required=True)
    args = parser.parse_args()
    main(args.root,args.branch)
