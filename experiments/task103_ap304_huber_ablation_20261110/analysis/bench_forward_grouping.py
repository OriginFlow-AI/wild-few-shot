"""Disposable Train-only probe: physical forward size vs fixed 16-window loss.

Does not edit any formal checkpoint, configuration, source binding or inputs.
Dropout RNG advancement is measured explicitly; a larger forward is not assumed
to reproduce the original stochastic trajectory.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch


def main(root, limit):
    root = Path(root).resolve()
    assert str(root).startswith('/pub_data/zjl/')
    sys.path.insert(0, str(root / 'code/scripts'))
    from common import atomic, settings, sha
    from train import Windows, network, objective, optimizer
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(limit * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
    for key, name in [('CUDA_CACHE_PATH', 'cuda_kernels'), ('TORCH_HOME', 'torch'), ('XDG_CACHE_HOME', 'xdg')]:
        os.environ[key] = str(root / 'cache' / name)
    configuration = settings()
    assert configuration['common_microbatch'] == 16 and configuration['global_batch'] == 256
    initial = torch.load(root / 'assets/common_stage2_initial_state_dict.pt', map_location='cpu', weights_only=True)
    dataset = Windows(root, 'train', 'raw_original')
    samples = [dataset[i] for i in range(256)]
    x = torch.from_numpy(np.stack([s[0] for s in samples])).cuda()
    y = torch.from_numpy(np.stack([s[1] for s in samples])).cuda()
    mask = torch.from_numpy(np.stack([s[2] for s in samples])).cuda()
    norm = dataset.data['force_norm']['raw']
    mean = torch.tensor(norm['mean'], device='cuda', dtype=torch.float32)[None, :, None]
    std = torch.tensor(norm['std'], device='cuda', dtype=torch.float32)[None, :, None]

    def one_step(net, physical, count):
        denominator = int(mask[:count].sum())
        total = 0.
        outputs = []
        for start in range(0, count, physical):
            end = min(count, start + physical)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z = net(x[start:end])
            pred = z.float() * std + mean
            terms = []
            for offset in range(0, end - start, 16):
                stop = min(end - start, offset + 16)
                loss, points = objective(pred[offset:stop], y[start+offset:start+stop],
                                         mask[start+offset:start+stop], 'raw_original')
                assert torch.isfinite(loss) and points > 0
                terms.append(loss * points / denominator)
            weighted = sum(terms)
            weighted.backward()
            total += float(weighted.detach())
            outputs.append(pred.detach().cpu())
        return total, torch.cat(outputs)

    def rng():
        return torch.cuda.get_rng_state().cpu().numpy().tobytes().hex()

    records = []
    reference_grads = reference_output = None
    for physical in (16, 32, 64):
        atomic(root / 'checks/forward_grouping_probe_progress.json', dict(physical_forward=physical, phase='disposable_train_probe'))
        try:
            torch.manual_seed(43)
            torch.cuda.manual_seed(43)
            net = network().cuda()
            net.load_state_dict(initial, strict=True)
            opt = optimizer(net, configuration)
            net.train()
            times = []
            first_rng = None
            torch.cuda.reset_peak_memory_stats()
            for step in range(3):
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                tick = time.monotonic()
                value, _ = one_step(net, physical, 256)
                grad = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True))
                opt.step()
                torch.cuda.synchronize()
                times.append(time.monotonic() - tick)
                if step == 0:
                    first_rng = rng()
            full_rng = rng()
            train_peak = torch.cuda.max_memory_allocated() / 2**30
            torch.cuda.manual_seed(43)
            opt.zero_grad(set_to_none=True)
            tail_loss, _ = one_step(net, physical, 111)
            tail_rng = rng()
            del opt
            net.load_state_dict(initial, strict=True)
            net.eval()  # deterministic grouping/gradient check; dropout intentionally disabled
            net.zero_grad(set_to_none=True)
            deterministic_loss, output = one_step(net, physical, 256)
            grads = torch.cat([p.grad.detach().float().flatten().cpu() for p in net.parameters() if p.grad is not None])
            if reference_grads is None:
                reference_grads, reference_output = grads, output
            result = dict(physical_forward=physical, logical_loss_group=16, global_batch=256,
                          windows_per_second=256/statistics.median(times[1:]), step_seconds=times,
                          peak_allocated_GiB=train_peak, train_loss=value, unclipped_gradient_norm=grad,
                          full256_after_first_update_cuda_rng=first_rng, full256_after_three_updates_cuda_rng=full_rng,
                          tail111_cuda_rng=tail_rng, tail111_loss=tail_loss,
                          deterministic_eval_mode_loss=deterministic_loss,
                          deterministic_prediction_max_abs_difference_N=float((output-reference_output).abs().max()),
                          deterministic_gradient_relative_l2=float(torch.linalg.vector_norm((grads-reference_grads).double()) /
                                                                   torch.linalg.vector_norm(reference_grads.double()).clamp_min(1e-30)),
                          cuda_rng_matches_physical16=None)
            if records and records[0].get('physical_forward') == 16 and 'status' not in records[0]:
                result['cuda_rng_matches_physical16'] = all(result[k] == records[0][k] for k in
                    ('full256_after_first_update_cuda_rng', 'full256_after_three_updates_cuda_rng', 'tail111_cuda_rng'))
            else:
                result['cuda_rng_matches_physical16'] = True
            records.append(result)
            del grads, output, net
        except torch.OutOfMemoryError as error:
            records.append(dict(physical_forward=physical, status='OOM_IN_DISPOSABLE_PROBE', error=str(error)[:350]))
            net = opt = None
            gc.collect()
            torch.cuda.empty_cache()
            break
        gc.collect()
        torch.cuda.empty_cache()
        atomic(root / 'analysis_output/forward_grouping_probe.json', dict(
            status='PROBE_RUNNING', attempts=records, formal_training_changed=False, test_metrics_read=False))
    result = dict(status='DISPOSABLE_TRAIN_ONLY_PROBE_COMPLETE', attempts=records,
                  memory_limit_GiB=limit, GPU=torch.cuda.get_device_name(0),
                  common_initial_sha256=sha(root / 'assets/common_stage2_initial_state_dict.pt'),
                  dataset_sha256=sha(root / 'prepared/dataset.json'),
                  formal_training_changed=False, test_metrics_read=False,
                  caveat='Eval-mode comparison tests deterministic grouping only. Train-mode dropout masks may differ even when RNG advancement matches; no equivalence or formal adoption is assumed.')
    atomic(root / 'analysis_output/forward_grouping_probe.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--limit', type=float, default=32)
    args = parser.parse_args()
    main(args.root, args.limit)
