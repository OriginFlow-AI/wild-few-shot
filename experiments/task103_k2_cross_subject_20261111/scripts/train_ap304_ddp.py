"""Train the AP304 branches on one shared Task103 dataset using local NCCL ranks."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from common import EXPERIMENT, atomic, load, settings, sha, source_binding
from train import Windows, epoch_order, evaluate_native, network, objective, optimizer, physical_forward_windows


class PaddedWindows(Windows):
    def __getitem__(self, index):
        if index == -1:
            return np.zeros((16, 4000), np.float32), np.zeros((5, 89), np.float32), np.zeros((5, 89), bool)
        return super().__getitem__(index)


def save(path, net, opt, state, binding, rank, world):
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state())
    received = [None] * world if rank == 0 else None
    dist.gather_object(rng, received, dst=0)
    if rank == 0:
        tmp = path.with_suffix('.partial.pt')
        torch.save(dict(model=net.state_dict(), optimizer=opt.state_dict(), state=state,
                        binding=binding, rank_rng=received), tmp)
        tmp.replace(path)


@torch.no_grad()
def evaluate(net, root, split, branch, mean, std, device, rank, world, force_loss):
    ds = Windows(root, split, branch)
    ids = sorted({row['id'] for row in ds.data['entries'] if row['split'] == split})
    local_ids = set(ids[rank::world])
    local_entries = {i for i, row in enumerate(ds.data['entries']) if row['id'] in local_ids}
    by_entry = defaultdict(list)
    for index, start in ds.index:
        if int(index) in local_entries:
            by_entry[int(index)].append(int(start))
    net.eval()
    result = evaluate_native(net, ds, by_entry, branch, mean, std, device, force_loss)
    gathered = [None] * world
    dist.all_gather_object(gathered, result['rows'])
    rows = sorted((row for group in gathered for row in group), key=lambda row: row['id'])
    assert [row['id'] for row in rows] == ids
    return dict(records=len(rows), loss=float(np.mean([row['loss'] for row in rows])), rows=rows)


def train(root, branch, smoke, rank, world, device):
    c = settings()
    ds = PaddedWindows(root, 'train', branch)
    data = ds.data
    force_loss = c.get('force_loss')
    if branch == 'calibrated_original':
        assert force_loss is not None and data['target'] == 'calibrated_CMCU_N'
    init = root / 'assets/common_stage2_initial_state_dict.pt'
    norm_path = root / 'prepared/emg_train_norm.npz'
    assert ds.native_only and data['source_code'] == source_binding()
    assert sha(EXPERIMENT / 'config.json') == data['config_sha256']
    assert sha(init) == data['common_initial_sha256'] and sha(norm_path) == data['emg_norm_sha256']
    assert sha(root / 'prepared/subject_split.json') == data['subject_split_sha256']
    assert all(sha(root / 'prepared' / f'{split}_windows.npy') == digest
               for split, digest in data['window_indices_sha256'].items())
    batch = c['global_batch']
    micro = c['microbatch']
    local_batch = batch // world
    assert batch % (world * micro) == 0 and micro == c['common_microbatch']
    binding = dict(config_sha256=data['config_sha256'], dataset_sha256=sha(root / 'prepared/dataset.json'),
                   shared_k2_dataset_sha256=data['shared_k2_dataset_sha256'],
                   common_initial_sha256=data['common_initial_sha256'], source_code=data['source_code'],
                   subject_split_sha256=data['subject_split_sha256'], EMG_norm_sha256=data['emg_norm_sha256'],
                   branch=branch, force_norm=data['force_norm'][ds.kind], world_size=world)
    norm = binding['force_norm']
    mean, std = (torch.tensor(norm[k], dtype=torch.float32, device=device)[None, :, None] for k in ('mean', 'std'))
    folder = root / ('checks' if smoke else 'runs') / (f'smoke_{branch}' if smoke else branch)
    folder.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        atomic(folder / 'CONTRACT.json', dict(binding=binding, configuration=c))
    torch.manual_seed(c['seed'] + rank)
    torch.cuda.manual_seed(c['seed'] + rank)
    net = network()
    net.load_state_dict(torch.load(init, map_location='cpu', weights_only=True), strict=True)
    net.to(device)
    model = DistributedDataParallel(net, device_ids=[device.index], broadcast_buffers=False, gradient_as_bucket_view=True)
    opt = optimizer(net, c)
    state = dict(epoch=0, next_update=0, global_update=0, best=None, early_reference=None, bad_epochs=0, history=[])
    last, best = folder / 'last.pt', folder / 'best.pt'
    if last.is_file() and not smoke:
        saved = torch.load(last, map_location='cpu', weights_only=True)
        assert saved['binding'] == binding
        net.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        state = saved['state']
        cpu_rng, cuda_rng = saved['rank_rng'][rank]
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        del saved
    updates = math.ceil(len(ds) / batch)
    warm = c['warmup_updates']
    for epoch in range(state['epoch'], c['max_epochs']):
        if state['bad_epochs'] >= c['patience']:
            break
        order = epoch_order(len(ds), c['seed'] + epoch, micro)
        order = np.pad(order, (0, updates * batch - len(order)), constant_values=-1)
        batches = order.reshape(-1, batch)[state['next_update']:]
        if smoke:
            batches = batches[:2]
        sampler = batches[:, rank * local_batch:(rank + 1) * local_batch].ravel().tolist()
        # Isolate worker seeding from the model RNG saved in checkpoints.
        loader_rng = torch.Generator().manual_seed(c['seed'] + epoch * world + rank)
        loader = DataLoader(ds, batch_size=local_batch, sampler=sampler, num_workers=c['train_workers'],
                            pin_memory=True, timeout=120 if c['train_workers'] else 0, generator=loader_rng)
        start_time = time.monotonic()
        model.train()
        for u, (x, target, valid) in enumerate(loader, state['next_update']):
            den = valid.sum().to(device=device, non_blocking=True)
            dist.all_reduce(den)
            if den.item() <= 0:
                raise ValueError('Empty global training update')
            opt.zero_grad(set_to_none=True)
            step = state['global_update']
            physical = physical_forward_windows(c, step, micro)
            assert physical % micro == 0
            reduced_loss = torch.zeros((), device=device)
            for start in range(0, local_batch, physical):
                stop = min(start + physical, local_batch)
                with model.no_sync() if stop < local_batch else nullcontext():
                    xx, yy, mask = (array[start:stop].to(device, non_blocking=True) for array in (x, target, valid))
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        pred = model(xx)
                    pred = pred.float() * std + mean
                    combined = pred.sum() * 0
                    for lo in range(0, len(xx), micro):
                        window = slice(lo, lo + micro)
                        loss, count = objective(pred[window], yy[window], mask[window], branch, force_loss)
                        combined = combined + loss * count
                    combined = combined / den
                    if not torch.isfinite(combined):
                        raise ValueError('Nonfinite loss')
                    (combined * world).backward()
                    reduced_loss += combined.detach()
            dist.all_reduce(reduced_loss)
            progress = min(1.0, (step - warm) / max(1, updates * c['max_epochs'] - warm - 1))
            factor = (step + 1) / warm if step < warm else .1 + .9 * (1 + math.cos(math.pi * progress)) / 2
            for group in opt.param_groups:
                group['lr'] = group['base_lr'] * factor
            grad = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True))
            opt.step()
            state.update(next_update=u + 1, global_update=step + 1)
            peak = torch.tensor(torch.cuda.max_memory_allocated() / 2**30, device=device)
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            if rank == 0:
                status = dict(phase='training', branch=branch, epoch=epoch + 1, epoch_update=u + 1,
                              updates_per_epoch=updates, global_update=step + 1, loss=float(reduced_loss),
                              valid_points=int(den), gradient_norm=grad, peak_allocated_GiB=float(peak),
                              world_size=world, physical_forward_windows=physical,
                              elapsed_epoch_s=time.monotonic() - start_time)
                atomic(folder / 'STATUS.json', status)
                with (folder / 'metrics.jsonl').open('a') as handle:
                    handle.write(json.dumps(status) + '\n')
                if smoke and u + 1 == len(batches):
                    atomic(folder / 'PASS.json', status)
                    print(status, flush=True)
            if smoke and u + 1 == len(batches):
                return
            if (step + 1) % c['save_every_updates'] == 0:
                save(last, net, opt, state, binding, rank, world)
        save(last, net, opt, state, binding, rank, world)
        if rank == 0:
            atomic(folder / 'STATUS.json', dict(phase='validation', epoch=epoch + 1,
                                              global_update=state['global_update']))
        val = evaluate(net, root, 'val', branch, mean, std, device, rank, world, force_loss)
        loss = val['loss']
        is_best = state['best'] is None or loss < state['best']
        if is_best:
            state['best'] = loss
        if state['early_reference'] is None or loss < state['early_reference'] - c['min_delta']:
            state.update(early_reference=loss, bad_epochs=0)
        else:
            state['bad_epochs'] += 1
        state['history'].append(dict(epoch=epoch + 1, val_loss=loss, global_update=state['global_update']))
        if rank == 0:
            atomic(folder / f'epoch{epoch+1:03d}_val.json', val)
        state.update(epoch=epoch + 1, next_update=0)
        if is_best:
            save(best, net, opt, state, binding, rank, world)
        save(last, net, opt, state, binding, rank, world)
    dist.barrier()
    checkpoint = torch.load(best, map_location='cpu', weights_only=True)
    net.load_state_dict(checkpoint['model'])
    del checkpoint
    if rank == 0:
        atomic(folder / 'STATUS.json', dict(phase='final_test', branch=branch, selected_checkpoint_sha256=sha(best)))
    test = evaluate(net, root, 'test', branch, mean, std, device, rank, world, force_loss)
    if rank == 0:
        atomic(folder / 'final_test.json', test)
        done = dict(phase='complete', branch=branch, history=state['history'], test_recordings=test['records'],
                    test_not_used_for_selection=True, best_sha256=sha(best), binding=binding)
        atomic(folder / 'DONE.json', done)
        atomic(folder / 'STATUS.json', done)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--branch', choices=('raw_huber', 'raw_original', 'calibrated_original'), required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    limit = float(os.environ.get('TASK103_GPU_MEMORY_GIB', settings()['gpu_memory_limit_GiB']))
    torch.cuda.set_per_process_memory_fraction(limit * 2**30 / torch.cuda.get_device_properties(device).total_memory)
    dist.init_process_group('nccl', device_id=device)
    try:
        train(args.root, args.branch, args.smoke, dist.get_rank(), dist.get_world_size(), device)
    finally:
        dist.destroy_process_group()
