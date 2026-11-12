"""Evaluate frozen K2 weights on held-out ordinary stages with common CMCU labels."""
import argparse
from collections import defaultdict
import csv
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import BRANCHES, atomic, interpolate, load, sha
from prepare_cross_subject import REFERENCE_STAGES
from score_force_thresholds import METRICS, score
from train import network


def predict_hand(source, entry, starts, net, emg_norm, force_norm, device):
    side = entry['side']
    folder = source / 'cache/emg' / entry['id']
    signal = np.load(folder / f'{side}.npy', mmap_mode='r')
    emg_valid = np.load(folder / f'{side}_valid.npy', mmap_mode='r')
    assert len(signal) == entry['samples']
    em, es = emg_norm
    fm, fs = force_norm
    offsets = (starts[:, None] + 470 + 40 * np.arange(89)).ravel()
    query = entry['start_ns'] + 500_000 * offsets
    keep = np.r_[True, offsets[1:] > np.maximum.accumulate(offsets)[:-1]]
    with np.load(source / 'cache/labels' / f"{entry['id']}.npz") as labels:
        t = labels[f'{side}_t']
        valid = labels[f'{side}_valid']
        raw, raw_mask = interpolate(t, labels[f'{side}_raw'], valid, query)
        cal, cal_mask = interpolate(t, labels[f'{side}_cal'], valid, query)
    assert np.array_equal(raw_mask, cal_mask)
    preds = []
    for chunk in np.array_split(starts, (len(starts) + 63) // 64):
        x = np.stack([((signal[s:s+4000] - em) / es).T for s in chunk])
        assert np.isfinite(x).all() and all(emg_valid[s:s+4000].all() for s in chunk)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            normalized = net(torch.from_numpy(x).to(device))
        preds.append((normalized.float() * fs + fm).transpose(1, 2).flatten(0, 1).cpu().numpy())
    prediction = np.concatenate(preds)
    assert np.isfinite(prediction).all()
    return prediction[keep], raw[keep], cal[keep], raw_mask[keep], query[keep]


@torch.inference_mode()
def evaluate(root, source, output, branch):
    previous = load(source / 'prepared/dataset.json')
    people = load(root / 'prepared/subject_split.json')['subjects']
    person_split = {person: split for split in ('val', 'test') for person in people[split]}
    entries = previous['entries']
    indices = defaultdict(list)
    for i, start in np.load(source / 'prepared/train_windows.npy'):
        entry = entries[i]
        if entry['subject'] in person_split and entry['stage'] not in REFERENCE_STAGES:
            indices[int(i)].append(int(start))
    records = defaultdict(list)
    for i, starts in indices.items():
        records[entries[i]['id']].append((entries[i], np.asarray(sorted(starts))))
    assert records and set(records).isdisjoint(
        e['id'] for e in load(root / 'prepared/dataset.json')['entries'] if e['split'] == 'train')
    exported = root / 'exported_models' / branch
    info = load(exported / 'inference_parameters.json')
    weight = exported / 'best_model_state_dict.pt'
    norms = exported / 'emg_train_norm.npz'
    assert sha(weight) == info['model_state_sha256']
    assert sha(norms) == info['emg_norm_sha256'] == sha(root / 'prepared/emg_train_norm.npz')
    done = load(root / 'runs' / branch / 'DONE.json')
    assert info['selected_training_checkpoint_sha256'] == done['best_sha256']
    torch.set_num_threads(2)
    device = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(12 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    net = network()
    net.load_state_dict(torch.load(weight, map_location='cpu', weights_only=True), strict=True)
    net.to(device).eval()
    with np.load(norms) as saved:
        emg_norm = saved['emg_mean'], saved['emg_std']
    force_norm = tuple(torch.tensor(info['force_norm'][key], dtype=torch.float32,
                                   device=device)[None, :, None] for key in ('mean', 'std'))
    rows = []
    shared = hashlib.sha256()
    for number, (record_id, hands) in enumerate(sorted(records.items()), 1):
        results = []
        for entry, starts in sorted(hands, key=lambda pair: pair[0]['side']):
            result = predict_hand(source, entry, starts, net, emg_norm, force_norm, device)
            results.append(result[:-1])
            shared.update(f"{record_id}/{entry['side']}".encode())
            for values in result[1:]:
                shared.update(values.tobytes())
        pred, raw, cal, mask = (np.concatenate(parts) for parts in zip(*results))
        assert mask.any()
        meta = hands[0][0]
        for label, target in (('raw_CMCU', raw), ('calibrated_CMCU', cal)):
            _, values, counts = score(pred, target, mask)
            rows.append(dict(split=person_split[meta['subject']], stage=meta['stage'],
                             subject=meta['subject'], recording_id=record_id, target=label,
                             **values, **{f'points_{k}': n for k, n in counts.items()}))
        if number % 100 == 0:
            print(f'{branch}: {number}/{len(records)} recordings', flush=True)
    atomic(output / f'{branch}.json', dict(
        branch=branch, epoch=info['selected_epoch'], checkpoint_sha256=done['best_sha256'],
        source_dataset_sha256=sha(source / 'prepared/dataset.json'),
        subject_split_sha256=sha(root / 'prepared/subject_split.json'),
        shared_scoring_sha256=shared.hexdigest(), rows=rows))


def summarize(output):
    receipts = [load(output / f'{branch}.json') for branch in BRANCHES]
    assert len({r['shared_scoring_sha256'] for r in receipts}) == 1
    combined = []
    for receipt in receipts:
        groups = defaultdict(lambda: defaultdict(list))
        for row in receipt['rows']:
            for stage in ('all', row['stage']):
                groups[row['split'], stage, row['target']][row['subject']].append(row)
        for (split, stage, target), by_person in sorted(groups.items()):
            flat = [row for rows in by_person.values() for row in rows]
            summary = dict(split=split, stage=stage, target=target, branch=receipt['branch'],
                           epoch=receipt['epoch'], subjects=len(by_person), recordings=len(flat))
            for key in (k for k in flat[0] if k.startswith('points_')):
                summary[key] = sum(row[key] for row in flat)
            for key in METRICS:
                means = [np.mean(values) for rows in by_person.values()
                         if (values := [row[key] for row in rows if row[key] is not None])]
                summary[key] = float(np.mean(means)) if means else None
            combined.append(summary)
    with (output / 'ordinary_stage_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(combined)
    atomic(output / 'ordinary_stage_metrics.json', dict(
        shared_scoring_sha256=receipts[0]['shared_scoring_sha256'], metrics=combined))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--branch', choices=BRANCHES)
    args = parser.parse_args()
    if args.branch:
        evaluate(args.root, args.source, args.output, args.branch)
    else:
        summarize(args.output)
