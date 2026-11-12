"""Prepare a person-only Task103 split with native CMCU targets in all stages."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
from pathlib import Path

import numpy as np

from common import EXPERIMENT, SIDES, atomic, interpolate, load, settings, sha, source_binding, windows
from prepare_cross_subject import SPLITS, normalized


def assign_people(rows, prior, seed):
    assignment = {p: split for split, people in prior.items() for p in people}
    added = sorted({r['subject'] for r in rows} - assignment.keys(),
                   key=lambda p: hashlib.sha256(f'{seed}:{p}'.encode()).digest())
    holdout = round(len(added) * 0.15)
    assignment.update({person: ('val' if i < holdout else 'test' if i < 2 * holdout else 'train')
                       for i, person in enumerate(added)})
    return assignment


def prepare_record(job):
    root, source, row = job
    record_id = row['id']
    base = source if (source / 'cache/native' / f'{record_id}.npz').is_file() else root
    signal = base / 'cache/emg' / record_id
    native = base / 'cache/native' / f'{record_id}.npz'
    if not native.is_file() or not (signal / 'done.json').is_file():
        return [], 'missing native stream cache'
    meta = load(signal / 'done.json')
    clock = meta['clock_offsets']
    if abs(clock['left'] - clock['right']) > 50e6:
        return [], 'left/right CMCU clock consistency failed'
    if meta['etag'] != row['etag'] or sha(native) != meta['native_sha256']:
        raise ValueError(f'Cache identity mismatch: {record_id}')
    labels = {}
    entries = []
    with np.load(native) as values:
        for side in SIDES:
            hand = meta['hands'][side]
            t = values[f'{side}_t']
            raw = values[f'{side}_force'].astype(np.float32)
            if len(t) < 2 or raw.shape != (len(t), 5) or np.any(np.diff(t) <= 0):
                raise ValueError(f'Invalid native force stream: {record_id}/{side}')
            valid = np.isfinite(raw) & (raw >= 0) & (raw <= 100)
            labels.update({f'{side}_t': t, f'{side}_raw': raw, f'{side}_valid': valid})
            emg_valid = np.load(signal / f'{side}_valid.npy', mmap_mode='r')
            assert len(emg_valid) == hand['samples']
            prefix = np.r_[0, np.cumsum(~emg_valid)]
            starts = np.asarray(windows(hand['samples']), dtype=np.int64)
            starts = starts[prefix[starts] == prefix[starts + 4000]]
            query = hand['start_ns'] + 500_000 * (starts[:, None] + 470 + 40 * np.arange(89))
            _, mask = interpolate(t, raw, valid, query.ravel())
            starts = starts[mask.reshape(len(starts), 89, 5).any(axis=(1, 2))]
            if not len(starts):
                continue
            entry = dict(id=record_id, side=side, split=row['split'], subject=row['subject'],
                         stage=row['stage'], start_ns=hand['start_ns'], samples=hand['samples'],
                         EMG=meta['processed_files'][f'{side}.npy'])
            stats = None
            if row['split'] == 'train':
                emg = np.load(signal / f'{side}.npy', mmap_mode='r')
                data = emg if emg_valid.all() else emg[emg_valid]
                force = np.where(valid, raw, 0).astype(np.float64)
                stats = (
                    np.stack([np.full(16, len(data)), data.sum(axis=0, dtype=np.float64),
                              np.einsum('ij,ij->j', data, data, dtype=np.float64)]),
                    np.stack([valid.sum(axis=0), force.sum(axis=0), (force * force).sum(axis=0)]),
                )
            entries.append((entry, starts, stats))
    if not entries:
        return [], 'no valid EMG windows with native CMCU targets'
    if base == source:
        link = root / 'cache/emg' / record_id
        if not link.exists():
            link.symlink_to(signal, target_is_directory=True)
        if link.resolve() != signal.resolve():
            raise ValueError(f'Unexpected EMG cache: {record_id}')
    path = root / 'cache/labels' / f'{record_id}.npz'
    temp = path.with_suffix('.partial.npz')
    np.savez_compressed(temp, **labels)
    temp.replace(path)
    digest = sha(path)
    for entry, _, _ in entries:
        entry['labels_sha256'] = digest
    return entries, None


def cross_split_duplicates(records, candidates):
    keys = defaultdict(list)
    for row, hands in zip(records, candidates):
        if not hands:
            continue
        keys['recording', row['recording_id']].append(row)
        if row['etag'] and row['etag'] != '00000000000000000000000000000000-1':
            keys['object', row['etag'], row['size']].append(row)
        for entry, _, _ in hands:
            keys['processed_EMG', entry['EMG']['sha256']].append(row)
    return {row['id'] for group in keys.values() if len({r['split'] for r in group}) > 1 for row in group}


def main(source, prior_root, root, workers):
    import torch
    from train import network

    plan_path = source / 'plan.json'
    people_path = prior_root / 'prepared/subject_split.json'
    config = settings()
    plan = load(plan_path)
    rows = plan['rows']
    prior = load(people_path)['subjects']
    people = assign_people(rows, prior, config['split_seed'])
    rows = [{**row, 'split': people[row['subject']]} for row in rows]
    extraction = load(root / 'checks/missing_native_extraction.json')
    assert extraction['phase'] == 'complete'
    for folder in ('cache/emg', 'cache/labels', 'prepared', 'assets'):
        (root / folder).mkdir(parents=True, exist_ok=True)
    exclusions = {}
    candidates = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = ((root, source, row) for row in rows)
        for i, (hands, reason) in enumerate(pool.map(prepare_record, jobs, chunksize=4)):
            if reason:
                row = rows[i]
                error = extraction['errors'].get(row['id'])
                exclusions[row['id']] = dict(stage=row['stage'], split=row['split'],
                                              reason=error['reason'] if error else reason)
            candidates.append(hands)
            if (i + 1) % 500 == 0:
                print(f'Prepared {i+1}/{len(rows)} recordings', flush=True)
    duplicates = cross_split_duplicates(rows, candidates)
    entries, included = [], []
    indices = {split: [] for split in SPLITS}
    emg_stats = np.zeros((3, 16))
    force_stats = np.zeros((3, 5))
    for row, hands in zip(rows, candidates):
        if row['id'] in duplicates:
            exclusions[row['id']] = dict(stage=row['stage'], split=row['split'],
                                          reason='cross-split duplicate content')
            continue
        if not hands:
            continue
        included.append({key: row[key] for key in ('id', 'recording_id', 'subject', 'stage', 'split')})
        for entry, starts, stats in hands:
            indices[entry['split']].extend((len(entries), int(start)) for start in starts)
            entries.append(entry)
            if stats is not None:
                emg_stats += stats[0]
                force_stats += stats[1]

    prepared = root / 'prepared'
    stages = {row['stage'] for row in rows}
    counts = {}
    for split in SPLITS:
        rs = [row for row in included if row['split'] == split]
        by_stage = dict(Counter(row['stage'] for row in rs))
        if set(by_stage) != stages:
            raise ValueError(f'{split} lacks eligible recordings for {sorted(stages - by_stage.keys())}')
        np.save(prepared / f'{split}_windows.npy', np.asarray(indices[split], dtype=np.int32))
        counts[split] = dict(people=len({row['subject'] for row in rs}), recordings=len(rs),
                             windows=len(indices[split]), by_stage=by_stage)
    with (prepared / 'recording_split.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(included[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(included)
    split_info = dict(
        seed=config['split_seed'], subject_disjoint=True, all_stages_per_split=True,
        prior_subject_split_sha256=sha(people_path), source_plan_sha256=sha(plan_path),
        policy='Keep prior assignments. Assign new people by seed/SHA256 order, 15% Val and 15% Test.',
        subjects={s: sorted(p for p, group in people.items() if group == s) for s in SPLITS},
        counts=counts, recording_split_sha256=sha(prepared / 'recording_split.csv'))
    atomic(prepared / 'subject_split.json', split_info)
    atomic(prepared / 'exclusions.json', exclusions)
    emg = normalized(emg_stats)
    np.savez(prepared / 'emg_train_norm.npz', emg_mean=np.asarray(emg['mean'], dtype=np.float32),
             emg_std=np.asarray(emg['std'], dtype=np.float32), emg_count=emg['count'])
    torch.set_num_threads(2)
    torch.manual_seed(config['seed'])
    init = root / 'assets/common_stage2_initial_state_dict.pt'
    model = network()
    previous = torch.load(prior_root / 'assets/common_stage2_initial_state_dict.pt', weights_only=True)
    assert all(torch.equal(value, previous[key]) for key, value in model.state_dict().items())
    torch.save(model.state_dict(), init)
    atomic(prepared / 'dataset.json', dict(
        target='native_CMCU_N', entries=entries, force_norm={'raw': normalized(force_stats)}, emg_norm=emg,
        windows={s: len(indices[s]) for s in SPLITS}, split_recordings=counts,
        config_sha256=sha(EXPERIMENT / 'config.json'), source_code=source_binding(),
        subject_split_sha256=sha(prepared / 'subject_split.json'),
        emg_norm_sha256=sha(prepared / 'emg_train_norm.npz'), common_initial_sha256=sha(init),
        window_indices_sha256={s: sha(prepared / f'{s}_windows.npy') for s in SPLITS},
        exclusions_sha256=sha(prepared / 'exclusions.json')))
    atomic(root / 'STATUS.json', dict(phase='prepared', counts=counts,
                                      dataset_sha256=sha(prepared / 'dataset.json')))
    print(counts, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--prior-root', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    main(args.source, args.prior_root, args.root, args.workers)
