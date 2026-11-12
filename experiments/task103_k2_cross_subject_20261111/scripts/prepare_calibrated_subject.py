"""Apply refitted test5 curves to the previously frozen Task103 person split."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

from bind_ap304_all_stages import bind_initialization, link
from common import EXPERIMENT, atomic, group, interpolate, load, settings, sha, source_binding
from prepare_cross_subject import SPLITS, normalized


@lru_cache(maxsize=128)
def model_for(path):
    model = load(path)
    for curve in model['channels'].values():
        x, y = np.asarray(curve['x']), np.asarray(curve['y'])
        assert len(x) == len(y) >= 2 and np.isfinite(x).all() and np.isfinite(y).all()
        assert np.all(np.diff(x) > 0) and np.all(np.diff(y) >= 0)
    return model


def apply_one(job):
    root, dataset_root, previous_ap304, refit, row, entries = job
    rid = row['id']
    raw_path = dataset_root / 'cache/labels' / f'{rid}.npz'
    native_path = previous_ap304 / 'cache/native' / f'{rid}.npz'
    if not native_path.is_file():
        native_path = dataset_root / 'cache/native' / f'{rid}.npz'
    curve_path = refit / 'cache/calibration' / f"{row['source_id']}.json"
    model = model_for(curve_path)
    assert sha(raw_path) == entries[0][1]['labels_sha256']
    point_counts = Counter()
    retained = []
    labels = {}
    with np.load(raw_path) as raw, np.load(native_path) as native:
        for i, entry, starts in entries:
            side = entry['side']
            t, raw_valid = (raw[f'{side}_{key}'] for key in ('t', 'valid'))
            ad = native[f'{side}_ad']
            assert np.array_equal(native[f'{side}_t'], t) and ad.shape == raw_valid.shape
            mask = np.zeros_like(raw_valid)
            force = np.zeros(raw_valid.shape, dtype=np.float32)
            point_counts['raw_valid_points'] += int(raw_valid.sum())
            for finger in range(5):
                channel = finger + (5 if side == 'right' else 0)
                curve = model['channels'].get(str(channel))
                if curve is None:
                    point_counts['missing_curve_points'] += int(raw_valid[:, finger].sum())
                    continue
                x, y = np.asarray(curve['x']), np.asarray(curve['y'])
                valid_ad = np.isfinite(ad[:, finger])
                point_counts['invalid_ad_points'] += int((raw_valid[:, finger] & ~valid_ad).sum())
                inside = (ad[:, finger] >= x[0]) & (ad[:, finger] <= x[-1])
                point_counts['out_of_domain_points'] += int((raw_valid[:, finger] & valid_ad & ~inside).sum())
                mask[:, finger] = raw_valid[:, finger] & inside & valid_ad
                use = mask[:, finger]
                force[use, finger] = np.interp(ad[use, finger], x, y)
            point_counts['cal_valid_points'] += int(mask.sum())
            query = entry['start_ns'] + 500_000 * (starts[:, None] + 470 + 40 * np.arange(89))
            _, use = interpolate(t, force, mask, query.ravel())
            keep = starts[use.reshape(len(starts), 89, 5).any(axis=(1, 2))]
            if not len(keep):
                continue
            labels.update({f'{side}_t': t, f'{side}_cal': force, f'{side}_valid': mask})
            force64 = force.astype(np.float64)
            stats = np.stack((mask.sum(axis=0), force64.sum(axis=0), (force64**2).sum(axis=0)))
            retained.append((i, entry, keep, stats))
    assert point_counts['raw_valid_points'] == sum(point_counts[k] for k in
            ('missing_curve_points', 'invalid_ad_points', 'out_of_domain_points', 'cal_valid_points'))
    if not retained:
        return [], dict(point_counts), 'no windows with valid calibrated force'
    path = root / 'cache/labels' / f'{rid}.npz'
    temp = path.with_suffix('.partial.npz')
    np.savez_compressed(temp, **labels)
    temp.replace(path)
    label_sha = sha(path)
    retained = [(i, {**entry, 'labels_sha256': label_sha, 'calibration_source_id': row['source_id']}, starts, stats)
                for i, entry, starts, stats in retained]
    return retained, dict(point_counts), ''


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main(root, dataset_root, previous_ap304, refit, workers):
    root, dataset_root, previous_ap304, refit = (p.resolve() for p in (root, dataset_root, previous_ap304, refit))
    source = dataset_root / 'prepared/dataset.json'
    data = load(source)
    config = settings()
    assert data['target'] == 'native_CMCU_N' and config['target'] == 'calibrated_CMCU_N'
    prior_manifest = dataset_root / 'prepared/subject_split.json'
    assert sha(prior_manifest) == data['subject_split_sha256']
    prior = load(prior_manifest)
    person_split = {p: s for s, people in prior['subjects'].items() for p in people}
    assert len(person_split) == sum(map(len, prior['subjects'].values()))
    plan_path = previous_ap304 / 'plan.json'
    plan = load(plan_path)
    assert sha(plan_path) == prior['source_plan_sha256']
    plan_by_id = {row['id']: row for row in plan['rows']}
    quality_path = refit / 'checks/calibration_quality.json'
    quality = {row['id']: row for row in load(quality_path)['records']}
    assert set(quality) == {row['id'] for row in plan['rows'] if row['stage'] == 'mix-test-5'}
    for directory in ('cache/labels', 'prepared', 'checks', 'assets'):
        (root / directory).mkdir(parents=True, exist_ok=True)
    link(dataset_root / 'cache/emg', root / 'cache/emg')
    emg_norm, verified = bind_initialization(root, previous_ap304)
    assert verified == 243

    starts_by_entry = defaultdict(list)
    for split, expected in data['window_indices_sha256'].items():
        path = dataset_root / 'prepared' / f'{split}_windows.npy'
        assert sha(path) == expected
        for entry, start in np.load(path, mmap_mode='r'):
            starts_by_entry[int(entry)].append(int(start))
    by_record = defaultdict(list)
    for i, entry in enumerate(data['entries']):
        assert person_split[entry['subject']] == entry['split']
        by_record[entry['id']].append((i, entry, np.asarray(starts_by_entry[i], dtype=np.int64)))
    jobs = []
    targets = {}
    for rid, entries in by_record.items():
        row = plan_by_id[rid]
        entry = entries[0][1]
        source_id = row['source_id']
        info = dict(id=rid, recording_id=row['recording_id'], object_key=row['key'], subject=entry['subject'],
                    split=entry['split'], stage=entry['stage'], calibration_source_id=source_id or '',
                    status='pending', raw_windows=sum(len(starts) for _, _, starts in entries),
                    cal_windows=0, raw_valid_points='', cal_valid_points='', missing_curve_points='',
                    invalid_ad_points='', out_of_domain_points='', retained_fraction='')
        targets[rid] = info
        if row['route_status'] != 'candidate':
            info['status'] = row['route_status']
            continue
        source_row = plan_by_id[source_id]
        assert source_row['stage'] == 'mix-test-5' and group(row) == group(source_row)
        assert person_split[source_row['subject']] == entry['split']
        if not quality[source_id]['channels']:
            info['status'] = 'nearest_test5_no_valid_curve'
            continue
        jobs.append((root, dataset_root, previous_ap304, refit, row, entries))

    usable = []
    stats = np.zeros((3, 5))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (job, result) in enumerate(zip(jobs, pool.map(apply_one, jobs, chunksize=8)), 1):
            retained, counts, reason = result
            info = targets[job[4]['id']]
            info.update({key: counts.get(key, 0) for key in ('raw_valid_points', 'cal_valid_points',
                        'missing_curve_points', 'invalid_ad_points', 'out_of_domain_points')})
            info['retained_fraction'] = info['cal_valid_points'] / max(1, info['raw_valid_points'])
            info['cal_windows'] = sum(len(starts) for _, _, starts, _ in retained)
            info['status'] = reason or 'retained'
            for _, entry, _, s in retained:
                if entry['split'] == 'train':
                    stats += s
            usable.extend(retained)
            if i % 500 == 0:
                print(f'Applied calibration to {i}/{len(jobs)} recordings', flush=True)

    entries = []
    indices = {split: [] for split in SPLITS}
    for _, entry, starts, _ in sorted(usable, key=lambda item: item[0]):
        indices[entry['split']].extend((len(entries), int(start)) for start in starts)
        entries.append(entry)
    prepared = root / 'prepared'
    counts = {}
    kept = [row for row in targets.values() if row['status'] == 'retained']
    for split in SPLITS:
        assert indices[split]
        np.save(prepared / f'{split}_windows.npy', np.asarray(indices[split], dtype=np.int32))
        rows = [row for row in kept if row['split'] == split]
        counts[split] = dict(people=len({row['subject'] for row in rows}), recordings=len(rows),
                             windows=len(indices[split]), by_stage=dict(Counter(row['stage'] for row in rows)),
                             missing_stages=sorted(set(prior['counts'][split]['by_stage']) - {row['stage'] for row in rows}))
    write_csv(prepared / 'recording_split.csv', [{key: row[key] for key in
              ('id', 'recording_id', 'subject', 'stage', 'split', 'calibration_source_id')} for row in kept])
    write_csv(root / 'checks/calibration_target_coverage.csv', list(targets.values()))
    split_info = dict(
        seed=prior['seed'], subject_disjoint=True, subjects=prior['subjects'], counts=counts,
        parent_subject_split_sha256=sha(prior_manifest), parent_dataset_sha256=sha(source),
        source_plan_sha256=sha(plan_path), source_refit_quality_sha256=sha(quality_path),
        recording_split_sha256=sha(prepared / 'recording_split.csv'),
        policy='Keep frozen subject assignments and original window starts with at least one calibrated point.')
    atomic(prepared / 'subject_split.json', split_info)
    atomic(prepared / 'dataset.json', dict(
        target='calibrated_CMCU_N', entries=entries, force_norm={'cal': normalized(stats)}, emg_norm=emg_norm,
        windows={s: len(indices[s]) for s in SPLITS}, split_recordings=counts,
        config_sha256=sha(EXPERIMENT / 'config.json'), source_code=source_binding(),
        shared_k2_dataset_sha256=sha(source), subject_split_sha256=sha(prepared / 'subject_split.json'),
        emg_norm_sha256=config['emg_norm_sha256'], common_initial_sha256=config['common_initial_sha256'],
        window_indices_sha256={s: sha(prepared / f'{s}_windows.npy') for s in SPLITS},
        source_refit_quality_sha256=sha(quality_path)))
    print(counts, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--previous-ap304', type=Path, required=True)
    parser.add_argument('--calibration-refit', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=20)
    args = parser.parse_args()
    main(args.root, args.dataset_root, args.previous_ap304, args.calibration_refit, args.workers)
