"""Check final selection, exact scoring inputs and published metrics against the run."""
import argparse
import collections
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import BRANCHES, atomic, interpolate, load, settings, sha, source_binding


TARGETS = ('raw', 'cal', 'bsq')
EXPERIMENTS = (
    (1, 'calibrated_huber', 'cal'),
    (2, 'calibrated_huber', 'bsq'),
    (3, 'raw_huber', 'bsq'),
    (4, 'raw_original', 'bsq'),
)


def scoring_inputs(root, data, split):
    """Rebuild each scored recording directly from the frozen window index and labels."""
    index = np.load(root / 'prepared' / f'{split}_windows.npy', mmap_mode='r')
    assert len(index) == data['windows'][split]
    entries = collections.defaultdict(list)
    for entry, start in index:
        entries[int(entry)].append(int(start))
    records = collections.defaultdict(list)
    for entry, starts in sorted(entries.items()):
        meta = data['entries'][entry]
        assert meta['split'] == split and starts == sorted(set(starts)), (split, entry)
        records[meta['id']].append((meta, starts))
    assert records.keys() == {entry['id'] for entry in data['entries'] if entry['split'] == split}
    for record_id, hands in records.items():
        label_file = root / 'cache/labels' / f'{record_id}.npz'
        assert all(meta['labels_sha256'] == sha(label_file) for meta, _ in hands), record_id
        parts = collections.defaultdict(list)
        with np.load(label_file) as labels:
            for meta, starts in hands:
                side = meta['side']
                finger = 0 if side == 'left' else 5
                previous_end = -1
                offsets = []
                for start in starts:
                    assert 0 <= start <= meta['samples'] - 4000
                    queries = start + 470 + 40 * np.arange(89, dtype=np.int64)
                    offsets.append(queries[queries > previous_end])
                    previous_end = int(queries[-1])
                t_ns = meta['start_ns'] + np.concatenate(offsets) * 500_000
                mask = np.ones((len(t_ns), 5), dtype=bool)
                for target in ('raw', 'cal'):
                    force, valid = interpolate(labels[f'{side}_t'], labels[f'{side}_{target}'],
                                               labels[f'{side}_valid'], t_ns)
                    parts[target].append(force)
                    mask &= valid
                bsq, valid = interpolate(labels['reference_t'],
                                         np.repeat(labels['reference_N'][:, None], 5, axis=1),
                                         labels['reference_valid'][:, finger:finger + 5], t_ns)
                parts['bsq'].append(bsq)
                parts['mask'].append(mask & valid)
                parts['finger'].append(np.broadcast_to(np.arange(finger, finger + 5), mask.shape))
                parts['t_ns'].append(np.broadcast_to(t_ns[:, None], mask.shape))
        arrays = {name: np.concatenate(values) for name, values in parts.items()}
        assert arrays['mask'].any(), record_id
        yield record_id, hands[0][0], arrays


def rows_for(receipt, expected):
    rows = {row['id']: row for row in receipt['rows']}
    assert len(rows) == len(receipt['rows']) == receipt['records'] == len(expected)
    assert rows.keys() == expected.keys()
    assert all(math.isfinite(row['loss']) for row in rows.values())
    assert math.isclose(receipt['loss'], np.mean([row['loss'] for row in rows.values()]),
                        rel_tol=0, abs_tol=1e-12)
    return rows


def check_val(root, data, histories):
    expected = {}
    for record_id, meta, arrays in scoring_inputs(root, data, 'val'):
        mask = arrays['mask']
        expected[record_id] = (meta['subject'], meta['stage'], int(mask.sum()),
                               {target: int(np.count_nonzero(mask & (abs(arrays[target]) >= .5)))
                                for target in TARGETS})
    for branch, history in histories.items():
        for epoch, loss in history:
            receipt = load(root / 'runs' / branch / f'epoch{epoch:03d}_val.json')
            assert receipt['loss'] == loss, (branch, epoch)
            for record_id, row in rows_for(receipt, expected).items():
                subject, stage, points, mape_points = expected[record_id]
                assert (row['subject'], row['stage']) == (subject, stage)
                for target in TARGETS:
                    actual = row['metrics'][target]
                    assert actual['points'] == points, (branch, epoch, record_id, target)
                    assert actual['mape_points'] == mape_points[target], (branch, epoch, record_id, target)
    return dict(recordings=len(expected), points=sum(item[2] for item in expected.values()),
                complete_epochs={branch: len(history) for branch, history in histories.items()})


def check_selection(root, branch, data, config, dataset_sha):
    import torch
    from audit_selected_validation import audit

    run = root / 'runs' / branch
    done = load(run / 'DONE.json')
    assert done['phase'] == 'complete' and done['branch'] == branch and done['test_not_used_for_selection']
    assert load(run / 'STATUS.json')['selected_checkpoint_sha256'] == done['best_sha256']
    contract = load(run / 'CONTRACT.json')
    assert done['binding'] == {key: value for key, value in contract.items() if key != 'configuration'}
    binding = done['binding']
    assert binding['dataset_sha256'] == dataset_sha
    assert binding['config_sha256'] == data['config_sha256'] and binding['source_code'] == data['source_code']
    assert binding['common_initial_sha256'] == data['common_initial_sha256']
    assert binding['common_initial_sha256'] == sha(root / 'assets/common_stage2_initial_state_dict.pt')
    assert binding['EMG_norm_sha256'] == data['emg_norm_sha256'] == sha(root / 'prepared/emg_train_norm.npz')
    assert binding['subject_split_sha256'] == data['subject_split_sha256'] == sha(root / 'prepared/subject_split.json')
    assert binding['force_norm'] == data['force_norm']['cal' if branch == 'calibrated_huber' else 'raw']
    updates = math.ceil(data['windows']['train'] / config['global_batch'])
    history = done['history']
    assert 0 < len(history) <= config['max_epochs']
    reference = math.inf
    bad = 0
    for epoch, row in enumerate(history, 1):
        assert row['epoch'] == epoch and row['global_update'] == epoch * updates
        assert bad < config['patience'] and math.isfinite(row['val_loss'])
        if row['val_loss'] < reference - config['min_delta']:
            reference = row['val_loss']
            bad = 0
        else:
            bad += 1
    assert len(history) == config['max_epochs'] or bad == config['patience']
    last = torch.load(run / 'last.pt', map_location='cpu', weights_only=True)
    state = last['state']
    assert last['binding'] == binding and state['history'] == history
    assert (state['epoch'], state['next_update'], state['global_update']) == (len(history), 0, len(history) * updates)
    assert (state['bad_epochs'], state['early_reference']) == (bad, reference)
    assert all(int(value['step']) == state['global_update'] for value in last['optimizer']['state'].values())
    del last
    best = min(history, key=lambda row: row['val_loss'])
    assert state['best'] == best['val_loss'] and sha(run / 'best.pt') == done['best_sha256']
    audit(root, branch, best['epoch'])
    return dict(epochs=len(history), best_epoch=best['epoch'], best_val_loss=best['val_loss'],
                best_sha256=done['best_sha256'], stop_reason='max_epochs' if len(history) == config['max_epochs']
                else 'val_patience', test_recordings=done['test_recordings'])


def check_test(root, data, selected):
    from train import measures

    entries = {entry['id']: entry for entry in data['entries'] if entry['split'] == 'test'}
    test_rows = {}
    for branch in BRANCHES:
        run = root / 'runs' / branch
        test_rows[branch] = rows_for(load(run / 'final_test.json'), entries)
        assert selected[branch]['test_recordings'] == len(entries)
        assert {path.stem for path in (run / 'test_predictions').glob('*.npz')} == entries.keys()
    points = collections.Counter()
    for record_id, meta, expected in scoring_inputs(root, data, 'test'):
        mask = expected['mask']
        points[meta['stage']] += int(mask.sum())
        for branch in BRANCHES:
            row = test_rows[branch][record_id]
            assert (row['subject'], row['stage']) == (meta['subject'], meta['stage'])
            path = root / 'runs' / branch / 'test_predictions' / f'{record_id}.npz'
            with np.load(path) as output:
                for name, array in expected.items():
                    assert np.array_equal(output[name], array), (branch, record_id, name)
                pred = output['pred']
                assert pred.shape == mask.shape and np.isfinite(pred).all(), (branch, record_id)
                for target in TARGETS:
                    assert row['metrics'][target] == measures(pred, expected[target], mask), (branch, record_id, target)
                bsq = expected['bsq']
                for finger in range(10):
                    assert row['by_finger'][str(finger)] == measures(
                        pred, bsq, mask & (expected['finger'] == finger)), (branch, record_id, finger)
                for band, lo, hi in [('0-2', 0, 2), ('2-10', 2, 10), ('10-25', 10, 25), ('25+', 25, math.inf)]:
                    assert row['BSQ_force_bands'][band] == measures(
                        pred, bsq, mask & (bsq >= lo) & (bsq < hi)), (branch, record_id, band)
            target = 'cal' if branch == 'calibrated_huber' else 'raw'
            metric = 'original_loss' if branch == 'raw_original' else 'huber_N2'
            assert row['loss'] == row['metrics'][target][metric]
    return test_rows, dict(recordings=len(entries), points=sum(points.values()), points_by_stage=dict(points))


def check_report(root, test_rows, scoring):
    folder = root / 'results'
    report = load(folder / 'FINAL_REPORT.json')
    assert report['status'] == 'COMPLETE' and report['point_and_ground_truth_identity_checked']
    assert report['common_test_recordings'] == scoring['recordings']
    experiments = report['experiments']
    assert [(row['experiment'], row['training_branch'], row['test_label']) for row in experiments] == list(EXPERIMENTS)
    for row in experiments:
        grouped = collections.defaultdict(list)
        records = test_rows[row['training_branch']]
        for record in records.values():
            grouped[record['subject']].append(record['metrics'][row['test_label']])
        assert row['recordings'] == len(records) and row['subjects'] == len(grouped)
        for metric in ('mae_N', 'mape_percent', 'corr', 'huber_N2'):
            values = [[item[metric] for item in group if item[metric] is not None] for group in grouped.values()]
            subjects = [np.mean(group) for group in values if group]
            trials = [value for group in values for value in group]
            assert row['valid_subjects_' + metric] == len(subjects)
            for weight, items in [('subject', subjects), ('trial', trials)]:
                assert math.isclose(row[f'{weight}_equal_{metric}'], np.mean(items), rel_tol=1e-12)
    names = ('FINAL_REPORT.json', 'four_experiments.csv', 'by_stage.csv', 'by_finger.csv',
             'by_force_band.csv', 'paired_person_differences.csv', 'calibration_quality_summary.json',
             'REPORT.txt', 'BSQ_MAE_by_stage.png', 'report.html')
    return {name: sha(folder / name) for name in names}


def main(root, val_only):
    root = Path(root)
    data_path = root / 'prepared/dataset.json'
    data = load(data_path)
    dataset_sha = sha(data_path)
    assert data['source_code'] == source_binding() and data['plan_sha256'] == sha(root / 'plan.json')
    split_file = root / 'prepared/subject_split.json'
    split = load(split_file)
    assert sha(split_file) == data['subject_split_sha256']
    assert sha(root / 'prepared/recording_split.csv') == split['recording_split_sha256']
    source_rows = {row['id']: row for row in load(root / 'plan.json')['rows']}
    record_splits, people_splits, calibration_splits = {}, {}, {}
    for entry in data['entries']:
        person, phase = entry['subject'], entry['split']
        record_splits.setdefault(entry['id'], set()).add(phase)
        people_splits.setdefault(person, set()).add(phase)
        calibration_splits.setdefault(entry['source_id'], set()).add(phase)
        assert person == source_rows[entry['source_id']]['subject']
        assert person in split['subjects'][phase]
    assert all(len(phases) == 1 for groups in (record_splits, people_splits, calibration_splits)
               for phases in groups.values())
    assert all(sha(root / 'prepared' / f'{phase}_windows.npy') == digest
               for phase, digest in data['window_indices_sha256'].items())
    histories = {}
    for branch in BRANCHES:
        run = root / 'runs' / branch
        if val_only:
            histories[branch] = [(int(path.stem[5:8]), load(path)['loss']) for path in sorted(run.glob('epoch*_val.json'))]
        else:
            histories[branch] = [(row['epoch'], row['val_loss']) for row in load(run / 'DONE.json')['history']]
        assert histories[branch]
    result = dict(dataset_sha256=dataset_sha, subject_split_sha256=sha(split_file),
                  people_disjoint=True, record_and_calibration_sources_disjoint=True,
                  validation=check_val(root, data, histories))
    if val_only:
        result['status'] = 'PASS_VAL_SCORING_INPUTS'
        destination = root / 'checks/val_scoring_input_audit.json'
    else:
        assert load(root / 'STATUS.json')['phase'] in ('report_ready', 'complete')
        config = settings()
        import torch
        torch.set_num_threads(2)
        selected = {branch: check_selection(root, branch, data, config, dataset_sha) for branch in BRANCHES}
        test_rows, scoring = check_test(root, data, selected)
        result.update(status='PASS_FINAL_RUN', models=selected, test=scoring,
                      report_sha256=check_report(root, test_rows, scoring))
        destination = root / 'checks/final_run_audit.json'
    atomic(destination, result)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--val-only', action='store_true')
    args = parser.parse_args()
    main(args.root, args.val_only)
