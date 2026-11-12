"""Verify and combine the finished native-force reports for publication."""
import argparse
from collections import defaultdict
import csv
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import atomic, load, sha
from score_force_thresholds import METRICS


COUNT_FIELDS = ('points', 'points_below_1N', 'mape_points', 'points_ge_1N')
NAMES = {
    'k2_raw_original': 'K2 5.56M · 原损失',
    'ap304_raw_huber': 'AP304 二阶段 · Huber δ=1 N',
    'ap304_raw_original': 'AP304 二阶段 · 原损失',
}


def report(root, branch, folder, status_path):
    run = root / 'runs' / branch
    done = load(run / 'DONE.json')
    assert done['phase'] == 'complete' and done['test_not_used_for_selection']
    assert done['best_sha256'] == sha(run / 'best.pt')
    assert done['binding']['dataset_sha256'] == sha(root / 'prepared/dataset.json')
    path = folder / 'all_stage_metrics.json'
    status = load(status_path)
    assert status['phase'] == 'complete' and status['report_sha256'] == sha(path)
    summary = load(path)
    epoch = min(done['history'], key=lambda row: row['val_loss'])['epoch']
    assert epoch == status['selected_epoch'] == summary['checkpoint_epoch']
    assert done['best_sha256'] == summary['selected_checkpoint_sha256']
    assert summary['target'] == 'native_CMCU_N'
    coverage = {}
    for split, file in [('val', f'epoch{epoch:03d}_val.json'), ('test', 'final_test.json')]:
        result = load(run / file)
        coverage[split] = {
            row['id']: (row['subject'], row['stage'], *(row['metrics']['raw'][key] for key in COUNT_FIELDS))
            for row in result['rows']
        }
        assert len(coverage[split]) == result['records'] == len(result['rows'])
    metadata = dict(selected_epoch=epoch, checkpoint_sha256=done['best_sha256'],
                    report_sha256=sha(path), runtime_dataset_sha256=done['binding']['dataset_sha256'],
                    history=done['history'])
    return summary, coverage, metadata


def verify_dataset(k2, ap304, data):
    split_rows = defaultdict(dict)
    labels = {}
    for entry in data['entries']:
        record_id = entry['id']
        split_rows[entry['split']][record_id] = (entry['subject'], entry['stage'])
        label = entry['labels_sha256']
        assert labels.setdefault(record_id, label) == label
    sets = [{subject for subject, _ in split.values()} for split in split_rows.values()]
    assert sum(map(len, sets)) == len(set.union(*sets))
    split_path = k2 / 'prepared/subject_split.json'
    assert sha(split_path) == data['subject_split_sha256']
    split_info = load(split_path)
    for split, file_sha in data['window_indices_sha256'].items():
        assert sha(k2 / f'prepared/{split}_windows.npy') == file_sha
        counts = data['split_recordings'][split]
        assert len(counts['by_stage']) == 20
        assert counts['recordings'] == len(split_rows[split])
    hasher = hashlib.sha256()
    for rid in sorted(split_rows['val'].keys() | split_rows['test'].keys()):
        assert sha(k2 / 'cache/labels' / f'{rid}.npz') == labels[rid], rid
        hasher.update(f'{rid} {labels[rid]}\n'.encode())
    if ap304:
        ap_data = load(ap304 / 'prepared/dataset.json')
        assert ap_data['shared_k2_dataset_sha256'] == sha(k2 / 'prepared/dataset.json')
        for key in ('entries', 'force_norm', 'target', 'window_indices_sha256', 'subject_split_sha256',
                    'split_recordings'):
            assert data[key] == ap_data[key], key
        for folder in ('emg', 'labels'):
            assert (k2 / 'cache' / folder).samefile(ap304 / 'cache' / folder)
        for split in data['window_indices_sha256']:
            path = f'prepared/{split}_windows.npy'
            assert (k2 / path).samefile(ap304 / path)
    return split_rows, dict(
        dataset_sha256=sha(k2 / 'prepared/dataset.json'), subject_split_sha256=data['subject_split_sha256'],
        window_indices_sha256=data['window_indices_sha256'], val_test_label_set_sha256=hasher.hexdigest(),
        recording_split_sha256=split_info['recording_split_sha256'],
        participant_disjoint=True, ap304_shared_labels_emg_windows=bool(ap304))


def markdown(path, table, counts):
    lines = [
        '# Task103 全 stage 跨人评测',
        '',
        'Train、Val、Test 仅按人划分，各覆盖 20 个 stage。下表以原始 CMCU force 为参考。',
        '指标先逐条 recording 计算，再对同人取平均，最后按人等权。力段阈值以参考力绝对值判断。',
        '',
        '| 集合 | 人数 | Recording | 窗口 |',
        '|---|---:|---:|---:|',
    ]
    for split in ('train', 'val', 'test'):
        c = counts[split]
        lines.append(f"| {split.title()} | {c['people']} | {c['recordings']:,} | {c['windows']:,} |")
    lines.extend([
        '',
        '| 模型 | 选中轮次 | 集合 | Corr | MAE (N) | MAE <1 N (N) | MAPE ≥0.5 N | MAPE ≥1 N |',
        '|---|---:|---|---:|---:|---:|---:|---:|',
    ])
    for row in table:
        if row['stage'] != 'all':
            continue
        values = ['—' if row[k] is None else f'{row[k]:.2f}%' if k.startswith('mape') else f'{row[k]:.3f}'
                  for k in METRICS]
        lines.append('| ' + ' | '.join([NAMES[row['model']], str(row['selected_epoch']),
                     row['split'].title(), *values]) + ' |')
    lines.extend(['', '[总体和逐 stage 完整指标](metrics.csv) · [数据和计分点核验](audit.json)', ''])
    path.write_text('\n'.join(lines), encoding='utf-8')


def main(k2, ap304, output):
    data = load(k2 / 'prepared/dataset.json')
    expected, audit = verify_dataset(k2, ap304, data)
    specs = [('k2_raw_original', k2, 'raw_original', k2 / 'results', k2 / 'STATUS.json')]
    if ap304:
        assert load(ap304 / 'STATUS.json')['phase'] == 'complete'
        for branch in ('raw_huber', 'raw_original'):
            folder = ap304 / 'results' / branch
            specs.append((f'ap304_{branch}', ap304, branch, folder, folder / 'REPORT_COMPLETE.json'))
    ref = None
    combined = []
    training = {}
    for name, root, branch, folder, status in specs:
        summary, counts, metadata = report(root, branch, folder, status)
        for split, recordings in counts.items():
            assert {rid: row[:2] for rid, row in recordings.items()} == expected[split]
        if ref is None:
            ref = counts
        assert counts == ref, name
        assert summary['coverage'] == data['split_recordings']
        assert len(summary['metrics']) == 42
        for row in summary['metrics']:
            combined.append(dict(model=name, selected_epoch=metadata['selected_epoch'], **row))
        training[name] = metadata
    audit.update(per_recording_scoring_counts_identical=True, models=training)
    output.mkdir(parents=True, exist_ok=True)
    atomic(output / 'audit.json', audit)
    atomic(output / 'metrics.json', dict(target=data['target'], coverage=data['split_recordings'],
                                         metrics=combined))
    with (output / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(combined[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(combined)
    markdown(output / 'README.md', combined, data['split_recordings'])
    print((output / 'README.md').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--k2-root', type=Path, required=True)
    parser.add_argument('--ap304-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.k2_root, args.ap304_root, args.output)
