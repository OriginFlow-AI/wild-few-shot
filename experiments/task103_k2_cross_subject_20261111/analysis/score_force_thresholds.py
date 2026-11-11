"""Score the four experiments on the same Test windows, with force-range metrics."""
import argparse
import collections
import csv
from pathlib import Path

import numpy as np

from audit_final_run import EXPERIMENTS, rows_for, scoring_inputs
from common import atomic, load, sha
from train import measures


METRICS = ('corr', 'mae_N', 'mae_below_1N', 'mape_ge_0p5N_percent', 'mape_ge_1N_percent')
LABELS_ZH = {'calibrated CMCU': '标定 CMCU', 'device CMCU force': '原始 CMCU force', 'BSQ': 'BSQ'}


def score(prediction, target, mask):
    base = measures(prediction, target, mask)
    force = target[mask].astype(np.float64)
    error = abs(prediction[mask].astype(np.float64) - force)
    low = abs(force) < 1
    above_half = abs(force) >= .5
    above_one = abs(force) >= 1
    values = dict(corr=base['corr'], mae_N=base['mae_N'],
                  mae_below_1N=float(error[low].mean()) if low.any() else None,
                  mape_ge_0p5N_percent=base['mape_percent'],
                  mape_ge_1N_percent=float(100 * np.mean(error[above_one] / abs(force[above_one])))
                  if above_one.any() else None)
    counts = dict(all=len(force), below_1N=int(low.sum()),
                  ge_0p5N=int(above_half.sum()), ge_1N=int(above_one.sum()))
    return base, values, counts


def model_receipt(root, branch, interim):
    run = root / 'runs' / branch
    if branch == 'raw_original' and interim:
        run = interim / 'runs' / branch
        done = load(interim / 'STATUS.json')
        assert done['phase'] == 'complete' and done['branch'] == branch
        return run, done['checkpoint_epoch'], done['selected_checkpoint_sha256'], done['test_recordings']
    done = load(run / 'DONE.json')
    assert done['phase'] == 'complete' and done['branch'] == branch
    assert done['best_sha256'] == sha(run / 'best.pt')
    epoch = min(done['history'], key=lambda row: row['val_loss'])['epoch']
    return run, epoch, done['best_sha256'], done['test_recordings']


def write_table(path, rows, data, interim):
    counts = {split: len({e['id'] for e in data['entries'] if e['split'] == split})
              for split in ('train', 'val', 'test')}
    lines = [
        '# 截至第 9 轮的 Test 对比' if interim else '# 最终 Test 对比',
        '',
        f"共同划分为普通 stage Train（{counts['train']:,} 条）、留出 Val 人员的 test5（{counts['val']:,} 条）"
        f"和留出 Test 人员的 test5 加 test7（{counts['test']:,} 条）。",
        '',
        '| 实验 | Train 标签 | Val 标签 | Test 真值 | 选中轮次 | Corr | MAE (N) | MAE <1 N (N) | MAPE ≥0.5 N | MAPE ≥1 N |',
        '|---|---|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        labels = [LABELS_ZH[row[f'{split}_label']] for split in ('train', 'val', 'test')]
        values = []
        for metric in METRICS:
            value = row[metric]
            values.append('—' if value is None else f'{value:.2f}%' if metric.startswith('mape') else f'{value:.3f}')
        cells = [str(row['experiment']), *labels, str(row['checkpoint_epoch']), *values]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.extend([
        '',
        '阈值按 Test 真值的绝对值筛点。指标先逐条记录计算，再对每人的有效记录取平均，最后按人等权。'
        'Corr 为 Pearson 相关。空子集和未定义的相关系数不参与相应指标的平均。',
        '',
        '实验 1 和 2 共享模型。实验 2、3、4 的 BSQ 真值与有效掩码逐点一致。'
        'test5 的 BSQ 参与离线标定，该 stage 的结果需要按这一口径解读。',
        '',
    ])
    path.write_text('\n'.join(lines), encoding='utf-8')


def main(root, output, interim):
    data_path = root / 'prepared/dataset.json'
    data = load(data_path)
    expected = {entry['id']: entry for entry in data['entries'] if entry['split'] == 'test'}
    runs = {branch: model_receipt(root, branch, interim) for _, branch, _ in EXPERIMENTS}
    rows = {}
    for branch, (run, _, _, recordings) in runs.items():
        assert recordings == len(expected)
        assert {p.stem for p in (run / 'test_predictions').glob('*.npz')} == expected.keys()
        rows[branch] = rows_for(load(run / 'final_test.json'), expected)

    per_person = collections.defaultdict(lambda: collections.defaultdict(list))
    point_counts = collections.defaultdict(collections.Counter)
    for record_id, meta, arrays in scoring_inputs(root, data, 'test'):
        for branch, (run, _, _, _) in runs.items():
            path = run / 'test_predictions' / f'{record_id}.npz'
            with np.load(path) as values:
                for name, reference in arrays.items():
                    assert np.array_equal(values[name], reference), (branch, record_id, name)
                prediction = values['pred']
            assert prediction.shape == arrays['mask'].shape and np.isfinite(prediction).all()
            for experiment, current, target in EXPERIMENTS:
                if current != branch:
                    continue
                base, metrics, counts = score(prediction, arrays[target], arrays['mask'])
                assert base == rows[branch][record_id]['metrics'][target], (branch, record_id, target)
                per_person[experiment][meta['subject']].append(metrics)
                point_counts[experiment].update(counts)

    result = []
    for experiment, branch, target in EXPERIMENTS:
        _, epoch, digest, _ = runs[branch]
        train_label = 'calibrated CMCU' if branch == 'calibrated_huber' else 'device CMCU force'
        row = dict(experiment=experiment, branch=branch, checkpoint_epoch=epoch, train_label=train_label,
                   val_label=train_label, test_label='BSQ' if target == 'bsq' else 'calibrated CMCU')
        groups = per_person[experiment]
        row.update(recordings=len(expected), subjects=len(groups), **{f'points_{key}': count
                   for key, count in point_counts[experiment].items()})
        for metric in METRICS:
            subjects = []
            for recording_metrics in groups.values():
                values = [r[metric] for r in recording_metrics if r[metric] is not None]
                if values:
                    subjects.append(np.mean(values))
            row[metric] = float(np.mean(subjects)) if subjects else None
            row[f'subjects_{metric}'] = len(subjects)
        row['checkpoint_sha256'] = digest
        result.append(row)
    assert all(result[1][key] == row[key] for row in result[2:] for key in result[1] if key.startswith('points_'))

    output.mkdir(parents=True, exist_ok=True)
    atomic(output / 'force_threshold_metrics.json',
           dict(status='INTERIM_THROUGH_EPOCH_9' if interim else 'FINAL', dataset_sha256=sha(data_path),
                aggregation='Mean per recording, then equally weight recordings within each subject and equally weight subjects. '
                'Undefined Pearson correlations and empty force ranges are skipped for their metric.',
                force_range='Thresholds apply to absolute Test ground truth in N.',
                common_test_recordings=len(expected), experiments=result))
    with (output / 'force_threshold_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(result)
    write_table(output / 'force_threshold_metrics.md', result, data, interim)
    for row in result:
        print(row['experiment'], row['checkpoint_epoch'], *(row[key] for key in METRICS))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--interim', type=Path, help='Run directory holding a separately evaluated epoch 9 for raw_original.')
    args = parser.parse_args()
    main(args.root, args.output, args.interim)
