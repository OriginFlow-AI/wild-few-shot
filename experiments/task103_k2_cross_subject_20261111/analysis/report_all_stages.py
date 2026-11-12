"""Audit the selected native-force model and report held-out metrics by stage."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_final_run import rows_for
from common import atomic, load, sha
from score_force_thresholds import METRICS
from train import network


def summarize(rows, split):
    scopes = defaultdict(lambda: defaultdict(list))
    for row in rows.values():
        for stage in ('all', row['stage']):
            scopes[stage][row['subject']].append(row['metrics']['raw'])
    table = []
    for stage, subjects in sorted(scopes.items()):
        metrics = [m for person in subjects.values() for m in person]
        aggregate = dict(split=split, stage=stage, subjects=len(subjects), recordings=len(metrics))
        for key in ('points', 'points_below_1N', 'mape_points', 'points_ge_1N'):
            aggregate[key] = sum(m[key] for m in metrics)
        for key in METRICS:
            means = [np.mean(values) for ms in subjects.values()
                     if (values := [m[key] for m in ms if m[key] is not None])]
            aggregate[key] = float(np.mean(means)) if means else None
        table.append(aggregate)
    return table


def main(root, branch='raw_original', output=None):
    data_path = root / 'prepared/dataset.json'
    data = load(data_path)
    assert data['target'] == 'native_CMCU_N'
    run = root / 'runs' / branch
    done = load(run / 'DONE.json')
    assert done['phase'] == 'complete' and done['test_not_used_for_selection']
    assert done['binding']['dataset_sha256'] == sha(data_path)
    best = run / 'best.pt'
    assert done['best_sha256'] == sha(best)
    selected = min(done['history'], key=lambda row: row['val_loss'])
    epoch = selected['epoch']
    state = torch.load(best, map_location='cpu', weights_only=True)
    assert state['binding'] == done['binding'] and state['state']['epoch'] == epoch
    assert all(torch.isfinite(v).all() for v in state['model'].values())
    net = network()
    net.load_state_dict(state['model'], strict=True)
    tables = []
    for split, path in [('val', run / f'epoch{epoch:03d}_val.json'), ('test', run / 'final_test.json')]:
        expected = {e['id']: e for e in data['entries'] if e['split'] == split}
        receipt = load(path)
        rows = rows_for(receipt, expected)
        assert all(row['stage'] == expected[rid]['stage'] and row['subject'] == expected[rid]['subject']
                   for rid, row in rows.items())
        if split == 'val':
            assert receipt['loss'] == selected['val_loss']
        table = summarize(rows, split)
        by_stage = {row['stage']: row['recordings'] for row in table if row['stage'] != 'all'}
        assert by_stage == data['split_recordings'][split]['by_stage']
        tables.extend(table)
    report_root = output or root / 'results'
    report_root.mkdir(parents=True, exist_ok=True)
    with (report_root / 'all_stage_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(tables[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(tables)
    atomic(report_root / 'all_stage_metrics.json', dict(branch=branch, target=data['target'], checkpoint_epoch=epoch,
           selected_checkpoint_sha256=done['best_sha256'], history=done['history'],
           dataset_sha256=done['binding']['dataset_sha256'],
           shared_k2_dataset_sha256=data.get('shared_k2_dataset_sha256'),
           aggregation='Per recording, then mean per person, then equal weight across people.',
           coverage=data['split_recordings'], metrics=tables))
    export = root / 'exported_models' / branch
    export.mkdir(parents=True, exist_ok=True)
    weights = export / 'best_model_state_dict.pt'
    torch.save(state['model'], weights)
    emg_norm = root / 'prepared/emg_train_norm.npz'
    assert sha(emg_norm) == data['emg_norm_sha256']
    shutil.copy2(emg_norm, export / emg_norm.name)
    atomic(export / 'inference_parameters.json', dict(selected_epoch=epoch,
           selected_training_checkpoint_sha256=done['best_sha256'], model_state_sha256=sha(weights),
           emg_norm_sha256=sha(emg_norm), force_norm=state['binding']['force_norm']))
    status = report_root / 'REPORT_COMPLETE.json' if output else root / 'STATUS.json'
    atomic(status, dict(phase='complete', branch=branch, selected_epoch=epoch,
           report_sha256=sha(report_root / 'all_stage_metrics.json'), model_state_sha256=sha(weights)))
    print([r for r in tables if r['stage'] == 'all'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--branch', choices=('raw_original', 'raw_huber'), default='raw_original')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    main(args.root, args.branch, args.output)
