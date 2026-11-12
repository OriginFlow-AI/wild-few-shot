"""Bind AP304 stage2 to the frozen K2 all-stage recording and window indices."""
import argparse
from pathlib import Path

import numpy as np
import torch

from common import EXPERIMENT, atomic, load, settings, sha, source_binding
from train import network


def link(source, target):
    source = source.resolve(strict=True)
    if target.exists() or target.is_symlink():
        if target.resolve(strict=True) != source:
            raise ValueError(f'Unexpected existing path: {target}')
    else:
        target.symlink_to(source, target_is_directory=source.is_dir())


def bind_initialization(root, previous_ap304):
    config = settings()
    inputs = {
        'common_initial_sha256': previous_ap304 / 'assets/common_stage2_initial_state_dict.pt',
        'stage1_export_sha256': previous_ap304 / 'assets/ap304_rank_best_model_state_dict.pt',
        'emg_norm_sha256': previous_ap304 / 'reference_stage2/stage1_emg_train_norm.npz',
        'model_source_sha256': EXPERIMENT / 'vendor/model_source.py',
    }
    for key, file in inputs.items():
        assert sha(file) == config[key], key
    initial = inputs['common_initial_sha256']
    norm = inputs['emg_norm_sha256']
    link(initial, root / 'assets/common_stage2_initial_state_dict.pt')
    link(norm, root / 'prepared/emg_train_norm.npz')
    torch.set_num_threads(2)
    exported = torch.load(inputs['stage1_export_sha256'], weights_only=True, map_location='cpu')
    prepared = torch.load(initial, weights_only=True, map_location='cpu')
    backbone = [key for key in prepared if not key.startswith('output_head.')]
    assert len(backbone) == 243 and set(exported) == set(prepared)
    assert all(torch.equal(prepared[key], exported[key]) for key in backbone)
    network().load_state_dict(prepared, strict=True)
    with np.load(norm) as n:
        assert np.isfinite(n['emg_mean']).all() and np.all(np.isfinite(n['emg_std']) & (n['emg_std'] > 0))
        emg = {key: n[f'emg_{key}'].tolist() for key in ('mean', 'std')}
    return emg, len(backbone)


def main(root, dataset_root, previous_ap304):
    root, dataset_root, previous_ap304 = (path.resolve() for path in (root, dataset_root, previous_ap304))
    source = dataset_root / 'prepared/dataset.json'
    data = load(source)
    config = settings()
    assert data['target'] == config['target'] == 'native_CMCU_N'
    assert load(dataset_root / 'prepared/subject_split.json')['all_stages_per_split']
    for folder in ('prepared', 'assets', 'checks', 'cache'):
        (root / folder).mkdir(parents=True, exist_ok=True)
    for name in ('emg', 'labels'):
        link(dataset_root / 'cache' / name, root / 'cache' / name)
    manifest = dataset_root / 'prepared/subject_split.json'
    assert sha(manifest) == data['subject_split_sha256']
    assert sha(dataset_root / 'prepared/recording_split.csv') == load(manifest)['recording_split_sha256']
    for name, digest in data['window_indices_sha256'].items():
        assert sha(dataset_root / f'prepared/{name}_windows.npy') == digest
        link(dataset_root / f'prepared/{name}_windows.npy', root / f'prepared/{name}_windows.npy')
    for name in ('subject_split.json', 'recording_split.csv', 'exclusions.json'):
        link(dataset_root / 'prepared' / name, root / 'prepared' / name)
    emg, backbone_count = bind_initialization(root, previous_ap304)
    receipt = dict(k2_dataset_sha256=sha(source), subject_split_sha256=data['subject_split_sha256'],
                   window_indices_sha256=data['window_indices_sha256'],
                   recording_split_sha256=sha(dataset_root / 'prepared/recording_split.csv'),
                   raw_force_norm=data['force_norm']['raw'], target=data['target'],
                   backbone_tensors_verified=backbone_count,
                   initialization_sha256=config['common_initial_sha256'],
                   stage1_export_sha256=config['stage1_export_sha256'],
                   emg_norm_sha256=config['emg_norm_sha256'])
    atomic(root / 'checks/shared_dataset_and_initialization.json', receipt)
    output = root / 'prepared/dataset.json'
    expected = dict(
        data, emg_norm=emg, emg_norm_sha256=config['emg_norm_sha256'],
        common_initial_sha256=config['common_initial_sha256'],
        config_sha256=sha(EXPERIMENT / 'config.json'), source_code=source_binding(),
        shared_k2_dataset_sha256=receipt['k2_dataset_sha256'])
    if output.is_file() and load(output) != expected:
        raise ValueError('Existing AP304 dataset binding differs')
    if not output.is_file():
        atomic(output, expected)
    print(receipt, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--previous-ap304', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.dataset_root, args.previous_ap304)
