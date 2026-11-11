"""Audit a completed Val that selected a new best, without running inference."""
import argparse
import json
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import BRANCHES, sha


def equal(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b



def audit(root, branch, epoch):
    root = Path(root)
    sys.path.insert(0, str(root / 'code/scripts'))
    from common import source_binding
    dataset_path = root / 'prepared/dataset.json'
    dataset = json.loads(dataset_path.read_text())
    dataset_sha = sha(dataset_path)
    expected = {e['id'] for e in dataset['entries'] if e['split'] == 'val'}
    directory = root / 'runs' / branch
    val = json.loads((directory / f'epoch{epoch:03d}_val.json').read_text())
    rows = {row['id']: row for row in val['rows']}
    assert len(rows) == len(val['rows']) == val['records'] == len(expected)
    assert set(rows) == expected
    assert math.isfinite(val['loss'])
    assert abs(sum(v['loss'] for v in rows.values()) / len(rows) - val['loss']) < 1e-10
    for name in BRANCHES:
        reference = json.loads((root / 'runs' / name / 'epoch001_val.json').read_text())
        reference_rows = {row['id']: row for row in reference['rows']}
        assert reference_rows.keys() == rows.keys()
        for record_id, row in rows.items():
            for target in ('raw', 'cal', 'bsq'):
                for key in ('points', 'mape_points'):
                    assert row['metrics'][target][key] == reference_rows[record_id]['metrics'][target][key]

    contract = json.loads((directory / 'CONTRACT.json').read_text())
    updates = math.ceil(dataset['windows']['train'] / contract['configuration']['global_batch'])
    candidate = root / 'archive/epoch_candidates' / branch / dataset_sha[:16] / f'epoch{epoch:03d}.pt'
    best = directory / 'best.pt'
    before = torch.load(candidate, map_location='cpu', weights_only=True)
    after = torch.load(best, map_location='cpu', weights_only=True)
    assert before['state']['epoch'] == epoch - 1
    assert before['state']['next_update'] == updates
    assert after['state']['epoch'] == epoch and after['state']['next_update'] == 0
    assert after['state']['global_update'] == before['state']['global_update'] == epoch * updates
    history = after['state']['history']
    assert before['state']['history'] == history[:-1] and len(history) == epoch
    assert history[-1]['val_loss'] == val['loss'] == after['state']['best']
    assert val['loss'] == min(h['val_loss'] for h in history)
    assert equal(before['binding'], after['binding'])
    assert after['binding'] == {k: v for k, v in contract.items() if k != 'configuration'}
    assert after['binding']['dataset_sha256'] == dataset_sha
    assert after['binding']['config_sha256'] == sha(root / 'code/config.json')
    assert after['binding']['source_code'] == source_binding()
    for component in ('model', 'optimizer', 'torch_rng', 'cuda_rng'):
        assert equal(before[component], after[component]), component
    assert all(torch.isfinite(t).all().item() for t in after['model'].values())
    assert all(int(v['step']) == epoch * updates for v in after['optimizer']['state'].values())
    result = dict(status='PASS_REAL_COMPLETE_VAL_NEW_BEST_AND_STATE_PRESERVATION', branch=branch,
                  epoch=epoch, global_update=epoch * updates, Val_recordings=len(rows),
                  prepared_Val_record_set_exactly_covered=True, trial_equal_validation_loss=val['loss'],
                  objective='(1-Corr)+0.3*MAPE+0.03*MAE' if branch == 'raw_original' else 'Huber_delta_1N',
                  selected_best_epoch=after['state']['epoch'], selected_best_sha256=sha(best),
                  pre_Val_candidate_sha256=sha(candidate),
                  model_optimizer_CPU_CUDA_RNG_unchanged_by_full_Val=True,
                  all_three_first_Val_record_sets_and_target_point_counts_identical=True,
                  Adam_steps_match_global_update=True, all_model_tensors_finite=True,
                  validation_history=history, bad_epochs=after['state']['bad_epochs'],
                  dataset_sha256=dataset_sha, test_metrics_read=False,
                  scoring_source_policy='self_test5_nearest_test5_for_other_stages_v1')
    output = root / 'checks' / f'{branch}_epoch{epoch:03d}_Val_and_state_audit.json'
    temporary = output.with_suffix('.partial.json')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(output)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--branch', choices=BRANCHES, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    args = parser.parse_args()
    audit(args.root, args.branch, args.epoch)
