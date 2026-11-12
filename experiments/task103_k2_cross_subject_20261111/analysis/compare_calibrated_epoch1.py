"""Compare epoch-one validation recordings and rerun available checkpoints on common points."""
import argparse
from collections import defaultdict
import csv
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import atomic, interpolate, load, sha
from report_all_stages import summarize
from score_force_thresholds import score as threshold_score


def epoch_rows(root, branch):
    result = load(root / 'runs' / branch / 'epoch001_val.json')
    return {row['id']: row for row in result['rows']}


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def model(root, checkpoint, name, device):
    spec = importlib.util.spec_from_file_location(name, root / 'code/vendor/model_source.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert state['binding']['dataset_sha256'] == sha(root / 'prepared/dataset.json')
    assert state['state']['epoch']==1 or (
        state['state']['epoch']==0 and state['state']['global_update'] == state['state']['next_update']
        == int(np.ceil(load(root/'prepared/dataset.json')['windows']['train']/256)))
    net = module.K2RingNetwork(module.K2RingConfig()).eval()
    net.load_state_dict(state['model'], strict=True)
    norm_path = root / 'prepared/emg_train_norm.npz'
    assert state['binding']['EMG_norm_sha256'] == sha(norm_path)
    with np.load(norm_path) as n:
        emg = tuple(np.asarray(n[f'emg_{k}'], np.float32) for k in ('mean', 'std'))
    force = tuple(torch.tensor(state['binding']['force_norm'][key], device=device,
                               dtype=torch.float32)[None,:,None] for key in ('mean','std'))
    return net.to(device), emg, force


def grid(start_ns, starts):
    offsets = (starts[:,None]+470+40*np.arange(89)).ravel()
    keep = np.r_[True, offsets[1:] > np.maximum.accumulate(offsets)[:-1]]
    return start_ns+offsets[keep]*500_000, keep


def val_entries(root):
    data = load(root / 'prepared/dataset.json')
    starts = defaultdict(list)
    for i, start in np.load(root/'prepared/val_windows.npy', mmap_mode='r'):
        starts[int(i)].append(int(start))
    return {(row['id'],row['side']):(row,np.asarray(starts[i])) for i,row in enumerate(data['entries'])
            if row['split']=='val'}


def score(prediction, target):
    _, metrics, counts = threshold_score(prediction, target, np.ones(len(target), dtype=bool))
    return dict(points=counts['all'],points_below_1N=counts['below_1N'],
                mape_points=counts['ge_0p5N'],points_ge_1N=counts['ge_1N'],**metrics)


@torch.no_grad()
def predict(spec, signal, starts, device):
    net, (em,es), (mean,std) = spec
    pieces = []
    for part in np.array_split(starts,(len(starts)+63)//64):
        batch = np.stack([((signal[s:s+4000]-em)/es).T for s in part])
        with torch.autocast('cuda',dtype=torch.bfloat16):
            result = net(torch.from_numpy(batch).to(device))
        pieces.append((result.float()*std+mean).transpose(1,2).flatten(0,1).cpu().numpy())
    return np.concatenate(pieces)


def rerun(cal_root, raw_root, models, device):
    raw_entries, cal_entries = val_entries(raw_root), val_entries(cal_root)
    by_record = defaultdict(list)
    for k, values in cal_entries.items():
        by_record[k[0]].append((k,values))
    rows = {name: {} for name in models}
    totals = defaultdict(lambda: np.zeros(6, dtype=np.int64))
    edges = [0,.5,1,2,5,10,np.inf]
    for number,(rid,hands) in enumerate(sorted(by_record.items()),1):
        cal_file = cal_root/'cache/labels'/f'{rid}.npz'
        raw_file = raw_root/'cache/labels'/f'{rid}.npz'
        predictions, targets = defaultdict(list), defaultdict(list)
        with np.load(raw_file) as raw, np.load(cal_file) as cal:
            for key,(entry,starts) in hands:
                side = key[1]
                t,keep = grid(entry['start_ns'], starts)
                raw_t,_ = grid(entry['start_ns'],raw_entries[key][1])
                y,cal_valid = interpolate(cal[f'{side}_t'],cal[f'{side}_cal'],cal[f'{side}_valid'],t)
                x,raw_valid = interpolate(raw[f'{side}_t'],raw[f'{side}_raw'],raw[f'{side}_valid'],t)
                valid = cal_valid & raw_valid & np.isin(t,raw_t)[:,None]
                signal = np.load(raw_root/'cache/emg'/rid/f'{side}.npy',mmap_mode='r')
                for name,spec in models.items():
                    predictions[name].append(predict(spec,signal,starts,device)[keep][valid])
                targets['raw'].append(x[valid])
                targets['cal'].append(y[valid])
        for kind,parts in targets.items():
            y = np.concatenate(parts).astype(np.float64)
            totals[kind] += np.histogram(abs(y),bins=edges)[0]
            for name,parts in predictions.items():
                row = rows[name].setdefault(rid,dict(id=rid,subject=entry['subject'],stage=entry['stage'],metrics={}))
                row['metrics'][kind] = score(np.concatenate(parts).astype(np.float64),y)
        if number%100==0:
            print(f'Epoch-one comparison {number}/{len(by_record)}',flush=True)
    return rows, totals


def main(cal_root, k2_root, ap_root, output):
    output.mkdir(parents=True,exist_ok=True)
    reported = {
        'cal_ap304_original': (epoch_rows(cal_root,'calibrated_original'),'cal'),
        'raw_ap304_original': (epoch_rows(ap_root,'raw_original'),'raw'),
        'raw_ap304_huber': (epoch_rows(ap_root,'raw_huber'),'raw'),
        'raw_k2_original': (epoch_rows(k2_root,'raw_original'),'raw'),
    }
    common = set.intersection(*(set(rows) for rows,_ in reported.values()))
    assert common == reported['cal_ap304_original'][0].keys()
    first = []
    for name,(rows,kind) in reported.items():
        matched = {rid:rows[rid] for rid in sorted(common)}
        first.extend(dict(model=name, reference=kind, **row) for row in summarize(matched,'val',kind))
    write_csv(output/'reported_epoch001_val_common_recordings.csv',first)

    torch.set_num_threads(2)
    device=torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(10*2**30/torch.cuda.get_device_properties(0).total_memory)
    (k2_epoch1,) = (k2_root/'archive/epoch_candidates/raw_original').glob('*/epoch001.pt')
    paths = {
        'cal_ap304_original': (cal_root,output/'calibrated_epoch001.pt'),
        'raw_ap304_huber': (ap_root,ap_root/'runs/raw_huber/best.pt'),
        'raw_k2_original': (k2_root,k2_epoch1),
    }
    models = {name:model(root,ckpt,name,device) for name,(root,ckpt) in paths.items()}
    rows,totals = rerun(cal_root,k2_root,models,device)
    matched = []
    for name,by_record in rows.items():
        assert by_record.keys()==common
        for kind in ('raw','cal'):
            matched.extend(dict(model=name,reference=kind,**row) for row in summarize(by_record,'val',kind))
    write_csv(output/'rerun_epoch001_val_common_points.csv',matched)
    atomic(output/'epoch001_comparison.json',dict(
        common_recordings=len(common),
        target_abs_force_edges_N=[0,.5,1,2,5,10,None],
        common_point_target_counts=dict(totals),
        reported=[row for row in first if row['stage']=='all'],
        rerun=[row for row in matched if row['stage']=='all'],
        missing_epoch_one_checkpoint=['raw_ap304_original'],
        checkpoint_sha256={name:sha(ckpt) for name,(_,ckpt) in paths.items()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cal-root','k2-root','ap-root','output'):
        parser.add_argument(f'--{name}',type=Path,required=True)
    args=parser.parse_args()
    main(args.cal_root,args.k2_root,args.ap_root,args.output)
