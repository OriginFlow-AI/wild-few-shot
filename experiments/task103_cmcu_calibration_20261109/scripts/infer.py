"""Frozen bundle inference; preserve native output times and context gaps."""
import argparse
import json
import sys
import time
import numpy as np
import torch
from common import ROOT, MODEL_DIR
sys.path.insert(0, str(MODEL_DIR))
from ap304_rank import RankPredictor
from ap304_rank.inference import window_starts


def infer_trial(row, predictor, batch_size):
    output = ROOT/'predictions'/(row['id']+'.npz')
    if output.exists(): return
    directory = ROOT/'emg'/row['id']; metadata = json.loads((directory/'done.json').read_text())
    arrays, audit = {}, {}
    for side in ['left', 'right']:
        raw = np.load(directory/(side+'.npy'), mmap_mode='r')
        valid = np.load(directory/(side+'_valid.npy')); bad = np.r_[0, np.cumsum(~valid)]
        starts = window_starts(len(raw)); scores, ticks = [], []; last = -1
        for begin in range(0, len(starts), batch_size):
            selected = np.array([s for s in starts[begin:begin+batch_size] if bad[s+4000] == bad[s]], int)
            if not len(selected): continue
            predictions = predictor.predict_windows(np.stack([raw[s:s+4000] for s in selected]))
            for start, prediction in zip(selected, predictions):
                indices = start+470+40*np.arange(89, dtype=np.int64); keep = indices > last
                if keep.any():
                    scores.append(prediction[keep]); ticks.append(indices[keep]); last = int(indices[keep][-1])
        if not ticks: raise ValueError('No valid reset windows: '+row['id'])
        arrays[side+'_t'] = metadata['hands'][side]['start_ns']+np.concatenate(ticks)*500000
        arrays[side+'_scores'] = np.concatenate(scores)
        audit[side] = dict(windows=len(starts), rejected_windows=int(sum(bad[s+4000] != bad[s] for s in starts)))
    arrays['clock_ok'] = np.array(metadata['clock_ok'])
    temporary = output.with_suffix('.partial.npz'); np.savez_compressed(temporary, **arrays); temporary.replace(output)
    output.with_suffix('.json').write_text(json.dumps(dict(id=row['id'], variant='best', device=str(predictor.device),
        precision=predictor.precision, batch_size=batch_size, clock_ok=metadata['clock_ok'], hands=audit)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--id', action='append', help='Optional smoke-check trial IDs')
    args = parser.parse_args(); torch.set_num_threads(4)
    (ROOT/'predictions').mkdir(parents=True, exist_ok=True)
    rows = json.loads((ROOT/'sources.json').read_text())
    if args.id: rows = [r for r in rows if r['id'] in args.id]
    if not rows: parser.error('No matching source trials')
    predictor = RankPredictor(variant='best', root=MODEL_DIR, device=args.device, precision=args.precision)
    errors, done = [], 0; started = time.monotonic()
    for row in rows:
        try: infer_trial(row, predictor, args.batch_size); done += 1
        except Exception as exc: errors.append(dict(id=row['id'], error=str(exc)))
        if (done+len(errors)) % 20 == 0: print(f'INFER {done}/{len(rows)}, errors={len(errors)}', flush=True)
    result = dict(done=done, total=len(rows), errors=errors, seconds=time.monotonic()-started,
                  device=args.device, precision=args.precision, batch_size=args.batch_size)
    name = 'inference_smoke_result.json' if args.id else 'inference_result.json'
    (ROOT/name).write_text(json.dumps(result, indent=2))
    if errors: raise SystemExit(f'{len(errors)} inference failures; see {name}')
