"""Portable raw EMG .npy -> native ranking-score .npz command line."""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--variant',choices=['best','last'],default='best')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    parser.add_argument('--batch-size',type=int,default=4,help='number of individual 2s windows per forward')
    parser.add_argument('--cpu-threads',type=int,default=2)
    args=parser.parse_args()
    if args.input.resolve()==args.output.resolve():parser.error('output must differ from input')
    import numpy as np
    import torch
    from ap304_rank import RankPredictor
    torch.set_num_threads(args.cpu_threads)
    if args.device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
    raw=np.load(args.input,mmap_mode='r',allow_pickle=False)
    result=RankPredictor(variant=args.variant,device=args.device,precision=args.precision).predict_trial(raw,batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('wb') as handle:np.savez(handle,**result)
    import json
    print(json.dumps(dict(output=str(args.output),points=len(result['scores']),shape=list(result['scores'].shape),
                         units=str(result['output_units']),first_sample=int(result['sample_indices'][0]),
                         last_sample=int(result['sample_indices'][-1])),ensure_ascii=False))


if __name__=='__main__':main()
