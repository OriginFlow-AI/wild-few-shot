"""Copy hash-checked LFS model assets into a fresh runtime root."""
import argparse
from pathlib import Path
import shutil
from common import EXPERIMENT, settings, sha

def main(root):
    root=Path(root).expanduser().resolve();(root/'assets').mkdir(parents=True,exist_ok=True)
    folder=EXPERIMENT.parents[1]/'models/ap304_stage1';c=settings()
    for name,key in [('ap304_rank_best_model_state_dict.pt','stage1_export_sha256'),
                     ('common_stage2_initial_state_dict.pt','common_initial_sha256')]:
        source=folder/name
        if sha(source)!=c[key]:raise ValueError('Missing or changed LFS asset; run git lfs pull: '+str(source))
        dest=root/'assets'/name
        if dest.exists():
            if sha(dest)!=c[key]:raise ValueError('Existing asset mismatch: '+str(dest))
        else:shutil.copyfile(source,dest)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)
