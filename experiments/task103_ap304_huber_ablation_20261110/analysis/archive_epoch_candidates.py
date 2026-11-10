"""Copy completed-epoch checkpoints without changing or interrupting trainers."""
import argparse
import fcntl
import math
import os
from pathlib import Path
import shutil
import sys
import time
import torch


def main(root):
    root=Path(root)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from common import BRANCHES,atomic,load,sha,settings
    torch.set_num_threads(2)
    updates=math.ceil(load(root/'prepared/dataset.json')['windows']['train']/settings()['global_batch'])
    output=root/'archive/epoch_candidates';output.mkdir(parents=True,exist_ok=True)
    with (output/'archiver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic(root/'checks/epoch_archive_status.json',dict(phase='watching',pid=os.getpid()))
        while True:
            finished=[]
            for branch in BRANCHES:
                folder=root/'runs'/branch;status=folder/'STATUS.json'
                if not status.exists():continue
                s=load(status);done=folder/'DONE.json'
                if done.exists():finished.append(branch)
                boundary=s.get('epoch_update')==s.get('updates_per_epoch') and s.get('epoch_update') is not None
                if not boundary and not done.exists():continue
                contract=load(folder/'CONTRACT.json')
                guess=s.get('epoch') if boundary else load(done)['history'][-1]['epoch']
                existing=output/branch/contract['dataset_sha256'][:16]/f'epoch{guess:03d}.pt'
                if existing.exists() and existing.with_suffix('.json').exists():continue
                source=folder/'last.pt'
                if not source.exists():continue
                # Keep this descriptor open across load and copy. Atomic trainer
                # replacement cannot substitute another update underneath us.
                with source.open('rb') as source_file:
                    checkpoint=torch.load(source_file,map_location='cpu',weights_only=True)
                    state=checkpoint['state']
                    epoch=state['epoch'] if state['next_update']==0 else state['epoch']+1 if state['next_update']==updates else None
                    if not epoch or state['global_update']!=epoch*updates:continue
                    binding=checkpoint['binding'];assert binding['branch']==branch
                    dest=output/branch/binding['dataset_sha256'][:16];dest.mkdir(parents=True,exist_ok=True)
                    target=dest/f'epoch{epoch:03d}.pt';receipt=target.with_suffix('.json')
                    if target.exists() and receipt.exists():continue
                    if shutil.disk_usage(root).free<10*2**30:raise RuntimeError('Epoch archive disk reserve below10GiB')
                    temporary=target.with_suffix('.partial.pt');source_file.seek(0)
                    with temporary.open('wb') as f:shutil.copyfileobj(source_file,f,8<<20)
                    os.replace(temporary,target)
                    atomic(receipt,dict(branch=branch,completed_epoch=epoch,global_update=state['global_update'],
                        dataset_sha256=binding['dataset_sha256'],configuration_sha256=binding['config_sha256'],
                        checkpoint_sha256=sha(target),bytes=target.stat().st_size,
                        source='Atomic completed-epoch last checkpoint, copied through one stable file descriptor',
                        validation_history_at_copy=state['history'],training_artifacts_modified=False))
                    print('ARCHIVED',branch,epoch,state['global_update'],str(target),flush=True)
                del checkpoint
            atomic(root/'checks/epoch_archive_status.json',dict(phase='complete' if len(finished)==len(BRANCHES) else 'watching',
                pid=os.getpid(),finished_branches=finished,archived_epochs=len(list(output.glob('*/*/epoch*.json')))))
            if len(finished)==len(BRANCHES):return
            time.sleep(10)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)
