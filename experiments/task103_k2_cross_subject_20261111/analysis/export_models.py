"""Export finite, strictly loadable best states only after final report validation."""
import argparse
from pathlib import Path
import shutil
import sys
import numpy as np
import torch


def main(root):
    root=Path(root)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from common import BRANCHES, EXPERIMENT, atomic, load, sha, source_binding
    from train import network
    report=load(root/'results/FINAL_REPORT.json')
    assert report['status']=='COMPLETE' and report['point_and_ground_truth_identity_checked']
    output=root/'exported_models';output.mkdir(exist_ok=True)
    records=[]
    for branch in BRANCHES:
        folder=root/'runs'/branch;done=load(folder/'DONE.json');checkpoint=folder/'best.pt'
        assert done['phase']=='complete' and sha(checkpoint)==done['best_sha256']
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert state['binding']==done['binding']
        assert state['binding']['source_code']==source_binding()
        assert all(torch.isfinite(t).all() for t in state['model'].values())
        net=network();net.load_state_dict(state['model'],strict=True)
        norm=state['binding']['force_norm']
        assert np.isfinite(norm['mean']).all() and np.isfinite(norm['std']).all() and min(norm['std'])>0
        dest=output/branch;dest.mkdir(exist_ok=True)
        weights=dest/'best_model_state_dict.pt';torch.save(state['model'],weights)
        emg=root/'prepared/emg_train_norm.npz';shutil.copy2(emg,dest/'emg_train_norm.npz')
        atomic(dest/'inference_parameters.json',dict(
            branch=branch,force_norm=norm,emg_norm_sha256=sha(emg),
            model_state_sha256=sha(weights),selected_training_checkpoint_sha256=sha(checkpoint),
            selected_epoch=state['state']['epoch'],selected_val_loss=state['state']['best'],
            inference='Apply the documented causal EMG preprocessing and shared EMG norm. The network output is normalized force: force_N = output * force_std + force_mean.',
            output_shape='batch, five fingers for one hand, 89 times',
            temporal_queries='start_sample + 470 + 40*k, k=0..88, on the 2000Hz input grid',
            training_binding=state['binding']))
        records.append(dict(branch=branch,weights_sha256=sha(weights),bytes=weights.stat().st_size,
                            selected_epoch=state['state']['epoch']))
        del net,state
    atomic(output/'EXPORT_COMPLETE.json',dict(status='COMPLETE',models=records,
        final_report_sha256=sha(root/'results/FINAL_REPORT.json'),point_identity_verified=True))
    print(records)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)
