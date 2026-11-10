"""Rebind paused checkpoints only after exact Train identity is proven."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch


def train_identity(root,dataset):
    entries=dataset['entries'];pairs=np.load(root/'prepared/train_windows.npy')
    identity=np.empty(len(pairs),dtype=[('id','S20'),('side','S5'),('start','<i8')])
    identity['id']=np.asarray([r['id'] for r in entries],dtype='S20')[pairs[:,0]]
    identity['side']=np.asarray([r['side'] for r in entries],dtype='S5')[pairs[:,0]]
    identity['start']=pairs[:,1]
    rows=[{k:r[k] for k in ['id','side','stage','source_id','start_ns','samples','EMG','labels_sha256']}
          for r in entries if r['split']=='train']
    return dict(ordered_windows_sha256=hashlib.sha256(identity.tobytes()).hexdigest(),
        training_record_inputs_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        windows=len(pairs),records=len({r['id'] for r in rows}),force_norm=dataset['force_norm'])


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main(root):
    root=Path(root);archive=root/'archive/nearest_other_test5_before_self_correction'
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from common import BRANCHES,EXPERIMENT,atomic,load,settings,sha,source_binding
    torch.set_num_threads(2)
    old=load(archive/'prepared/dataset.json');new=load(root/'prepared/dataset.json');config=settings()
    assert new['config_sha256']==sha(EXPERIMENT/'config.json') and new['source_code']==source_binding()
    previous_config=load(archive/'code/config.json')
    keys=['seed','global_batch','common_microbatch','max_epochs','patience','min_delta','lr_body','lr_head',
          'weight_decay','warmup_updates','scheduler','huber_delta_N','mape_activity','train_reduction',
          'objective_domain','emg_norm_sha256','model_source_sha256','common_initial_sha256']
    assert all(previous_config[k]==config[k] for k in keys)
    before=train_identity(archive,old);after=train_identity(root,new);assert before==after
    assert old['source_code']['scripts/train.py']==new['source_code']['scripts/train.py']
    assert all(v==new['source_code'][k] for k,v in old['source_code'].items() if k.startswith('vendor/'))
    rows=[]
    for branch in BRANCHES:
        process=load(root/'checks'/('process_'+branch+'.json'));p=Path('/proc')/str(process['pid'])/'cmdline'
        assert not p.exists() or b'train.py' not in p.read_bytes(), 'Trainer must be paused'
        source=archive/'runs'/branch/'last.pt';checkpoint=torch.load(source,map_location='cpu',weights_only=True)
        assert checkpoint['state']['epoch']==0 and checkpoint['state']['history']==[] and checkpoint['state']['best'] is None
        assert not list((root/'runs'/branch).glob('epoch*_val.json'))
        binding=dict(checkpoint['binding']);old_binding=dict(binding)
        binding.update(config_sha256=new['config_sha256'],dataset_sha256=sha(root/'prepared/dataset.json'),source_code=source_binding())
        kind='cal' if branch=='calibrated_huber' else 'raw'
        assert binding['force_norm']==new['force_norm'][kind]
        assert binding['common_initial_sha256']==sha(root/'assets/common_stage2_initial_state_dict.pt')
        assert binding['EMG_norm_sha256']==sha(EXPERIMENT/'reference/stage1_emg_train_norm.npz')
        checkpoint['binding']=binding
        target=root/'runs'/branch/'last.pt';tmp=target.with_suffix('.rebind.partial.pt');torch.save(checkpoint,tmp);tmp.replace(target)
        restored=torch.load(target,map_location='cpu',weights_only=True)
        assert restored['binding']==binding
        for key in ['model','optimizer','state','torch_rng','cuda_rng']:assert equal(checkpoint[key],restored[key]),key
        atomic(root/'runs'/branch/'CONTRACT.json',dict(**binding,configuration=config))
        rows.append(dict(branch=branch,global_update=restored['state']['global_update'],
            source_checkpoint_sha256=sha(source),rebound_checkpoint_sha256=sha(target),
            previous_binding=old_binding,new_binding=binding,
            exact_model_optimizer_cursor_CPU_CUDA_RNG_preserved=True))
        del restored,checkpoint
    atomic(root/'checks/validation_only_protocol_migration.json',dict(status='PASS',
        reason='Restore original nearest-test5 scope including itself for test5; ordinary-stage Train is exactly identical',
        old_train_identity=before,new_train_identity=after,branches=rows,
        prior_complete_Val_results=0,neural_Test_metrics_read=False,training_restarted_from_initial_weights=False))
    print('PASS: exact Train inputs/order/norm, unchanged training algorithm, exact model/Adam/cursor/CPU-CUDA RNG; only Val/Test protocol and bindings changed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)
