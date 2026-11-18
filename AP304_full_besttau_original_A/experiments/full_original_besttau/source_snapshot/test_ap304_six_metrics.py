import contextlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from emg2force.ap304_six_metrics import METHODS, moments, merge, scores, summarize_six


def fixture_trials():
    rng=np.random.default_rng(237)
    result=[]
    # Unequal trial counts, lengths, subjects/stages, and one missing stage cell.
    for subject,stage,count in [('a',1,1),('a',2,3),('b',1,2),('c',1,2),('c',2,1)]:
        for trial in range(count):
            y=rng.normal(size=(5,13+trial))+stage*2
            p=(.2+trial)*y+subject.count('b')*5+rng.normal(size=y.shape)
            result.append(dict(trial_key=f'{subject}-{stage}-{trial}',subject=subject,stage=stage,p=p,y=y,
                statistics=moments(p,y),per_finger_statistics=[moments(p[f],y[f]) for f in range(5)]))
    return result


def direct(p,y):
    p,y=np.asarray(p).ravel(),np.asarray(y).ravel();active=np.abs(y)>=.5
    return np.array([np.corrcoef(p,y)[0,1],100*np.mean(np.abs(p[active]-y[active])/np.abs(y[active])),np.mean(np.abs(p-y))])


def reference(rows):
    def trial_mean(rs):return np.mean([direct(x['p'],x['y']) for x in rs],axis=0)
    def pooled(rs):return direct(np.concatenate([x['p'] for x in rs],axis=-1),np.concatenate([x['y'] for x in rs],axis=-1))
    subjects=sorted({x['subject'] for x in rows});stages=sorted({x['stage'] for x in rows})
    person_means=[];person_stage_means=[];person_pooled=[];person_stage_pooled=[]
    for person in subjects:
        rs=[x for x in rows if x['subject']==person];available=sorted({x['stage'] for x in rs})
        cells=[[x for x in rs if x['stage']==s] for s in available]
        person_means.append(trial_mean(rs));person_pooled.append(pooled(rs))
        person_stage_means.append(np.mean([trial_mean(cell) for cell in cells],axis=0))
        person_stage_pooled.append(np.mean([pooled(cell) for cell in cells],axis=0))
    return [trial_mean(rows),np.mean([trial_mean([x for x in rows if x['stage']==s]) for s in stages],axis=0),
        np.mean(person_means,axis=0),np.mean(person_stage_means,axis=0),
        np.mean(person_pooled,axis=0),np.mean(person_stage_pooled,axis=0)]


def vector(x):return [x['corr'],x['mape_percent'],x['mae_n']]


def test_all_six_match_direct_array_reference_after_persistence():
    rows=fixture_trials()
    stored=json.loads(json.dumps([{k:v for k,v in x.items() if k not in ('p','y')} for x in rows],allow_nan=False))
    report=summarize_six(stored)
    for key,expected in zip(METHODS,reference(rows)):
        np.testing.assert_allclose(vector(report['pooled_five_fingers']['methods'][key]),expected,rtol=1e-12,atol=1e-12)
    for f in range(5):
        finger_rows=[dict(x,p=x['p'][f],y=x['y'][f]) for x in rows]
        for key,expected in zip(METHODS,reference(finger_rows)):
            np.testing.assert_allclose(vector(report['per_finger'][f]['methods'][key]),expected,rtol=1e-12,atol=1e-12)
    assert not np.isclose(report['pooled_five_fingers']['methods']['3_subject_trial_equal']['corr'],
        report['pooled_five_fingers']['methods']['5_subject_pooled']['corr'])


def test_between_trial_means_undefined_and_duplicate_rejection():
    p1,y1=np.array([10.,11.]),np.array([0.,1.])
    p2,y2=np.array([0.,1.]),np.array([10.,11.])
    combined=merge(moments(p1,y1),moments(p2,y2))
    np.testing.assert_allclose(vector(scores(combined)),direct(np.r_[p1,p2],np.r_[y1,y2]),atol=1e-12)
    assert scores(combined)['corr']<0
    s=scores(moments(np.ones(10),np.zeros(10)))
    assert s['corr'] is None and s['mape_percent'] is None and s['mae_n']==1
    rows=fixture_trials()
    with pytest.raises(ValueError):summarize_six(rows+rows[:1])


def test_cpu_evaluation_persists_statistics_and_epoch_checkpoint(tmp_path):
    from emg2force import ap304_stage2_six_train as trainer
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo',init_method='file://'+str(tmp_path/'dist'),rank=0,world_size=1)
    rows=[dict(trial_key=str(i),subject='a' if i<2 else 'b',stage=1+i%2,ftype='1x5x1',n_samples=4000) for i in range(4)]
    class Reader:
        def __init__(self,*args):pass
        def window(self,trial,wid):
            x=np.full((16,4000),trial/10,np.float32)
            y=np.arange(445,dtype=np.float32).reshape(5,89)/100+trial/20
            return x,y,470+40*np.arange(89)
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.scale=torch.nn.Parameter(torch.tensor(.7))
        def forward(self,x):
            base=torch.arange(445,dtype=torch.float32).reshape(1,5,89)/100
            return base*self.scale+x[:,0,0,None,None]
    config=dict(run_root=str(tmp_path),source_checkpoint_sha256='fixture',precision='bf16_mixed',
        index_sha256='fixture',mape_threshold_n=.5,eval_batch=8,corr_weight=1.,mape_weight=.3,mae_weight=.03)
    (tmp_path/'resolved_config.yaml').write_text('fixture: true\n');(tmp_path/'train_norm.npz').write_bytes(b'fixture')
    checkpoint=tmp_path/'source.pt';torch.save(dict(fixture=True),checkpoint)
    net=Model();optimizer=torch.optim.AdamW(net.parameters())
    try:
        with patch.object(trainer,'WindowReader',Reader),patch.object(trainer.torch,'autocast',lambda *a,**kw:contextlib.nullcontext()):
            result=trainer.evaluate(net,config,rows,'val','epoch001_val',0,1,torch.device('cpu'),torch.zeros(1,5,1),torch.ones(1,5,1),checkpoint)
        folder=tmp_path/'evaluations/epoch001_val';receipt=json.loads((folder/'COMPLETE.json').read_text())
        assert len(result['six_aggregations'])==6
        for name,digest in receipt['files'].items():assert trainer.sha(folder/name)==digest
        records=[json.loads(line) for line in (folder/'rank0_trials.jsonl').read_text().splitlines()]
        regenerated=summarize_six(records)
        assert regenerated['pooled_five_fingers']['methods']==result['six_aggregations']
        state=dict(epoch=1,next_update=0,global_update=1)
        with patch.object(trainer,'rng_state',lambda:dict(fixture=True)):
            trainer.save(net,optimizer,state,config,tmp_path,0,1)
        saved=json.loads((tmp_path/'checkpoints/epoch001.json').read_text())
        assert trainer.sha(saved['checkpoint'])==saved['sha256']
        assert saved['evaluation_receipt_sha256']==trainer.sha(folder/'COMPLETE.json')
        assert torch.load(saved['checkpoint'],weights_only=True)['state']==state
        assert not torch.cuda.is_initialized()
    finally:torch.distributed.destroy_process_group()
