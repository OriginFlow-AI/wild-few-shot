"""Meaningful CPU checks of loss units, missing labels and recording isolation."""
import tempfile
import collections
from pathlib import Path
import numpy as np
import torch
from common import atomic, interpolate, load, settings
from prepare import freeze_plan, wall_center, CALIBRATION_VERSION
from common import group
from train import objective, measures

def main():
    # Missing labels must neither contribute to the objective nor its gradient.
    p=torch.tensor([.5,float('nan'),2.],requires_grad=True)
    y=torch.tensor([0.,float('nan'),0.]);mask=torch.tensor([True,False,True])
    # Prediction nonfinite is intentionally forbidden; use finite masked prediction.
    p=torch.tensor([.5,99.,2.],requires_grad=True)
    h,n=objective(p,y,mask,'raw_huber');assert n==2
    assert torch.allclose(h,torch.tensor((.125+1.5)/2));h.backward();assert p.grad[1]==0
    a=torch.tensor([.2,1.,3.]);b=torch.tensor([0.,1.5,2.]);m=torch.ones(3,dtype=torch.bool)
    loss,_=objective(a,b,m,'raw_original')
    expected=measures(a.numpy(),b.numpy(),m.numpy())['original_loss']
    assert abs(float(loss)-expected)<1e-6
    silence=measures(np.zeros(3),np.zeros(3),np.ones(3,bool))
    assert silence['mape_percent'] is None and silence['mape_points']==0 and silence['corr'] is None
    assert silence['original_loss']==1.  # Finite training/selection convention remains unchanged.
    # Unequal valid counts must still equal one global Huber mean/gradient.
    p=torch.tensor([.1,.8,2.,3.],requires_grad=True);target=torch.zeros(4)
    full,_=objective(p,target,torch.ones(4,dtype=torch.bool),'raw_huber')
    g=torch.autograd.grad(full,p,retain_graph=True)[0]
    first,n1=objective(p[:1],target[:1],torch.ones(1,dtype=torch.bool),'raw_huber')
    rest,n2=objective(p[1:],target[1:],torch.ones(3,dtype=torch.bool),'raw_huber')
    gg=torch.autograd.grad((first*n1+rest*n2)/(n1+n2),p)[0]
    assert torch.allclose(g,gg)
    # Never bridge a masked endpoint, extrapolate, or bridge long label gaps.
    t=np.array([0,10,100],np.int64);x=np.array([[0.],[1.],[10.]],float)
    _,ok=interpolate(t,x,np.ones_like(x,bool),np.array([-1,5,50,101]),max_gap_ns=20)
    assert ok[:,0].tolist()==[False,True,False,False]
    with tempfile.TemporaryDirectory() as d:
        plan=freeze_plan(d);rows=plan['rows'];by_id={r['id']:r for r in rows}
        assert (plan['test5_val'],plan['test5_test'])==(988,989)
        assert plan['calibration_version']==CALIBRATION_VERSION
        assert all(r['source_id']!=r['id'] for r in rows)
        assert len({r['recording_id'] for r in rows})==len(rows)
        candidates_by_group=collections.defaultdict(list)
        for r in rows:
            if r['stage']=='mix-test-5':candidates_by_group[group(r)].append(r)
        for r in rows:
            if r['route_status']=='candidate':
                source=by_id[r['source_id']]
                assert source['stage']=='mix-test-5'
                assert all(source[k]==r[k] for k in ['subject','pc','boot'])
                assert source['key'].split('/')[2]==r['key'].split('/')[2]
                candidates=[x for x in candidates_by_group[group(r)] if x['id']!=r['id']]
                expected=min(candidates,key=lambda x:(abs(wall_center(x)-wall_center(r)),x['id']))
                assert source['id']==expected['id']
        assert freeze_plan(d)==plan
    print('PASS: physical-N Huber, original formula, masked gradients, gap masks and frozen recording/source isolation')

if __name__=='__main__':main()
