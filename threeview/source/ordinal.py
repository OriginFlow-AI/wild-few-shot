"""Pairwise logistic surrogate; discrete Kendall is evaluated separately."""
import math
import torch
import torch.nn.functional as F

PAIRS=2048

def indices(shape,seed):
    b,f,n=shape;g=torch.Generator(device='cpu').manual_seed(seed)
    a=torch.randint(n,(b,f,PAIRS),generator=g)
    bidx=torch.randint(n-1,(b,f,PAIRS),generator=g)
    bidx=bidx+(bidx>=a)
    return a,bidx

def differences(y,seed):
    a,b=indices(y.shape,seed);a=a.to(y.device);b=b.to(y.device)
    return torch.gather(y.float(),2,a)-torch.gather(y.float(),2,b),a,b

def ordinal(pred,raw,temperature,seed):
    dy,a,b=differences(raw,seed)
    dp=torch.gather(pred.float(),2,a)-torch.gather(pred.float(),2,b)
    valid=dy!=0
    t=torch.as_tensor(temperature,device=pred.device,dtype=torch.float32).view(1,5,1)
    assert torch.isfinite(t).all() and (t>0).all()
    terms=F.softplus(-dy.sign()*dp/t)/math.log(2)
    return (terms*valid).sum()/valid.sum().clamp_min(1),dict(valid_pairs=int(valid.sum()),total_pairs=valid.numel())

def selftest():
    y=torch.arange(64,dtype=torch.float32).view(1,1,64).expand(2,5,64).clone()
    p=y.clone().requires_grad_(True)
    good,_=ordinal(p,y,[1]*5,42);bad,_=ordinal(-p,y,[1]*5,42)
    assert good<bad and torch.isfinite(torch.autograd.grad(good,p)[0]).all()
    flat=p*0;zero,coverage=ordinal(p,flat,[1]*5,42)
    assert zero==0 and torch.equal(torch.autograd.grad(zero,p)[0],torch.zeros_like(p)) and coverage['valid_pairs']==0
    shifted,_=ordinal(p+17,y,[1]*5,42);assert torch.allclose(good,shifted,atol=1e-7)
    scaled,_=ordinal(p*3,y*3,[3]*5,42);assert torch.allclose(good,scaled,atol=1e-7)
    a,b=indices(y.shape,42);assert not (a==b).any()
    return dict(status='PASS_FINITE_ORDER_TIES_SHIFT_SCALE_AND_DETERMINISTIC_PAIRS',pairs_per_cell=PAIRS)
