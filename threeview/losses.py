"""Five-finger original weighting/threshold semantics; user's new coefficients."""
import torch
if __package__:
    from .source.ordinal import ordinal
else:
    from source.ordinal import ordinal
def a_loss(pred,raw,norm):
    mean=torch.as_tensor(norm['force_mean'],device=pred.device).view(1,5,1)
    scale=torch.as_tensor(norm['force_std'],device=pred.device).view(1,5,1)
    p=pred*scale+mean;y=(raw-mean)/scale
    pp=pred.transpose(0,1).reshape(5,-1);yy=y.transpose(0,1).reshape(5,-1)
    pc=pp-pp.mean(1,keepdim=True);yc=yy-yy.mean(1,keepdim=True)
    vp=pc.square().sum(1);vy=yc.square().sum(1)
    corr=(pc*yc).sum(1)/(vp*vy).clamp_min(1e-8).sqrt()
    w=raw.abs().mean((0,2));wc=w*((vp>1e-8)&(vy>1e-8))
    corrw=torch.where(wc.sum()>1e-8,(wc*corr).sum()/wc.sum().clamp_min(1e-8),corr.sum()*0)
    eps=torch.minimum(raw.abs().amax()*.05,raw.new_tensor(.5)).clamp_min(1e-6)
    valid=raw.abs()>eps;n=valid.sum((0,2));wm=w*(n>0)
    mape=((p-raw).abs()/raw.abs().clamp_min(eps)*valid).sum((0,2))/n.clamp_min(1)
    m=torch.where(wm.sum()>1e-8,(wm*mape).sum()/wm.sum().clamp_min(1e-8),mape.sum()*0)
    mae=(p-raw).abs().mean();a=.5*(1-corrw)+.1*m+.15*mae
    return a,dict(corr=corrw,mape_ratio=m,mae=mae),p
def total(pred,raw,norm,temp,seed,phase,coef):
    a,parts,p=a_loss(pred,raw,norm);rank,cov=ordinal(p,raw,temp,seed)
    objective=rank+coef*a if phase==1 else a+coef*rank
    return objective,a,rank,parts,cov
