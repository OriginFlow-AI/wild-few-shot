"""Native FP32 CUDA forward, original ATen linear Jacobian on CPU for ordered backward.

No change to 89->3572 align_corners=True coordinates, output bytes or loss.
This execution adapter is installed only in the isolated training forward.
"""
import torch
from torch.nn import functional as F

class OrderedLinearBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        assert x.dtype==torch.float32 and x.ndim==3 and x.shape[-1]==89
        ctx.input_size=list(x.shape)
        return F.interpolate(x,size=3572,mode='linear',align_corners=True)
    @staticmethod
    def backward(ctx,grad):
        assert grad.dtype==torch.float32 and grad.shape[-1]==3572
        cpu=grad.detach().contiguous().cpu()
        result=torch.ops.aten.upsample_linear1d_backward.default(cpu,[3572],ctx.input_size,True,None)
        return result.to(device=grad.device)

def aligned_prediction(model,emg):
    native=model(emg)
    assert native.shape==(emg.shape[0],5,89)
    return OrderedLinearBackward.apply(native)

def install():
    import two_rank_forward
    two_rank_forward.aligned_prediction=aligned_prediction

def cpu_oracle():
    # Independent analytic float64 Jacobian, plus native CPU autograd reference.
    x=torch.arange(2*5*89,dtype=torch.float32).reshape(2,5,89).remainder(191)/191
    u=torch.arange(2*5*3572,dtype=torch.float32).reshape(2,5,3572).remainder(127)/1270
    t=torch.arange(3572,dtype=torch.float64)*88/3571
    lo=t.floor().long();hi=(lo+1).clamp_max(88);w=t-lo
    matrix=torch.zeros(3572,89,dtype=torch.float64)
    matrix[torch.arange(3572),lo]+=1-w;matrix[torch.arange(3572),hi]+=w
    truth=u.double().reshape(-1,3572)@matrix
    z=x.clone().requires_grad_();a=OrderedLinearBackward.apply(z)
    ga=torch.autograd.grad(a,z,u)[0]
    z=x.clone().requires_grad_();b=F.interpolate(z,size=3572,mode='linear',align_corners=True)
    gb=torch.autograd.grad(b,z,u)[0]
    # Float32 accumulation is the unchanged dtype. The independent oracle is reported,
    # not confused with the fixed 1e-7 complete-model recovery gate.
    assert torch.equal(a,b) and torch.equal(ga,gb)
    return dict(forward_native_CPU_exact=True,backward_native_CPU_exact=True,
        analytic_float64_max_abs=float((ga.double().reshape(-1,89)-truth).abs().max()),
        analytic_float64_max_relative=float((ga.double().reshape(-1,89)-truth).abs().max()/truth.abs().max()),
        original_coordinates='89_to3572_align_corners_True',original_dtype='float32')
