"""Explicit acceleration, preserving original Parameter identities/state_dict."""
from types import MethodType


def sparse_forward(self, features, weights):
    if not features.is_cuda or features.ndim != 3 or weights.ndim != 3:
        raise ValueError('Sparse training backend requires CUDA BTC/BTE tensors')
    if any(expert[2].p != 0 for expert in self.experts):
        raise ValueError('This qualified sparse backend requires expert dropout=0')
    from .vendor.rq_expert_execution import sparse_experts
    return sparse_experts(self, features, weights, 8)


def configure(model, backend):
    if backend not in ('reference', 'fla', 'sparse', 'combined'):
        raise ValueError(backend)
    from .vendor.experts import AffineExpertBank
    for block in model.encoder.blocks:
        block.temporal_attention.backend = 'fla' if backend in ('fla', 'combined') else 'torch'
    for member in model.members:
        for bank in member.experts:
            bank.forward = MethodType(sparse_forward if backend in ('sparse', 'combined')
                                      else AffineExpertBank.forward, bank)
    return model
