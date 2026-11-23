from __future__ import annotations
import math, os
import torch
from torch import nn
from torch.nn import functional as F

def _top_k(weights: torch.Tensor, top_k: int | None, *, straight_through: bool=False) -> torch.Tensor:
    if top_k is None or top_k >= weights.shape[-1]:
        return weights
    values, indices = torch.topk(weights, k=top_k, dim=-1)
    sparse = torch.zeros_like(weights).scatter(-1, indices, values)
    sparse = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    if straight_through:
        return sparse.detach() - weights.detach() + weights
    return sparse

class AffineExpertBank(nn.Module):
    """Zero-output residual MLP experts for BTC Ring features."""

    def __init__(self, channels: int, num_experts: int, bottleneck: int=128, dropout: float=0.0, top_k: int | None=None, output_init_std: float=0.0, **unused: object) -> None:
        super().__init__()
        del unused
        self.num_experts = int(num_experts)
        self.top_k = top_k
        self.execution_backend = os.environ.get('RQ_EXPERT_BACKEND', 'reference')
        self.runtime_sparse_top_k = None
        if output_init_std < 0:
            raise ValueError('output_init_std must be non-negative')
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            expert = nn.Sequential(nn.Linear(channels, bottleneck), nn.SiLU(), nn.Dropout(dropout), nn.Linear(bottleneck, channels))
            if output_init_std == 0:
                nn.init.zeros_(expert[-1].weight)
            else:
                nn.init.normal_(expert[-1].weight, mean=0.0, std=output_init_std)
            nn.init.zeros_(expert[-1].bias)
            self.experts.append(expert)

    def forward(self, features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError('expert bank expects BTC features')
        if weights.ndim not in {2, 3} or weights.shape[0] != features.shape[0]:
            raise ValueError('router weights must be BE or BTE')
        if weights.shape[-1] != self.num_experts:
            raise ValueError('router/expert count mismatch')
        if weights.ndim == 3 and weights.shape[1] != features.shape[1]:
            raise ValueError('temporal weights must match feature length')
        backend = self.execution_backend
        if backend not in {'reference', 'dense', 'triton_sparse'}:
            raise ValueError(f'unknown expert execution backend: {backend}')
        eligible = weights.ndim == 3 and features.is_cuda and all((expert[2].p == 0 for expert in self.experts))
        if eligible and backend == 'dense' and (not hasattr(self.experts[0][0], 'rank')):
            from emg2pose.rq_expert_execution import dense_experts
            return dense_experts(self, features, weights)
        if eligible and backend == 'triton_sparse' and (self.runtime_sparse_top_k is not None):
            from emg2pose.rq_expert_execution import sparse_experts
            return sparse_experts(self, features, weights, self.runtime_sparse_top_k)
        residual = torch.zeros_like(features, dtype=torch.promote_types(features.dtype, weights.dtype))
        for index, expert in enumerate(self.experts):
            if weights.ndim == 2 and self.top_k is not None:
                active = torch.nonzero(weights[:, index] != 0, as_tuple=False).flatten()
                if active.numel() == 0:
                    continue
                value = expert(features.index_select(0, active))
                value = value * weights.index_select(0, active)[:, index, None, None]
                residual = residual.index_add(0, active, value.to(residual.dtype))
            else:
                scale = weights[:, index, None, None] if weights.ndim == 2 else weights[:, :, index, None]
                residual = residual + expert(features) * scale
        return residual
