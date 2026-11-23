from __future__ import annotations
import math, os
import torch
from torch import nn
from torch.nn import functional as F
from .experts import _top_k

class CausalDecayLinearAttentionRouter(nn.Module):
    """Small causal linear-attention router with fixed temporal half-lives.

    The exponentially decayed key/value states are evaluated with parallel
    cumulative sums.  Consequently, each token's route depends only on its
    current and preceding features without introducing a serial Python loop.
    """

    def __init__(self, channels: int, num_experts: int, *, model_channels: int=32, num_heads: int=4, feedforward_channels: int=64, half_life_tokens: tuple[float, ...]=(12.5, 25.0, 50.0, 100.0), dropout: float=0.0, temperature: float=1.0, top_k: int | None=8, dense_warmup_epochs: int=0, top_k_anneal: tuple[int, ...]=(), straight_through: bool=False) -> None:
        super().__init__()
        if model_channels < 1 or model_channels % num_heads:
            raise ValueError('router width must be divisible by its head count')
        if len(half_life_tokens) != num_heads:
            raise ValueError('one temporal half-life is required per router head')
        if any((value <= 0 for value in half_life_tokens)):
            raise ValueError('router temporal half-lives must be positive')
        if top_k is not None and (not 1 <= top_k <= num_experts):
            raise ValueError('invalid router Top-K')
        if dense_warmup_epochs < 0:
            raise ValueError('dense router warm-up must be non-negative')
        if any((not 1 <= value <= num_experts for value in top_k_anneal)):
            raise ValueError('router Top-K anneal values must be valid')
        self.num_experts = int(num_experts)
        self.num_heads = int(num_heads)
        self.head_channels = model_channels // num_heads
        self.temperature = float(temperature)
        self.top_k = top_k
        self.dense_warmup_epochs = int(dense_warmup_epochs)
        self.top_k_anneal = tuple((int(value) for value in top_k_anneal))
        self.straight_through = bool(straight_through)
        self._routing_epoch = 0
        decay = torch.tensor([2.0 ** (-1.0 / float(value)) for value in half_life_tokens], dtype=torch.float32)
        self.register_buffer('decay', decay)
        self.input_projection = nn.Linear(channels, model_channels)
        self.pre_attention_norm = nn.LayerNorm(model_channels)
        self.qkv_projection = nn.Linear(model_channels, 3 * model_channels, bias=False)
        self.attention_projection = nn.Linear(model_channels, model_channels)
        self.pre_ffn_norm = nn.LayerNorm(model_channels)
        self.feedforward = nn.Sequential(nn.Linear(model_channels, feedforward_channels), nn.SiLU(), nn.Dropout(dropout), nn.Linear(feedforward_channels, model_channels))
        self.output_norm = nn.LayerNorm(model_channels)
        self.expert_head = nn.Linear(model_channels, num_experts)
        nn.init.normal_(self.expert_head.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.expert_head.bias)
        self.dropout = nn.Dropout(dropout)
        self.last_dense_weights: torch.Tensor | None = None
        self.last_logits: torch.Tensor | None = None
        self.last_effective_top_k: int | None = top_k

    def set_epoch(self, epoch: int) -> None:
        self._routing_epoch = int(epoch)

    def _effective_top_k(self) -> int | None:
        if not self.training:
            return self.top_k
        if self._routing_epoch < self.dense_warmup_epochs:
            return None
        index = self._routing_epoch - self.dense_warmup_epochs
        if index < len(self.top_k_anneal):
            return self.top_k_anneal[index]
        return self.top_k

    def _linear_attention(self, features: torch.Tensor) -> torch.Tensor:
        batch, n_time, width = features.shape
        normalized = self.pre_attention_norm(features)
        query, key, value = self.qkv_projection(normalized).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch, n_time, self.num_heads, self.head_channels).permute(0, 2, 1, 3)
        query = F.elu(split_heads(query).float()) + 1.0
        key = F.elu(split_heads(key).float()) + 1.0
        value = split_heads(value).float()
        time = torch.arange(n_time, device=features.device, dtype=torch.float32)
        powers = self.decay.float()[:, None].pow(time[None, :])
        vector_scale = powers[None, :, :, None]
        matrix_scale = powers[None, :, :, None, None]
        key_value = key.unsqueeze(-1) * value.unsqueeze(-2)
        state = torch.cumsum(key_value / matrix_scale, dim=2) * matrix_scale
        normalizer_state = torch.cumsum(key / vector_scale, dim=2) * vector_scale
        numerator = torch.einsum('bhtd,bhtde->bhte', query, state)
        denominator = torch.einsum('bhtd,bhtd->bht', query, normalizer_state).unsqueeze(-1)
        attended = numerator / denominator.clamp_min(1e-06)
        attended = attended.permute(0, 2, 1, 3).reshape(batch, n_time, width)
        return attended.to(dtype=features.dtype)

    def forward(self, features: torch.Tensor, subject_id: torch.Tensor | None=None) -> torch.Tensor:
        del subject_id
        if features.ndim != 3:
            raise ValueError('linear-decay router expects BTC features')
        hidden = self.input_projection(features)
        attended = self._linear_attention(hidden)
        hidden = hidden + self.dropout(self.attention_projection(attended))
        hidden = hidden + self.dropout(self.feedforward(self.pre_ffn_norm(hidden)))
        logits = self.expert_head(self.output_norm(hidden))
        dense = torch.softmax(logits / max(self.temperature, 1e-06), dim=-1)
        effective_top_k = self._effective_top_k()
        weights = _top_k(dense, effective_top_k, straight_through=self.training and self.straight_through)
        self.last_logits = logits
        self.last_dense_weights = dense
        self.last_effective_top_k = effective_top_k
        return weights
