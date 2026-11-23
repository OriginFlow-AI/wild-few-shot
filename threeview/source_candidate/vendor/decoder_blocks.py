from __future__ import annotations
import math, os
import torch
from torch import nn
from torch.nn import functional as F

class CausalDepthwiseResidualBlock(nn.Module):
    """A causal depthwise-separable temporal residual block for BTC features."""

    def __init__(self, channels: int, kernel_size: int=9, dilation: int=1) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError('channels must be positive')
        if kernel_size <= 0:
            raise ValueError('kernel_size must be positive')
        if dilation <= 0:
            raise ValueError('dilation must be positive')
        self.left_padding = dilation * (kernel_size - 1)
        self.depthwise = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.LayerNorm(channels)
        self.activation = nn.LeakyReLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features
        features = features.transpose(1, 2)
        features = torch.nn.functional.pad(features, (self.left_padding, 0))
        features = self.depthwise(features)
        features = self.pointwise(features).transpose(1, 2)
        features = self.activation(self.norm(features))
        return residual + features

class CausalSelfAttentionBlock(nn.Module):
    """Pre-norm causal self-attention with a token-wise feed-forward block."""

    def __init__(self, channels: int=64, num_heads: int=4, feedforward_channels: int=256, dropout: float=0.0, position_encoding: str='sinusoidal', attention_cache_length: int | None=None) -> None:
        super().__init__()
        if channels % num_heads:
            raise ValueError('channels must be divisible by num_heads')
        if position_encoding not in {'sinusoidal', 'rope'}:
            raise ValueError("position_encoding must be 'sinusoidal' or 'rope'")
        if position_encoding == 'rope' and channels // num_heads % 2:
            raise ValueError('RoPE attention head channels must be even')
        self.position_encoding = position_encoding
        if attention_cache_length is not None and attention_cache_length < 1:
            raise ValueError('attention_cache_length must be positive')
        self.attention_cache_length = attention_cache_length
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.feedforward = nn.Sequential(nn.Linear(channels, feedforward_channels), nn.GELU(), nn.Linear(feedforward_channels, channels))
        self.dropout = nn.Dropout(dropout)

    def _rope_attention(self, features: torch.Tensor) -> torch.Tensor:
        """Run MHA with RoPE applied after the learned Q/K projections."""
        batch, n_time, channels = features.shape
        heads = self.attention.num_heads
        head_channels = channels // heads
        query, key, value = F.linear(features, self.attention.in_proj_weight, self.attention.in_proj_bias).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, n_time, heads, head_channels).transpose(1, 2)
        query = _apply_rotary_position_embedding(split_heads(query))
        key = _apply_rotary_position_embedding(split_heads(key))
        value = split_heads(value)
        row = torch.arange(n_time, device=features.device)[:, None]
        column = torch.arange(n_time, device=features.device)[None, :]
        allowed = column <= row
        if self.attention_cache_length is not None:
            allowed = allowed & (column > row - int(self.attention_cache_length))
        mask = torch.zeros(n_time, n_time, device=features.device, dtype=query.dtype).masked_fill(~allowed, float('-inf'))
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=self.attention.dropout if self.training else 0.0, is_causal=False)
        attended = attended.transpose(1, 2).reshape(batch, n_time, channels)
        return self.attention.out_proj(attended)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(features)
        if self.position_encoding == 'rope':
            attended = self._rope_attention(normalized)
        else:
            n_time = features.shape[1]
            causal_mask = torch.ones(n_time, n_time, device=features.device, dtype=torch.bool).triu(1)
            if self.attention_cache_length is not None:
                row = torch.arange(n_time, device=features.device)[:, None]
                column = torch.arange(n_time, device=features.device)[None, :]
                causal_mask = causal_mask | (column <= row - int(self.attention_cache_length))
            attended, _ = self.attention(normalized, normalized, normalized, attn_mask=causal_mask, need_weights=False)
        features = features + self.dropout(attended)
        return features + self.dropout(self.feedforward(self.norm2(features)))
