"""Multi-scale factorized encoder for a circular sixteen-electrode array."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def circular_fourier_encoding(electrodes: int, width: int) -> torch.Tensor:
    """Return a fixed real Fourier basis without the constant component."""
    if electrodes < 2 or width < 1:
        raise ValueError("electrodes and width must be positive")
    positions = torch.arange(electrodes, dtype=torch.float32)
    angle = 2.0 * math.pi * positions / electrodes
    components: list[torch.Tensor] = []
    for harmonic in range(1, (electrodes + 1) // 2):
        components.extend(
            (torch.sin(harmonic * angle), torch.cos(harmonic * angle))
        )
    if electrodes % 2 == 0:
        components.append(torch.cos((electrodes // 2) * angle))
    basis = torch.stack(components, dim=-1)
    return basis.repeat(1, math.ceil(width / basis.shape[-1]))[:, :width]


class MultiScaleSharedPatchMLP(nn.Module):
    """Apply shared causal patch MLPs independently to every electrode."""

    def __init__(
        self,
        receptive_fields: tuple[int, ...],
        left_paddings: tuple[int, ...],
        stride: int,
        hidden_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()
        if len(receptive_fields) != len(left_paddings) or not receptive_fields:
            raise ValueError("receptive_fields and left_paddings must match")
        self.receptive_fields = tuple(int(value) for value in receptive_fields)
        self.left_paddings = tuple(int(value) for value in left_paddings)
        self.stride = int(stride)
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Linear(receptive_field, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, output_channels),
            )
            for receptive_field in self.receptive_fields
        )
        self.branch_logits = nn.Parameter(torch.zeros(len(self.branches)))
        self.norm = nn.LayerNorm(output_channels)

    def forward(self, emg: torch.Tensor) -> torch.Tensor:
        if emg.ndim != 3:
            raise ValueError("EMG must use BCT layout")
        outputs = []
        lengths = []
        for receptive_field, padding, branch in zip(
            self.receptive_fields, self.left_paddings, self.branches
        ):
            patches = F.pad(emg, (padding, 0)).unfold(
                dimension=-1,
                size=receptive_field,
                step=self.stride,
            )
            lengths.append(patches.shape[-2])
            outputs.append(branch(patches))
        if len(set(lengths)) != 1:
            raise ValueError(f"multi-scale branches are misaligned: {lengths}")
        weights = torch.softmax(self.branch_logits, dim=0)
        fused = sum(weight * output for weight, output in zip(weights, outputs))
        return self.norm(fused).permute(0, 2, 1, 3)


class TemporalDecayLinearAttention(nn.Module):
    """Chunkwise causal linear attention with one learned decay per head."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        dropout: float = 0.1,
        chunk_size: int = 32,
    ) -> None:
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.channels = channels
        self.num_heads = num_heads
        self.head_channels = channels // num_heads
        self.chunk_size = chunk_size
        self.qkv = nn.Linear(channels, 3 * channels)
        self.output = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)

        half_lives = torch.logspace(
            math.log10(4.0), math.log10(256.0), num_heads
        )
        decay = torch.exp(-math.log(2.0) / half_lives)
        self.decay_logit = nn.Parameter(torch.logit(decay))

    @staticmethod
    def _feature_map(value: torch.Tensor) -> torch.Tensor:
        return F.elu(value, alpha=1.0) + 1.0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, length, _ = inputs.shape
        query, key, value = self.qkv(inputs).chunk(3, dim=-1)
        shape = (batch, length, self.num_heads, self.head_channels)
        query = self._feature_map(query.reshape(shape).transpose(1, 2))
        key = self._feature_map(key.reshape(shape).transpose(1, 2))
        value = value.reshape(shape).transpose(1, 2)
        decay = torch.sigmoid(self.decay_logit).to(dtype=query.dtype)

        state = query.new_zeros(
            batch, self.num_heads, self.head_channels, self.head_channels
        )
        normalizer_state = query.new_zeros(
            batch, self.num_heads, self.head_channels
        )
        outputs = []
        for start in range(0, length, self.chunk_size):
            stop = min(start + self.chunk_size, length)
            query_chunk, key_chunk, value_chunk = (
                tensor[:, :, start:stop] for tensor in (query, key, value)
            )
            chunk_length = stop - start
            positions = torch.arange(chunk_length, device=inputs.device)
            distance = positions[:, None] - positions[None, :]
            causal_decay = torch.where(
                distance >= 0,
                decay[:, None, None] ** distance.clamp_min(0)[None],
                torch.zeros((), device=inputs.device, dtype=query.dtype),
            )[None]
            carry_decay = (
                decay[:, None] ** (positions + 1)[None]
            )[None]
            update_decay = (
                decay[:, None]
                ** (chunk_length - 1 - positions)[None]
            )[None]
            state_decay = (decay**chunk_length)[None]

            scores = torch.einsum(
                "bhid,bhjd->bhij", query_chunk, key_chunk
            )
            scores = scores * causal_decay
            numerator = torch.einsum(
                "bhij,bhjd->bhid", scores, value_chunk
            )
            denominator = scores.sum(dim=-1)
            numerator = numerator + carry_decay[:, :, :, None] * torch.einsum(
                "bhid,bhde->bhie", query_chunk, state
            )
            denominator = denominator + carry_decay * torch.einsum(
                "bhid,bhd->bhi", query_chunk, normalizer_state
            )
            outputs.append(
                numerator / denominator.clamp_min(1e-6)[..., None]
            )

            state = state_decay[:, :, None, None] * state + torch.einsum(
                "bhj,bhjd,bhje->bhde", update_decay, key_chunk, value_chunk
            )
            normalizer_state = (
                state_decay[:, :, None] * normalizer_state
                + torch.einsum("bhj,bhjd->bhd", update_decay, key_chunk)
            )

        result = torch.cat(outputs, dim=2).transpose(1, 2).reshape(
            batch, length, self.channels
        )
        return self.dropout(self.output(result))


class FactorizedLinearBlock(nn.Module):
    """Spatial softmax attention followed by causal temporal attention."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        feedforward_channels: int,
        dropout: float,
        temporal_chunk_size: int,
    ) -> None:
        super().__init__()
        self.spatial_norm = nn.LayerNorm(channels)
        self.spatial_attention = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(channels)
        self.temporal_attention = TemporalDecayLinearAttention(
            channels, num_heads, dropout, temporal_chunk_size
        )
        self.feedforward_norm = nn.LayerNorm(channels)
        self.feedforward = nn.Sequential(
            nn.Linear(channels, feedforward_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_channels, channels),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, n_time, electrodes, width = tokens.shape
        spatial = tokens.reshape(batch * n_time, electrodes, width)
        normalized = self.spatial_norm(spatial)
        attended, _ = self.spatial_attention(
            normalized, normalized, normalized, need_weights=False
        )
        spatial = spatial + self.dropout(attended)
        tokens = spatial.reshape(batch, n_time, electrodes, width)

        temporal = tokens.permute(0, 2, 1, 3).reshape(
            batch * electrodes, n_time, width
        )
        temporal = temporal + self.temporal_attention(
            self.temporal_norm(temporal)
        )
        temporal = temporal + self.feedforward(
            self.feedforward_norm(temporal)
        )
        return temporal.reshape(
            batch, electrodes, n_time, width
        ).permute(0, 2, 1, 3)


class RingFactorizedEncoder(nn.Module):
    """The electrode-token encoder from the 224-wide multi-query Ring model."""

    def __init__(
        self,
        input_channels: int = 16,
        model_channels: int = 304,
        patch_hidden_channels: int = 96,
        receptive_fields: tuple[int, ...] = (40, 128),
        left_paddings: tuple[int, ...] = (16, 98),
        patch_stride: int = 40,
        context_samples: int = 1_790,
        encoder_layers: int = 9,
        num_heads: int = 8,
        feedforward_channels: int = 1216,
        dropout: float = 0.1,
        temporal_chunk_size: int = 32,
    ) -> None:
        super().__init__()
        if model_channels % num_heads:
            raise ValueError("model_channels must be divisible by num_heads")
        self.input_channels = int(input_channels)
        self.model_channels = int(model_channels)
        self.patch_stride = int(patch_stride)
        self.left_context = int(context_samples)
        self.right_context = 0
        self.context_tokens = round(context_samples / patch_stride)
        self.patch_embedding = MultiScaleSharedPatchMLP(
            tuple(receptive_fields),
            tuple(left_paddings),
            patch_stride,
            patch_hidden_channels,
            model_channels,
        )
        self.register_buffer(
            "electrode_fourier",
            circular_fourier_encoding(input_channels, model_channels)[
                None, None
            ],
            persistent=True,
        )
        self.blocks = nn.ModuleList(
            FactorizedLinearBlock(
                model_channels,
                num_heads,
                feedforward_channels,
                dropout,
                temporal_chunk_size,
            )
            for _ in range(encoder_layers)
        )
        self.final_norm = nn.LayerNorm(model_channels)

    def forward(self, emg: torch.Tensor) -> torch.Tensor:
        if emg.ndim != 3 or emg.shape[1] != self.input_channels:
            raise ValueError(
                f"expected B,{self.input_channels},T EMG, got {tuple(emg.shape)}"
            )
        tokens = self.patch_embedding(emg)
        tokens = tokens + self.electrode_fourier.to(dtype=tokens.dtype)
        for block in self.blocks:
            if torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                tokens = checkpoint(block, tokens, use_reentrant=False, preserve_rng_state=True)
            else:
                tokens = block(tokens)
        return self.final_norm(tokens)
