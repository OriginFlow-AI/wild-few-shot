"""K=2 ring-factorized Transformer adapted to the frozen C7 Fz contract.

Only the network is new.  The surrounding task remains the C7 task:

* input: 16-channel, 2 kHz EMG windows with 4,000 samples;
* target: five normalized Fz channels;
* target crop: 428 samples on the left and none on the right;
* benchmark alignment: ``linear`` interpolation with ``align_corners=True``.

The architecture follows the supplied K2 Ring Transformer specification.  Its
Pose-specific dimensions are mapped mechanically to the Fz task: 20 outputs
become five, 100 native patch tokens are produced, 11 context tokens are
removed, and the affine head therefore emits 89 native force tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import torch

import torch.nn.functional as F

from torch import nn

@dataclass(frozen=True)
class K2RingConfig:
    input_channels: int = 16
    output_channels: int = 5
    input_samples: int = 4_000
    left_context: int = 428
    right_context: int = 0
    d_model: int = 304
    num_heads: int = 8
    dim_feedforward: int = 1216
    encoder_blocks: int = 9
    decoder_blocks: int = 4
    pool_queries: int = 2
    dropout: float = 0.1
    temporal_chunk_size: int = 32
    patch_hidden: int = 96
    patch_stride: int = 40
    patch_receptive_fields: tuple[int, int] = (40, 128)
    patch_left_paddings: tuple[int, int] = (16, 98)
    output_scale: float = 0.01
    gate_bias: float = 4.0
    num_position_steps: int = 500

    @property
    def input_tokens(self) -> int:
        receptive_field = self.patch_receptive_fields[0]
        left_padding = self.patch_left_paddings[0]
        return (
            (self.input_samples + left_padding - receptive_field)
            // self.patch_stride
            + 1
        )

    @property
    def context_tokens(self) -> int:
        return round(self.left_context / self.patch_stride)

    @property
    def native_output_tokens(self) -> int:
        return self.input_tokens - self.context_tokens

    @property
    def position_anchor_tokens(self) -> int:
        # Intentionally use Python's bankers-rounding, matching the reference.
        return round(self.num_position_steps / self.patch_stride)

MODEL_NAME = "k2_ring_transformer_affine_w304_e9_d4"

MODEL_CONFIGS = {MODEL_NAME: K2RingConfig()}

EXPECTED_PARAMETERS = 18357945

class SharedPatchMLP(nn.Module):
    """Two shared left-padded patch MLPs with a learned branch mixture."""

    def __init__(self, config: K2RingConfig) -> None:
        super().__init__()
        self.config = config
        rf40, rf128 = config.patch_receptive_fields
        self.branch40 = nn.Sequential(
            nn.Linear(rf40, config.patch_hidden),
            nn.GELU(),
            nn.Linear(config.patch_hidden, config.d_model),
        )
        self.branch128 = nn.Sequential(
            nn.Linear(rf128, config.patch_hidden),
            nn.GELU(),
            nn.Linear(config.patch_hidden, config.d_model),
        )
        self.branch_logits = nn.Parameter(torch.zeros(2))
        self.output_norm = nn.LayerNorm(config.d_model)

    def _patch(
        self,
        emg: torch.Tensor,
        receptive_field: int,
        left_padding: int,
    ) -> torch.Tensor:
        padded = F.pad(emg, (left_padding, 0))
        return padded.unfold(
            dimension=-1,
            size=receptive_field,
            step=self.config.patch_stride,
        )

    def forward_branches(
        self, emg: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rf40, rf128 = self.config.patch_receptive_fields
        pad40, pad128 = self.config.patch_left_paddings
        branch40 = self.branch40(self._patch(emg, rf40, pad40))
        branch128 = self.branch128(self._patch(emg, rf128, pad128))
        if branch40.shape[:-1] != branch128.shape[:-1]:
            raise RuntimeError(
                "K2 patch branches are misaligned: "
                f"RF40={tuple(branch40.shape)}, RF128={tuple(branch128.shape)}"
            )
        return branch40, branch128

    def forward(self, emg: torch.Tensor) -> torch.Tensor:
        branch40, branch128 = self.forward_branches(emg)
        weights = self.branch_logits.softmax(dim=0)
        fused = weights[0] * branch40 + weights[1] * branch128
        # [B, C, T, D] -> [B, T, C, D]
        return self.output_norm(fused).permute(0, 2, 1, 3)

def circular_fourier_encoding(
    channels: int,
    width: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Construct the fixed real Fourier basis for a 16-electrode ring."""
    if channels != 16:
        raise ValueError(f"the fixed K2 ring encoding requires 16 channels, got {channels}")
    theta = 2.0 * math.pi * torch.arange(channels, dtype=torch.float64) / channels
    components: list[torch.Tensor] = []
    for harmonic in range(1, 8):
        components.extend(
            (torch.sin(harmonic * theta), torch.cos(harmonic * theta))
        )
    components.append(torch.cos(8 * theta))
    basis = torch.stack(components, dim=-1)
    repeats = math.ceil(width / basis.shape[-1])
    return basis.repeat(1, repeats)[:, :width].to(dtype=dtype)

class TemporalDecayLinearAttention(nn.Module):
    """Strictly causal normalized linear attention with learned head decay.

    Computation is vectorized within chunks while the compressed KV and K
    normalizer states are carried from one chunk to the next.  Accumulators are
    float32 under mixed precision to keep the positive-feature normalization
    stable.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        chunk_size: int,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.chunk_size = chunk_size
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.output_dropout = nn.Dropout(dropout)

        half_lives = torch.logspace(
            math.log10(4.0), math.log10(256.0), steps=num_heads
        )
        decay = torch.exp(-math.log(2.0) / half_lives)
        self.decay_logits = nn.Parameter(torch.logit(decay))

    def _chunk_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        decay: torch.Tensor,
        kv_state: torch.Tensor,
        k_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Inputs are [N, H, L, M], states are [N, H, M, M]/[N, H, M].
        length = query.shape[2]
        positions = torch.arange(length, device=query.device)
        distance = positions[:, None] - positions[None, :]
        causal = distance >= 0
        local_decay = decay[:, None, None].pow(distance.clamp_min(0)[None])
        local_decay = local_decay * causal.to(dtype=local_decay.dtype)[None]

        similarity = torch.einsum("nhid,nhjd->nhij", query, key)
        weights = similarity * local_decay[None]
        numerator = torch.einsum("nhij,nhjd->nhid", weights, value)
        denominator = weights.sum(dim=-1)

        carry_decay = decay[:, None].pow(positions[None] + 1)
        numerator = numerator + (
            torch.einsum("nhid,nhde->nhie", query, kv_state)
            * carry_decay[None, :, :, None]
        )
        denominator = denominator + (
            torch.einsum("nhid,nhd->nhi", query, k_state)
            * carry_decay[None]
        )
        output = numerator / denominator.clamp_min(1e-6)[..., None]

        end_weights = decay[:, None].pow(
            (length - 1 - positions)[None]
        )
        state_decay = decay.pow(length)
        weighted_key = key * end_weights[None, :, :, None]
        kv_state = (
            kv_state * state_decay[None, :, None, None]
            + torch.einsum("nhid,nhie->nhde", weighted_key, value)
        )
        k_state = (
            k_state * state_decay[None, :, None]
            + weighted_key.sum(dim=2)
        )
        return output, kv_state, k_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        qkv = self.qkv(x).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query = (F.elu(query) + 1.0).float()
        key = (F.elu(key) + 1.0).float()
        value = value.float()
        decay = self.decay_logits.sigmoid().float()

        kv_state = query.new_zeros(
            batch, self.num_heads, self.head_dim, self.head_dim
        )
        k_state = query.new_zeros(batch, self.num_heads, self.head_dim)
        chunks: list[torch.Tensor] = []
        for start in range(0, length, self.chunk_size):
            stop = min(length, start + self.chunk_size)
            output, kv_state, k_state = self._chunk_forward(
                query[:, :, start:stop],
                key[:, :, start:stop],
                value[:, :, start:stop],
                decay,
                kv_state,
                k_state,
            )
            chunks.append(output)
        attended = torch.cat(chunks, dim=2)
        attended = attended.permute(0, 2, 1, 3).reshape(batch, length, self.d_model)
        attended = attended.to(dtype=x.dtype)
        return self.output_dropout(self.output(attended))

class FactorizedLinearBlock(nn.Module):
    """Spatial full attention, temporal decay attention, then token FFN."""

    def __init__(self, config: K2RingConfig) -> None:
        super().__init__()
        width = config.d_model
        self.spatial_norm = nn.LayerNorm(width)
        self.spatial_attention = nn.MultiheadAttention(
            width,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.spatial_dropout = nn.Dropout(config.dropout)
        self.temporal_norm = nn.LayerNorm(width)
        self.temporal_attention = TemporalDecayLinearAttention(
            width,
            config.num_heads,
            config.dropout,
            config.temporal_chunk_size,
        )
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, width),
            nn.Dropout(config.dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, channels, width = tokens.shape
        spatial = tokens.reshape(batch * length, channels, width)
        normalized = self.spatial_norm(spatial)
        attended, _ = self.spatial_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        spatial = spatial + self.spatial_dropout(attended)
        tokens = spatial.reshape(batch, length, channels, width)

        temporal = tokens.permute(0, 2, 1, 3).reshape(
            batch * channels, length, width
        )
        temporal = temporal + self.temporal_attention(
            self.temporal_norm(temporal)
        )
        temporal = temporal + self.ffn(self.ffn_norm(temporal))
        return temporal.reshape(batch, channels, length, width).permute(0, 2, 1, 3)

def _apply_rope(x: torch.Tensor, base: float = 10_000.0) -> torch.Tensor:
    """Apply rotary positions to [B, H, T, head_dim] Q or K."""
    head_dim = x.shape[-1]
    if head_dim % 2:
        raise ValueError("RoPE head dimension must be even")
    positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
    inv_frequency = base ** (
        -torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32)
        / head_dim
    )
    phase = positions[:, None] * inv_frequency[None]
    cosine = phase.cos().to(dtype=x.dtype)[None, None]
    sine = phase.sin().to(dtype=x.dtype)[None, None]
    even = x[..., 0::2]
    odd = x[..., 1::2]
    return torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine),
        dim=-1,
    ).flatten(-2)

class CausalSelfAttentionBlock(nn.Module):
    """Pre-norm RoPE causal Transformer block used by both K streams."""

    def __init__(self, config: K2RingConfig) -> None:
        super().__init__()
        width = config.d_model
        if width % config.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = config.num_heads
        self.head_dim = width // config.num_heads
        self.attention_dropout = config.dropout
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, config.dim_feedforward),
            nn.GELU(),
            nn.Linear(config.dim_feedforward, width),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(self.attention_norm(x)).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query = _apply_rope(query)
        key = _apply_rope(key)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        x = x + self.residual_dropout(self.output(attended))
        return x + self.ffn(self.ffn_norm(x))

def diagonal_affine_scan(
    multiplier: torch.Tensor,
    bias: torch.Tensor,
    initial: torch.Tensor,
) -> torch.Tensor:
    """Inclusive Hillis-Steele scan for independent affine recurrences."""
    if multiplier.shape != bias.shape or multiplier.ndim != 3:
        raise ValueError(
            f"expected matching [B,T,C] tensors, got {multiplier.shape}/{bias.shape}"
        )
    if initial.shape != (multiplier.shape[0], multiplier.shape[2]):
        raise ValueError(
            f"expected initial {(multiplier.shape[0], multiplier.shape[2])}, "
            f"got {tuple(initial.shape)}"
        )
    prefix_a = multiplier
    prefix_b = bias
    offset = 1
    while offset < multiplier.shape[1]:
        identity_a = torch.ones_like(prefix_a[:, :offset])
        identity_b = torch.zeros_like(prefix_b[:, :offset])
        left_a = torch.cat((identity_a, prefix_a[:, :-offset]), dim=1)
        left_b = torch.cat((identity_b, prefix_b[:, :-offset]), dim=1)
        prefix_b = prefix_b + prefix_a * left_b
        prefix_a = prefix_a * left_a
        offset *= 2
    return prefix_a * initial[:, None, :] + prefix_b

class K2RingNetwork(nn.Module):
    """Raw EMG to native-rate force tokens."""

    def __init__(self, config: K2RingConfig) -> None:
        super().__init__()
        self.config = config
        self.left_context = config.left_context
        self.right_context = config.right_context
        self.patch_embedding = SharedPatchMLP(config)
        self.register_buffer(
            "electrode_fourier",
            circular_fourier_encoding(config.input_channels, config.d_model),
            persistent=True,
        )
        self.encoder_blocks = nn.ModuleList(
            FactorizedLinearBlock(config) for _ in range(config.encoder_blocks)
        )
        self.encoder_norm = nn.LayerNorm(config.d_model)

        self.pool_query = nn.Parameter(
            torch.empty(1, config.pool_queries, config.d_model)
        )
        nn.init.trunc_normal_(self.pool_query, std=0.02)
        self.channel_pool = nn.MultiheadAttention(
            config.d_model,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder_projection = nn.Linear(config.d_model, config.d_model)
        self.decoder_blocks = nn.ModuleList(
            CausalSelfAttentionBlock(config) for _ in range(config.decoder_blocks)
        )
        self.decoder_norm = nn.LayerNorm(config.d_model)

        self.fusion_query = nn.Parameter(torch.empty(config.d_model))
        nn.init.trunc_normal_(self.fusion_query, std=0.02)
        self.output_head = nn.Linear(
            config.d_model, 3 * config.output_channels
        )
        with torch.no_grad():
            self.output_head.bias[2 * config.output_channels :].fill_(
                config.gate_bias
            )

    def encode(self, emg: torch.Tensor) -> torch.Tensor:
        if emg.ndim != 3:
            raise ValueError(f"expected EMG [B,C,T], got {tuple(emg.shape)}")
        if emg.shape[1:] != (self.config.input_channels, self.config.input_samples):
            raise ValueError(
                "the frozen C7 contract requires EMG shape "
                f"[B,{self.config.input_channels},{self.config.input_samples}], "
                f"got {tuple(emg.shape)}"
            )
        tokens = self.patch_embedding(emg)
        if tokens.shape[1] != self.config.input_tokens:
            raise RuntimeError(
                f"expected {self.config.input_tokens} patch tokens, got {tokens.shape[1]}"
            )
        tokens = tokens + self.electrode_fourier[None, None].to(tokens.dtype)
        for block in self.encoder_blocks:
            tokens = block(tokens)
        return self.encoder_norm(tokens)

    def decode_streams(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, channels, width = tokens.shape
        values = tokens.reshape(batch * length, channels, width)
        queries = self.pool_query.expand(batch * length, -1, -1)
        pooled, _ = self.channel_pool(
            queries,
            values,
            values,
            need_weights=False,
        )
        pooled = self.decoder_projection(pooled)
        streams = pooled.reshape(
            batch, length, self.config.pool_queries, width
        ).permute(0, 2, 1, 3).reshape(
            batch * self.config.pool_queries, length, width
        )
        for block in self.decoder_blocks:
            streams = block(streams)
        streams = self.decoder_norm(streams)
        streams = streams.reshape(
            batch, self.config.pool_queries, length, width
        ).permute(0, 2, 1, 3)
        # Crop only after both shared decoder blocks and their final norm.
        streams = streams[:, self.config.context_tokens :]
        if streams.shape[1] != self.config.native_output_tokens:
            raise RuntimeError(
                f"expected {self.config.native_output_tokens} native tokens, "
                f"got {streams.shape[1]}"
            )
        return streams

    def fuse_streams(
        self, streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_streams = F.normalize(streams.float(), dim=-1)
        normalized_query = F.normalize(self.fusion_query.float(), dim=0)
        logits = torch.einsum(
            "btkd,d->btk", normalized_streams, normalized_query
        ) * math.sqrt(self.config.d_model)
        weights = logits.softmax(dim=-1).to(dtype=streams.dtype)
        fused = torch.einsum("btk,btkd->btd", weights, streams)
        return fused, weights

    def affine_decode(
        self, fused: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        position, velocity, gate_logit = self.output_head(fused).chunk(3, dim=-1)
        position = self.config.output_scale * position
        velocity = self.config.output_scale * velocity
        gate = gate_logit.sigmoid()
        anchor_tokens = self.config.position_anchor_tokens
        gate = torch.cat(
            (torch.zeros_like(gate[:, :anchor_tokens]), gate[:, anchor_tokens:]),
            dim=1,
        )
        bias = gate * velocity + (1.0 - gate) * position
        initial = bias.new_zeros(bias.shape[0], bias.shape[2])
        force = diagonal_affine_scan(gate, bias, initial)
        return force.transpose(1, 2), {
            "position": position,
            "velocity": velocity,
            "gate": gate,
        }

    def forward_with_aux(
        self, emg: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        encoded = self.encode(emg)
        streams = self.decode_streams(encoded)
        fused, fusion_weights = self.fuse_streams(streams)
        native, affine = self.affine_decode(fused)
        return native, {
            "encoded": encoded,
            "streams": streams,
            "fusion_weights": fusion_weights,
            **affine,
        }

    def forward(self, emg: torch.Tensor) -> torch.Tensor:
        native, _ = self.forward_with_aux(emg)
        return native

def count_params(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
