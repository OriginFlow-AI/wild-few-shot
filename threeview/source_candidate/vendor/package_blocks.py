from __future__ import annotations
import math, os
import torch
from torch import nn
from torch.nn import functional as F
from .encoder_reference import TemporalDecayLinearAttention

class SpatialAttInvAttention(nn.MultiheadAttention):

    def enable_attinv(self, electrodes=16):
        assert self.batch_first and self._qkv_same_embed_dim
        assert self.bias_k is None and self.bias_v is None and (not self.add_zero_attn)
        self.electrodes = electrodes
        self.attinv_enabled = True
        with torch.random.fork_rng(devices=[]):
            self.diff_gate = nn.Linear(self.embed_dim, self.num_heads, device=self.in_proj_weight.device, dtype=self.in_proj_weight.dtype)
            nn.init.zeros_(self.diff_gate.weight)
            nn.init.zeros_(self.diff_gate.bias)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):
        assert query is key and key is value
        assert key_padding_mask is None and attn_mask is None and (not is_causal)
        assert not need_weights
        batch, electrodes, width = query.shape
        assert electrodes == self.electrodes and width == self.embed_dim
        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.reshape(batch, electrodes, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        low = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        if self.attinv_enabled:
            gate = self.diff_gate(query).tanh().permute(0, 2, 1).unsqueeze(-1)
            low = low + gate * (v - low)
        out = low.transpose(1, 2).contiguous().reshape(batch, electrodes, width)
        return (F.linear(out, self.out_proj.weight, self.out_proj.bias), None)

class StarReLU(nn.Module):

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.scale * torch.relu(x).square() + self.bias

class RingFrequencyScale(nn.Module):

    def __init__(self, channels=224, electrodes=16, groups=16, filters=4, seed=7300):
        super().__init__()
        assert channels % groups == 0 and electrodes == 16
        self.channels, self.electrodes = (channels, electrodes)
        self.groups, self.filters = (groups, filters)
        self.enabled = True
        with torch.random.fork_rng(devices=[]):
            private_cpu = torch.Generator(device='cpu').manual_seed(seed)
            torch.set_rng_state(private_cpu.get_state())
            self.route_fc1 = nn.Linear(channels, channels // 16, bias=False)
            self.route_act = StarReLU()
            self.route_fc2 = nn.Linear(channels // 16, groups * filters, bias=False)
            self.bases = nn.Parameter(torch.zeros(filters, channels // groups, electrodes // 2 + 1))

    def forward(self, x):
        rows, electrodes, channels = x.shape
        assert electrodes == self.electrodes and channels == self.channels
        if not self.enabled:
            return x
        routing = self.route_fc2(self.route_act(self.route_fc1(x.mean(dim=1))))
        routing = routing.tanh().reshape(rows, self.groups, self.filters)
        with torch.autocast(device_type=x.device.type, enabled=False):
            spectrum = torch.fft.rfft(x.float(), dim=1, norm='ortho')
            weights = torch.einsum('bgf,fck->bgck', routing.float(), self.bases.float())
            weights = weights.reshape(rows, channels, electrodes // 2 + 1).transpose(1, 2)
            delta = torch.fft.irfft(spectrum * weights, n=electrodes, dim=1, norm='ortho')
            return (x.float() + delta).to(x.dtype)

class SpatialRingFreqAttention(SpatialAttInvAttention):

    def forward(self, query, key, value, **kwargs):
        output, weights = super().forward(query, key, value, **kwargs)
        return (self.ring_freq(output), weights)

class MLPRingFrequencyScale(RingFrequencyScale):

    def forward(self, x):
        be, time, width = x.shape
        assert be % self.electrodes == 0
        batch = be // self.electrodes
        spatial = x.reshape(batch, self.electrodes, time, width).permute(0, 2, 1, 3)
        spatial = super().forward(spatial.reshape(batch * time, self.electrodes, width))
        return spatial.reshape(batch, time, self.electrodes, width).permute(0, 2, 1, 3).reshape_as(x)

class TemporalAttInvAttention(TemporalDecayLinearAttention):

    def enable_attinv(self):
        self.attinv_enabled = True
        with torch.random.fork_rng(devices=[]):
            self.diff_gate = nn.Linear(self.channels, self.num_heads, device=self.qkv.weight.device, dtype=self.qkv.weight.dtype)
            nn.init.zeros_(self.diff_gate.weight)
            nn.init.zeros_(self.diff_gate.bias)

    def _mix_attinv(self, inputs, aggregated, current_value):
        if not self.attinv_enabled:
            return aggregated
        b, t, _ = inputs.shape
        shape = (b, t, self.num_heads, self.head_channels)
        gate = self.diff_gate(inputs).tanh().unsqueeze(-1)
        a = aggregated.reshape(shape)
        v = current_value.reshape(shape).to(a.dtype)
        return (a + gate * (v - a)).reshape_as(aggregated)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.backend == 'fla':
            return self._forward_fla(inputs)
        batch, length, _ = inputs.shape
        query, key, value = self.qkv(inputs).chunk(3, dim=-1)
        shape = (batch, length, self.num_heads, self.head_channels)
        current_value = value.reshape(shape)
        query = self._feature_map(query.reshape(shape).transpose(1, 2))
        key = self._feature_map(key.reshape(shape).transpose(1, 2))
        value = value.reshape(shape).transpose(1, 2)
        decay = torch.sigmoid(self.decay_logit).to(dtype=query.dtype)
        state = query.new_zeros(batch, self.num_heads, self.head_channels, self.head_channels)
        normalizer_state = query.new_zeros(batch, self.num_heads, self.head_channels)
        outputs = []
        for start in range(0, length, self.chunk_size):
            stop = min(start + self.chunk_size, length)
            query_chunk, key_chunk, value_chunk = (tensor[:, :, start:stop] for tensor in (query, key, value))
            chunk_length = stop - start
            positions = torch.arange(chunk_length, device=inputs.device)
            distance = positions[:, None] - positions[None, :]
            causal_decay = torch.where(distance >= 0, decay[:, None, None] ** distance.clamp_min(0)[None], torch.zeros((), device=inputs.device, dtype=query.dtype))[None]
            carry_decay = (decay[:, None] ** (positions + 1)[None])[None]
            update_decay = (decay[:, None] ** (chunk_length - 1 - positions)[None])[None]
            state_decay = (decay ** chunk_length)[None]
            scores = torch.einsum('bhid,bhjd->bhij', query_chunk, key_chunk)
            scores = scores * causal_decay
            numerator = torch.einsum('bhij,bhjd->bhid', scores, value_chunk)
            denominator = scores.sum(dim=-1)
            numerator = numerator + carry_decay[:, :, :, None] * torch.einsum('bhid,bhde->bhie', query_chunk, state)
            denominator = denominator + carry_decay * torch.einsum('bhid,bhd->bhi', query_chunk, normalizer_state)
            outputs.append(numerator / denominator.clamp_min(1e-06)[..., None])
            state = state_decay[:, :, None, None] * state + torch.einsum('bhj,bhjd,bhje->bhde', update_decay, key_chunk, value_chunk)
            normalizer_state = state_decay[:, :, None] * normalizer_state + torch.einsum('bhj,bhjd->bhd', update_decay, key_chunk)
        result = torch.cat(outputs, dim=2).transpose(1, 2).reshape(batch, length, self.channels)
        result = self._mix_attinv(inputs, result, current_value)
        return self.dropout(self.output(result))

    def _forward_fla(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.device.type != 'cuda' or torch.onnx.is_in_onnx_export():
            raise RuntimeError("FLA attention requires CUDA and does not support ONNX export; use temporal_attention_backend='torch' for CPU/export.")
        try:
            from fla.ops.simple_gla import chunk_simple_gla
        except ImportError as error:
            raise ImportError("FLA backend requires fla-core[cuda] (tested with 0.5.2), PyTorch >= 2.7 and Triton >= 3.3. Install these in a separate environment or use temporal_attention_backend='torch'.") from error
        if os.environ.get('TRITON_F32_DEFAULT') not in {'tf32x3', 'ieee'}:
            raise RuntimeError("FLA encoder requires TRITON_F32_DEFAULT=tf32x3 (or ieee) before starting Python to preserve FP32 attention/gradient accuracy; Triton's default tf32 is insufficient.")
        batch, length, _ = inputs.shape
        shape = (batch, length, self.num_heads, self.head_channels)
        projected = self.qkv(inputs)
        current_value = projected[..., 2 * self.channels:].reshape(shape)
        result_dtype = projected.dtype
        fused = self.fla_fused_boundary and self.fla_kernel_dtype == 'bfloat16' and (result_dtype == torch.bfloat16)
        if fused:
            from emg2pose.ring_fla_fused import normalize_bf16, prepare_bf16
            query, key, value = prepare_bf16(projected, self.num_heads)
        else:
            query, key, value = projected.chunk(3, dim=-1)
            query = self._feature_map(query.reshape(shape))
            key = self._feature_map(key.reshape(shape))
            value = value.reshape(shape)
        kernel_dtype = getattr(torch, self.fla_kernel_dtype)
        if kernel_dtype == torch.bfloat16:
            log_decay = F.logsigmoid(self.decay_logit.float())
        else:
            decay = torch.sigmoid(self.decay_logit).to(dtype=query.dtype)
            log_decay = decay.float().log()
        gates = log_decay[None, None, :].expand(batch, length, -1).contiguous()
        if not fused:
            query, key, value = (tensor.to(kernel_dtype).contiguous() for tensor in (query, key, value))
            value = torch.cat((value, torch.ones_like(value[..., :1])), dim=-1)
        del projected
        attended, _ = chunk_simple_gla(q=query, k=key, v=value, g=gates, scale=1.0, initial_state=None, output_final_state=False)
        if fused:
            result = normalize_bf16(attended)
        else:
            attended = attended.float()
            result = attended[..., :-1] / attended[..., -1:].clamp_min(1e-06)
        result = result.reshape(batch, length, self.channels).to(result_dtype)
        result = self._mix_attinv(inputs, result, current_value)
        return self.dropout(self.output(result))

class DynamicTemporalAttInvAttention(TemporalAttInvAttention):

    def enable_dynamic_decay(self):
        self.decay_enabled = True
        self.max_logit_shift = 1.0
        with torch.random.fork_rng(devices=[]):
            self.decay_gate = nn.Linear(2 * self.channels, self.num_heads, device=self.qkv.weight.device, dtype=self.qkv.weight.dtype)
            nn.init.zeros_(self.decay_gate.weight)
            nn.init.zeros_(self.decay_gate.bias)

    def _dynamic_delta(self, inputs):
        difference = torch.cat((torch.zeros_like(inputs[:, :1]), inputs[:, 1:] - inputs[:, :-1]), dim=1)
        return self.max_logit_shift * self.decay_gate(torch.cat((inputs, difference), dim=-1)).tanh()

    def _dynamic_log_decay(self, inputs, native_log_decay):
        batch, length, _ = inputs.shape
        if not self.decay_enabled:
            return native_log_decay[None, None, :].expand(batch, length, -1).contiguous()
        delta = self._dynamic_delta(inputs)
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            logits = self.decay_logit.float()
            correction = F.logsigmoid(logits[None, None, :] + delta.float()) - F.logsigmoid(logits)[None, None, :]
            return (native_log_decay.float()[None, None, :] + correction).clamp_max(0).contiguous()

    def forward(self, inputs):
        if self.backend == 'fla':
            return self._forward_fla(inputs)
        batch, length, _ = inputs.shape
        shape = (batch, length, self.num_heads, self.head_channels)
        q, k, v = self.qkv(inputs).chunk(3, dim=-1)
        result_dtype = v.dtype
        q = self._feature_map(q.reshape(shape))
        k = self._feature_map(k.reshape(shape))
        v = v.reshape(shape)
        if self.fla_kernel_dtype == 'bfloat16':
            native = F.logsigmoid(self.decay_logit.float())
        else:
            native = self.decay_logit.sigmoid().to(q.dtype).float().log()
        gates = self._dynamic_log_decay(inputs, native)
        state = torch.zeros(batch, self.num_heads, self.head_channels, self.head_channels, device=inputs.device, dtype=torch.float32)
        normalizer = torch.zeros(batch, self.num_heads, self.head_channels, device=inputs.device, dtype=torch.float32)
        result = []
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            for t in range(length):
                decay = gates[:, t].exp()
                qt, kt, vt = (q[:, t].float(), k[:, t].float(), v[:, t].float())
                state = decay[..., None, None] * state + kt[..., None] * vt[..., None, :]
                normalizer = decay[..., None] * normalizer + kt
                numerator = torch.einsum('bhd,bhde->bhe', qt, state)
                denominator = (qt * normalizer).sum(-1).clamp_min(1e-06)
                result.append(numerator / denominator[..., None])
        result = torch.stack(result, 1).reshape(batch, length, self.channels).to(result_dtype)
        return self.dropout(self.output(self._mix_attinv(inputs, result, v)))

    def _forward_fla(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.device.type != 'cuda' or torch.onnx.is_in_onnx_export():
            raise RuntimeError("FLA attention requires CUDA and does not support ONNX export; use temporal_attention_backend='torch' for CPU/export.")
        try:
            from fla.ops.simple_gla import chunk_simple_gla
        except ImportError as error:
            raise ImportError("FLA backend requires fla-core[cuda] (tested with 0.5.2), PyTorch >= 2.7 and Triton >= 3.3. Install these in a separate environment or use temporal_attention_backend='torch'.") from error
        if os.environ.get('TRITON_F32_DEFAULT') not in {'tf32x3', 'ieee'}:
            raise RuntimeError("FLA encoder requires TRITON_F32_DEFAULT=tf32x3 (or ieee) before starting Python to preserve FP32 attention/gradient accuracy; Triton's default tf32 is insufficient.")
        batch, length, _ = inputs.shape
        shape = (batch, length, self.num_heads, self.head_channels)
        projected = self.qkv(inputs)
        current_value = projected[..., 2 * self.channels:].reshape(shape)
        result_dtype = projected.dtype
        fused = self.fla_fused_boundary and self.fla_kernel_dtype == 'bfloat16' and (result_dtype == torch.bfloat16)
        if fused:
            from emg2pose.ring_fla_fused import normalize_bf16, prepare_bf16
            query, key, value = prepare_bf16(projected, self.num_heads)
        else:
            query, key, value = projected.chunk(3, dim=-1)
            query = self._feature_map(query.reshape(shape))
            key = self._feature_map(key.reshape(shape))
            value = value.reshape(shape)
        kernel_dtype = getattr(torch, self.fla_kernel_dtype)
        if kernel_dtype == torch.bfloat16:
            log_decay = F.logsigmoid(self.decay_logit.float())
        else:
            decay = torch.sigmoid(self.decay_logit).to(dtype=query.dtype)
            log_decay = decay.float().log()
        gates = self._dynamic_log_decay(inputs, log_decay)
        if not fused:
            query, key, value = (tensor.to(kernel_dtype).contiguous() for tensor in (query, key, value))
            value = torch.cat((value, torch.ones_like(value[..., :1])), dim=-1)
        del projected
        attended, _ = chunk_simple_gla(q=query, k=key, v=value, g=gates, scale=1.0, initial_state=None, output_final_state=False)
        if fused:
            result = normalize_bf16(attended)
        else:
            attended = attended.float()
            result = attended[..., :-1] / attended[..., -1:].clamp_min(1e-06)
        result = result.reshape(batch, length, self.channels).to(result_dtype)
        result = self._mix_attinv(inputs, result, current_value)
        return self.dropout(self.output(result))
