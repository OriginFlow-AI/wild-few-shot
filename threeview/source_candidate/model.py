"""Force Package Encoder + three W128 readouts; isolated architecture smoke.

Inputs are already normalized B,16,N EMG. Outputs are normalized B,T,5 Force.
No labels or target masks enter forward. Each call resets all sequence state.
Random frozen tokenizer construction is allowed ONLY with synthetic_smoke=True.
"""
from pathlib import Path
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from .vendor.encoder_reference import RingFactorizedEncoder
from .vendor.package_blocks import (SpatialRingFreqAttention, RingFrequencyScale,
                                   MLPRingFrequencyScale, DynamicTemporalAttInvAttention)
from .vendor.decoder_blocks import CausalDepthwiseResidualBlock, CausalSelfAttentionBlock
from .vendor.experts import AffineExpertBank
from .vendor.router import CausalDecayLinearAttentionRouter
from .vendor.finger import PoseFrameDecoderFingers
from .vendor.motion import MotionDecoder, decode_sequence


class RelativeLagAttention(CausalSelfAttentionBlock):
    """Original pre-norm block, relative lags replace absolute temporal PE."""
    def __init__(self):
        super().__init__(128, 4, 256, 0., attention_cache_length=250)
        self.relative_bias = nn.Parameter(torch.zeros(4, 250))

    def forward(self, x):
        batch, time, width = x.shape
        normalized = self.norm1(x)
        q, k, v = F.linear(normalized, self.attention.in_proj_weight,
                           self.attention.in_proj_bias).chunk(3, -1)
        q, k, v = [a.reshape(batch, time, 4, 32).transpose(1, 2) for a in (q, k, v)]
        index = torch.arange(time, device=x.device)
        lag = index[:, None] - index[None, :]
        bias = self.relative_bias[:, lag.clamp(0, 249)].to(q.dtype)
        bias = bias.masked_fill((lag < 0) | (lag >= 250), -torch.inf)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=bias[None])
        attended = attended.transpose(1, 2).reshape(batch, time, width)
        x = x + self.dropout(self.attention.out_proj(attended))
        return x + self.dropout(self.feedforward(self.norm2(x)))


class FrozenForceTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.codebooks = nn.Parameter(torch.randn(5, 2, 128, 8) * .02)
        self.frame = PoseFrameDecoderFingers([[i] for i in range(5)], 64, 8)
        self.motion = MotionDecoder(128, 5, 40)
        self.requires_grad_(False)

    def forward(self, logits):
        # Frozen parameters must still pass gradients back to the EMG logits.
        with torch.autocast(device_type=logits.device.type, enabled=False):
            weights = logits.float().unflatten(-1, (5, 2, 128)).softmax(-1)
            latent = torch.einsum('btglk,glkd->btgd', weights, self.codebooks.float())
            latent = latent.flatten(-2)
            observation = self.frame(latent.transpose(1, 2)).transpose(1, 2)
            return decode_sequence(self.motion, latent, observation)


class ForceReadout(nn.Module):
    def __init__(self, encoder_width):
        super().__init__()
        self.pool_query = nn.Parameter(torch.empty(1, 2, encoder_width))
        self.fusion_query = nn.Parameter(torch.empty(128))
        nn.init.trunc_normal_(self.pool_query, std=.02)
        nn.init.trunc_normal_(self.fusion_query, std=.02)
        self.channel_pool = nn.MultiheadAttention(encoder_width, 8, dropout=.1, batch_first=True)
        self.projection = nn.Linear(encoder_width, 128)
        self.temporal = nn.Sequential(*[CausalDepthwiseResidualBlock(128, 9, d) for d in (1, 2)])
        self.attention = nn.ModuleList([RelativeLagAttention(), RelativeLagAttention()])
        self.condition = nn.Linear(138, 128)
        with torch.no_grad():
            self.condition.weight.zero_(); self.condition.bias.zero_()
            self.condition.weight[:, :128].copy_(torch.eye(128))
        self.logit_head = nn.Linear(128, 1280)
        self.experts = nn.ModuleList([AffineExpertBank(128, 32, 128, 0., top_k=8) for _ in range(3)])
        # Explicit reference backend: never inherit unrelated process env settings.
        for expert in self.experts:
            expert.execution_backend = 'reference'
        self.routers = nn.ModuleList([CausalDecayLinearAttentionRouter(128, 32) for _ in range(3)])

    @staticmethod
    def flatten(streams):
        b, t, k, w = streams.shape
        return streams.permute(0, 2, 1, 3).reshape(b*k, t, w)

    @staticmethod
    def unflatten(flat, batch):
        return flat.reshape(batch, 2, flat.shape[1], 128).permute(0, 2, 1, 3)

    def adapt(self, x, position):
        router = self.routers[position]
        weights = router(x)
        out = x + self.experts[position](x, weights)
        # No diagnostics may keep a training graph alive between calls.
        router.last_logits = router.last_dense_weights = None
        return out

    def fuse(self, streams):
        weights = torch.softmax(math.sqrt(128) * torch.einsum(
            'btkd,d->btk', F.normalize(streams.float(), dim=-1),
            F.normalize(self.fusion_query.float(), dim=0)), -1).to(streams.dtype)
        return torch.einsum('btk,btkd->btd', weights, streams)

    def forward(self, tokens, tokenizer, context_tokens=45):
        b, t, e, w = tokens.shape
        values = tokens.reshape(b*t, e, w)
        pooled, _ = self.channel_pool(self.pool_query.expand(b*t, -1, -1),
                                      values, values, need_weights=False)
        streams = self.projection(pooled).reshape(b, t, 2, 128)
        flat = self.adapt(self.flatten(streams), 0)
        first = self.adapt(self.attention[0](self.temporal(flat)), 1)
        coarse = tokenizer(self.logit_head(self.fuse(self.unflatten(first, b)[:, context_tokens:])))
        zero = torch.zeros_like(coarse[:, :1])
        previous = torch.cat([zero, coarse[:, :-1]], 1)
        older = torch.cat([zero, previous[:, :-1]], 1)
        history = F.pad(torch.cat([previous, previous-older], -1), (0, 0, context_tokens, 0))
        history = history[:, None].expand(-1, 2, -1, -1).reshape(b*2, t, 10).to(first.dtype)
        second = self.adapt(self.attention[1](self.condition(torch.cat([first, history], -1))), 2)
        fine = tokenizer(self.logit_head(self.fuse(self.unflatten(second, b)[:, context_tokens:])))
        return fine, coarse


class ForceModel(nn.Module):
    def __init__(self, *, width=304, layers=9, synthetic_smoke=False):
        super().__init__()
        if not synthetic_smoke:
            raise RuntimeError('No fitted tokenizer/norm/data binding yet; synthetic smoke only')
        if (width, layers) not in {(304, 9), (224, 5)}:
            raise ValueError('Only approved r2 or historical r1 comparison is implemented')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4301)
            self.encoder = RingFactorizedEncoder(model_channels=width, encoder_layers=layers,
                num_heads=8, feedforward_channels=4*width, patch_hidden_channels=96,
                temporal_attention_backend='torch', temporal_fla_kernel_dtype='float32',
                temporal_fla_fused_boundary=False)
            for i, block in enumerate(self.encoder.blocks):
                attention = block.spatial_attention
                attention.__class__ = SpatialRingFreqAttention
                attention.enable_attinv(16)
                attention.ring_freq = RingFrequencyScale(channels=width, seed=7300+i)
                block.feedforward.add_module('ring_freq', MLPRingFrequencyScale(channels=width, seed=8300+i))
                temporal = block.temporal_attention
                temporal.__class__ = DynamicTemporalAttInvAttention
                temporal.enable_attinv(); temporal.enable_dynamic_decay()
        members = []
        for seed in (4301, 4302, 4305):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                members.append(ForceReadout(width))
        self.members = nn.ModuleList(members)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4307)
            self.tokenizer = FrozenForceTokenizer()
        order = torch.randperm(width, generator=torch.Generator().manual_seed(4303))
        masks = torch.ones(3, width, dtype=torch.bool)
        for i in range(3):
            masks[i, order[i*24:(i+1)*24]] = False
        self.register_buffer('view_masks', masks)

    def train(self, mode=True):
        super().train(mode)
        self.tokenizer.eval()
        return self

    def forward(self, emg):
        if emg.ndim != 3 or emg.shape[1] != 16 or emg.shape[-1] != 4000:
            raise ValueError('This smoke-qualified API accepts B,16,4000 independent windows only')
        tokens = self.encoder(emg)
        outputs = [m(tokens*self.view_masks[i], self.tokenizer, self.encoder.context_tokens)
                   for i, m in enumerate(self.members)]
        fine = torch.stack([o[0] for o in outputs])
        return {'members': fine, 'ensemble': fine.mean(0),
                'coarse': torch.stack([o[1] for o in outputs]),
                'sample_indices': torch.arange(55, device=emg.device)*40+1829}

    def parameter_counts(self):
        count = lambda module: sum(p.numel() for p in module.parameters())
        frozen = count(self.tokenizer)
        return dict(encoder=count(self.encoder), decoder_per_view=[count(m) for m in self.members],
                    shared_frozen_tokenizer=frozen, total=count(self),
                    trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    decoder_side=count(self.members)+frozen)


def synthetic_masked_loss(predictions, target, mask):
    """Engineering MSE only. Masked NaN targets cannot poison forward/gradients."""
    if mask.dtype != torch.bool or target.shape != mask.shape:
        raise ValueError('Target mask must be bool with B,T,5 shape')
    if not mask.any() or not torch.isfinite(target[mask]).all():
        raise ValueError('Need at least one finite valid target; all-invalid batch is rejected')
    return (predictions[:, mask].float()-target[mask].float()).square().mean()
