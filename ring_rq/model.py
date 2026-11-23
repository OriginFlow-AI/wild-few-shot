"""Scratch full-trainable ex45 W304/E9/D4 plus Train-only frozen soft Ring-RQ."""
import sys
from pathlib import Path
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source'))
from ring_model import K2RingNetwork,K2RingConfig,FactorizedLinearBlock
from rq_model import RQReadout
from stable_interpolation import OrderedLinearBackward
class MemoryBlock(FactorizedLinearBlock):
    split_checkpoint_enabled=True
    def forward(self,tokens):
        if not self.training or not self.split_checkpoint_enabled:return super().forward(tokens)
        batch,length,channels,width=tokens.shape
        spatial=tokens.reshape(batch*length,channels,width)
        def spatial_delta(x):
            normalized=self.spatial_norm(x)
            attended,_=self.spatial_attention(normalized,normalized,normalized,need_weights=False)
            return self.spatial_dropout(attended)
        spatial=spatial+checkpoint(spatial_delta,spatial,use_reentrant=False,preserve_rng_state=True)
        temporal=spatial.reshape(batch,length,channels,width).permute(0,2,1,3).reshape(batch*channels,length,width)
        def temporal_delta(x):return self.temporal_attention(self.temporal_norm(x))
        temporal=temporal+checkpoint(temporal_delta,temporal,use_reentrant=False,preserve_rng_state=True)
        def ffn_delta(x):return self.ffn(self.ffn_norm(x))
        temporal=temporal+checkpoint(ffn_delta,temporal,use_reentrant=False,preserve_rng_state=True)
        return temporal.reshape(batch,channels,length,width).permute(0,2,1,3)
class RingRQ(K2RingNetwork):
    def __init__(self,rq_path,activation_checkpoint=True):
        super().__init__(K2RingConfig())
        reference=self.encoder_blocks
        with torch.random.fork_rng(devices=[]):
            blocks=nn.ModuleList(MemoryBlock(self.config) for _ in reference)
        blocks.load_state_dict(reference.state_dict(),strict=True)
        self.encoder_blocks=blocks
        self.output_head=nn.Linear(304,5*2*128)
        self.rq_readout=RQReadout(rq_path)
        self.activation_checkpoint=activation_checkpoint
    def encode(self,emg):
        assert emg.shape[1:]==(16,4000) and emg.dtype==torch.float32
        embedded=checkpoint(self.patch_embedding,emg,use_reentrant=False,preserve_rng_state=True) if self.training and self.activation_checkpoint else self.patch_embedding(emg)
        tokens=embedded+self.electrode_fourier[None,None]
        for block in self.encoder_blocks:
            block.split_checkpoint_enabled=self.activation_checkpoint
            tokens=block(tokens)  # Retain each inner sublayer checkpoint; no duplicate outer recompute.
        return self.encoder_norm(tokens)
    def affine_decode(self,fused):return self.rq_readout(self.output_head(fused)).transpose(1,2),{}
    def prediction(self,emg):return OrderedLinearBackward.apply(self(emg))
def build(seed=42):
    torch.manual_seed(seed)
    return RingRQ(HERE/'data/RQ_FROZEN.pt')
