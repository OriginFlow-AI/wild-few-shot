"""Checkpoint-compatible three-view architecture, independent reset2s windows."""
from pathlib import Path
import hashlib,json
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
if __package__:
 from .source_candidate.model import ForceModel
 from .source_candidate.accelerate import configure
 from .optimizations import install as optimize
 from .streams import install as stream_blocks
else:
 from source_candidate.model import ForceModel
 from source_candidate.accelerate import configure
 from optimizations import install as optimize
 from streams import install as stream_blocks
ROOT=Path(__file__).resolve().parent
GRID=torch.arange(55)*40+1829
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
class ThreeViewModel(nn.Module):
 def __init__(self,*,tokenizer=None,backend='combined',synthetic=False,streams_enabled=True):
  super().__init__();self.physical=16
  if tokenizer is None and not synthetic:raise ValueError('A fitted tokenizer is required')
  self.core=ForceModel(synthetic_smoke=True)
  if tokenizer is not None:self.core.tokenizer.load_state_dict(tokenizer,strict=True)
  self.core.tokenizer.requires_grad_(False);self.core.tokenizer.eval()
  self.core=configure(self.core,backend)
  self.register_buffer('target_indices',GRID.clone())
  optimize(self)
  if streams_enabled and backend=='combined':stream_blocks(self)
 def chunk(self,x):return self.core(x)['members'].permute(0,1,3,2)
 def members_prediction(self,x):
  if x.ndim!=3 or x.shape[1:]!=(16,4000) or x.dtype!=torch.float32:raise ValueError('Expected FP32 [B,16,4000]')
  if torch.is_autocast_enabled('cuda') or torch.is_autocast_enabled('cpu'):raise RuntimeError('FP32 only')
  outputs=[]
  for part in x.split(self.physical):
   outputs.append(checkpoint(self.chunk,part,use_reentrant=False,preserve_rng_state=True) if self.training and torch.is_grad_enabled() else self.chunk(part))
  return torch.cat(outputs,dim=1)
 def forward(self,x):return self.members_prediction(x).mean(0)
 def prediction(self,x):return self(x)
def build(seed=42,*,streams_enabled=True):
 torch.manual_seed(seed);p=ROOT/'tokenizer/TOKENIZER_FROZEN.pt'
 receipt=json.loads((ROOT/'tokenizer/TOKENIZER_FROZEN.json').read_bytes())
 assert receipt['status']=='PASS_TRAIN_FIT_VAL_BEST_FRAME_AND_MOTION' and sha(p)==receipt['sha256']
 payload=torch.load(p,map_location='cpu',weights_only=False)
 for key,n in [('norm_sha256','data/norm.npz'),('split_sha256','data/FULL_SPLIT.json.gz'),('label_policy_sha256','LABEL_POLICY.json')]:assert payload['binding'][key]==sha(ROOT/n),key
 return ThreeViewModel(tokenizer=payload['model'],backend='combined',streams_enabled=streams_enabled)
