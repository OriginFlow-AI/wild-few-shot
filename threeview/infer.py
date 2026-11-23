"""Three-view ensemble inference, native55 support, no GT or label LUT input."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,hashlib
import numpy as np
import torch
if __package__:
 from .model import ThreeViewModel
else:
 from model import ThreeViewModel
class ThreeViewInference:
 def __init__(self,checkpoint,norm,device='cuda:0',backend=None):
  self.device=torch.device(device);self.backend=backend or ('combined' if self.device.type=='cuda' else 'reference')
  if self.backend not in ('reference','combined'):raise ValueError('Use combined CUDA or explicit reference math')
  if self.backend=='combined' and self.device.type!='cuda':raise ValueError('Combined kernels require CUDA')
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
  torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
  with np.load(norm,allow_pickle=False) as z:self.norm={k:z[k].copy() for k in ('emg_mean','emg_std','force_mean','force_std')}
  for k,n in [('emg_mean',16),('emg_std',16),('force_mean',5),('force_std',5)]:
   a=self.norm[k]
   if a.shape!=(n,) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Invalid norm: '+k)
  if (self.norm['emg_std']<=0).any() or (self.norm['force_std']<=0).any():raise ValueError('Norm scales must be positive')
  payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
  binding=payload.get('binding',{})
  if binding.get('arm')!='candidate':raise ValueError('Expected a candidate three-view checkpoint')
  if binding.get('norm_sha256')!=hashlib.sha256(Path(norm).read_bytes()).hexdigest():raise ValueError('Checkpoint/norm mismatch')
  if not any(k.startswith('core.tokenizer.') for k in payload['model']):raise ValueError('Checkpoint is missing its frozen tokenizer')
  # Synthetic construction is immediately replaced by the complete, strictly
  # loaded checkpoint, including fitted codebooks/frame/Motion and view masks.
  self.model=ThreeViewModel(synthetic=True,backend=self.backend).to(self.device).eval()
  masks=self.model.core.view_masks.detach().cpu().clone()
  self.model.load_state_dict(payload['model'],strict=True)
  if not torch.equal(self.model.core.view_masks.cpu(),masks):raise ValueError('Unexpected fixed view masks')
  if not torch.equal(self.model.target_indices.cpu(),torch.arange(55)*40+1829):raise ValueError('Unexpected time grid')
  if any(p.dtype!=torch.float32 or not torch.isfinite(p).all() for p in self.model.parameters()):raise ValueError('Finite FP32 weights required')
 @torch.inference_mode()
 def predict(self,emg):
  """Filtered raw-scale uV float32 [B,4000,16] -> members [3,B,55,5]."""
  a=np.asarray(emg)
  if a.ndim!=3 or a.shape[1:]!=(4000,16) or not len(a) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Expected finite nonempty float32 [B,4000,16]')
  x=(a-self.norm['emg_mean'])/self.norm['emg_std']
  x=torch.from_numpy(np.ascontiguousarray(x.transpose(0,2,1))).to(self.device)
  members=self.model.members_prediction(x);ensemble=members.mean(0)
  scale=torch.as_tensor(self.norm['force_std'],device=self.device).view(1,5,1)
  mean=torch.as_tensor(self.norm['force_mean'],device=self.device).view(1,5,1)
  return dict(ensemble_normalized=ensemble.transpose(1,2).cpu().numpy(),ensemble_N=(ensemble*scale+mean).transpose(1,2).cpu().numpy(),members_normalized=members.permute(0,1,3,2).cpu().numpy(),members_N=(members*scale+mean).permute(0,1,3,2).cpu().numpy(),input_sample_indices=np.arange(55,dtype=np.int64)*40+1829)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('checkpoint','norm','emg','output'):p.add_argument('--'+name,required=True)
 p.add_argument('--device',default='cuda:0');p.add_argument('--backend',choices=('combined','reference'));p.add_argument('--batch-size',type=int,default=16)
 a=p.parse_args()
 if a.batch_size<1:p.error('--batch-size must be positive')
 target=Path(a.output)
 if target.exists():raise FileExistsError(target)
 emg=np.load(a.emg,mmap_mode='r',allow_pickle=False)
 if emg.dtype!=np.float32 or emg.ndim!=3 or emg.shape[1:]!=(4000,16) or not len(emg):raise ValueError('Expected float32 [B,4000,16]')
 runtime=ThreeViewInference(a.checkpoint,a.norm,a.device,a.backend)
 rows=[runtime.predict(emg[i:i+a.batch_size]) for i in range(0,len(emg),a.batch_size)]
 out={k:np.concatenate([v[k] for v in rows],axis=1 if k.startswith('members_') else 0) for k in ('ensemble_N','ensemble_normalized','members_N','members_normalized')}
 out['input_sample_indices']=rows[0]['input_sample_indices'];target.parent.mkdir(parents=True,exist_ok=True)
 with target.open('xb') as f:np.savez(f,**out)
if __name__=='__main__':main()
