"""Train-only frame EMA RQ and autonomous 50Hz Motion; Val MAE selects both."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import json,hashlib,time,random,traceback,sys
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'source_candidate'))
from io_local import read,write,sha,closed_rows
from fast_batches import Batcher
from source.rq_model import ForceTokenizer
from source_candidate.vendor.finger import PoseFrameDecoderFingers
from source_candidate.vendor.motion import MotionDecoder,decode_sequence
OUT=ROOT/'tokenizer';OUT.mkdir(exist_ok=True)
GRID=np.arange(55)*40+1829
class Frame(ForceTokenizer):
 def __init__(self):
  super().__init__()
  self.decoders=nn.ModuleList([nn.Sequential(nn.Linear(8,64),nn.LeakyReLU(.2),nn.Linear(64,1)) for _ in range(5)])
 def latent(self,y):
  zs=[]
  for f in range(5):
   z=self.encoders[f](y[:,f:f+1]);res=z;q=torch.zeros_like(z)
   for level in range(2):
    book=self.books[f,level];v=book[self.nearest(res,book)];q=q+v;res=res-v
   zs.append(q)
  return torch.cat(zs,-1)
def save(p,v):
 tmp=p.with_suffix('.tmp');torch.save(v,tmp)
 with tmp.open('rb') as stream:os.fsync(stream.fileno())
 tmp.replace(p)
def checkpoint(path,model,opt,epoch,history,best,stale,extra=None):
 save(path,dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,history=history,best_epoch_so_far=best['epoch'],best_value_so_far=best['value'],no_improve=stale,rng=dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state()),extra=extra))
def panels():
 path=OUT/'PANELS.npz'
 if path.exists():
  binding=read(OUT/'PANELS.json');assert sha(path)==binding['sha256'];return dict(np.load(path)),binding
 norm=dict(np.load(ROOT/'data/norm.npz'));records={};arrays={}
 for role in ['train','val']:
  b=Batcher(role,norm)
  if role=='train':specs=[(*b.index.sample(0,k),k) for k in range(2304)]
  else:
   rows=closed_rows()['val'];stages={x['stage'] for x in closed_rows()['train']};seen={};specs=[]
   for k,(i,start) in enumerate(b.index.fixed_windows()):
    row=rows[i];key=(row['subject'],row['stage'])
    if row['stage'] in stages and seen.get(key,0)<2:specs.append((i,start,k));seen[key]=seen.get(key,0)+1
  chunks=[];ids=[]
  for off in range(0,len(specs),64):
   q=b.prepare(specs[off:off+64]);v=q['raw'][...,GRID];assert np.isfinite(v).all();chunks.append(v.transpose(0,2,1))
   ids.extend([dict(id=i,start=s,subject=p,stage=g) for i,s,p,g in zip(q['ids'],q['starts'],q['subjects'],q['stages'])])
   write(OUT/'PROGRESS.json',dict(stage='read_'+role,windows=min(off+64,len(specs)),total=len(specs),time=time.time()))
  b.close();arrays[role]=np.concatenate(chunks);records[role]=ids
 assert set(x['subject'] for x in records['train']).isdisjoint(x['subject'] for x in records['val'])
 np.savez(path,**arrays);binding=dict(sha256=sha(path),norm_sha256=sha(ROOT/'data/norm.npz'),split_sha256=sha(ROOT/'data/FULL_SPLIT.json.gz'),label_policy_sha256=sha(ROOT/'LABEL_POLICY.json'),common_target_indices=GRID.tolist(),all_five_valid=True,continuous50Hz=True,records=records,Val_gradient_fit=False,Test_read=False)
 write(OUT/'PANELS.json',binding);return arrays,binding
def train_frame(arrays):
 torch.manual_seed(4307);m=Frame().cuda();train=torch.from_numpy(arrays['train'].reshape(-1,5)).cuda();val=torch.from_numpy(arrays['val'].reshape(-1,5)).cuda()
 norm=dict(np.load(ROOT/'data/norm.npz'));mean=torch.as_tensor(norm['force_mean'],device='cuda');std=torch.as_tensor(norm['force_std'],device='cuda');train=(train-mean)/std;val=(val-mean)/std
 m.initialize(train);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);history=[];best={'epoch':None,'value':float('inf')};stale=0
 for ep in range(1,51):
  m.train();order=torch.randperm(len(train),device='cuda');loss_sum=0
  for off in range(0,len(train),2048):
   y=train[order[off:off+2048]];opt.zero_grad(set_to_none=True);p,commit,_=m(y,update=True);loss=(p-y).abs().mean()+.25*commit;loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step();loss_sum+=float(loss.detach())
  m.eval();sums=torch.zeros(5,device='cuda')
  with torch.no_grad():
   for y in val.split(4096):p,_,_=m(y);sums+=(p-y).abs().sum(0)
  per=(sums/len(val)).cpu().tolist();score=sum(per)/5;improved=score<best['value'];stale=0 if improved else stale+1
  if improved:best=dict(epoch=ep,value=score)
  rec=dict(epoch=ep,Val_normalized_MAE=score,Val_per_finger_MAE=per,selection_metric='Val_normalized_MAE',direction='min',best_epoch_so_far=best['epoch'],best_value_so_far=best['value']);history.append(rec)
  checkpoint(OUT/'frame_last.pt',m,opt,ep,history,best,stale)
  if improved:checkpoint(OUT/'frame_best.pt',m,opt,ep,history,best,stale)
  write(OUT/'FRAME_HISTORY.json',history);write(OUT/'PROGRESS.json',dict(stage='frame',epoch=ep,Val=score,best=best,no_improve=stale,time=time.time()))
  if stale>=10:break
 payload=torch.load(OUT/'frame_best.pt',map_location='cpu',weights_only=False);m.load_state_dict(payload['model']);m.requires_grad_(False);m.eval();return m,norm,best
def train_motion(frame,norm,arrays,frame_best):
 torch.manual_seed(4309);m=MotionDecoder(128,5,40).cuda();opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4)
 mean=torch.as_tensor(norm['force_mean'],device='cuda');std=torch.as_tensor(norm['force_std'],device='cuda');data={}
 with torch.no_grad():
  for role in ['train','val']:
   y=(torch.from_numpy(arrays[role]).cuda()-mean)/std;n,t,f=y.shape;q=frame.latent(y.reshape(-1,5));r=torch.cat([d(q[:,i*8:(i+1)*8]) for i,d in enumerate(frame.decoders)],-1);data[role]=(q.reshape(n,t,40),r.reshape(n,t,5),y)
 history=[];best={'epoch':None,'value':float('inf')};stale=0
 for ep in range(1,51):
  m.train();q,r,y=data['train'];order=torch.randperm(len(y),device='cuda')
  for off in range(0,len(y),128):
   ix=order[off:off+128];opt.zero_grad(set_to_none=True);p=decode_sequence(m,q[ix],r[ix]);loss=(p-y[ix]).abs().mean();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step()
  m.eval();q,r,y=data['val'];sums=torch.zeros(5,device='cuda')
  with torch.no_grad():
   for off in range(0,len(y),128):p=decode_sequence(m,q[off:off+128],r[off:off+128]);sums+=(p-y[off:off+128]).abs().sum((0,1))
  per=(sums/(len(y)*55)).cpu().tolist();score=sum(per)/5;improved=score<best['value'];stale=0 if improved else stale+1
  if improved:best=dict(epoch=ep,value=score)
  rec=dict(epoch=ep,Val_normalized_MAE=score,Val_per_finger_MAE=per,selection_metric='Val_normalized_MAE',direction='min',best_epoch_so_far=best['epoch'],best_value_so_far=best['value']);history.append(rec)
  checkpoint(OUT/'motion_last.pt',m,opt,ep,history,best,stale)
  if improved:checkpoint(OUT/'motion_best.pt',m,opt,ep,history,best,stale)
  write(OUT/'MOTION_HISTORY.json',history);write(OUT/'PROGRESS.json',dict(stage='motion',epoch=ep,Val=score,best=best,no_improve=stale,time=time.time()))
  if stale>=10:break
 payload=torch.load(OUT/'motion_best.pt',map_location='cpu',weights_only=False);m.load_state_dict(payload['model']);m.requires_grad_(False);m.eval()
 output={'codebooks':frame.books.detach().cpu()};dest=PoseFrameDecoderFingers([[i] for i in range(5)],64,8)
 with torch.no_grad():
  for i,d in enumerate(frame.decoders):
   for src,layer in [(d[0],dest.nets[i].net[0].conv),(d[2],dest.nets[i].net[1].conv)]:layer.weight.copy_(src.weight.cpu().unsqueeze(-1));layer.bias.copy_(src.bias.cpu())
 output.update({'frame.'+k:v.cpu() for k,v in dest.state_dict().items()});output.update({'motion.'+k:v.cpu() for k,v in m.state_dict().items()})
 with torch.no_grad():
  q,r,y=data['val'];decoded=dest.cuda()(q[:32].transpose(1,2)).transpose(1,2);assert torch.allclose(decoded,r[:32],atol=1e-6,rtol=1e-6)
 binding=dict(norm_sha256=sha(ROOT/'data/norm.npz'),split_sha256=sha(ROOT/'data/FULL_SPLIT.json.gz'),label_policy_sha256=sha(ROOT/'LABEL_POLICY.json'),panel_sha256=sha(OUT/'PANELS.npz'),frame_best=frame_best,motion_best=best,frame_best_sha256=sha(OUT/'frame_best.pt'),motion_best_sha256=sha(OUT/'motion_best.pt'),Train_only_fit=True,Val_selection_only=True,GT_history=False,rate_hz=50,Test_read=False)
 save(OUT/'TOKENIZER_FROZEN.pt',dict(model=output,binding=binding));write(OUT/'TOKENIZER_FROZEN.json',dict(status='PASS_TRAIN_FIT_VAL_BEST_FRAME_AND_MOTION',sha256=sha(OUT/'TOKENIZER_FROZEN.pt'),binding=binding,time=time.time()))
def main():
 __import__('argparse').ArgumentParser(description=__doc__).parse_args()
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;random.seed(42);np.random.seed(42);torch.manual_seed(42)
 assert (ROOT/'data/norm.npz').is_file() and (ROOT/'data/INDEX.json.gz').is_file();assert not (OUT/'TOKENIZER_FROZEN.json').exists()
 assert not (ROOT/'TRAINING_STARTED.json').exists(),'Never refit an active experiment'
 arrays,_=panels();frame,norm,best=train_frame(arrays);train_motion(frame,norm,arrays,best)
if __name__=='__main__':
 try:main()
 except Exception:write(OUT/'FAILED.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
