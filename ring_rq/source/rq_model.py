"""Small Force adaptation of Pose Ring-RQ: G=5,L=2,K=128,D=8, soft readout."""
from types import MethodType
from pathlib import Path
import torch
from torch import nn

class ForceTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders=nn.ModuleList([nn.Sequential(nn.Linear(1,64),nn.SiLU(),nn.Linear(64,8)) for _ in range(5)])
        self.decoders=nn.ModuleList([nn.Sequential(nn.Linear(8,64),nn.SiLU(),nn.Linear(64,1)) for _ in range(5)])
        self.register_buffer('books',torch.zeros(5,2,128,8))
        self.register_buffer('usage',torch.ones(5,2,128))
        self.register_buffer('sums',torch.zeros(5,2,128,8))

    @staticmethod
    def nearest(z,c):
        return (z.square().sum(-1,keepdim=True)+c.square().sum(-1)[None]-2*z@c.T).argmin(-1)

    @torch.no_grad()
    def initialize(self,y):
        for g in range(5):
            v=y[:,g:g+1].sort(dim=0).values
            values=v[torch.linspace(0,len(v)-1,128,device=v.device).long()]
            z=self.encoders[g](y[:,g:g+1]);c=self.encoders[g](values)
            self.books[g,0].copy_(c)
            residual=z-c[self.nearest(z,c)]
            # Quantile of residual projection preserves both ends, deterministic.
            order=residual[:,0].argsort();indices=order[torch.linspace(0,len(order)-1,128,device=v.device).long()]
            self.books[g,1].copy_(residual[indices])
        self.sums.copy_(self.books);self.usage.fill_(1)

    def forward(self,y,update=False):
        outputs=[];commit=0.;hits=[]
        for g in range(5):
            z=self.encoders[g](y[:,g:g+1]);res=z.detach();total=torch.zeros_like(z)
            for level in range(2):
                book=self.books[g,level];ids=self.nearest(res,book);q=book[ids].clone()
                if update:
                    with torch.no_grad():
                        count=torch.bincount(ids,minlength=128).float()
                        sums=torch.zeros_like(book).index_add_(0,ids,res)
                        self.usage[g,level].mul_(.95).add_(count,alpha=.05)
                        self.sums[g,level].mul_(.95).add_(sums,alpha=.05)
                        book.copy_(self.sums[g,level]/self.usage[g,level].clamp_min(1e-5)[:,None])
                        dead=self.usage[g,level]<.01
                        if dead.any():
                            replacements=res[torch.arange(int(dead.sum()),device=res.device)*37%len(res)]
                            book[dead]=replacements;self.sums[g,level,dead]=replacements;self.usage[g,level,dead]=1
                total=total+q;res=res-q;hits.append(ids)
            commit=commit+(z-total.detach()).square().mean()/5
            outputs.append(self.decoders[g](z+(total-z).detach()))
        return torch.cat(outputs,-1),commit,hits

class RQReadout(nn.Module):
    def __init__(self,path):
        super().__init__();payload=torch.load(path,map_location='cpu',weights_only=False)
        tokenizer=ForceTokenizer();tokenizer.load_state_dict(payload['model'])
        self.register_buffer('books',tokenizer.books.clone())
        self.decoders=tokenizer.decoders
        self.requires_grad_(False);self.eval()

    def train(self,mode=True):
        return super().train(False)

    def forward(self,logits):
        with torch.autocast(device_type=logits.device.type,enabled=False):
            p=logits.float().unflatten(-1,(5,2,128)).softmax(-1)
            z=torch.einsum('btglk,glkd->btgd',p,self.books.float())
            return torch.cat([decoder(z[:,:,g]) for g,decoder in enumerate(self.decoders)],-1)

def direct_decode(self,fused):
    prediction=self.rq_readout(self.output_head(fused))
    return prediction.transpose(1,2),{}

def install(model,path):
    net=model.network
    net.output_head=nn.Linear(net.config.d_model,5*2*128)
    net.rq_readout=RQReadout(path)
    net.affine_decode=MethodType(direct_decode,net)
    return model

def assert_frozen(readout,path):
    # Frozen assets are small; exact check also runs at every full checkpoint.
    payload=torch.load(Path(path),map_location='cpu',weights_only=False)['model']
    assert torch.equal(readout.books.cpu(),payload['books'])
    for k,v in readout.decoders.state_dict().items():assert torch.equal(v.cpu(),payload['decoders.'+k])
    assert not readout.training and all(p.grad is None and not p.requires_grad for p in readout.parameters())
