"""Force3 Ring-RQ causal streaming replay; W100 decoder KV, continuous encoder.
No MA filter. Encoder recurrent states and affine state remain continuous.
Independent evaluation copy; never patches live training or immutable assets.
"""
import sys
import torch
import torch.nn.functional as F

class ChunkStream:
    def __init__(self,net,capacity,window=295):
        self.window=window
        self.net=net;self.capacity=capacity;self.i=0;self.buffer=torch.zeros((1,16,128),device='cuda');self.previous=None
        self.temporal=[];self.kv=[None for _ in net.decoder_blocks]
        for block in net.encoder_blocks:
            a=block.temporal_attention
            self.temporal.append((torch.zeros((16,a.num_heads,a.head_dim,a.head_dim),device='cuda'),torch.zeros((16,a.num_heads,a.head_dim),device='cuda')))

    def rope(self,x):
        d=x.shape[-1];pos=torch.arange(self.i,self.i+x.shape[-2],device=x.device,dtype=torch.float32)
        inv=10000.**(-torch.arange(0,d,2,device=x.device,dtype=torch.float32)/d);phase=pos[:,None]*inv[None]
        co=phase.cos()[None,None];si=phase.sin()[None,None];a=x[...,0::2];b=x[...,1::2]
        return torch.stack((a*co-b*si,a*si+b*co),dim=-1).flatten(-2)

    def push(self,new):
        assert new.shape[:2]==(1,16) and new.dtype==torch.float32
        size=(new.shape[-1]+10)//40 if self.i==0 else new.shape[-1]//40
        assert size>0 and new.shape[-1]==size*40-(10 if self.i==0 else 0) and self.i+size<=self.capacity
        n=self.net;merged=torch.cat((self.buffer,new),-1)
        patches=merged[...,30 if self.i==0 else 40:].unfold(-1,128,40)
        assert patches.shape==(1,16,size,128)
        self.buffer=merged[...,-128:].clone();pe=n.patch_embedding;weights=pe.branch_logits.softmax(0)
        tokens=pe.output_norm(weights[0]*pe.branch40(patches[...,-46:-6])+weights[1]*pe.branch128(patches)).permute(0,2,1,3)
        tokens=tokens+n.electrode_fourier[None,None]
        for j,b in enumerate(n.encoder_blocks):
            spatial=tokens.reshape(size,16,n.config.d_model);z=b.spatial_norm(spatial);v,_=b.spatial_attention(z,z,z,need_weights=False)
            spatial=spatial+b.spatial_dropout(v);temporal=spatial.reshape(1,size,16,n.config.d_model).permute(0,2,1,3).reshape(16,size,n.config.d_model)
            x=b.temporal_norm(temporal);att=b.temporal_attention;qkv=att.qkv(x).reshape(16,size,3,att.num_heads,att.head_dim).permute(2,0,3,1,4)
            q,k,v=qkv.unbind(0);q=F.elu(q)+1.;k=F.elu(k)+1.;ks,kn=self.temporal[j];outs=[]
            for start in range(0,size,att.chunk_size):
                end=min(start+att.chunk_size,size);o,ks,kn=att._chunk_forward(q[:,:,start:end],k[:,:,start:end],v[:,:,start:end],att.decay_logits.sigmoid(),ks,kn);outs.append(o)
            self.temporal[j]=(ks,kn);o=torch.cat(outs,2).permute(0,2,1,3).reshape(16,size,n.config.d_model)
            temporal=temporal+att.output_dropout(att.output(o));temporal=temporal+b.ffn(b.ffn_norm(temporal));tokens=temporal.reshape(1,16,size,n.config.d_model).permute(0,2,1,3)
        tokens=n.encoder_norm(tokens);values=tokens.reshape(size,16,n.config.d_model)
        pooled,_=n.channel_pool(n.pool_query.expand(size,-1,-1),values,values,need_weights=False)
        streams=n.decoder_projection(pooled).reshape(1,size,2,n.config.d_model).permute(0,2,1,3).reshape(2,size,n.config.d_model)
        end=self.i+size
        for j,b in enumerate(n.decoder_blocks):
            qkv=b.qkv(b.attention_norm(streams)).reshape(2,size,3,b.num_heads,b.head_dim).permute(2,0,3,1,4)
            q,k,v=qkv.unbind(0);q=self.rope(q);k=self.rope(k)
            old=self.kv[j];previous=0 if old is None else old[0].shape[-2]
            keys=k if old is None else torch.cat((old[0],k),2)
            values=v if old is None else torch.cat((old[1],v),2)
            qids=torch.arange(self.i,end,device=new.device)[:,None]
            kids=torch.arange(self.i-previous,end,device=new.device)[None,:]
            mask=kids<=qids
            if self.window is not None:mask=mask & (kids>=qids-self.window+1)
            o=F.scaled_dot_product_attention(q,keys,values,attn_mask=mask[None,None],dropout_p=0.,is_causal=False).transpose(1,2).reshape(2,size,n.config.d_model)
            if self.window is None:self.kv[j]=(keys,values)
            else:self.kv[j]=(keys[:,:,-(self.window-1):].clone(),values[:,:,-(self.window-1):].clone())
            streams=streams+b.residual_dropout(b.output(o));streams=streams+b.ffn(b.ffn_norm(streams))
        streams=n.decoder_norm(streams).reshape(1,2,size,n.config.d_model).permute(0,2,1,3)
        trim=max(0,min(size,n.config.context_tokens-self.i));first_valid=self.i+trim;self.i=end
        if trim==size:return torch.empty((0,5),device='cuda')
        fused,_=n.fuse_streams(streams[:,trim:])
        return n.rq_readout(n.output_head(fused))[0]
