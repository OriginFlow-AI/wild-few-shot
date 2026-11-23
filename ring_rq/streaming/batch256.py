"""Fixed256 Encoder Graph + exact-shape eager tails; one loaded immutable model.

Canonical batch replay: zero or more full256 blocks and at most one terminal
short block. Never pad a tail. Decoder/KV/Affine equations stay outside Graph.
"""
import ast
import inspect
import sys
import textwrap
import time

import torch
import torch.nn.functional as F


def make_batch_class(base):
    tree=ast.parse(textwrap.dedent(inspect.getsource(base.push)))
    nodes=tree.body[0].body
    def assignment(node,name):
        return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)
    start=next(i for i,n in enumerate(nodes) if assignment(n,'n'))
    stop=next(i for i,n in enumerate(nodes) if assignment(n,'end'))
    encoder=ast.unparse(ast.Module(body=nodes[start:stop],type_ignores=[]))
    changes={
        '30 if self.i == 0 else 40':'offset',
        'self.buffer = merged[..., -128:].clone()':'self.buffer.copy_(merged[..., -128:])',
        'weights = pe.branch_logits.softmax(0)':'weights = self.patch_weights',
        'att.decay_logits.sigmoid()':'self.decays[j]',
        'self.temporal[j] = (ks, kn)':'self.temporal[j][0].copy_(ks)\n    self.temporal[j][1].copy_(kn)',
    }
    for a,b in changes.items():
        assert encoder.count(a)==1,a
        encoder=encoder.replace(a,b)
    encoder='def encode(self,new,size,offset):\n'+textwrap.indent(encoder,'    ')+'\n    return streams\n'
    tree.body[0].body=nodes[:start]+ast.parse('n = self.net\nstreams = self.encoder_step(new,size)').body+nodes[stop:]
    decoder=next(n for n in tree.body[0].body if isinstance(n,ast.For) and ast.unparse(n.iter)=='enumerate(n.decoder_blocks)')
    left=next(i for i,n in enumerate(decoder.body) if assignment(n,'qids'))
    right=next(i for i,n in enumerate(decoder.body) if assignment(n,'o'))
    decoder.body[left:right]=ast.parse('mask = self.mask_for(size,previous)').body
    scope=dict(torch=torch,F=F,sys=sys)
    exec(compile(encoder+'\n'+ast.unparse(tree),'<ex45_batch256_graph>','exec'),scope)

    class Batch256Graph(base):
        def __init__(self,net,capacity,window=100):
            assert net.config.d_model==304 and len(net.encoder_blocks)==9 and len(net.decoder_blocks)==4
            assert window==100 and all(b.head_dim==38 for b in net.decoder_blocks)
            super().__init__(net,capacity,window)
            tick=time.perf_counter()
            self.patch_weights=net.patch_embedding.branch_logits.softmax(0)
            self.decays=[b.temporal_attention.decay_logits.sigmoid() for b in net.encoder_blocks]
            self.inv=10000.**(-torch.arange(0,38,2,device='cuda',dtype=torch.float32)/38)
            self.masks={};self.mask_for(256,0);self.mask_for(256,99)
            self.graph_input=torch.zeros((1,16,256*40),device='cuda',dtype=torch.float32)
            self.capture_count=0;self.full_replays=0;self.tail_calls=0
            self.reset()
            warm=torch.cuda.Stream();warm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm):
                for _ in range(3):self.encode(self.graph_input,256,40)
            torch.cuda.current_stream().wait_stream(warm)
            self.reset();torch.cuda.synchronize()
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):self.encoded=self.encode(self.graph_input,256,40)
            self.capture_count+=1;self.reset();torch.cuda.synchronize()
            self.startup_seconds=time.perf_counter()-tick
            self.fixed_addresses=self.addresses()

        def addresses(self):
            return [self.buffer.data_ptr(),self.graph_input.data_ptr()]+[t.data_ptr() for pair in self.temporal for t in pair]

        def reset(self):
            self.i=0;self.previous=None;self.kv=[None for _ in self.net.decoder_blocks]
            self.tail_seen=False;self.rope_key=None;self.co=self.si=None
            self.buffer.zero_();self.graph_input.zero_()
            for a,b in self.temporal:a.zero_();b.zero_()
            if hasattr(self,'fixed_addresses'):assert self.addresses()==self.fixed_addresses

        def mask_for(self,size,previous):
            assert 1<=size<=256 and previous in (0,99)
            key=(size,previous)
            if key not in self.masks:
                q=torch.arange(size,device='cuda')[:,None]
                k=torch.arange(-previous,size,device='cuda')[None,:]
                self.masks[key]=(k<=q)&(k>=q-self.window+1)
            return self.masks[key]

        def rope(self,x):
            key=(self.i,x.shape[-2]);assert x.shape[-1]==38
            if self.rope_key!=key:
                pos=torch.arange(self.i,self.i+x.shape[-2],device=x.device,dtype=torch.float32)
                phase=pos[:,None]*self.inv[None]
                self.co=phase.cos()[None,None];self.si=phase.sin()[None,None];self.rope_key=key
            a,b=x[...,0::2],x[...,1::2]
            return torch.stack((a*self.co-b*self.si,a*self.si+b*self.co),dim=-1).flatten(-2)

        def encoder_step(self,new,size):
            assert 1<=size<=256 and not self.tail_seen,'canonical full blocks followed by one terminal tail only'
            if size==256:
                if self.i==0:
                    self.graph_input.zero_();self.graph_input[...,-new.shape[-1]:].copy_(new)
                else:self.graph_input.copy_(new)
                self.graph.replay();self.full_replays+=1
                return self.encoded
            self.tail_seen=True;self.tail_calls+=1
            return self.encode(new,size,30 if self.i==0 else 40)

    Batch256Graph.encode=scope['encode'];Batch256Graph.push=scope['push']
    Batch256Graph.encoder_generated_source=encoder
    return Batch256Graph
