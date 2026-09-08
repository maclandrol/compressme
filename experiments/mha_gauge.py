"""Exact MHA gauge fixing with implicit identity submatrices.

The compact checkpoint omits r*r coefficients per head for Q and V. Evaluation
materializes and caches dense effective Q/K/V weights to retain native PyTorch
fast paths; this is checkpoint compression, not a resident-memory reduction.
"""
import copy
import numpy as np
import scipy.linalg
import torch
from torch import nn
from torch.nn import functional as F


def _fix(matrix,max_condition):
    _,_,columns=scipy.linalg.qr(matrix.numpy(),pivoting=True,mode="economic")
    rank=matrix.shape[0];pivots=np.array(columns[:rank],copy=True)
    other=np.array(sorted(set(range(matrix.shape[1]))-set(pivots)),dtype=np.int64)
    pivot=matrix[:,pivots]
    condition=float(torch.linalg.cond(pivot))
    if not np.isfinite(condition) or condition>max_condition:
        raise ValueError(f"Unstable pivot block condition={condition:.3g}")
    normalized=torch.linalg.solve(pivot,matrix)
    return normalized[:,other],torch.tensor(pivots),torch.tensor(other),pivot,condition


class GaugeFixedMHA(nn.Module):
    merge_masks=nn.MultiheadAttention.merge_masks

    def __init__(self,source,max_condition=1000.0):
        super().__init__()
        if type(source) is not nn.MultiheadAttention or not source._qkv_same_embed_dim:
            raise TypeError("Requires standard common-width MultiheadAttention")
        if not source.batch_first or source.bias_k is not None or source.bias_v is not None or source.add_zero_attn:
            raise ValueError("Unsupported MHA layout or added tokens")
        if any("forward" in vars(m) or m._forward_hooks or m._forward_pre_hooks for m in source.modules()):
            raise ValueError("Hooked or overridden attention is not supported")
        D,H=source.embed_dim,source.num_heads;C=D//H
        device,dtype=source.in_proj_weight.device,source.in_proj_weight.dtype
        w=source.in_proj_weight.detach().cpu().double().reshape(3,H,C,D)
        b=torch.zeros(3,H,C,dtype=torch.double) if source.in_proj_bias is None else source.in_proj_bias.detach().cpu().double().reshape(3,H,C)
        old_out=source.out_proj.weight.detach().cpu().double().reshape(D,H,C)
        qf=[];vf=[];qp=[];vp=[];qr=[];vr=[];keys=[];qb=[];vb=[];outs=[];conditions=[]
        for head in range(H):
            fq,pq,rq,P,cq=_fix(w[0,head],max_condition)
            fv,pv,rv,T,cv=_fix(w[2,head],max_condition)
            qf.append(fq);vf.append(fv);qp.append(pq);vp.append(pv);qr.append(rq);vr.append(rv)
            keys.append(P.T@w[1,head]);qb.append(torch.linalg.solve(P,b[0,head]));vb.append(torch.linalg.solve(T,b[2,head]))
            outs.append(old_out[:,head]@T);conditions.append({"query":cq,"value":cv})
        def parameter(value,requires_grad):
            return nn.Parameter(value.to(dtype=dtype).to(device=device),requires_grad=requires_grad)
        self.q_free=parameter(torch.stack(qf),source.in_proj_weight.requires_grad)
        self.v_free=parameter(torch.stack(vf),source.in_proj_weight.requires_grad)
        self.k_weight=parameter(torch.stack(keys),source.in_proj_weight.requires_grad)
        grad_bias=source.in_proj_bias is not None and source.in_proj_bias.requires_grad
        self.q_bias=parameter(torch.stack(qb),grad_bias)
        self.v_bias=parameter(torch.stack(vb),grad_bias)
        for name,value in (("q_pivots",qp),("v_pivots",vp),("q_other",qr),("v_other",vr)):
            self.register_buffer(name,torch.stack(value).to(device=device))
        with torch.random.fork_rng(devices=[]):
            self.out_proj=nn.Linear(D,D,bias=source.out_proj.bias is not None,device="cpu",dtype=dtype)
        with torch.no_grad():
            self.out_proj.weight.copy_(torch.stack(outs,dim=1).reshape(D,D).to(dtype=dtype))
            if source.out_proj.bias is not None:
                self.out_proj.bias.copy_(source.out_proj.bias.detach().cpu())
        self.out_proj.to(device=device)
        self.out_proj.weight.requires_grad_(source.out_proj.weight.requires_grad or source.in_proj_weight.requires_grad)
        if self.out_proj.bias is not None: self.out_proj.bias.requires_grad_(source.out_proj.bias.requires_grad)
        self.embed_dim=D;self.num_heads=H;self.head_dim=C;self.batch_first=True
        self.kdim=D;self.vdim=D;self._qkv_same_embed_dim=True
        self.dropout=source.dropout;self.bias_k=None;self.bias_v=None;self.add_zero_attn=False
        self.conditions=conditions;self._dense_cache=None;self._dense_versions=None
        self.train(source.training)

    def _effective(self,free,pivots,other):
        H,C,_=free.shape;D=self.embed_dim
        output=free.new_zeros(H,C,D)
        output=output.scatter(-1,other[:,None,:].expand(H,C,D-C),free)
        identity=torch.eye(C,device=free.device,dtype=free.dtype).expand(H,C,C)
        output=output.scatter(-1,pivots[:,None,:].expand(H,C,C),identity)
        return output.reshape(D,D)

    @property
    def in_proj_weight(self):
        sources=(self.q_free,self.k_weight,self.v_free,self.q_pivots,self.q_other,self.v_pivots,self.v_other)
        versions=tuple((t._version,str(t.device),str(t.dtype),t.data_ptr()) for t in sources)
        cache=not self.training and not torch.is_grad_enabled()
        if cache and self._dense_cache is not None and versions==self._dense_versions:
            return self._dense_cache
        weight=torch.cat([self._effective(self.q_free,self.q_pivots,self.q_other),
                          self.k_weight.reshape(self.embed_dim,self.embed_dim),
                          self._effective(self.v_free,self.v_pivots,self.v_other)])
        if cache:
            self._dense_cache=weight;self._dense_versions=versions
        return weight

    @property
    def in_proj_bias(self):
        return torch.cat([self.q_bias.flatten(),torch.zeros_like(self.q_bias).flatten(),self.v_bias.flatten()])

    def forward(self,query,key,value,key_padding_mask=None,need_weights=True,attn_mask=None,
                average_attn_weights=True,is_causal=False):
        output,weights=F.multi_head_attention_forward(
            query.transpose(0,1),key.transpose(0,1),value.transpose(0,1),
            self.embed_dim,self.num_heads,self.in_proj_weight,self.in_proj_bias,
            None,None,False,self.dropout,self.out_proj.weight,self.out_proj.bias,
            training=self.training,key_padding_mask=key_padding_mask,need_weights=need_weights,
            attn_mask=attn_mask,average_attn_weights=average_attn_weights,is_causal=is_causal)
        return output.transpose(0,1),weights


def fix_moljepa_transformer_gauge(model,inplace=False,max_condition=1000.0):
    converted=model if inplace else copy.deepcopy(model)
    root=converted.backbone if hasattr(converted,"backbone") else converted
    before=sum(p.numel() for p in converted.parameters())
    reports=[]
    for layer in root.model.transformer_head.transformer.layers:
        layer.self_attn=GaugeFixedMHA(layer.self_attn,max_condition=max_condition)
        reports.append(layer.self_attn.conditions)
    return converted,{"before":before,"after":sum(p.numel() for p in converted.parameters()),
                      "conditions":reports,"scope":"checkpoint compression; dense runtime cache"}
