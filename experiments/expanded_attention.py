"""Exact inference contraction for a linear expansion followed by self-attention.

Input features have width r, an affine map expands them to d, and attention
then uses head width c. The output projection is included. This representation
is profitable only when its actual resulting tensor count is smaller.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import to_dense_batch


def _copy(target,value):
    target.copy_(value.to(dtype=target.dtype).to(device=target.device))


def _linear(in_features,out_features,*,bias=True,dtype):
    # Construct on CPU and restore its RNG state. No device RNG is touched.
    with torch.random.fork_rng(devices=[]):
        return nn.Linear(in_features,out_features,bias=bias,device="cpu",dtype=dtype)


class ExpandedInputSelfAttention(nn.Module):
    """Consume raw features for a projection-before-padding self-attention.

    ``padding_after_projection=True`` is an explicit contract: padded raw rows
    are zero, and the original projected rows were replaced by zero afterwards.
    This changes the inner module input width; the enclosing encoder adapter
    preserves the public model API. Unsupported behavior is rejected.
    """
    def __init__(self,projection,attention,*,padding_after_projection):
        super().__init__()
        if padding_after_projection is not True:
            raise ValueError("Requires explicit padding_after_projection=True contract")
        if type(projection) is not nn.Linear or type(attention) is not nn.MultiheadAttention:
            raise TypeError("Requires standard Linear and MultiheadAttention")
        for parent in (projection,attention):
            for module in parent.modules():
                if "forward" in vars(module) or any(getattr(module,name,{}) for name in
                    ("_forward_hooks","_forward_pre_hooks","_backward_hooks","_backward_pre_hooks")):
                    raise ValueError("Hooks and instance forward overrides are not supported")
        if not attention.batch_first or not attention._qkv_same_embed_dim:
            raise ValueError("Requires batch-first self attention with shared qkv width")
        if attention.add_zero_attn or attention.bias_k is not None or attention.bias_v is not None:
            raise ValueError("Extra key/value tokens are not supported")
        D=attention.embed_dim;H=attention.num_heads;C=D//H;R=projection.in_features
        if projection.out_features!=D:
            raise ValueError("Projection output must equal attention embed_dim")
        device,dtype=attention.in_proj_weight.device,attention.in_proj_weight.dtype
        parameters=list(projection.parameters())+list(attention.parameters())
        if device.type=="meta" or any(p.device!=device or p.dtype!=dtype for p in parameters):
            raise ValueError("Source parameters must share one actual device and dtype")
        if dtype not in {torch.float16,torch.bfloat16,torch.float32,torch.float64}:
            raise ValueError("Unsupported source weight dtype")
        if any(not p.detach().cpu().isfinite().all().item() for p in parameters):
            raise ValueError("Source parameters must be finite")
        A=projection.weight.detach().cpu().double()
        a=torch.zeros(D,dtype=torch.double) if projection.bias is None else projection.bias.detach().cpu().double()
        W=attention.in_proj_weight.detach().cpu().double().reshape(3,D,D)
        b=torch.zeros(3,D,dtype=torch.double) if attention.in_proj_bias is None else attention.in_proj_bias.detach().cpu().double().reshape(3,D)
        q,k,v=(W@A).reshape(3,H,C,R)
        qb,kb,vb=(torch.einsum("sij,j->si",W,a)+b).reshape(3,H,C)
        O=attention.out_proj.weight.detach().cpu().double().reshape(D,H,C)
        self.input_dim=R;self.embed_dim=D;self.num_heads=H;self.head_dim=C
        self.batch_first=True;self.dropout=attention.dropout
        self.score=_linear(R,H*R,dtype=dtype)
        self.padding_score=nn.Parameter(torch.empty(H,R,device="cpu",dtype=dtype))
        self.output=_linear(H*(R+1),D,bias=attention.out_proj.bias is not None,dtype=dtype)
        with torch.no_grad():
            _copy(self.score.weight,torch.einsum("hci,hcj->hij",k,q).reshape(H*R,R))
            _copy(self.score.bias,torch.einsum("hci,hc->hi",k,qb).reshape(H*R))
            _copy(self.padding_score,torch.einsum("hci,hc->hi",k,b[0].reshape(H,C)))
            ov=torch.einsum("dhc,hcr->dhr",O,v)
            ovb=torch.einsum("dhc,hc->dh",O,vb).unsqueeze(-1)
            _copy(self.output.weight,torch.cat([ov,ovb],dim=-1).reshape(D,H*(R+1)))
            if attention.out_proj.bias is not None:
                _copy(self.output.bias,attention.out_proj.bias.detach().cpu().double())
        score_weight_grad=projection.weight.requires_grad or attention.in_proj_weight.requires_grad
        attention_bias_grad=attention.in_proj_bias is not None and attention.in_proj_bias.requires_grad
        projection_bias_grad=projection.bias is not None and projection.bias.requires_grad
        score_bias_grad=score_weight_grad or attention_bias_grad or projection_bias_grad
        self.score.weight.requires_grad_(score_weight_grad)
        self.score.bias.requires_grad_(score_bias_grad)
        self.padding_score.requires_grad_(attention.in_proj_bias is not None and (score_weight_grad or attention_bias_grad))
        self.output.weight.requires_grad_(score_bias_grad or attention.out_proj.weight.requires_grad)
        if self.output.bias is not None:
            self.output.bias.requires_grad_(attention.out_proj.bias.requires_grad)
        self.to(device=device)
        self.train(attention.training)

    def forward(self,x,key_padding_mask=None,need_weights=True,average_attn_weights=True):
        if x.ndim!=3 or x.shape[-1]!=self.input_dim:
            raise ValueError("Expected batch-first raw input features")
        B,L,R=x.shape;H=self.num_heads
        query=self.score(x).reshape(B,L,H,R).transpose(1,2)
        if key_padding_mask is not None:
            if key_padding_mask.dtype is not torch.bool or key_padding_mask.shape!=(B,L):
                raise ValueError("Requires boolean key padding mask")
            if bool(torch.any(x[key_padding_mask]!=0)):
                raise ValueError("Raw padded rows must be zero under padding_after_projection=True")
            # Source code pads AFTER its affine input projection. These query
            # rows therefore use q=bq, not q=Q*a+bq. Retain that distinction.
            query=torch.where(key_padding_mask[:,None,:,None],self.padding_score[None,:,None,:],query)
        scores=torch.matmul(query,x[:,None,:,:].transpose(-1,-2))*(self.head_dim**-0.5)
        if key_padding_mask is not None:
            scores=scores.masked_fill(key_padding_mask[:,None,None,:],float("-inf"))
        alpha=scores.softmax(-1)
        weights=F.dropout(alpha.reshape(B*H,L,L),p=self.dropout,training=self.training).reshape(B,H,L,L)
        ones=torch.ones(B,L,1,device=x.device,dtype=x.dtype)
        augmented=torch.cat([x,ones],dim=-1)
        summary=torch.matmul(weights,augmented[:,None,:,:])
        output=self.output(summary.transpose(1,2).reshape(B,L,H*(R+1)))
        if not need_weights:
            return output,None
        return output,weights.mean(1) if average_attn_weights else weights


class ContractedAtomsEncoder(nn.Module):
    def __init__(self,source):
        super().__init__()
        self.input_proj=source.input_proj
        self.self_attn_layers=source.self_attn_layers
        self.norms1=source.norms1;self.norms2=source.norms2
        self.drop=source.drop;self.ffns=source.ffns
        self.pooling=source.pooling;self.out=source.out
        self.self_attn_layers[0]=ExpandedInputSelfAttention(self.input_proj,self.self_attn_layers[0],padding_after_projection=True)
        self.train(source.training)

    def forward(self,x,batch):
        raw,mask=to_dense_batch(x,batch)
        dense,_=to_dense_batch(self.input_proj(x),batch)
        for index,(attention,norm1,norm2,ffn) in enumerate(zip(self.self_attn_layers,self.norms1,self.norms2,self.ffns)):
            if index==0:
                update,_=attention(raw,key_padding_mask=~mask)
            else:
                update,_=attention(dense,dense,dense,key_padding_mask=~mask)
            dense=norm1(dense+self.drop(update))
            dense=norm2(dense+ffn(dense))
        pooled=self.pooling(dense[mask],batch)
        return self.out(pooled)


def contract_moljepa_uma(model,inplace=False):
    converted=model if inplace else copy.deepcopy(model)
    base=converted.backbone if hasattr(converted,"backbone") else converted
    before=sum(p.numel() for p in converted.parameters())
    if "uma" not in base.model.encoders:
        return converted,{"before":before,"after":before,"saved":0,"skipped":"UMA encoder absent"}
    source=base.model.encoders["uma"]
    base.model.encoders["uma"]=ContractedAtomsEncoder(source)
    after=sum(p.numel() for p in converted.parameters())
    return converted,{"before":before,"after":after,"saved":before-after,
                      "guarantee":"exact affine/bilinear contraction in real arithmetic; all modalities preserved"}
