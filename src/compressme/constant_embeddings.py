"""Lossless storage deduplication for frozen, bitwise-identical embedding rows.

The token-index contract remains intact: ``weight`` is the original vocabulary
shape and forward calls ordinary torch.nn.functional.embedding. No row is
rounded or discarded. Expanded views share one stored row; operations that
explicitly ask for a contiguous full weight may materialize that matrix.

This is a frozen representation. It does not support subsequently unfreezing
individual vocabulary rows, max_norm's in-place renormalization, sparse
gradients, custom hooks, or externally shared original Parameters.
The weight view preserves shape and values, not contiguous strides; custom
direct-weight consumers must support expanded tensors or be validated separately.
"""
from __future__ import annotations
from collections import Counter
import math
import torch
from torch import nn
from torch.nn import functional as F

_DTYPES=(torch.float16,torch.bfloat16,torch.float32,torch.float64)

def _has_hooks(module):
    return any(bool(value) for name,value in vars(module).items()
               if name.endswith('_hooks') and isinstance(value,dict))

def _validate_source(source):
    if type(source) is not nn.Embedding:
        raise ValueError('Only unmodified torch.nn.Embedding is supported')
    if _has_hooks(source):
        raise ValueError('Embedding hooks require a separate adapter')
    if 'forward' in vars(source):
        raise ValueError('Instance-overridden embedding forward is unsupported')
    if source.weight.requires_grad:
        raise ValueError('Embedding must already be frozen')
    if source.max_norm is not None or source.sparse:
        raise ValueError('max_norm and sparse embeddings are unsupported')
    if source.weight.layout!=torch.strided or source.weight.dtype not in _DTYPES or source.weight.device.type=='meta':
        raise ValueError('Expected an ordinary real floating-point embedding')
    if tuple(source.weight.shape)!=(source.num_embeddings,source.embedding_dim):
        raise ValueError('Embedding attributes and weight shape differ')
    if source.num_embeddings<=1 or source.embedding_dim<=0:
        raise ValueError('No row storage saving is possible')
    if source.padding_idx is not None and not -source.num_embeddings<=source.padding_idx<source.num_embeddings:
        raise ValueError('Embedding padding index is outside the vocabulary')
    if not isinstance(source.scale_grad_by_freq,bool):
        raise ValueError('Expected an ordinary boolean scale_grad_by_freq')

class ConstantRowEmbedding(nn.Module):
    """Frozen embedding backed by one row, retaining the full-shaped weight view."""
    def __init__(self,row,num_embeddings,*,padding_idx=None,norm_type=2.0,scale_grad_by_freq=False):
        super().__init__()
        if not isinstance(row,torch.Tensor) or row.ndim!=2 or row.shape[0]!=1 or row.shape[1]<1:
            raise ValueError('Expected one nonempty row with shape (1, embedding_dim)')
        if row.layout!=torch.strided or row.dtype not in _DTYPES:
            raise ValueError('Expected an ordinary real floating-point row')
        if not isinstance(num_embeddings,int) or isinstance(num_embeddings,bool) or num_embeddings<1:
            raise ValueError('num_embeddings must be a positive integer')
        if padding_idx is not None and (not isinstance(padding_idx,int) or isinstance(padding_idx,bool)
                                       or not -num_embeddings<=padding_idx<num_embeddings):
            raise ValueError('padding_idx must be an integer in the vocabulary or None')
        if not isinstance(norm_type,(int,float)) or isinstance(norm_type,bool) or not math.isfinite(norm_type):
            raise ValueError('norm_type must be a finite real number')
        if not isinstance(scale_grad_by_freq,bool):
            raise ValueError('scale_grad_by_freq must be boolean')
        if not torch.isfinite(row.detach().cpu()).all().item():
            raise ValueError('Nonfinite embedding values are unsupported')
        self.num_embeddings=num_embeddings
        self.embedding_dim=row.shape[1]
        self.padding_idx=padding_idx
        self.max_norm=None
        self.norm_type=norm_type
        self.scale_grad_by_freq=scale_grad_by_freq
        self.sparse=False
        self._row=nn.Parameter(row.detach().clone().contiguous(),requires_grad=False)

    @classmethod
    def from_embedding(cls,source):
        _validate_source(source)
        # No combined device/dtype conversion: copy bits at the original dtype.
        stored=source.weight.detach().cpu().contiguous()
        if not torch.isfinite(stored).all().item():
            raise ValueError('Nonfinite embedding values are unsupported')
        raw=stored.view(torch.uint8).reshape(source.num_embeddings,-1)
        if not torch.equal(raw,raw[:1].expand_as(raw)):
            raise ValueError('Embedding rows are not bitwise identical')
        result=cls(source.weight[:1],source.num_embeddings,padding_idx=source.padding_idx,
                   norm_type=source.norm_type,scale_grad_by_freq=source.scale_grad_by_freq)
        result.train(source.training)
        return result

    @property
    def weight(self):
        return self._row.expand(self.num_embeddings,-1)

    def requires_grad_(self,requires_grad=True):
        if requires_grad:
            raise ValueError('Constant-row storage requires frozen weights; materialize nn.Embedding to unfreeze')
        return super().requires_grad_(False)

    def forward(self,input):
        if self._row.requires_grad:
            raise ValueError('Constant-row storage requires frozen weights; materialize nn.Embedding to unfreeze')
        if self.max_norm is not None or self.sparse:
            raise ValueError('Constant-row storage does not support max_norm or sparse gradients')
        return F.embedding(input,self.weight,self.padding_idx,self.max_norm,self.norm_type,
                           self.scale_grad_by_freq,self.sparse)

    def extra_repr(self):
        return (f'{self.num_embeddings}, {self.embedding_dim}, padding_idx={self.padding_idx}, '
                f'stored_rows=1, frozen=True')

    def recipe(self):
        """JSON-safe replay metadata; state_dict stores only ``_row``.

        Apply this shape recipe BEFORE strict state_dict loading. A freshly
        constructed original embedding need not itself have constant rows.
        The artifact's one stored row is authoritative after loading.
        """
        return {'kind':'ConstantRowEmbedding','num_embeddings':self.num_embeddings,
                'embedding_dim':self.embedding_dim,'padding_idx':self.padding_idx,
                'norm_type':self.norm_type,'scale_grad_by_freq':self.scale_grad_by_freq,
                'dtype':str(self._row.dtype).removeprefix('torch.'),'training':self.training}

    @classmethod
    def from_recipe(cls,spec,*,device='cpu'):
        if spec.get('kind')!='ConstantRowEmbedding':
            raise ValueError('Unsupported embedding recipe')
        dtype=getattr(torch,spec['dtype'],None)
        if dtype not in _DTYPES:
            raise ValueError('Unsupported embedding recipe dtype')
        # Initialize only one row; strict state loading supplies its true bits.
        result=cls(torch.zeros(1,spec['embedding_dim'],dtype=dtype,device=device),
                   spec['num_embeddings'],padding_idx=spec['padding_idx'],
                   norm_type=spec['norm_type'],scale_grad_by_freq=spec['scale_grad_by_freq'])
        return result.train(spec.get('training',False))

def deduplicate_embeddings(model,*,inplace=False):
    """Replace profitable eligible embeddings without changing token inputs.

    Returns ``(model, report)``. Shared modules/Parameters are retained because
    replacing one side would break alias semantics or leave dense storage alive.
    """
    from .compiler import _copy_with_compiler_metadata
    param_aliases=Counter(id(p) for _,p in model.named_parameters(remove_duplicate=False))
    module_aliases=Counter(id(m) for _,m in model.named_modules(remove_duplicate=False))
    def storage_key(tensor):
        if tensor.layout!=torch.strided or tensor.device.type=='meta' or tensor.numel()==0:
            return None
        storage=tensor.untyped_storage()
        return (str(tensor.device),storage.data_ptr(),storage.nbytes())
    # Distinct Parameters and buffers can still view the same storage.
    stored_tensors=list(model.named_parameters(remove_duplicate=False))+list(model.named_buffers(remove_duplicate=False))
    storage_aliases=Counter(storage_key(t) for _,t in stored_tensors if storage_key(t) is not None)
    originals=dict(model.named_modules())
    candidate=model if inplace else _copy_with_compiler_metadata(model)
    before=sum(p.numel() for p in model.parameters())
    records=[]
    for path,module in list(candidate.named_modules()):
        if not isinstance(module,nn.Embedding):
            continue
        original=originals[path]
        reason=None
        if (module_aliases[id(original)]>1 or param_aliases[id(original.weight)]>1
                or storage_aliases[storage_key(original.weight)]>1):
            reason='Shared embedding module or Parameter is retained'
        ancestors=['']+['.'.join(path.split('.')[:i]) for i in range(1,len(path.split('.')))] if path else []
        if any(_has_hooks(originals[ancestor]) for ancestor in ancestors):
            reason='Ancestor module hooks require a separate adapter'
        try:
            if reason is not None:raise ValueError(reason)
            replacement=ConstantRowEmbedding.from_embedding(module)
        except ValueError as exc:
            records.append({'path':path,'status':'retained','reason':str(exc)})
            continue
        if path:
            parent,_,name=path.rpartition('.')
            setattr(candidate.get_submodule(parent) if parent else candidate,name,replacement)
        else:
            candidate=replacement
        records.append({'path':path,'status':'deduplicated','parameters_before':module.weight.numel(),
                        'parameters_after':replacement._row.numel(),'recipe':replacement.recipe()})
    return candidate,{'method':'exact_frozen_constant_embedding_rows','parameters_before':before,
                      'parameters_after':sum(p.numel() for p in candidate.parameters()),'layers':records,
                      'guarantee':'Identical stored row bits and unchanged F.embedding token-index lookup; frozen weights only',
                      'performance':'Reduces stored weight bytes; full-weight contiguous consumers can materialize the original shape'}
