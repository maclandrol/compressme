"""Exact few-active-key inference path for PyTorch self-attention.

Retains every original MHA parameter and uses the original implementation for
unsupported inputs or training/autograd. No parameter compression is claimed.
"""
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F
from compressme.runtime import autocast_enabled


def _keep_parent_python_path(module, inputs):
    # TransformerEncoderLayer checks descendant hooks before its native fused
    # forward, which otherwise reads our weights and bypasses our forward.
    return None


class FewActiveKeysMHA(nn.MultiheadAttention):
    def __init__(self, source: nn.MultiheadAttention):
        if type(source) is not nn.MultiheadAttention:
            raise TypeError("Requires standard nn.MultiheadAttention")
        if any(m._forward_hooks or m._forward_pre_hooks for m in source.modules()):
            raise ValueError("Source forward hooks require a separate adapter")
        dtype = source.out_proj.weight.dtype
        device = source.out_proj.weight.device
        with torch.random.fork_rng(devices=[]):
            super().__init__(source.embed_dim, source.num_heads, source.dropout,
                bias=source.in_proj_bias is not None, add_bias_kv=source.bias_k is not None,
                add_zero_attn=source.add_zero_attn, kdim=source.kdim, vdim=source.vdim,
                batch_first=source.batch_first, dtype=dtype)
        self.to(device)
        self.load_state_dict(source.state_dict(), strict=True)
        for (_, p), (_, original) in zip(self.named_parameters(), source.named_parameters()):
            p.requires_grad_(original.requires_grad)
        self.train(source.training)
        self.register_forward_pre_hook(_keep_parent_python_path)
        self.fast_calls = 0
        self.fallback_calls = 0

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        eligible = (
            not self.training and not torch.is_grad_enabled() and self.batch_first
            and query is key and key is value and isinstance(query, torch.Tensor)
            and not query.is_nested and query.ndim == 3 and self._qkv_same_embed_dim
            and self.bias_k is None and self.bias_v is None and not self.add_zero_attn
            and attn_mask is None and not is_causal
            and query.dtype == self.in_proj_weight.dtype
            and query.dtype in (torch.float32, torch.float64)
            and not autocast_enabled(query.device.type)
        )
        active = None
        if eligible:
            B, L, D = query.shape
            eligible = B > 0 and L > 0
            if eligible and key_padding_mask is None:
                eligible = L <= 2
                if eligible:
                    active = torch.arange(L, device=query.device)
            elif eligible:
                eligible = (isinstance(key_padding_mask, torch.Tensor)
                            and key_padding_mask.shape == (B, L)
                            and key_padding_mask.device == query.device)
                if eligible:
                    if key_padding_mask.dtype == torch.bool:
                        missing = key_padding_mask
                    elif key_padding_mask.is_floating_point():
                        missing = torch.isneginf(key_padding_mask)
                        eligible = bool(((key_padding_mask == 0) | missing).all())
                    else:
                        eligible = False
                if eligible:
                    eligible = torch.equal(missing, missing[:1].expand_as(missing))
                if eligible:
                    active = (~missing[0]).nonzero().flatten()
                    eligible = active.numel() in (1, 2)
        if not eligible:
            self.fallback_calls += 1
            return super().forward(query, key, value, key_padding_mask=key_padding_mask,
                need_weights=need_weights, attn_mask=attn_mask,
                average_attn_weights=average_attn_weights, is_causal=is_causal)
        self.fast_calls += 1
        H, C = self.num_heads, self.head_dim
        K = active.numel()
        xk = query.index_select(1, active)
        Wq, Wk, Wv = self.in_proj_weight.chunk(3, dim=0)
        bq = bv = None
        if self.in_proj_bias is not None:
            bq, _, bv = self.in_proj_bias.chunk(3, dim=0)
        if K == 1:
            alpha = torch.ones(B, H, L, 1, device=query.device, dtype=query.dtype)
        else:
            delta = xk[:, 1] - xk[:, 0]
            key_delta = F.linear(delta, Wk).reshape(B, H, C)
            query_metric = torch.einsum('bhc,hcd->bhd', key_delta, Wq.reshape(H, C, D))
            difference = torch.einsum('bld,bhd->bhl', query, query_metric)
            if bq is not None:
                difference = difference + (key_delta*bq.reshape(H, C)).sum(-1)[:, :, None]
            p1 = torch.sigmoid(difference/math.sqrt(C))
            alpha = torch.stack((1-p1, p1), dim=-1)
        values = F.linear(xk, Wv, bv).reshape(B, K, H, C)
        projected_values = torch.einsum('bkhc,ohc->bhko', values,
                                       self.out_proj.weight.reshape(D, H, C))
        out = torch.einsum('bhlk,bhko->blo', alpha, projected_values)
        if self.out_proj.bias is not None:
            out = out+self.out_proj.bias
        if not need_weights:
            return out, None
        attention = torch.zeros(B, H, L, L, device=query.device, dtype=query.dtype)
        attention[..., active] = alpha
        if average_attn_weights:
            attention = attention.mean(1)
        return out, attention


def install_few_active_keys(model: nn.Module):
    """Return a copied model with standard MHA modules using this optional path."""
    result = copy.deepcopy(model)
    for path, module in list(result.named_modules()):
        if type(module) is nn.MultiheadAttention:
            replacement = FewActiveKeysMHA(module)
            if not path:
                result = replacement
            else:
                parent, _, leaf = path.rpartition('.')
                owner = result.get_submodule(parent) if parent else result
                owner._modules[leaf] = replacement
    return result
