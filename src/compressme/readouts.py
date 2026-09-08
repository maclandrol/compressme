"""Batch independent affine readouts without copying or dropping parameters."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class BatchedLinearHeads(nn.Module):
    """Independent equal-shape Linear modules stored as one batched operator.

    Input (..., heads, in_features) -> (..., heads, out_features). Each head
    retains its own complete weight and bias. Integer indexing returns a callable
    view for compatibility with a loop over the original heads.
    """
    _compressme_runtime_only = True
    def __init__(self, heads):
        super().__init__()
        heads = list(heads)
        if not heads or any(type(h) is not nn.Linear for h in heads):
            raise TypeError("Requires one or more standard Linear readouts")
        first = heads[0]
        self.num_heads = len(heads)
        self.in_features, self.out_features = first.in_features,first.out_features
        params = [p for h in heads for p in h.parameters()]
        if len({id(p) for p in params}) != len(params):
            raise ValueError("Shared readout parameters require an explicit adapter")
        for h in heads:
            if h.weight.shape != first.weight.shape or (h.bias is None) != (first.bias is None):
                raise ValueError("Readouts must have equal dimensions and bias configuration")
            if h.weight.device != first.weight.device or h.weight.dtype != first.weight.dtype:
                raise ValueError("Readouts must have the same dtype and device")
            if h.weight.requires_grad != first.weight.requires_grad or (
                h.bias is not None and h.bias.requires_grad != first.bias.requires_grad):
                raise ValueError("Readouts must share their trainability configuration")
            if h._forward_hooks or h._forward_pre_hooks or h._backward_hooks or "forward" in vars(h):
                raise ValueError("Custom readout behaviour cannot be batched")
        self.weight = nn.Parameter(torch.stack([h.weight.detach() for h in heads]),
                                   requires_grad=first.weight.requires_grad)
        self.bias = (nn.Parameter(torch.stack([h.bias.detach() for h in heads]),
                                  requires_grad=first.bias.requires_grad)
                     if first.bias is not None else None)
        self.train(first.training)

    def __len__(self):
        return self.num_heads

    def __getitem__(self,index):
        if not isinstance(index,int) or not -self.num_heads <= index < self.num_heads:
            raise IndexError("Invalid readout index")
        return lambda x: F.linear(x,self.weight[index],None if self.bias is None else self.bias[index])

    def forward(self,x):
        if x.ndim < 2 or x.shape[-2:] != (self.num_heads,self.in_features):
            raise ValueError("Expected (..., heads, in_features) readout inputs")
        leading = x.shape[:-2]
        xh = x.reshape(-1,self.num_heads,self.in_features).transpose(0,1)
        output = torch.bmm(xh,self.weight.transpose(1,2)).transpose(0,1)
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output.reshape(*leading,self.num_heads,self.out_features)
