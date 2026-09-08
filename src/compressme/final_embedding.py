"""Explicit inference-only final embeddings from an already selected backbone.

No Transformers dependency is imported. This helper changes the requested output
contract: it returns only the backbone's last_hidden_state. It is not a drop-in
replacement for a complete language model, its logits/loss/cache/attention API,
or an arbitrary model's hidden_states[-1] (which can have different semantics).
"""
from __future__ import annotations
from collections.abc import Mapping
import torch
from torch import nn


class FinalEmbedding(nn.Module):
    """Run the exact selected backbone without retaining all layer outputs.

    The caller must explicitly choose output_contract="last_hidden_state" and
    supply the already audited backbone, for example ``masked_lm.esm``. The
    wrapper shares that module; it does not clone weights, rewrite operations,
    infer a backbone path, or mutate model configuration. Releasing a former
    task-head owner can then release parameters unique to the unused head.

    Only output_hidden_states=False and return_dict=True are supplied. Attention
    flags, attention implementation, dtype, device and all other inputs remain as
    supplied, avoiding backend changes caused by forcing attentions off. Other
    requested backbone outputs are intentionally not exposed by this contract.
    Inference uses torch.no_grad, matching typical embedding preprocessing.
    """
    def __init__(self, backbone: nn.Module, *, output_contract: str):
        super().__init__()
        if output_contract != 'last_hidden_state':
            raise ValueError('Explicitly choose output_contract="last_hidden_state"')
        if not isinstance(backbone, nn.Module):
            raise TypeError('backbone must be the explicit torch module to execute')
        if any(module.training for module in backbone.modules()):
            raise ValueError('The selected backbone and its children must already be in evaluation mode')
        self.backbone = backbone
        self.training = False
        self.output_contract = output_contract

    def train(self, mode=True):
        if mode is not False:
            raise ValueError('FinalEmbedding is an explicit inference-only output contract')
        return super().train(False)

    def forward(self, *args, **kwargs):
        if self.output_contract != 'last_hidden_state' or self.training or any(m.training for m in self.backbone.modules()):
            raise ValueError('Preserve the final-hidden-state contract and evaluation mode')
        if kwargs.get('output_hidden_states', False) is not False:
            raise ValueError('This contract does not expose all-layer hidden states')
        if kwargs.get('return_dict', True) is not True:
            raise ValueError('This contract requires the named last_hidden_state output')
        kwargs['output_hidden_states'] = False
        kwargs['return_dict'] = True
        with torch.no_grad():
            result = self.backbone(*args, **kwargs)
        value = result.get('last_hidden_state') if isinstance(result, Mapping) else getattr(result, 'last_hidden_state', None)
        if not isinstance(value, torch.Tensor):
            raise TypeError('The selected backbone must return a Tensor named last_hidden_state')
        return value
