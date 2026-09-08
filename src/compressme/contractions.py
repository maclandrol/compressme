"""Explicit channel-wise token contractions with one matrix packing per input."""
from __future__ import annotations
import torch


def channelwise_token_contraction(a: torch.Tensor, b: torch.Tensor, *, direction='outgoing') -> torch.Tensor:
    """Contract a shared token axis independently for each batch and channel.

    Outgoing inputs are [B,I,K,D] and [B,J,K,D], with output
    C[b,i,j,d] = sum_k a[b,i,k,d] * b[b,j,k,d]. Incoming inputs are
    [B,K,I,D] and [B,K,J,D], with the same output indices and shared K.
    I and J may differ. Batch, reduction and channel dimensions must match;
    implicit broadcasting is deliberately outside this helper's contract.

    Each input is permuted and packed once into contiguous batched matrices.
    torch.bmm performs the reduction, then the output view restores [B,I,J,D]
    with channel-major backing storage. No dtype conversion, detach, custom
    backward or explicit autocast override is introduced. PyTorch autograd flows
    through the packing, bmm and output view normally.

    This is the same real-arithmetic bilinear operation as the corresponding
    einsum. Layout changes can change backend kernel selection and floating-point
    summation, so byte identity and runtime gains require checks on each actual
    backend, dtype and workload. No universal floating-point equality or speed
    improvement is promised. Dense strided materialized inputs are required.
    """
    if direction not in ('outgoing', 'incoming'):
        raise ValueError('direction must be outgoing or incoming')
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError('Both inputs must be tensors')
    if a.ndim != 4 or b.ndim != 4 or a.layout != torch.strided or b.layout != torch.strided:
        raise ValueError('Expected dense strided rank-four inputs')
    if a.device.type == 'meta' or b.device.type == 'meta':
        raise ValueError('Expected materialized inputs')
    if a.dtype != b.dtype or a.device != b.device:
        raise ValueError('Input dtypes and devices must match')
    if a.shape[0] != b.shape[0] or a.shape[3] != b.shape[3]:
        raise ValueError('Batch and channel dimensions must match; no broadcasting')
    B, D = a.shape[0], a.shape[3]
    if direction == 'outgoing':
        I, K, J = a.shape[1], a.shape[2], b.shape[1]
        if K != b.shape[2]:raise ValueError('Reduction dimensions must match')
        left = a.permute(0, 3, 1, 2).contiguous().reshape(B*D, I, K)
        right = b.permute(0, 3, 2, 1).contiguous().reshape(B*D, K, J)
    else:
        K, I, J = a.shape[1], a.shape[2], b.shape[2]
        if K != b.shape[1]:raise ValueError('Reduction dimensions must match')
        left = a.permute(0, 3, 2, 1).contiguous().reshape(B*D, I, K)
        right = b.permute(0, 3, 1, 2).contiguous().reshape(B*D, K, J)
    return torch.bmm(left, right).reshape(B, D, I, J).permute(0, 2, 3, 1)
