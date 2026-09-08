"""Lossless, CPU-resident block storage for frozen float32 embeddings.

Only storage changes. Requested rows are decoded to their original byte patterns
and copied to the requested device. No result cache, numerical approximation or
shape-dependent floating-point table is used. CPU decoding and device transfers
can make inference slower. Registered buffers include every compressed byte.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from .packing import pack_bytes, unpack_bytes


class PackedFrozenEmbedding(nn.Module):
    """A frozen embedding with independently lossless-compressed row blocks.

The public weight property materializes the complete original tensor on demand;
    its returned value is an independent snapshot; mutating it does not change
    the packed weights. Device
placement moves the zero-sized output anchor, while packed storage stays on CPU.
No training parameterization or mutable weight alias is preserved.
"""
    def __init__(self, source: nn.Embedding, *, block_rows=16, level=3):
        super().__init__()
        if (type(source) is not nn.Embedding or source.training or source.weight.requires_grad
                or source.weight.dtype != torch.float32 or source.weight.layout != torch.strided
                or source.weight.device.type == 'meta' or source.max_norm is not None or source.sparse
                or set(source._parameters) != {'weight'} or source._buffers or source._modules
                or not all(source.weight.shape) or not math.isfinite(source.norm_type)
                or 'forward' in vars(source)
                or any(getattr(source, attr, {}) for attr in ('_forward_hooks', '_forward_pre_hooks',
                                                             '_backward_hooks', '_backward_pre_hooks'))):
            raise ValueError('Expected an ordinary hook-free frozen float32 evaluation Embedding without max_norm/sparse')
        if type(block_rows) is not int or not 1 <= block_rows <= 1024:
            raise ValueError('block_rows must be an integer between 1 and 1024')
        self.num_embeddings, self.embedding_dim = source.weight.shape
        self.padding_idx, self.norm_type = source.padding_idx, source.norm_type
        self.max_norm, self.sparse = None, False
        self.scale_grad_by_freq = source.scale_grad_by_freq
        self.block_rows = block_rows
        self._shape = tuple(source.weight.shape)
        weight = source.weight.detach().cpu().resolve_neg().contiguous()
        parts, offsets = [], [0]
        for start in range(0, self.num_embeddings, block_rows):
            data = weight[start:start + block_rows].view(torch.uint8).numpy().tobytes()
            part = pack_bytes(data, level=level)
            parts.append(torch.frombuffer(bytearray(part), dtype=torch.uint8))
            offsets.append(offsets[-1] + len(part))
        with torch.inference_mode(False):
            self.register_buffer('payload', torch.cat(parts))
            self.register_buffer('offsets', torch.tensor(offsets, dtype=torch.int64))
            self.register_buffer('_anchor', torch.empty(0, device=source.weight.device, dtype=torch.float32))
        self.eval()

    def _apply(self, fn, recurse=True):
        # Keeping the immutable codec on CPU avoids a full GPU roundtrip to
        # access a compressed block. Count this storage in unified-memory use.
        anchor = fn(self._anchor)
        if anchor.dtype != torch.float32 or anchor.device.type not in ('cpu', 'mps', 'cuda'):
            raise ValueError('Packed embeddings preserve float32 and support CPU, MPS or CUDA outputs')
        self._buffers['_anchor'] = anchor
        return self

    def requires_grad_(self, requires_grad=True):
        if requires_grad:
            raise ValueError('Packed embeddings support frozen inference only')
        return self

    def _check(self):
        if self.training or any(p.requires_grad for p in self.parameters()):
            raise ValueError('Packed embeddings require frozen evaluation')
        if (set(self._buffers) != {'payload', 'offsets', '_anchor'} or self._modules or self._parameters
                or self._non_persistent_buffers_set or self._shape != (self.num_embeddings, self.embedding_dim)
                or type(self.block_rows) is not int or not 1 <= self.block_rows <= 1024
                or self.max_norm is not None or self.sparse is not False):
            raise ValueError('Packed embedding structure changed')
        if (self.payload.dtype != torch.uint8 or self.offsets.dtype != torch.int64
                or self.payload.device.type != 'cpu' or self.offsets.device.type != 'cpu'
                or self._anchor.dtype != torch.float32 or self._anchor.numel()
                or self.payload.ndim != 1 or not self.payload.is_contiguous()
                or self.offsets.shape != (math.ceil(self.num_embeddings / self.block_rows) + 1,)
                or not self.offsets.is_contiguous() or self._anchor.device.type not in ('cpu', 'mps', 'cuda')):
            raise ValueError('Packed storage must remain on CPU and output dtype float32')

    def _block(self, index):
        start, stop = self.offsets[index:index + 2].tolist()
        rows = min(self.block_rows, self.num_embeddings - index * self.block_rows)
        raw_size = rows * self.embedding_dim * 4
        raw = unpack_bytes(self.payload[start:stop].numpy().tobytes(), max_output_bytes=raw_size)
        if len(raw) != raw_size:
            raise ValueError('Decoded embedding block has an invalid length')
        return torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(rows, self.embedding_dim)

    @property
    def weight(self):
        self._check()
        parts = [self._block(i) for i in range(math.ceil(self.num_embeddings / self.block_rows))]
        return torch.cat(parts).to(self._anchor.device)

    def forward(self, indices):
        self._check()
        if not isinstance(indices, torch.Tensor) or indices.dtype not in (torch.int32, torch.int64):
            raise RuntimeError('Expected tensor for indices with scalar type Long or Int')
        if indices.layout != torch.strided or indices.device != self._anchor.device:
            raise ValueError('Indices must use the output device and dense strided layout')
        local = indices.detach().cpu().reshape(-1).to(torch.int64)
        if local.numel() and (int(local.min()) < 0 or int(local.max()) >= self.num_embeddings):
            raise IndexError('index out of range in self')
        result = torch.empty((len(local), self.embedding_dim), dtype=torch.float32)
        blocks = torch.div(local, self.block_rows, rounding_mode='floor')
        for block_index in torch.unique(blocks).tolist():
            positions = torch.nonzero(blocks == block_index, as_tuple=True)[0]
            decoded = self._block(block_index)
            result[positions] = decoded[local[positions] - block_index * self.block_rows]
        return result.reshape(*indices.shape, self.embedding_dim).to(self._anchor.device)

    def recipe(self):
        self._check()
        return {'kind': 'PackedFrozenEmbedding', 'version': 1, 'num_embeddings': self.num_embeddings,
                'embedding_dim': self.embedding_dim, 'padding_idx': self.padding_idx,
                'norm_type': self.norm_type, 'scale_grad_by_freq': self.scale_grad_by_freq,
                'block_rows': self.block_rows, 'payload_bytes': self.payload.numel()}

    @classmethod
    def from_recipe(cls, spec):
        if (not isinstance(spec, dict) or set(spec) != {'kind', 'version', 'num_embeddings', 'embedding_dim',
                'padding_idx', 'norm_type', 'scale_grad_by_freq', 'block_rows', 'payload_bytes'}
                or spec.get('kind') != 'PackedFrozenEmbedding' or type(spec.get('version')) is not int
                or spec['version'] != 1):
            raise ValueError('Unsupported packed embedding recipe')
        n, width, rows, size = (spec.get(k) for k in ('num_embeddings', 'embedding_dim', 'block_rows', 'payload_bytes'))
        if any(type(x) is not int or x < 1 for x in (n, width, rows, size)) or rows > 1024:
            raise ValueError('Invalid packed embedding dimensions')
        padding = spec['padding_idx']
        if ((padding is not None and (type(padding) is not int or not 0 <= padding < n))
                or type(spec['norm_type']) not in (int, float) or not math.isfinite(spec['norm_type'])
                or type(spec['scale_grad_by_freq']) is not bool):
            raise ValueError('Invalid packed embedding metadata')
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.num_embeddings, model.embedding_dim, model.block_rows = n, width, rows
        model._shape = (n, width)
        model.padding_idx, model.norm_type = spec.get('padding_idx'), spec.get('norm_type', 2.0)
        model.scale_grad_by_freq = spec.get('scale_grad_by_freq', False)
        model.max_norm, model.sparse = None, False
        model.register_buffer('payload', torch.empty(size, dtype=torch.uint8))
        model.register_buffer('offsets', torch.empty(math.ceil(n / rows) + 1, dtype=torch.int64))
        model.register_buffer('_anchor', torch.empty(0, dtype=torch.float32))
        return model.eval()

    def validate_payload(self):
        self._check()
        offsets = self.offsets.tolist()
        if (len(offsets) != math.ceil(self.num_embeddings / self.block_rows) + 1
                or offsets[0] != 0 or offsets[-1] != self.payload.numel()
                or any(a >= b for a, b in zip(offsets, offsets[1:]))):
            raise ValueError('Invalid packed embedding block offsets')
        for index in range(len(offsets) - 1):
            self._block(index)
