"""Research prototype: lossless float32 profile tables using exact integer XOR.

No floating-point subtraction, rounding, quantization, or source weights enter
encoding/reconstruction. Sparse high-bit exceptions are padded per token row so
runtime shapes are known without device synchronization. Every padded element
counts toward storage. This prototype is not a production compressor pass.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def registered_bytes(module):
    seen = set()
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage._cdata)
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
    return total


def bytes_equal(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                            right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)))


def _copy_array(array):
    return torch.from_numpy(np.array(array, copy=True))


class _Delta(nn.Module):
    def __init__(self, mode, buffers):
        super().__init__()
        self.mode = mode
        for name, tensor in buffers.items():
            self.register_buffer(name, tensor)
        self.eval()

    def decode(self, indices, base):
        if self.mode == 'same':
            return base
        if self.mode == 'full':
            if self.full.dtype != torch.float32:
                raise ValueError('Float32 storage dtype must be preserved')
            return F.embedding(indices, self.full)
        low_bits = int(self.mode)
        code = F.embedding(indices, self.low).to(torch.int32)
        if low_bits == 16:
            code = code.bitwise_and(0xffff)
        if self.columns.shape[1]:
            columns = F.embedding(indices, self.columns).to(torch.int64)
            if self.columns.dtype == torch.int16:
                columns = columns.bitwise_and(0xffff)
            high = F.embedding(indices, self.high).to(torch.int32).bitwise_left_shift(low_bits)
            # Each real exception column is unique. Padding adds zero at column0.
            # Aligned upper bits and low bits have disjoint bit positions.
            code.scatter_add_(-1, columns, high)
        return torch.bitwise_xor(base.view(torch.int32), code).view(torch.float32)


def _encode_delta(base, target, *, low_bits='auto'):
    a = base.detach().cpu().contiguous().numpy().view(np.uint32)
    b = target.detach().cpu().contiguous().numpy().view(np.uint32)
    xor = np.bitwise_xor(a, b)
    if not np.any(xor):
        return _Delta('same', {})
    options = []
    n, width = xor.shape
    column_bytes = 2 if width <= 65536 else 4
    for bits in (8, 16) if low_bits == 'auto' else (low_bits,):
        if bits not in (8, 16):
            raise ValueError('Choose low_bits=8, 16, or auto')
        exception = xor >= (1 << bits)
        slots = int(exception.sum(axis=1).max())
        cost = xor.size * (bits // 8) + n * slots * (column_bytes + (4 if bits == 8 else 2))
        options.append((cost, bits, slots, exception))
    cost, bits, slots, exception = min(options, key=lambda item: item[0])
    if cost >= target.numel() * target.element_size():
        return _Delta('full', {'full': target.detach().cpu().clone()})
    low = (xor & ((1 << bits) - 1)).astype(np.uint8 if bits == 8 else np.uint16)
    if bits == 16:
        low = low.view(np.int16)
    columns = np.zeros((n, slots), dtype=np.uint16 if column_bytes == 2 else np.int32)
    high = np.zeros((n, slots), dtype=np.int32 if bits == 8 else np.uint16)
    for row in range(n):
        locations = np.flatnonzero(exception[row])
        columns[row, :len(locations)] = locations
        high[row, :len(locations)] = xor[row, locations] >> bits
    if column_bytes == 2:
        columns = columns.view(np.int16)
    if bits == 16:
        high = high.view(np.int16)
    return _Delta(str(bits), {'low': _copy_array(low), 'columns': _copy_array(columns), 'high': _copy_array(high)})


class XorProfileLookup(nn.Module):
    """Gather one selected route, reconstructing exactly the stored float32 bits.

    All table routes have the same [vocabulary, width] shape. The base route is
    intentionally explicit: making the usual bulk route direct avoids decoding
    work on common prefill calls. Construction happens on CPU; .to('mps') moves
    the frozen encoded state without changing its numerical representation.
    """
    def __init__(self, tables, *, base_route=0, low_bits='auto'):
        super().__init__()
        if (not isinstance(tables, (list, tuple)) or not tables or type(base_route) is not int
                or not 0 <= base_route < len(tables)):
            raise ValueError('Supply nonempty tables and a valid base route')
        base = tables[base_route]
        if (base.dtype != torch.float32 or base.ndim != 2 or not all(base.shape)
                or base.layout != torch.strided
                or any(t.dtype != base.dtype or t.shape != base.shape or t.layout != torch.strided for t in tables)):
            raise ValueError('This prototype accepts matching dense float32 matrices')
        if low_bits not in (8, 16, 'auto'):
            raise ValueError('Choose low_bits=8, 16, or auto')
        self.base_route = base_route
        self.register_buffer('base', base.detach().cpu().clone())
        self.routes = nn.ModuleList([_encode_delta(base, tensor, low_bits=low_bits) for tensor in tables])
        self.eval()

    def forward(self, indices, *, route=0):
        if type(route) is not int or not 0 <= route < len(self.routes):
            raise ValueError('Unknown profile route')
        if not isinstance(indices, torch.Tensor) or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError('Expected integer token IDs')
        if self.base.dtype != torch.float32 or self.training or any(t.requires_grad for t in self.buffers()):
            raise ValueError('Preserve float32 dtype and frozen evaluation mode')
        encoded = self.routes[route]
        if encoded.mode == 'full':
            return encoded.decode(indices, None)
        base = F.embedding(indices, self.base)
        return encoded.decode(indices, base)

    def requires_grad_(self, requires_grad=True):
        if requires_grad:
            raise ValueError('Lossless profile storage does not preserve a training parameterization')
        return super().requires_grad_(False)

    def recipe(self):
        return {'kind': 'XorProfileLookupResearch', 'version': 1, 'base_route': self.base_route,
                'shape': list(self.base.shape),
                'routes': [{'mode': route.mode,
                            'buffers': {name: {'shape': list(value.shape), 'dtype': str(value.dtype).removeprefix('torch.')}
                                        for name, value in route.named_buffers()}}
                           for route in self.routes]}

    @classmethod
    def from_recipe(cls, spec):
        # Research-only closed recipe. Safe tensor loading is performed separately.
        if spec.get('kind') != 'XorProfileLookupResearch' or spec.get('version') != 1:
            raise ValueError('Unsupported XOR recipe')
        shape = spec.get('shape')
        if (not isinstance(shape, list) or len(shape) != 2 or any(type(n) is not int or n < 1 for n in shape)
                or not isinstance(spec.get('routes'), list) or not spec['routes']
                or type(spec.get('base_route')) is not int or not 0 <= spec['base_route'] < len(spec['routes'])):
            raise ValueError('Invalid XOR dimensions/routes')
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.base_route = spec['base_route']
        obj.register_buffer('base', torch.zeros(shape, dtype=torch.float32))
        routes = []
        for item in spec['routes']:
            mode, descriptions = item.get('mode'), item.get('buffers')
            if mode not in ('same', 'full', '8', '16') or not isinstance(descriptions, dict):
                raise ValueError('Invalid XOR mode')
            required = set() if mode == 'same' else {'full'} if mode == 'full' else {'low', 'columns', 'high'}
            if set(descriptions) != required:
                raise ValueError('Invalid XOR buffers')
            buffers = {}
            for name, value in descriptions.items():
                size = value.get('shape')
                dtype = getattr(torch, value.get('dtype', ''), None)
                if (not isinstance(size, list) or len(size) != 2 or size[0] != shape[0]
                        or any(type(n) is not int or n < 0 for n in size)
                        or dtype not in (torch.float32, torch.uint8, torch.int16, torch.int32)):
                    raise ValueError('Invalid XOR buffer shape/dtype')
                buffers[name] = torch.zeros(size, dtype=dtype)
            if mode == 'full' and (buffers['full'].shape != tuple(shape) or buffers['full'].dtype != torch.float32):
                raise ValueError('Invalid full route')
            if mode in ('8', '16'):
                if (tuple(buffers['low'].shape) != tuple(shape)
                        or buffers['low'].dtype != (torch.uint8 if mode == '8' else torch.int16)
                        or buffers['columns'].shape != buffers['high'].shape
                        or buffers['columns'].dtype != (torch.int16 if shape[1] <= 65536 else torch.int32)
                        or buffers['high'].dtype != (torch.int32 if mode == '8' else torch.int16)):
                    raise ValueError('Inconsistent XOR code buffers')
            routes.append(_Delta(mode, buffers))
        obj.routes = nn.ModuleList(routes)
        obj.eval()
        return obj
