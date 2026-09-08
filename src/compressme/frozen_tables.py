"""Lossless resident storage for families of related float32 lookup tables.

No floating-point subtraction, rounding, quantization, or source weights enter
encoding/reconstruction. Sparse high-bit exceptions are padded per token row so
runtime shapes are known without device synchronization. Every padded element
counts toward storage. This stores frozen lookup values; it does not reparameterize a trainable model.
"""
from __future__ import annotations
import json
import torch
from torch import nn
from torch.nn import functional as F
from .finite_lookup import _ValidatedLookup, _has_hooks, _resident_storage_bytes
from .runtime import autocast_enabled


def registered_bytes(module):
    return _resident_storage_bytes(module)


def _table_storage_bytes(tables):
    allocations = {}
    for tensor in tables:
        storage = tensor.untyped_storage()
        allocations[(str(tensor.device), storage._cdata)] = storage.nbytes()
    return sum(allocations.values())


def bytes_equal(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.detach().cpu().resolve_neg().contiguous().reshape(-1).view(torch.uint8),
                            right.detach().cpu().resolve_neg().contiguous().reshape(-1).view(torch.uint8)))


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
    # Integer arithmetic only. The int64 mask gives unsigned 32-bit XOR words
    # without requiring experimental uint32 operators or an extra dependency.
    a = base.detach().cpu().resolve_neg().contiguous().view(torch.int32)
    b = target.detach().cpu().resolve_neg().contiguous().view(torch.int32)
    xor = torch.bitwise_xor(a, b).to(torch.int64).bitwise_and(0xffffffff)
    if not torch.any(xor):
        return _Delta('same', {})
    options = []
    n, width = xor.shape
    column_bytes = 2 if width <= 65536 else 4
    for bits in (8, 16) if low_bits == 'auto' else (low_bits,):
        exception = xor >= (1 << bits)
        slots = int(exception.sum(dim=1).max())
        cost = xor.numel() * (bits // 8) + n * slots * (column_bytes + (4 if bits == 8 else 2))
        options.append((cost, bits, slots, exception))
    cost, bits, slots, exception = min(options, key=lambda item: item[0])
    if cost >= target.numel() * target.element_size():
        return _Delta('full', {'full': target.detach().cpu().resolve_neg().clone()})
    low = xor.bitwise_and((1 << bits) - 1).to(torch.uint8 if bits == 8 else torch.int16)
    columns = torch.zeros((n, slots), dtype=torch.int16 if column_bytes == 2 else torch.int32)
    high = torch.zeros((n, slots), dtype=torch.int32 if bits == 8 else torch.int16)
    for row in range(n):
        locations = torch.nonzero(exception[row], as_tuple=True)[0]
        columns[row, :len(locations)] = locations.to(columns.dtype)
        high[row, :len(locations)] = xor[row, locations].bitwise_right_shift(bits).to(high.dtype)
    return _Delta(str(bits), {'low': low.clone(), 'columns': columns, 'high': high})


class XorProfileLookup(_ValidatedLookup):
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
        if (type(base) not in (torch.Tensor, nn.Parameter) or base.device.type == 'meta' or base.dtype != torch.float32 or base.ndim != 2 or not all(base.shape)
                or base.layout != torch.strided
                or any(type(t) not in (torch.Tensor, nn.Parameter) or t.device.type == 'meta' or t.dtype != base.dtype
                       or t.shape != base.shape or t.layout != torch.strided for t in tables)):
            raise ValueError('Supply matching materialized dense float32 matrices')
        if not (type(low_bits) is str and low_bits == 'auto' or type(low_bits) is int and low_bits in (8, 16)):
            raise ValueError('Choose low_bits=8, 16, or auto')
        self.base_route = base_route
        self._shape = tuple(base.shape)
        # Do not inherit inference tensors from an enclosing inference context:
        # normal versioned buffers are needed for mutation-aware export reports.
        with torch.inference_mode(False), torch.no_grad():
            self.register_buffer('base', base.detach().cpu().resolve_neg().contiguous().clone())
            self.routes = nn.ModuleList([_encode_delta(base, tensor, low_bits=low_bits) for tensor in tables])
        self.eval()
        self._validate_structure()

    def forward(self, indices, *, route=0):
        self._validate_structure()
        if type(route) is not int or not 0 <= route < len(self.routes):
            raise ValueError('Unknown profile route')
        if (not isinstance(indices, torch.Tensor) or indices.dtype not in (torch.int32, torch.int64)
                or indices.layout != torch.strided or indices.device != self.base.device):
            raise ValueError('Expected integer token IDs')
        if autocast_enabled(self.base.device.type):
            raise ValueError('Use explicit float32 dtypes, not autocast')
        encoded = self.routes[route]
        if encoded.mode == 'full':
            return encoded.decode(indices, None)
        base = F.embedding(indices, self.base)
        return encoded.decode(indices, base)

    def requires_grad_(self, requires_grad=True):
        if requires_grad:
            raise ValueError('Lossless profile storage does not preserve a training parameterization')
        return super().requires_grad_(False)

    def _validate_structure(self):
        """Cheap shape/type checks; no tensor values or device synchronization."""
        if (type(self.routes) is not nn.ModuleList or not self.routes
                or type(self.base_route) is not int or not 0 <= self.base_route < len(self.routes)
                or set(self._buffers) != {'base'} or set(self._modules) != {'routes'}
                or self._parameters or self._non_persistent_buffers_set):
            raise ValueError('The frozen table structure changed')
        if any(m.training or _has_hooks(m) or 'forward' in vars(m) for m in self.modules()):
            raise ValueError('Use hook-free frozen evaluation modules')
        shape, device = self._shape, self.base.device
        if not isinstance(shape, tuple) or len(shape) != 2 or any(type(n) is not int or n < 1 for n in shape):
            raise ValueError('The declared table dimensions changed')
        def buffer_ok(tensor, expected_shape, expected_dtype):
            return (type(tensor) in (torch.Tensor, nn.Parameter) and tensor.layout == torch.strided
                    and tuple(tensor.shape) == tuple(expected_shape) and tensor.dtype == expected_dtype
                    and tensor.device == device and device.type != 'meta' and not tensor.requires_grad
                    and not tensor.is_neg() and not tensor.is_conj())
        if not buffer_ok(self.base, shape, torch.float32):
            raise ValueError('Preserve the materialized float32 base shape, dtype and frozen state')
        for route in self.routes:
            if (type(route) is not _Delta or route._modules or route._parameters
                    or 'decode' in vars(route)
                    or route._non_persistent_buffers_set or route.mode not in ('same', 'full', '8', '16')):
                raise ValueError('The encoded route structure changed')
            required = set() if route.mode == 'same' else {'full'} if route.mode == 'full' else {'low', 'columns', 'high'}
            if set(route._buffers) != required:
                raise ValueError('The encoded route buffers changed')
            if route.mode == 'full':
                if not buffer_ok(route.full, shape, torch.float32):
                    raise ValueError('Preserve the full-table shape and float32 dtype')
            elif route.mode in ('8', '16'):
                low_dtype = torch.uint8 if route.mode == '8' else torch.int16
                high_dtype = torch.int32 if route.mode == '8' else torch.int16
                column_dtype = torch.int16 if shape[1] <= 65536 else torch.int32
                if (route.columns.ndim != 2 or route.columns.shape[0] != shape[0]
                        or not 0 <= route.columns.shape[1] <= shape[1]
                        or not buffer_ok(route.low, shape, low_dtype)
                        or not buffer_ok(route.columns, route.columns.shape, column_dtype)
                        or not buffer_ok(route.high, route.columns.shape, high_dtype)):
                    raise ValueError('Preserve the integer-code shapes, dtypes and frozen state')
        if self.routes[self.base_route].mode != 'same':
            raise ValueError('The base route must use its direct stored table')

    def _validate_payload(self):
        """Validate persisted exception indices once at export/load, on CPU."""
        self._validate_structure()
        width = self._shape[1]
        for route in self.routes:
            if route.mode not in ('8', '16') or not route.columns.shape[1]:
                continue
            columns = route.columns.detach().cpu().to(torch.int64)
            if route.columns.dtype == torch.int16:
                columns = columns.bitwise_and(0xffff)
            high = route.high.detach().cpu().to(torch.int64)
            if (torch.any(columns < 0) or torch.any(columns >= width)
                    or route.mode == '8' and (torch.any(high < 0) or torch.any(high > 0xffffff))):
                raise ValueError('Invalid encoded exception index or high bits')
            nonzero = high != 0
            ordered = torch.where(nonzero, columns, width).sort(dim=1).values
            if ordered.shape[1] > 1 and torch.any((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] != width)):
                raise ValueError('Nonzero XOR exceptions must have unique columns within each row')

    def recipe(self):
        self._validate_structure()
        return {'kind': 'XorProfileLookup', 'version': 1, 'base_route': self.base_route,
                'shape': list(self._shape), 'training': False,
                'routes': [{'mode': route.mode,
                            'buffers': {name: {'shape': list(value.shape), 'dtype': str(value.dtype).removeprefix('torch.')}
                                        for name, value in route.named_buffers()}}
                           for route in self.routes]}

    def _validation_signature(self):
        try:
            signature = super()._validation_signature()
            if signature is None:
                return None
            return signature[0], json.dumps(self.recipe(), sort_keys=True)
        except (ValueError, TypeError, RuntimeError, AttributeError):
            return None

    @classmethod
    def from_recipe(cls, spec, *, device="cpu"):
        """Rebuild a closed representation; payload integrity is checked by load."""
        if (not isinstance(spec, dict) or spec.get('kind') != 'XorProfileLookup'
                or type(spec.get('version')) is not int or spec['version'] != 1
                or spec.get('training') is not False):
            raise ValueError('Unsupported XOR recipe')
        shape = spec.get('shape')
        if (not isinstance(shape, list) or len(shape) != 2 or any(type(n) is not int or n < 1 for n in shape)
                or not isinstance(spec.get('routes'), list) or not spec['routes']
                or type(spec.get('base_route')) is not int or not 0 <= spec['base_route'] < len(spec['routes'])):
            raise ValueError('Invalid XOR dimensions/routes')
        route_specs = []
        for item in spec['routes']:
            if not isinstance(item, dict) or set(item) != {'mode', 'buffers'}:
                raise ValueError('Invalid XOR route descriptor')
            mode, descriptions = item['mode'], item['buffers']
            if type(mode) is not str or mode not in ('same', 'full', '8', '16') or not isinstance(descriptions, dict):
                raise ValueError('Invalid XOR mode')
            required = set() if mode == 'same' else {'full'} if mode == 'full' else {'low', 'columns', 'high'}
            if set(descriptions) != required:
                raise ValueError('Invalid XOR buffers')
            checked = {}
            for name, value in descriptions.items():
                if not isinstance(value, dict) or set(value) != {'shape', 'dtype'}:
                    raise ValueError('Invalid XOR buffer descriptor')
                size, dtype_name = value['shape'], value['dtype']
                dtype = getattr(torch, dtype_name, None) if type(dtype_name) is str else None
                if (not isinstance(size, list) or len(size) != 2 or size[0] != shape[0]
                        or any(type(n) is not int or n < 0 for n in size)
                        or dtype not in (torch.float32, torch.uint8, torch.int16, torch.int32)):
                    raise ValueError('Invalid XOR buffer shape/dtype')
                checked[name] = (size, dtype)
            if mode == 'full' and checked['full'] != (shape, torch.float32):
                raise ValueError('Invalid full route')
            if mode in ('8', '16'):
                if (checked['low'] != (shape, torch.uint8 if mode == '8' else torch.int16)
                        or checked['columns'][0] != checked['high'][0]
                        or checked['columns'][0][1] > shape[1]
                        or checked['columns'][1] != (torch.int16 if shape[1] <= 65536 else torch.int32)
                        or checked['high'][1] != (torch.int32 if mode == '8' else torch.int16)):
                    raise ValueError('Inconsistent XOR code buffers')
            route_specs.append((mode, checked))
        if route_specs[spec['base_route']][0] != 'same':
            raise ValueError('The base route must use its direct stored table')
        if torch.device(device).type == 'meta':
            raise ValueError('Materialized table storage is required')
        with torch.inference_mode(False), torch.no_grad():
            obj = cls.__new__(cls)
            _ValidatedLookup.__init__(obj)
            obj.base_route = spec['base_route']
            obj._shape = tuple(shape)
            obj.register_buffer('base', torch.zeros(shape, dtype=torch.float32, device=device))
            obj.routes = nn.ModuleList([
                _Delta(mode, {name: torch.zeros(size, dtype=dtype, device=device)
                              for name, (size, dtype) in checked.items()}) for mode, checked in route_specs])
            obj.eval()
        obj._validate_structure()
        return obj


def pack_lookup_tables(tables, *, base_route=0, low_bits='auto'):
    """Snapshot and byte-verify a frozen table family; report both byte counts.

    Returns a CompressionResult with an explicit route API, not a model rewrite.
    ``storage_reduced`` requires savings in both logical tensor bytes and unique
    source backing allocations. Aliased input tables cannot manufacture a
    resident saving. Full-table fallback may produce a non-saving result, which
    the caller can retain or decline. External owners of source storage are not
    inspected, so these counts do not imply that an allocation has been freed.
    """
    from .compiler import CompressionResult, state_bytes
    candidate = XorProfileLookup(tables, base_route=base_route, low_bits=low_bits)
    before = sum(value.numel() * value.element_size() for value in tables)
    before_resident = _table_storage_bytes(tables)
    candidate._validate_payload()
    ids = torch.arange(candidate.base.shape[0])
    with torch.inference_mode():
        for route, value in enumerate(tables):
            if not bytes_equal(value, candidate(ids, route=route)):
                raise RuntimeError('Lossless table reconstruction changed value bytes')
    after, after_resident = state_bytes(candidate), _resident_storage_bytes(candidate)
    logical_saving, resident_saving = after < before, after_resident < before_resident
    report = {'method': 'lossless_xor_lookup_tables', 'status': 'byte_verified',
              'storage_reduced': logical_saving and resident_saving,
              'logical_storage_reduced': logical_saving, 'resident_storage_reduced': resident_saving,
              'tensor_bytes_before': before, 'tensor_bytes_after': after,
              'resident_storage_bytes_before': before_resident, 'resident_storage_bytes_after': after_resident,
              'storage_accounting': 'Logical supplied tensor values and unique backing allocations, including view padding; every integer buffer and padded exception slot counted',
              'ownership_scope': 'Supplied tables only; external storage owners are not inspected or freed',
              'routes': len(tables), 'base_route': base_route,
              'validation': {'accepted': True, 'kind': 'complete_table_value_bytes', 'compared_values': sum(t.numel() for t in tables)},
              'guarantee': 'Integer XOR reconstructs every supplied float32 bit; no numerical approximation',
              'scope': 'Stored table identity; original-model table construction and complete-output checks remain separate',
              'performance': 'Selected alternate rows require decoding; no speedup is assumed'}
    candidate._seal_validation()
    return CompressionResult(candidate, report)
