"""Experimental bounded cache of parameter-only rows at original GEMM shapes.

The source is passed on every call and is not owned/duplicated by this helper.
Only outputs of known constant parameter rows are stored. Other rows of the
temporary tensor are zeros, never molecule/cell inputs or predicted outputs.
This preserves the source operation shapes, not a universal bitwise theorem
about opaque backend implementations. Validate on the target device first.
"""
from __future__ import annotations

from collections import OrderedDict
import math
import weakref
import torch
from torch import nn
from .runtime import autocast_enabled
from .finite_lookup import RowNormalize, _POINTWISE, _flatten, _has_hooks


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        return None


def _configuration(source, width):
    """Audited dense row-local grammar, with mutable settings in the cache key."""
    if type(source) is not nn.Sequential:
        raise ValueError("Use an ordinary nn.Sequential to establish the dense row-local graph")
    if any(m.training or _has_hooks(m) or "forward" in vars(m) for m in source.modules()):
        raise ValueError("Shape-matched constants require unmodified, hook-free evaluation modules")
    fingerprint = []
    for layer in _flatten(source):
        if type(layer) is nn.Linear:
            if layer.in_features != width or tuple(layer.weight.shape) != (layer.out_features, width):
                raise ValueError("Linear does not match the input row width")
            if layer.bias is not None and tuple(layer.bias.shape) != (layer.out_features,):
                raise ValueError("Linear bias does not match its output row width")
            width = layer.out_features
        elif type(layer) is nn.LayerNorm:
            if tuple(layer.normalized_shape) != (width,) or not math.isfinite(layer.eps) or layer.eps <= 0:
                raise ValueError("LayerNorm must operate only on the final feature dimension")
        elif type(layer) is RowNormalize:
            if layer.p != 2 or not math.isfinite(layer.eps) or layer.eps <= 0:
                raise ValueError("Invalid last-dimension normalization")
        elif type(layer) in _POINTWISE or type(layer) is nn.Dropout:
            pass
        else:
            raise ValueError(f"Unproved row-local operation: {type(layer).__name__}")
        scalar_settings = tuple(sorted((name, value) for name, value in vars(layer).items()
            if not name.startswith("_") and isinstance(value, (str, int, float, bool, tuple, type(None)))))
        fingerprint.append((weakref.ref(layer), type(layer), scalar_settings))
    return tuple(fingerprint), width


class ShapeMatchedConstantRows:
    """Cache known rows of a source Sequential at an exact leading tensor shape.

    call(source, input_shape, assignments) returns selected output rows in the
    order of assignments. Each assignment is (flat_row_index, constant_tensor).
    Pass the original stable parameter/buffer tensors as constants, not freshly
    cloned views; their identities and mutation versions establish cache reuse.

    Weight/settings/device/dtype changes invalidate entries. Inference tensors
    without version counters and calls needing gradients are evaluated without
    caching. Fully frozen source tensors and constants can be cached even with
    global gradient recording enabled. Direct unsafe .data/storage writes bypass PyTorch version tracking
    and are outside the supported mutation contract. clear() is explicit.

    A cache miss evaluates a zero template with the complete original shape,
    so first-call work is the full encoder and temporary memory scales with
    total_rows times the encoder's input/intermediate widths. The byte limit
    bounds only persistent cached selected rows, not that temporary workspace,
    caller-held outputs or backend allocator reservations. Cache hits copy only
    selected_rows times output_width values. Report cold and warm timing
    separately; this helper is experimental and has no universal bitwise claim.
    """
    def __init__(self, max_entries=8, max_bytes=16 * 1024 * 1024):
        if type(max_entries) is not int or max_entries < 0 or type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("Cache limits must be nonnegative integers")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.clear()

    def clear(self):
        self._cache = OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0

    def __getstate__(self):
        return {"max_entries": self.max_entries, "max_bytes": self.max_bytes}

    def __setstate__(self, state):
        self.max_entries = state["max_entries"]
        self.max_bytes = state["max_bytes"]
        self.clear()

    def __call__(self, source, input_shape, assignments):
        shape = tuple(input_shape)
        if len(shape) < 1 or any(type(n) is not int or n < 0 for n in shape) or shape[-1] < 1:
            raise ValueError("Expected a complete dense input shape with positive feature width")
        row_count = math.prod(shape[:-1])
        pairs = tuple(assignments)
        if not pairs:
            raise ValueError("Provide at least one known constant row")
        indices, constants = zip(*pairs)
        if any(type(i) is not int or not 0 <= i < row_count for i in indices) or len(set(indices)) != len(indices):
            raise ValueError("Constant row positions must be distinct valid flat row indices")
        if any(not isinstance(x, torch.Tensor) or x.numel() != shape[-1] or not x.is_floating_point() for x in constants):
            raise ValueError("Each known constant must contain exactly one real input row")
        device, dtype = constants[0].device, constants[0].dtype
        if any(x.device != device or x.dtype != dtype for x in constants):
            raise ValueError("All constant rows must share an explicit device/dtype")
        if autocast_enabled(device.type):
            raise ValueError("Shape-matched constants require explicit dtypes, not autocast")
        configuration, output_width = _configuration(source, shape[-1])
        state = tuple(source.parameters()) + tuple(source.buffers())
        if any(x.device != device or x.device.type == "meta" for x in state):
            raise ValueError("Source and constants must share a materialized device")
        if any(x.layout != torch.strided or x.is_quantized or x.is_complex()
               or (x.is_floating_point() and x.dtype != dtype) for x in state):
            raise ValueError("Source floating tensors and constants must share the same explicit dtype")
        if any(getattr(x, "_backward_hooks", None) for x in source.parameters()):
            raise ValueError("Parameter hooks are unsupported")
        versions = tuple(_version(x) for x in state + constants)
        needs_gradients = torch.is_grad_enabled() and any(x.requires_grad for x in state + constants)
        cacheable = (not needs_gradients and all(v is not None for v in versions)
                     and self.max_entries > 0 and self.max_bytes > 0)
        execution_settings = (torch.get_float32_matmul_precision(), torch.get_num_threads(),
                              torch.are_deterministic_algorithms_enabled(), torch.is_inference_mode_enabled())
        key = (weakref.ref(source), shape, str(device), dtype, tuple(indices), configuration, execution_settings,
               tuple((weakref.ref(x), version, str(x.dtype), str(x.device)) for x, version in zip(state + constants, versions)))
        if cacheable and key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            # Return an independent tensor so consumer mutation cannot corrupt
            # a future row. This copies only the selected constant outputs.
            return self._cache[key].clone()
        self.misses += 1
        template = torch.zeros(shape, dtype=dtype, device=device)
        row_ids = torch.tensor(indices, dtype=torch.long, device=device)
        known = torch.stack([x.reshape(-1) for x in constants])
        template.reshape(row_count, shape[-1]).index_copy_(0, row_ids, known)
        output = source(template)
        if output.shape != (*shape[:-1], output_width) or output.dtype != dtype:
            raise ValueError("Source violated the audited row shape/dtype contract")
        selected = output.reshape(row_count, output_width).index_select(0, row_ids)
        if not torch.isfinite(selected).all():
            raise ValueError("Known constant outputs are nonfinite")
        size = selected.numel() * selected.element_size()
        if (cacheable and size <= self.max_bytes
                and versions == tuple(_version(x) for x in state + constants)
                and configuration == _configuration(source, shape[-1])[0]):
            # Detached output rows own fresh storage; the large template/output
            # tensor and all data-independent zero rows are released on return.
            stored = selected.detach().clone()
            previous = self._cache.pop(key, None)
            if previous is not None:
                self._bytes -= previous.numel() * previous.element_size()
            self._cache[key] = stored
            self._bytes += size
            while len(self._cache) > self.max_entries or self._bytes > self.max_bytes:
                _, removed = self._cache.popitem(last=False)
                self._bytes -= removed.numel() * removed.element_size()
        return selected
