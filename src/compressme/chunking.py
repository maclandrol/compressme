"""Remove an unnecessary output copy from an audited one-chunk computation.

The input preparation and fallback copy follow OpenFold's Apache-2.0 chunk_layer
implementation (AlQuraishi Laboratory, 2021), as vendored by Nesso commit
6c72f66720d9d3447fd73c515cda963e39128b1f. This helper does not trace arbitrary
Python or certify that a caller-supplied layer is pure.
"""
from __future__ import annotations

from collections.abc import Callable
from functools import partial
import torch


def _storage(tensor):
    return (str(tensor.device), tensor.untyped_storage()._cdata)


def _map(fn, value):
    if type(value) is dict:
        return {key: _map(fn, item) for key, item in value.items()}
    if type(value) is tuple:
        return tuple(_map(fn, item) for item in value)
    if type(value) is list:
        return [_map(fn, item) for item in value]
    if isinstance(value, torch.Tensor):
        return fn(value)
    raise ValueError(f"Tree of type {type(value)} not supported")


def _copied_output(output, flat_batch_dim, chunk_size, orig_batch_dims):
    """Finish the original one-chunk allocation/assignment without rerunning it."""
    out = _map(lambda t: t.new_zeros((flat_batch_dim,) + t.shape[1:]), output)
    if type(output) is dict:
        def assign(destination, source):
            for key, value in destination.items():
                if type(value) is dict:
                    assign(value, source[key])
                else:
                    value[:chunk_size] = source[key]
        assign(out, output)
    elif type(output) is tuple:
        for destination, source in zip(out, output):
            destination[:chunk_size] = source
    elif type(output) is torch.Tensor:
        out[:chunk_size] = output
    else:
        raise ValueError("Not supported")
    return _map(lambda t: t.view(orig_batch_dims + t.shape[1:]), out)


_UNPREPARED = object()


def _audit_owner(layer):
    """Return immutable owner storage identities, or None for an unsafe owner."""
    owner = layer.func if isinstance(layer, partial) else layer
    owner = getattr(owner, "__self__", owner)
    owner_tensors = ()
    if isinstance(owner, torch.nn.Module):
        if any(m.training or "forward" in vars(m)
               or getattr(m, "_compiled_call_impl", None) is not None
               or any(getattr(m, name, {}) for name in (
                   "_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks"))
               for m in owner.modules()):
            return None
        owner_tensors = tuple(owner.parameters()) + tuple(owner.buffers())
        if any(t.requires_grad or t.layout != torch.strided or t.device.type == "meta"
               for t in owner_tensors):
            return None
    return frozenset(_storage(t) for t in owner_tensors)


def _dispatch_single_chunk(
    layer: Callable,
    inputs: dict,
    chunk_size: int,
    no_batch_dims: int,
    *,
    fallback: Callable,
    immutable_inference: bool = False,
    low_mem: bool = False,
    _out=None,
    _add_into_out: bool = False,
    statistics: dict | None = None,
    owner_storages=_UNPREPARED,
):
    """Use the original flattened layer call while eliding a redundant copy.

``fallback`` must implement the OpenFold chunk_layer calling convention. The
caller explicitly attests to an immutable, exclusively owned inference call:
the layer is deterministic, does not mutate its inputs, and does not expose its
temporary outputs through hooks or other side effects. This trust boundary is
needed because returning a temporary changes its aliases if another observer
retains it. Gradient-enabled calls, unsupported trees/layouts/flags and multiple
chunks use the original dispatcher unchanged. A non-contiguous, broadcasted or
input-aliasing output uses the original allocation/copy, without a second call.

No parameters or activations are retained. Only ordinary tensors, non-empty
batch dimensions and one output tensor with independently owned contiguous
storage qualify for copy elision. All floating-point operators inside the layer
receive the original shapes, strides and values.
"""
    def record(key):
        if statistics is not None:
            statistics[key] = statistics.get(key, 0) + 1

    def original():
        record("fallback_calls")
        return fallback(layer, inputs, chunk_size, no_batch_dims,
                        low_mem=low_mem, _out=_out, _add_into_out=_add_into_out)

    if (immutable_inference is not True or torch.is_grad_enabled()
            or low_mem is not False or _out is not None or _add_into_out is not False
            or type(chunk_size) is not int or chunk_size < 1
            or type(no_batch_dims) is not int or no_batch_dims < 1
            or type(inputs) is not dict or not inputs
            or any(type(key) is not str for key in inputs)):
        return original()
    leaves = tuple(inputs.values())
    if any(type(t) is not torch.Tensor or t.layout != torch.strided
           or t.device.type == "meta" or t.ndim < no_batch_dims
           or any(type(d) is not int or d < 1 for d in t.shape[:no_batch_dims])
           for t in leaves):
        return original()
    if owner_storages is _UNPREPARED:
        owner_storages = _audit_owner(layer)
    if owner_storages is None:
        return original()
    initial_dims = [t.shape[:no_batch_dims] for t in leaves]
    orig_batch_dims = tuple(max(s) for s in zip(*initial_dims))
    flat_batch_dim = 1
    for dimension in orig_batch_dims:
        flat_batch_dim *= dimension
    if flat_batch_dim > chunk_size:
        return original()

    # Preserve the original broadcast exception for all-singleton batch axes,
    # flattening and first-chunk slicing. Calling layer(**inputs) would change
    # GEMM ranks/layouts and can change floating-point results.
    chunks = {}
    for key, tensor in inputs.items():
        if sum(tensor.shape[:no_batch_dims]) != no_batch_dims:
            tensor = tensor.expand(orig_batch_dims + tensor.shape[no_batch_dims:])
        tensor = tensor.reshape(-1, *tensor.shape[no_batch_dims:])
        chunks[key] = tensor[:chunk_size] if tensor.shape[0] != 1 else tensor
    output = layer(**chunks)
    if (type(output) is torch.Tensor and output.layout == torch.strided
            and output.device.type != "meta" and output.ndim >= 1
            and output.shape[0] == flat_batch_dim and output.is_contiguous()
            and not output.is_conj() and not output.is_neg()
            and output.storage_offset() == 0
            and output.untyped_storage().nbytes() == output.numel() * output.element_size()
            and _storage(output) not in owner_storages
            and _storage(output) not in {_storage(t) for t in leaves + tuple(chunks.values())}):
        record("single_chunk_elisions")
        return output.view(orig_batch_dims + output.shape[1:])
    record("single_chunk_copies")
    return _copied_output(output, flat_batch_dim, chunk_size, orig_batch_dims)


def single_chunk_or_fallback(
    layer: Callable,
    inputs: dict,
    chunk_size: int,
    no_batch_dims: int,
    *,
    fallback: Callable,
    immutable_inference: bool = False,
    low_mem: bool = False,
    _out=None,
    _add_into_out: bool = False,
    statistics: dict | None = None,
):
    """Use the original flattened layer call while eliding a redundant copy.

``fallback`` must implement the OpenFold chunk_layer calling convention. The
caller explicitly attests to an immutable, exclusively owned inference call:
the layer is deterministic, does not mutate its inputs, and does not expose its
temporary outputs through hooks or other side effects. This trust boundary is
needed because returning a temporary changes its aliases if another observer
retains it. Gradient-enabled calls, unsupported trees/layouts/flags and multiple
chunks use the original dispatcher unchanged. A non-contiguous, broadcasted or
input-aliasing output uses the original allocation/copy, without a second call.

No parameters or activations are retained. Only ordinary tensors, non-empty
batch dimensions and one output tensor with independently owned contiguous
storage qualify for copy elision. All floating-point operators inside the layer
receive the original shapes, strides and values.
"""
    return _dispatch_single_chunk(
        layer, inputs, chunk_size, no_batch_dims, fallback=fallback,
        immutable_inference=immutable_inference, low_mem=low_mem, _out=_out,
        _add_into_out=_add_into_out, statistics=statistics,
    )


def prepare_single_chunk(
    layer: Callable,
    *,
    fallback: Callable,
    immutable_inference: bool = False,
    statistics: dict | None = None,
):
    """Prepare a dispatcher for an explicitly immutable inference owner.

    This is the same copy-elision operation as ``single_chunk_or_fallback``,
    with the module hierarchy and owned storage identities audited once. The
    caller must keep the callable, its configuration, weights, buffers, hooks,
    training state and device unchanged for the returned dispatcher's entire
    lifetime. Discard and recreate it after any such change. Audited request
    scopes can enforce model-state checks at their entry and exit boundaries.
    Unsafe ``.data`` mutation and external aliases remain outside that contract.

    Dynamic input trees, layouts, chunk counts, gradient mode, flags and output
    aliases are still checked on every call. Unsupported owners always use the
    original dispatcher. No tensors or parameter copies are cached: the closure
    retains the supplied callable and only owner-storage identity metadata.
    This does not establish the purity of arbitrary caller-provided Python code.
    """
    if immutable_inference is not True or torch.is_grad_enabled():
        raise ValueError("Preparation requires immutable_inference=True in no-grad/inference mode")
    owner_storages = _audit_owner(layer)

    def dispatch(inputs, chunk_size, no_batch_dims, *, low_mem=False, _out=None,
                 _add_into_out=False):
        return _dispatch_single_chunk(
            layer, inputs, chunk_size, no_batch_dims, fallback=fallback,
            immutable_inference=True, low_mem=low_mem, _out=_out,
            _add_into_out=_add_into_out, statistics=statistics,
            owner_storages=owner_storages,
        )

    return dispatch
