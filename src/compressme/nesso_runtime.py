"""Opt-in, request-scoped exact-operation reuse for pinned Nesso-1 inference.

No weights, precision, output fields or random-number calls are removed. These
optimizations reduce temporary copies/repeated conditioning computation; actual
speed and complete output agreement must be checked for the chosen workload.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import partial
import hashlib
import inspect
from pathlib import Path
from types import MethodType
import torch
from torch import nn

from .chunking import prepare_single_chunk
from .contractions import channelwise_token_contraction


NESSO_REVISION = "6c72f66720d9d3447fd73c515cda963e39128b1f"
_SOURCE_HASHES = {
    "models/nesso1.py": "279ce81038665773610d0eae98c9414ec299a2be3e24731f261e65abb33fd6be",
    "modules/esm_module.py": "32dca1d5f5df84cca4e6936f712548f779a12c3484b686d5d9eccd76166c5baa",
    "layers/triangular_attention/attention.py": "0cfcf385560e06b24e9c770cdd4c2445b9956ce703c31d73818c7b8b3157f422",
    "layers/triangular_attention/utils.py": "78a0ff1a0f748a724c13df207d8c769412e97e7d5ed6a245004555a802c01db0",
    "layers/triangular_attention/primitives.py": "1a32ae7ce437e636a1f95abab5bae22911395c66d063f672661c226107c0f7ff",
    "layers/triangular_mult.py": "746b1b227f7e252226cae19df931327bcb25e2174d352231c03d8781d06e6a31",
}


def _verified_types():
    # Optional Nesso/Lightning imports happen only when its adapter is requested.
    from nesso.model.models.nesso1 import Nesso1
    from nesso.model.modules.esm_module import ESMModule
    from nesso.model.layers.triangular_attention.attention import (
        TriangleAttention, TriangleAttentionEndingNode,
    )
    from nesso.model.layers.triangular_attention.primitives import Attention
    from nesso.model.layers.triangular_attention.utils import chunk_layer
    source = Path(inspect.getsourcefile(Nesso1)).resolve().parent.parent
    for relative, expected in _SOURCE_HASHES.items():
        file = source / relative
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Nesso runtime requires source revision {NESSO_REVISION}: {relative}")
    return Nesso1, ESMModule, (TriangleAttention, TriangleAttentionEndingNode), Attention, chunk_layer


def _autocast():
    states = []
    for device in ("cpu", "cuda", "mps"):
        try:
            enabled = torch.is_autocast_enabled(device)
        except TypeError:  # PyTorch 2.2's device-specific query API.
            enabled = (torch.is_autocast_cpu_enabled() if device == "cpu"
                       else torch.is_autocast_enabled() if device == "cuda" else False)
        states.append((device, enabled))
    return tuple(states)


def _tensor_signature(tensor):
    if type(tensor) not in (torch.Tensor, nn.Parameter) or tensor.layout != torch.strided:
        raise ValueError("Nesso runtime requires ordinary dense tensors")
    try:
        version = tensor._version
    except RuntimeError:
        version = None  # Inference tensors rely on the explicit immutability contract.
    return (id(tensor), str(tensor.device), tensor.dtype, tuple(tensor.shape),
            tuple(tensor.stride()), tensor.storage_offset(), tensor.data_ptr(),
            version, tensor.requires_grad, tensor.is_conj(), tensor.is_neg())


def _state_signature(model):
    # Inspect all registered state in one ownership walk. Keep every aliased
    # path and the original complete tensor metadata, but avoid recursively
    # rediscovering the same modules for parameters and then again for buffers.
    modules, parameters, buffers = [], [], []
    for name, module in model.named_modules(remove_duplicate=False):
        modules.append((name, id(module), type(module), module.training))
        for key, tensor in module._parameters.items():
            if tensor is not None:
                parameters.append((name, key, _tensor_signature(tensor)))
        for key, tensor in module._buffers.items():
            if tensor is not None:
                buffers.append((name, key, _tensor_signature(tensor)))
    return tuple(modules), tuple(parameters), tuple(buffers)


def _require_frozen(model):
    if torch.is_grad_enabled() or any(enabled for _, enabled in _autocast()):
        raise ValueError("Nesso runtime requires no-grad/inference mode with autocast disabled")
    if any(getattr(nn.modules.module, name, {}) for name in (
            "_global_forward_hooks", "_global_forward_pre_hooks", "_global_backward_hooks",
            "_global_backward_pre_hooks")):
        raise ValueError("Nesso runtime cannot preserve global hooks")
    for module in model.modules():
        if (module.training or "forward" in vars(module)
                or getattr(module, "_compiled_call_impl", None) is not None
                or any(getattr(module, name, {}) for name in (
                    "_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks"))):
            raise ValueError("Nesso runtime requires hook-free evaluation modules without forward overrides or compiled calls")
    if any(t.requires_grad for t in list(model.parameters()) + list(model.buffers())):
        raise ValueError("Nesso runtime requires frozen parameters and buffers")


@contextmanager
def nesso_inference_optimizations(
    model: nn.Module,
    *,
    single_chunk: bool = True,
    cache_esm: bool = False,
    pack_triangles: bool = False,
    immutable_request: bool = False,
):
    """Temporarily optimize an exclusively owned, frozen Nesso inference model.

The caller must explicitly pass ``immutable_request=True`` and run inside
``torch.no_grad()`` or ``torch.inference_mode()`` with autocast disabled. Model
bindings/configuration/weights and request inputs must not be changed until the
call returns. Tensor identity/layout/version checks catch ordinary changes, but
unsafe ``.data`` writes and unversioned inference tensors are not a universal
immutability proof. Do not share this mutable module instance across threads.

Single-chunk attention keeps the original flattened arguments and operations.
Optional ESM reuse computes the original conditioning expressions once for each
consecutive static input binding (full then cropped input), and retains only its
pair update until that top-level forward finishes. Cropped inputs are recomputed
with their original shapes, never substituted by a sliced full-shape GEMM result.
The yielded dictionary records calls and peak temporary cached tensor bytes.
Every replaced instance method is restored on normal or exceptional exit; no
global function is patched and no state_dict key is added or removed.

Optional triangle packing prepares contiguous matrices before the same batched
contraction. Its real-arithmetic identity is general, but changed layouts can
select different floating-point kernels: validate complete outputs on the chosen
backend. It falls back to the original path when CUDA kernels are requested.
"""
    if immutable_request is not True or any(type(flag) is not bool for flag in (single_chunk, cache_esm, pack_triangles)):
        raise ValueError("Explicit immutable_request=True and boolean optimization flags are required")
    types = _verified_types()
    Nesso1, ESMModule, triangle_types, Attention, chunk_layer = types
    if type(model) is not Nesso1:
        raise ValueError("Expected the original pinned Nesso1 class")
    if hasattr(model, "_compressme_nesso_runtime_owner"):
        raise ValueError("Nesso runtime scopes cannot be nested or shared")
    _require_frozen(model)
    triangles = [module for module in model.modules() if type(module) in triangle_types]
    if any("_chunk" in vars(module) or type(module.mha) is not Attention for module in triangles):
        raise ValueError("Nesso triangle attention has custom chunk/attention behavior")
    if cache_esm:
        esm = model.esm_module
        expected = (nn.LayerNorm, nn.Linear, nn.ReLU, nn.Dropout, nn.Linear)
        if (type(esm) is not ESMModule or type(esm.esm_mlp) is not nn.Sequential
                or tuple(type(child) for child in esm.esm_mlp) != expected
                or any(type(getattr(esm, name)) is not nn.Linear
                       for name in ("s_inputs_proj", "esm_z_1", "esm_z_2"))):
            raise ValueError("ESM reuse requires the original exact ESMModule children")
    triangle_products = []
    if pack_triangles:
        from nesso.model.layers.triangular_mult import TriangleMultiplicationIncoming, TriangleMultiplicationOutgoing
        triangle_products = [module for module in model.modules()
                             if type(module) in (TriangleMultiplicationIncoming, TriangleMultiplicationOutgoing)]
    seal = _state_signature(model)
    autocast = _autocast()
    report = {"nesso_revision": NESSO_REVISION, "requests": 0,
              "single_chunk_modules": len(triangles) if single_chunk else 0,
              "single_chunk_elisions": 0, "single_chunk_copies": 0, "fallback_calls": 0,
              "esm_computations": 0, "esm_reuses": 0, "peak_cached_bytes": 0}
    report['packed_triangle_modules'] = len(triangle_products) if pack_triangles else 0
    report['packed_triangle_calls'] = 0
    cache = {}
    active = False
    patched = []
    owner = object()

    def install(module, name, function):
        # All affected attributes were inherited methods at entry.
        if name in vars(module):
            raise ValueError(f"Cannot replace custom {name}")
        setattr(module, name, MethodType(function, module))
        patched.append((module, name))

    def runtime_guard():
        if torch.is_grad_enabled() or _autocast() != autocast:
            raise ValueError("Nesso runtime grad/autocast state changed during its scope")

    def make_chunk(module, original):
        # The enclosing scope audits and seals all owner bindings once, then
        # checks that seal before and after each request. Reuse the immutable
        # storage audit across the hundreds of attention calls in that request.
        prepared = prepare_single_chunk(
            partial(module.mha, use_kernels=False), fallback=chunk_layer,
            immutable_inference=True, statistics=report,
        )

        def chunk(module, x, tri_bias, mask_bias, mask, chunk_size, use_kernels=False):
            if not active or use_kernels is not False:
                return original(x, tri_bias, mask_bias, mask, chunk_size, use_kernels)
            runtime_guard()
            return prepared(
                {"q_x": x, "kv_x": x, "tri_bias": tri_bias, "mask_bias": mask_bias, "mask": mask},
                chunk_size, len(x.shape[:-2]),
            )
        return chunk

    original_forward = model.forward
    original_esm = model.esm_module.forward if cache_esm else None

    def make_product(original, direction):
        def product(module, x, mask, use_kernels=False):
            if not active or use_kernels:
                return original(x, mask, use_kernels)
            runtime_guard()
            # Preserve all learned projections, masks, casts and gates in their
            # original order; only the contraction's input layout is changed.
            x = module.norm_in(x)
            x_in = x
            x = module.p_in(x) * module.g_in(x).sigmoid()
            x = x * mask.unsqueeze(-1)
            a, b = torch.chunk(x.float(), 2, dim=-1)
            x = channelwise_token_contraction(a, b, direction=direction)
            report['packed_triangle_calls'] += 1
            return module.p_out(module.norm_out(x)) * module.g_out(x_in).sigmoid()
        return product

    def esm_forward(module, z, s_inputs, s_esm, pair_mask, use_kernels=False):
        if not active:
            return original_esm(z, s_inputs, s_esm, pair_mask, use_kernels)
        runtime_guard()
        key = tuple(_tensor_signature(t) for t in (s_inputs, s_esm, pair_mask))
        if cache.get("key") != key:
            # Release a previous full-shape update before constructing the
            # cropped replacement. Never retain a dictionary of request sizes.
            cache.clear()
            # These expressions retain the pinned source's evaluation order.
            if s_esm.dim() == 4 and module.use_esm_all_layers:
                weights = module.esm_layer_weights.softmax(0)
                s_esm = torch.einsum("bnld,l->bnd", s_esm, weights)
            s_esm_proj = module.esm_mlp(s_esm)
            s = module.s_inputs_proj(s_inputs) + s_esm_proj
            left = module.esm_z_1(s)
            right = module.esm_z_2(s)
            delta = left[:, :, None, :] + right[:, None, :, :]
            delta = delta * pair_mask.unsqueeze(-1)
            cache.update(key=key, delta=delta)
            report["esm_computations"] += 1
            report["peak_cached_bytes"] = max(report["peak_cached_bytes"], delta.numel() * delta.element_size())
        else:
            report["esm_reuses"] += 1
        return z + cache["delta"]

    def forward(module, *args, **kwargs):
        nonlocal active
        runtime_guard()
        if active:
            raise ValueError("Concurrent or recursive Nesso requests are unsupported")
        if _state_signature(module) != seal:
            raise ValueError("Nesso model state changed; exit and reopen the runtime scope")
        active = True
        cache.clear()
        report["requests"] += 1
        error = None
        try:
            return original_forward(*args, **kwargs)
        except BaseException as exc:
            error = exc
            raise
        finally:
            cache.clear()
            active = False
            try:
                if _state_signature(module) != seal:
                    raise ValueError("Nesso model mutated during an immutable request")
            except BaseException as audit_error:
                if error is None:
                    raise
                if hasattr(error, "add_note"):
                    error.add_note(f"Nesso runtime exit audit also failed: {audit_error}")

    try:
        model._compressme_nesso_runtime_owner = owner
        if single_chunk:
            for module in triangles:
                install(module, "_chunk", make_chunk(module, module._chunk))
        if cache_esm:
            install(model.esm_module, "forward", esm_forward)
        if pack_triangles:
            for module in triangle_products:
                direction = 'outgoing' if type(module) is TriangleMultiplicationOutgoing else 'incoming'
                install(module, 'forward', make_product(module.forward, direction))
        install(model, "forward", forward)
        yield report
    finally:
        cache.clear()
        active = False
        for module, name in reversed(patched):
            # Removing only instance overrides restores the original class
            # method without copying modules or changing parameter ownership.
            vars(module).pop(name, None)
        if getattr(model, "_compressme_nesso_runtime_owner", None) is owner:
            delattr(model, "_compressme_nesso_runtime_owner")
