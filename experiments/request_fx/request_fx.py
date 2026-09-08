"""Experimental request-local partial evaluation of a caller-audited FX graph.

This is not an arbitrary-code sandbox or a tracing API. The caller supplies a
trusted GraphModule and declares immutable static tensor arguments. Original
operators and operand layouts are preserved; no weights are fitted or folded.
"""
from __future__ import annotations

import inspect
import math
import operator
from contextlib import contextmanager

import torch
from torch import nn, fx
from torch.nn import functional as F


_FUNCTIONS = {
    operator.add, operator.sub, operator.mul, operator.truediv, operator.neg,
    operator.getitem, torch.add, torch.sub, torch.mul, torch.div, torch.neg,
    torch.sigmoid, torch.relu, torch.cat, torch.unsqueeze,
    F.linear, F.layer_norm, F.relu, F.silu,
}
_METHODS = {"view", "reshape", "unsqueeze", "squeeze", "transpose", "permute"}
_MODULES = {nn.Linear, nn.LayerNorm, nn.Sigmoid, nn.ReLU, nn.SiLU, nn.Identity}


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        return None


def _tensor_signature(tensor):
    if type(tensor) not in (torch.Tensor, nn.Parameter) or tensor.layout != torch.strided or tensor.device.type == "meta":
        raise ValueError("Only ordinary materialized strided tensors are supported")
    return (id(tensor), tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset(),
            tensor.dtype, str(tensor.device), tensor.untyped_storage()._cdata,
            _version(tensor), tensor.requires_grad, tensor.is_neg(), tensor.is_conj())


def _autocast():
    result = []
    for device in ("cpu", "cuda", "mps"):
        try:
            result.append((device, torch.is_autocast_enabled(device), torch.get_autocast_dtype(device)))
        except (TypeError, RuntimeError):
            if device == "cuda":
                result.append((device, torch.is_autocast_enabled(), torch.get_autocast_gpu_dtype()))
    return tuple(result)


def _ambient_guard():
    if torch.is_grad_enabled():
        raise ValueError("Request partial evaluation requires no-grad or inference mode")
    if any(enabled for _, enabled, _ in _autocast()):
        raise ValueError("Autocast is outside this experiment's execution contract")
    from torch.nn.modules import module as module_api
    for name in ("_global_forward_hooks", "_global_forward_pre_hooks", "_global_backward_hooks", "_global_backward_pre_hooks"):
        if getattr(module_api, name, {}):
            raise ValueError("Global module hooks are unsupported")


def _module_attrs(module):
    if type(module) is nn.Linear:
        return (module.in_features, module.out_features)
    if type(module) is nn.LayerNorm:
        return (tuple(module.normalized_shape), module.eps, module.elementwise_affine)
    if type(module) in (nn.ReLU, nn.SiLU):
        return (module.inplace,)
    return ()


def _module_guard(module):
    for child in module.modules():
        if child.training:
            raise ValueError("All source descendants must be in evaluation mode")
        if any(getattr(child, name, {}) for name in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")):
            raise ValueError("Module hooks are unsupported")
        if "forward" in vars(child) or getattr(child, "_compiled_call_impl", None) is not None:
            raise ValueError("Instance forward overrides and compiled wrappers are unsupported")
    for tensor in list(module.parameters()) + list(module.buffers()):
        if tensor.requires_grad:
            raise ValueError("Source parameters and buffers must be frozen")
        _tensor_signature(tensor)


def _source_signature(source):
    _module_guard(source)
    modules = tuple((name, id(module), type(module), id(type(module).forward), _module_attrs(module))
                    for name, module in source.named_modules(remove_duplicate=False))
    state = tuple((kind, name, _tensor_signature(tensor))
                  for kind, entries in (("parameter", source.named_parameters(remove_duplicate=False)),
                                         ("buffer", source.named_buffers(remove_duplicate=False)))
                  for name, tensor in entries)
    return modules, state, str(source.graph)


def _getattr(root, target):
    value = root
    for part in target.split("."):
        value = getattr(value, part)
    return value


def _literal(value):
    if isinstance(value, fx.Node):
        return
    if value is None or type(value) in (str, bool, int, torch.dtype, torch.device):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, (tuple, list)):
        for child in value:
            _literal(child)
        return
    if isinstance(value, dict) and all(type(key) is str for key in value):
        for child in value.values():
            _literal(child)
        return
    if isinstance(value, slice):
        _literal(value.start); _literal(value.stop); _literal(value.step)
        return
    raise ValueError(f"Unsupported FX literal: {type(value).__name__}")


def _audit_graph(source):
    if not isinstance(source, fx.GraphModule):
        raise TypeError("Supply a caller-audited torch.fx.GraphModule; this API does not trace source code")
    _module_guard(source)
    for node in source.graph.nodes:
        _literal(node.args); _literal(node.kwargs)
        if node.op == "placeholder":
            if type(node.target) is not str or not node.target.isidentifier():
                raise ValueError("Variadic FX placeholders are unsupported")
        elif node.op == "get_attr":
            if not isinstance(_getattr(source, node.target), torch.Tensor):
                raise ValueError("Only registered tensor get_attr nodes are supported")
            registered = dict(source.named_parameters(remove_duplicate=False)) | dict(source.named_buffers(remove_duplicate=False))
            if node.target not in registered:
                raise ValueError("Unregistered tensor attributes are unsupported")
        elif node.op == "call_module":
            child = source.get_submodule(node.target)
            if type(child) not in _MODULES or getattr(child, "inplace", False):
                raise ValueError(f"Module outside pure grammar: {type(child).__name__}")
        elif node.op == "call_function":
            if node.target not in _FUNCTIONS or node.kwargs.get("out") is not None or node.kwargs.get("inplace", False):
                raise ValueError(f"Function outside pure grammar: {node.target}")
            if node.target in (F.relu, F.silu) and len(node.args) > 1 and node.args[1] is not False:
                raise ValueError("Positional inplace activation is outside pure grammar")
        elif node.op == "call_method":
            if node.target not in _METHODS:
                raise ValueError(f"Method outside pure grammar: {node.target}")
        elif node.op != "output":
            raise ValueError(f"Unsupported FX node: {node.op}")


def _tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [tensor for child in value for tensor in _tensors(child)]
    if isinstance(value, dict):
        return [tensor for child in value.values() for tensor in _tensors(child)]
    return []


def _dynamic_values(value):
    """Reject subclass dispatch/custom containers before executing the graph."""
    if isinstance(value, torch.Tensor):
        _tensor_signature(value)
        if value.requires_grad:
            raise ValueError("Dynamic input gradients are outside this inference contract")
        return
    if type(value) in (tuple, list):
        for child in value:
            _dynamic_values(child)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for child in value.values():
            _dynamic_values(child)
        return
    if isinstance(value, (tuple, list, dict)):
        raise ValueError("Custom dynamic containers are unsupported")
    _literal(value)


def _resident(tensors):
    storages = {}
    for tensor in tensors:
        if tensor.numel():
            storage = tensor.untyped_storage()
            storages[(str(tensor.device), storage._cdata)] = storage.nbytes()
    return sum(storages.values())


def _registered(module):
    return list(module.parameters()) + list(module.buffers())


class RequestGraph(nn.Module):
    """A live request's graph; use ``partial_evaluate`` as a context manager.

    Dynamic arguments retain their FX names/order. Static arguments are bound
    once and omitted from the dynamic call. An optional static keyword must be
    the identical bound tensor; a replacement is rejected, not compared on GPU.

    Every source weight remains explicitly registered under ``source``. Static
    inputs are registered under ``bindings`` and only static/dynamic boundary
    values are retained in ``dynamic_graph``. This is compute reuse, not weight
    compression. Returned static tensors and all retained state are immutable.
    """
    _compressme_runtime_only = True

    def __init__(self, source, static_inputs, *, allow_unversioned_tensors=False):
        super().__init__()
        _ambient_guard(); _audit_graph(source)
        if type(static_inputs) is not dict or not static_inputs:
            raise ValueError("Declare one or more named static tensor arguments")
        if type(allow_unversioned_tensors) is not bool:
            raise TypeError("allow_unversioned_tensors must be a bool")
        placeholders = {node.target: node for node in source.graph.nodes if node.op == "placeholder"}
        if set(static_inputs) - set(placeholders):
            raise ValueError("Static argument name is absent from the FX graph")
        self.source = source
        self.bindings = nn.Module()
        self.bindings.training = False
        self._input_names = tuple(static_inputs)
        self._input_slots = {}
        for index, (name, value) in enumerate(static_inputs.items()):
            _tensor_signature(value)
            if value.requires_grad:
                raise ValueError("Static inputs must not require gradients")
            slot = f"value_{index}"
            self.bindings.register_buffer(slot, value, persistent=False)
            self._input_slots[name] = slot
        tracked = _registered(source) + list(static_inputs.values())
        if not allow_unversioned_tensors and any(_version(tensor) is None for tensor in tracked):
            raise ValueError("Unversioned inference tensors require an explicit immutable-owner contract")

        self._source_bound = _source_signature(source)
        self._inputs_bound = {name: _tensor_signature(value) for name, value in static_inputs.items()}
        self._settings = (torch.get_float32_matmul_precision(), _autocast())
        static, values = {}, {}
        evaluated = []
        # Each invariant node runs with its original tensor operands, including
        # shape and strides. No operator reassociation or representative row.
        for node in source.graph.nodes:
            if node.op == "placeholder":
                static[node] = node.target in static_inputs
                if static[node]:
                    values[node] = static_inputs[node.target]
            elif node.op == "get_attr":
                static[node] = True
                values[node] = _getattr(source, node.target)
            elif node.op == "output":
                static[node] = False
            else:
                static[node] = all(static[parent] for parent in node.all_input_nodes)
                if static[node]:
                    args = fx.map_arg(node.args, lambda parent: values[parent])
                    kwargs = fx.map_arg(node.kwargs, lambda parent: values[parent])
                    # Normal, versioned outputs make ordinary output mutation
                    # detectable even if the request entered inference_mode.
                    with torch.inference_mode(False), torch.no_grad():
                        if node.op == "call_module":
                            value = source.get_submodule(node.target)(*args, **kwargs)
                        elif node.op == "call_function":
                            value = node.target(*args, **kwargs)
                        else:
                            value = getattr(args[0], node.target)(*args[1:], **kwargs)
                    if not isinstance(value, torch.Tensor):
                        raise ValueError("Static computation must produce tensors in this experiment")
                    values[node] = value
                    evaluated.append(node)

        boundary = [node for node in source.graph.nodes if static[node]
                    and any(not static[user] for user in node.users)]
        holder = nn.Module()
        holder.add_module("_source", source)
        graph, mapped, constant_names = fx.Graph(), {}, []
        boundary_set = set(boundary)
        for node in source.graph.nodes:
            if static[node]:
                if node not in boundary_set:
                    continue
                if node.op == "get_attr":
                    mapped[node] = graph.get_attr("_source." + node.target)
                else:
                    name = f"_constant_{len(constant_names)}"
                    holder.register_buffer(name, values[node], persistent=False)
                    constant_names.append(name)
                    mapped[node] = graph.get_attr(name)
            elif node.op == "call_module":
                mapped[node] = graph.create_node(node.op, "_source." + node.target,
                    fx.map_arg(node.args, lambda parent: mapped[parent]),
                    fx.map_arg(node.kwargs, lambda parent: mapped[parent]), name=node.name)
            else:
                mapped[node] = graph.node_copy(node, lambda parent: mapped[parent])
        graph.lint()
        self.dynamic_graph = fx.GraphModule(holder, graph)
        self.dynamic_graph.training = False
        for child in self.dynamic_graph.modules():
            # Only fresh synthetic namespace holders can differ in mode; the
            # shared source leaves were already checked to be eval.
            child.training = False
        self._dynamic_signature = inspect.signature(self.dynamic_graph.forward)
        self._dynamic_bound = _source_signature(self.dynamic_graph)
        self._constant_names = tuple(constant_names)
        self.training = False
        self._active = True
        source_state = _registered(source)
        prepared_state = _registered(self)
        constants = [getattr(self.dynamic_graph, name) for name in constant_names]
        self.report = {
            "method": "declared_static_fx_partial_evaluation",
            "static_inputs": list(static_inputs),
            "dynamic_inputs": list(self._dynamic_signature.parameters),
            "evaluated_nodes": [node.name for node in evaluated],
            "boundary_nodes": [node.name for node in boundary],
            "source_parameter_values": sum(parameter.numel() for parameter in source.parameters()),
            "retained_parameter_values": sum(parameter.numel() for parameter in self.parameters()),
            "source_registered_storage_bytes": _resident(source_state),
            "bound_input_storage_bytes": _resident(list(static_inputs.values())),
            "boundary_storage_bytes": _resident(constants),
            "retained_registered_storage_bytes": _resident(prepared_state),
            "preparation_live_tensor_storage_bytes": _resident(source_state + _tensors(tuple(values.values()))),
            "preparation_memory_scope": "Source and all retained static intermediates at end of preparation; excludes allocator/kernel workspaces",
            "allow_unversioned_tensors": allow_unversioned_tensors,
            "closed": False,
            "guarantee_scope": "Same real-arithmetic pure graph under immutable static inputs/state; floating bit equality requires backend/shape validation",
        }
        # values/holder are local only; all non-boundary static intermediates
        # become releasable here, and no activation values enter the report.

    def _check(self):
        if not self._active:
            raise RuntimeError("Request scope is closed")
        _ambient_guard()
        if self.training or self._settings != (torch.get_float32_matmul_precision(), _autocast()):
            raise ValueError("Execution settings changed during the request")
        if self._source_bound != _source_signature(self.source):
            raise ValueError("Source graph/module/weight contract changed during the request")
        if self._dynamic_bound != _source_signature(self.dynamic_graph):
            raise ValueError("Prepared graph/constant contract changed during the request")
        for name, slot in self._input_slots.items():
            if _tensor_signature(getattr(self.bindings, slot)) != self._inputs_bound[name]:
                raise ValueError("Static input contract changed during the request")

    def forward(self, *args, **kwargs):
        self._check()
        kwargs = dict(kwargs)
        for name in self._input_names:
            if name in kwargs:
                if kwargs.pop(name) is not getattr(self.bindings, self._input_slots[name]):
                    raise ValueError("A supplied static argument differs from its bound tensor")
        bound = self._dynamic_signature.bind(*args, **kwargs)
        for value in bound.arguments.values():
            _dynamic_values(value)
        return self.dynamic_graph(*bound.args, **bound.kwargs)

    def close(self):
        if self._active:
            self._active = False
            # FX graphs have Python reference cycles. Clear owned constant
            # slots explicitly instead of waiting for cyclic GC to release GPU
            # tensors. Borrowed source modules themselves are not modified.
            for name in self._constant_names:
                self.dynamic_graph._buffers.pop(name, None)
            self.dynamic_graph._modules.clear()
            self._modules.clear()
            self._input_slots.clear()
            self.report["closed"] = True

    def _apply(self, fn, recurse=True):
        raise ValueError("Move source and inputs before preparing a request graph")

    def state_dict(self, *args, **kwargs):
        raise ValueError("Request-local state is not a portable model artifact")


@contextmanager
def partial_evaluate(graph, static_inputs, *, allow_unversioned_tensors=False):
    """Bind immutable static inputs for one request and release them on exit.

    Unsafe writes through .data/foreign aliases are outside the ownership
    contract: metadata/version guards cannot observe every such write. The
    explicit unversioned option is appropriate only for a proven immutable
    inference owner. This context never traces arbitrary Python, modifies the
    supplied graph, or retains predictions across requests.
    """
    prepared = RequestGraph(graph, static_inputs, allow_unversioned_tensors=allow_unversioned_tensors)
    try:
        yield prepared
    finally:
        prepared.close()
