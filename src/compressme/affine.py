"""Exact affine reachability rewrites; no rank truncation and no training.

The reusable primitive composes two affine maps. The FX compiler finds such
edges in ordinary PyTorch forwards, including fan-out with a retained residual.
Architecture adapters can use the same primitive across opaque graph kernels.
"""
from __future__ import annotations

import copy
from collections import Counter
import fnmatch
import torch
from torch import nn, fx


def compose_affine(first: nn.Module, second: nn.Module) -> nn.Linear:
    """Return second(first(x)) as one affine map, exact in real arithmetic.

    Accepts affine modules with 2D .weight and optional .bias (e.g. PyG Linear).
    The caller must establish that their forward really is an affine map.
    Conversion is CPU float64; the returned operator uses the second map's
    original dtype/device. Changed arithmetic order can change last bits.
    Independent optimisation of the fused parameters is a new parameterization.
    """
    for layer in (first, second):
        weight = layer.weight
        if weight.ndim != 2 or not weight.is_floating_point() or weight.device.type == "meta":
            raise ValueError("Affine composition requires materialized real 2D weights")
        if not torch.isfinite(weight).all():
            raise ValueError("Non-finite affine weight")
        bias = getattr(layer, "bias", None)
        if bias is not None and (bias.shape != (weight.shape[0],) or not torch.isfinite(bias).all()):
            raise ValueError("Invalid affine bias")
    a, w = (layer.weight.detach().cpu().double() for layer in (first, second))
    if a.shape[0] != w.shape[1]:
        raise ValueError("Affine dimensions do not compose")
    if first.weight.dtype != second.weight.dtype or first.weight.device != second.weight.device:
        raise ValueError("Affine maps must have the same dtype and device")
    ab, wb = getattr(first, "bias", None), getattr(second, "bias", None)
    bias = torch.zeros(w.shape[0], dtype=torch.float64)
    if ab is not None:
        bias += w @ ab.detach().cpu().double()
    if wb is not None:
        bias += wb.detach().cpu().double()
    # Initialisation must not consume the caller's random sequence.
    with torch.random.fork_rng(devices=[]):
        result = nn.Linear(a.shape[1], w.shape[0], bias=ab is not None or wb is not None,
                           dtype=second.weight.dtype)
    result = result.to(second.weight.device)
    with torch.no_grad():
        result.weight.copy_((w @ a).to(dtype=result.weight.dtype).to(result.weight.device))
        if result.bias is not None:
            result.bias.copy_(bias.to(dtype=result.weight.dtype).to(result.weight.device))
    result.weight.requires_grad_(first.weight.requires_grad or second.weight.requires_grad)
    if result.bias is not None:
        result.bias.requires_grad_((ab is not None and
                                    (ab.requires_grad or second.weight.requires_grad)) or
                                   (wb is not None and wb.requires_grad))
    result.train(second.training)
    return result


def _input(node):
    if len(node.args) == 1 and not node.kwargs:
        return node.args[0]
    if not node.args and set(node.kwargs) == {"input"}:
        return node.kwargs["input"]
    return None


def compile_affine(model: nn.Module, *, include=None, exclude=None,
                   validation=None, output_tolerance=1e-5, absolute_tolerance=1e-5):
    """Discover affine chains, fan-outs and normalization sandwiches with FX.

    Preserves its forward arguments/outputs; returns a GraphModule and report.
    LayerNorm can be crossed with an exact denominator statistic; other
    nonlinearities are barriers. Shared parameters, repeated consumer calls and
    direct .weight uses are retained conservatively. Custom hooks and dynamic
    Python control flow are rejected; use an explicit adapter in those cases.
    This is an inference compiler, not a way to preserve original parameter names
    or an optimizer trajectory. No dimension/rank approximation is performed.
    Repeated calls retain an existing portable graph if changed filters would
    require a second replay recipe; start from the original model to use new
    filters. A same-filter call is allowed when it makes no further rewrites.
    """
    from .compiler import (CompressionResult, parameter_count, state_bytes, _set,
                           _copy_with_compiler_metadata, _preserve_compiler_metadata)
    from .validation import validate
    from .norm_sandwich import compose_norm_sandwich
    from .constant_embeddings import ConstantRowEmbedding
    from .finite_lookup import FiniteTokenLookup, ProjectedEmbeddingLookup
    from .validation import seeded
    if any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks for m in model.modules()):
        raise ValueError("FX rewriting does not support custom module hooks")
    previous_recipe = getattr(model, "_compressme_affine_fx_spec", None)
    requested_recipe = {"include": include, "exclude": exclude}
    if previous_recipe is not None and previous_recipe != requested_recipe:
        candidate = _copy_with_compiler_metadata(model)
        report = {"method": "exact_affine_fx", "status": "retained_existing_affine_recipe",
                  "parameters_before": parameter_count(model), "parameters_after": parameter_count(candidate),
                  "tensor_bytes_before": state_bytes(model), "tensor_bytes_after": state_bytes(candidate),
                  "rewrites": [], "validation": None,
                  "reason": "Changed affine filters require starting from the original model; the existing portable graph is retained"}
        if validation is not None:
            report["validation"] = validate(model, candidate, validation,
                relative_tolerance=output_tolerance, absolute_tolerance=absolute_tolerance)
            if not report["validation"]["accepted"]:
                report["status"] = "rejected_and_rolled_back"
        return CompressionResult(candidate, report)
    if type(model) is nn.Linear:
        return CompressionResult(_copy_with_compiler_metadata(model), {
            "method": "exact_affine_fx", "status": "no_affine_edges",
            "parameters_before": parameter_count(model), "parameters_after": parameter_count(model),
            "tensor_bytes_before": state_bytes(model), "tensor_bytes_after": state_bytes(model),
            "rewrites": [], "validation": None})
    aliases = Counter(id(p) for _, p in model.named_parameters(remove_duplicate=False))
    shared = {name for name, m in model.named_modules()
              if any(aliases[id(p)] > 1 for p in m.parameters(recurse=False))}
    try:
        # FX can evaluate random calls without Proxy arguments during tracing.
        # Reject such captures and restore RNG state instead of silently turning
        # a stochastic model into a deterministic one.
        with seeded(9127):
            rng = torch.random.get_rng_state().clone()
            class ExactTracer(fx.Tracer):
                def is_leaf_module(self,module,qualified_name):
                    return isinstance(module,(ConstantRowEmbedding,FiniteTokenLookup,ProjectedEmbeddingLookup)) or super().is_leaf_module(module,qualified_name)
            source_copy = _copy_with_compiler_metadata(model)
            candidate = fx.GraphModule(source_copy,ExactTracer().trace(source_copy))
            _preserve_compiler_metadata(source_copy, candidate)
            if not torch.equal(rng,torch.random.get_rng_state()):
                raise ValueError("FX trace consumed random numbers; stochastic captures are unsupported")
    except (fx.proxy.TraceError, TypeError, RuntimeError) as exc:
        raise ValueError("Forward is not safely FX traceable; use an architecture adapter") from exc
    original_attrs = set(dict(model.named_buffers())) | set(dict(model.named_parameters()))
    for node in candidate.graph.nodes:
        if node.op == "get_attr" and str(node.target).startswith("_tensor_constant") and str(node.target) not in original_attrs:
            raise ValueError("FX captured a new tensor constant; register constants as buffers or use an adapter")
    candidate._compressme_source_tensor_dtypes = copy.deepcopy(getattr(model,"_compressme_source_tensor_dtypes",
        {n:str(t.dtype).removeprefix("torch.") for n,t in model.state_dict().items()}))
    # Keep the storage recipe when composing exact passes, so replay can start
    # from the caller's original dense embedding factory.
    candidate._compressme_constant_embedding_rewrites = [
        {"path":path,**module.recipe()} for path,module in model.named_modules()
        if isinstance(module,ConstantRowEmbedding)]
    candidate._compressme_finite_lookup_rewrites = [
        {"path":path,**module.recipe()} for path,module in model.named_modules()
        if isinstance(module,(FiniteTokenLookup,ProjectedEmbeddingLookup))]
    calls = Counter(n.target for n in candidate.graph.nodes if n.op == "call_module")
    direct_attrs = [str(n.target) for n in candidate.graph.nodes if n.op == "get_attr"]
    records = []
    for node in list(candidate.graph.nodes):
        if node.op != "call_module" or calls[node.target] != 1:
            continue
        parent = _input(node)
        if not isinstance(parent, fx.Node) or parent.op != "call_module":
            continue
        raw = _input(parent)
        if not isinstance(raw, fx.Node):
            continue
        if include is not None and not any(fnmatch.fnmatchcase(str(node.target), p) for p in include):
            continue
        if exclude and any(fnmatch.fnmatchcase(str(node.target), p) for p in exclude):
            continue
        paths = (str(parent.target), str(node.target))
        if any(p in shared or any(a.startswith(p + ".") for a in direct_attrs) for p in paths):
            continue
        first, second = (candidate.get_submodule(p) for p in paths)
        if type(first) is nn.LayerNorm and type(second) is nn.Linear:
            expansion_node = _input(parent)
            if not isinstance(expansion_node,fx.Node) or expansion_node.op != "call_module":
                continue
            expansion = candidate.get_submodule(str(expansion_node.target))
            original_input = _input(expansion_node)
            if type(expansion) is not nn.Linear or not isinstance(original_input,fx.Node):
                continue
            expansion_path = str(expansion_node.target)
            if expansion_path in shared or any(a.startswith(expansion_path+".") for a in direct_attrs):
                continue
            replacement, local_report = compose_norm_sandwich(expansion,first,second)
            if state_bytes(replacement) >= state_bytes(second):
                continue
            candidate = _set(candidate,str(node.target),replacement)
            node.args,node.kwargs = (original_input,),{}
            records.append({"producer":expansion_path,"normalization":str(parent.target),
                            "consumer":str(node.target),"method":"exact_normalization_statistic",
                            "isolated_block":local_report.to_dict()})
            if not parent.users:
                candidate.graph.erase_node(parent)
            if not expansion_node.users:
                candidate.graph.erase_node(expansion_node)
            continue
        if type(first) is not nn.Linear or type(second) is not nn.Linear:
            continue
        # Require a per-consumer saving even when other branches retain first.
        if first.in_features >= first.out_features:
            continue
        replacement = compose_affine(first, second)
        if state_bytes(replacement) >= state_bytes(second):
            continue
        candidate = _set(candidate, str(node.target), replacement)
        node.args, node.kwargs = (raw,), {}
        records.append({"producer": str(parent.target), "consumer": str(node.target),
                        "input_width_before": first.out_features,
                        "input_width_after": first.in_features,
                        "consumer_parameters_before": parameter_count(second),
                        "consumer_parameters_after": parameter_count(replacement)})
        if not parent.users:
            candidate.graph.erase_node(parent)
    candidate.graph.lint()
    candidate.recompile()
    # Only prune modules that were actually rewritten away, never unrelated
    # registered state or an opaque module's inner parameter dependencies.
    live = {str(n.target) for n in candidate.graph.nodes if n.op in {"call_module", "get_attr"}}
    removable = {r["producer"] for r in records} | {r["normalization"] for r in records if "normalization" in r}
    for path in removable:
        if not any(p == path or p.startswith(path + ".") for p in live):
            candidate.delete_submodule(path)
    incremental_records = records if previous_recipe is not None else []
    if incremental_records:
        candidate = _copy_with_compiler_metadata(model)
        records = []
    elif previous_recipe is None:
        candidate._compressme_affine_fx_spec = requested_recipe
    report = {"method": "exact_affine_fx", "status": "compiled",
              "parameters_before": parameter_count(model), "parameters_after": parameter_count(candidate),
              "tensor_bytes_before": state_bytes(model), "tensor_bytes_after": state_bytes(candidate),
              "rewrites": records,
              "guarantee": "same mathematical forward in real arithmetic; floating-point order changes",
              "validation": None}
    if incremental_records:
        report.update(status="retained_existing_affine_recipe", proposed_rewrites=incremental_records,
                      reason="Further affine rewrites require a second replay recipe; start from the original model with the intended filters")
    if validation is not None:
        report["validation"] = validate(model, candidate, validation,
                                          relative_tolerance=output_tolerance,
                                          absolute_tolerance=absolute_tolerance)
        if not report["validation"]["accepted"]:
            report["status"] = "rejected_and_rolled_back"
            report["proposal_parameters_after"] = report["parameters_after"]
            candidate = _copy_with_compiler_metadata(model)
            report["parameters_after"] = parameter_count(model)
            report["tensor_bytes_after"] = state_bytes(model)
    return CompressionResult(candidate, report)
