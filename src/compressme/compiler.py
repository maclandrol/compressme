"""Conservative, reversible module compiler. Unsupported operators stay dense."""
from __future__ import annotations
import copy
from dataclasses import dataclass
import fnmatch
import math
from collections import Counter
from typing import Callable

import torch
from torch import nn
from .linear import InputMoments, LowRankLinear, NormLinear, factorize_linear, factorize_norm_linear
from .validation import Example, validate


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def state_bytes(model: nn.Module) -> int:
    """Logical serialized tensor bytes (including buffers, excluding file metadata)."""
    return sum(t.numel() * t.element_size() for t in model.state_dict().values())


def _preserve_compiler_metadata(source, candidate):
    """Retain private replay recipes across GraphModule reconstruction."""
    for name, value in vars(source).items():
        if name.startswith("_compressme_"):
            setattr(candidate, name, copy.deepcopy(value))
    return candidate


def _copy_with_compiler_metadata(model):
    # GraphModule.__deepcopy__ omits ordinary custom attributes, including the
    # recipe needed to reconstruct eliminated modules in an original factory.
    return _preserve_compiler_metadata(model, copy.deepcopy(model))


def _set(model, path, replacement):
    if not path:
        for name in ("_compressme_source_tensor_dtypes", "_compressme_affine_fx_spec"):
            if hasattr(model,name) and not hasattr(replacement,name):
                setattr(replacement,name,getattr(model,name))
        return replacement
    parent, _, leaf = path.rpartition(".")
    owner = model.get_submodule(parent) if parent else model
    owner._modules[leaf] = replacement
    return model


def _is_linear(module, extra_types=()):
    return type(module) is nn.Linear or type(module) in extra_types


def _blocked_paths(model):
    # These can bypass child forward() and consume packed / dense weights.
    return [name for name, module in model.named_modules()
            if isinstance(module, (nn.MultiheadAttention, nn.TransformerEncoderLayer,
                                   nn.TransformerDecoderLayer))]


def _under(path, ancestors):
    return any(not p or path == p or path.startswith(p + ".") for p in ancestors)


@dataclass
class CompressionResult:
    model: nn.Module
    report: dict

    def save(self, directory, *, packing=False):
        from .serialization import save
        return save(self, directory, packing=packing)


def collect_moments(model: nn.Module, examples: list[Example], *, linear_types=(),
                    max_rows_per_call=2048) -> dict[str, InputMoments]:
    """Forward-only calibration of local inputs, with bounded rows per call.

    Statistics cover sampled rows only. They do not establish a deployment domain.
    """
    if max_rows_per_call < 1 or not examples:
        raise ValueError("Provide calibration examples and a positive row limit")
    stats, handles = {}, []
    blocked = _blocked_paths(model)

    def hook(name):
        def observe(module, args):
            if not args or not isinstance(args[0], torch.Tensor):
                return
            x = args[0].detach().reshape(-1, module.weight.shape[1])
            if len(x) == 0:
                return
            if len(x) > max_rows_per_call:
                idx = torch.linspace(0, len(x)-1, max_rows_per_call, device=x.device).long()
                x = x[idx]
            x = x.cpu().double()
            if not torch.isfinite(x).all():
                raise ValueError(f"Non-finite calibration input at {name}")
            n, mean = len(x), x.mean(0)
            m2 = (x-mean).T @ (x-mean)
            if name not in stats:
                stats[name] = [n, mean, m2]
            else:
                old_n, old_mean, old_m2 = stats[name]
                delta = mean-old_mean
                total = old_n+n
                stats[name] = [total, old_mean+delta*(n/total),
                               old_m2+m2+torch.outer(delta,delta)*(old_n*n/total)]
        return observe
    modes = [(m, m.training) for m in model.modules()]
    try:
        for name, module in model.named_modules():
            if _is_linear(module, linear_types) and not _under(name, blocked):
                handles.append(module.register_forward_pre_hook(hook(name)))
        model.eval()
        with torch.inference_mode():
            for example in examples:
                example.call(model)
    finally:
        for handle in handles:
            handle.remove()
        for module, training in modes:
            module.training = training
    return {name: InputMoments(n, mean, m2/n) for name, (n, mean, m2) in stats.items()}


def compress(model: nn.Module, *, ratio=0.5, max_relative_operator_error=0.1,
             min_parameters=4096, include: list[str] | None = None,
             exclude: list[str] | None = None, calibration: list[Example] | None = None,
             validation: list[Example] | None = None, output_tolerance=0.01,
             output_selector: Callable | None = None, linear_types=()) -> CompressionResult:
    """Return a copy with smaller operators, without training or distillation.

    ratio is the requested fraction of each eligible matrix's parameters. It is
    not a promise about the complete checkpoint. A strict operator budget may
    retain every layer. Set max_relative_operator_error=None only when knowingly
    relying on empirical final-output validation. Failed validation rolls back
    the complete proposal; it never silently returns a failed compressed model.

    Custom Python code that consumes .weight directly needs a composite adapter.
    The compiler skips native attention blocks with known bypasses. It cannot
    prove arbitrary third-party Python control flow safe.
    """
    if not 0 < ratio < 1:
        raise ValueError("ratio must lie strictly between zero and one")
    if max_relative_operator_error is not None and max_relative_operator_error < 0:
        raise ValueError("Operator error budget must be nonnegative")
    moments = collect_moments(model, calibration, linear_types=linear_types) if calibration else {}
    candidate = _copy_with_compiler_metadata(model)
    candidate._compressme_source_tensor_dtypes = copy.deepcopy(getattr(model,"_compressme_source_tensor_dtypes",
        {n:str(t.dtype).removeprefix("torch.") for n,t in model.state_dict().items()}))
    original_count, original_bytes = parameter_count(model), state_bytes(model)
    parameter_aliases = Counter(id(p) for _, p in model.named_parameters(remove_duplicate=False))
    module_aliases = Counter(id(m) for _, m in model.named_modules(remove_duplicate=False))
    blocked = _blocked_paths(model)
    handled, records = [], []
    for path, source in model.named_modules():
        if _under(path, handled) or _under(path, blocked):
            continue
        if include is not None and not any(fnmatch.fnmatchcase(path, pattern) for pattern in include):
            continue
        if exclude and any(fnmatch.fnmatchcase(path, pattern) for pattern in exclude):
            continue
        norm_pair = (type(source) is nn.Sequential and len(source) == 2 and
                     type(source[0]) is nn.LayerNorm and type(source[1]) is nn.Linear)
        if not norm_pair and not _is_linear(source, linear_types):
            continue
        if any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks for m in source.modules()):
            records.append({"path": path, "status": "retained_custom_hooks"})
            # A hooked sequential pair must not be partly rewritten later.
            if norm_pair:
                handled.append(path)
            continue
        if module_aliases[id(source)] > 1 or any(parameter_aliases[id(p)] > 1 for p in source.parameters()):
            records.append({"path": path, "status": "retained_shared_parameters"})
            continue
        linear = source[1] if norm_pair else source
        m, d = linear.weight.shape
        rank = math.floor(ratio*m*d/(m+d))
        if m*d < min_parameters or rank < 1 or rank >= min(m,d):
            records.append({"path": path, "status": "retained_no_useful_rank"})
            continue
        if norm_pair:
            replacement = factorize_norm_linear(source[0], linear, rank)
            method = "normalization_domain_svd"
        else:
            replacement = factorize_linear(linear, rank, moments=moments.get(path))
            method = "activation_covariance_svd" if path in moments else "spectral_svd"
        bound = replacement.bound.to_dict()
        record = {"path": path, "method": method, "rank": rank,
                  "parameters_before": parameter_count(source),
                  "parameters_after": parameter_count(replacement), "bound": bound}
        if state_bytes(replacement) >= state_bytes(source):
            record["status"] = "retained_no_byte_saving"
        elif max_relative_operator_error is not None and bound["relative_operator_error"] > max_relative_operator_error:
            record["status"] = "retained_error_budget"
        else:
            record["status"] = "compressed"
            candidate = _set(candidate, path, replacement)
            handled.append(path)
        records.append(record)
    report = {"format_version": 1, "method": "training_free_operator_rewrites",
              "requested_matrix_ratio": ratio, "parameters_before": original_count,
              "parameters_after": parameter_count(candidate),
              "tensor_bytes_before": original_bytes, "tensor_bytes_after": state_bytes(candidate),
              "layers": records, "known_bypass_blocks_skipped": blocked,
              "guarantee_scope": "local analytic residual bounds; no universal end-to-end guarantee",
              "validation": None}
    if validation is not None:
        report["validation"] = validate(model, candidate, validation, relative_tolerance=output_tolerance,
                                          output_selector=output_selector)
        if not report["validation"]["accepted"]:
            report["proposal_parameters_after"] = report["parameters_after"]
            report["proposal_tensor_bytes_after"] = report["tensor_bytes_after"]
            report["status"] = "rejected_and_rolled_back"
            candidate = _copy_with_compiler_metadata(model)
            report["parameters_after"] = original_count
            report["tensor_bytes_after"] = original_bytes
            for record in records:
                if record["status"] == "compressed":
                    record["status"] = "proposal_rolled_back"
        else:
            report["status"] = "accepted_on_validation_examples"
    else:
        report["status"] = "compiled_without_final_output_validation"
    return CompressionResult(candidate, report)
