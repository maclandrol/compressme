"""Finite token partial evaluation with named outputs and explicit residuals.

Only the declared row-local grammar is executed. This is a frozen inference
representation, not a replacement for arbitrary continuous embedding inputs.
Complete token enumeration and shape probes are numerical evidence, not a
universal floating-point guarantee for unseen devices or later model layers.
"""
from __future__ import annotations

import copy
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from .finite_lookup import (_DTYPES, _POINTWISE, RowNormalize, _ValidatedLookup, _audit, _bits_equal,
                            _flatten, _has_hooks, _resident_storage_bytes)
from .runtime import autocast_enabled


class Float32RMSNorm(nn.Module):
    """Declared RMS formula: float32 variance, then input-dtype value and scale.

    y = weight * (x.float() * rsqrt(mean(x.float()**2) + eps)).to(x.dtype)
    This explicitly defines the operation; it does not certify an arbitrary
    custom module merely because that module has a weight and epsilon.
    """
    def __init__(self, weight, eps=1e-6):
        super().__init__()
        if not isinstance(weight, torch.Tensor) or weight.ndim != 1 or not weight.is_floating_point():
            raise ValueError("RMS scale must be a real one-dimensional tensor")
        if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("RMS epsilon must be finite and positive")
        self.weight = weight if isinstance(weight, nn.Parameter) else nn.Parameter(weight)
        self.eps = float(eps)

    def forward(self, x):
        values = x.to(torch.float32)
        values = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * values.to(x.dtype)


class FiniteTokenFanout(nn.Module):
    """Explicit source graph; each output is an independent numeric tensor.

    Branches consume shared_prefix(embedding(indices)). The optional named
    residual returns the original embedding. In-place operations are refused
    by compilation. Repeated/shared branches are permitted and accounted once
    in source storage. Output clones make the value contract independent of
    internal aliases and permit storage deduplication without output coupling.
    """
    def __init__(self, embedding, shared_prefix, branches, *, residual_key=None):
        super().__init__()
        if type(embedding) is not nn.Embedding or type(shared_prefix) is not nn.Sequential:
            raise ValueError("Use ordinary Embedding and Sequential source modules")
        if not isinstance(branches, dict) or not branches:
            raise ValueError("Declare at least one named output branch")
        if any(type(name) is not str or not name or "." in name for name in branches):
            raise ValueError("Branch names must be nonempty module-safe strings")
        if residual_key is not None and (type(residual_key) is not str or not residual_key or residual_key in branches):
            raise ValueError("Residual output needs a distinct nonempty name")
        self.embedding = embedding
        self.shared_prefix = shared_prefix
        self.branches = nn.ModuleDict(branches)
        self.residual_key = residual_key

    def forward(self, indices):
        embedded = self.embedding(indices)
        shared = self.shared_prefix(embedded)
        outputs = {name: branch(shared).clone() for name, branch in self.branches.items()}
        if self.residual_key is not None:
            outputs[self.residual_key] = embedded.clone()
        return outputs


def _owner_audit(source, owner_model):
    if owner_model is None:
        return
    paths = [n for n, m in owner_model.named_modules(remove_duplicate=False) if m is source]
    if len(paths) != 1:
        raise ValueError("Source fanout must occur exactly once in its owner")
    path = paths[0]
    def inside(name):
        return not path or name == path or name.startswith(path + ".")
    own_modules = {id(m) for m in source.modules()}
    if any(id(m) in own_modules and not inside(n) for n, m in owner_model.named_modules(remove_duplicate=False)):
        raise ValueError("Source modules have external consumers")
    own = {(str(t.device), t.untyped_storage()._cdata)
           for t in list(source.parameters()) + list(source.buffers())}
    tensors = list(owner_model.named_parameters(remove_duplicate=False)) + list(owner_model.named_buffers(remove_duplicate=False))
    if any(not inside(n) and (str(t.device), t.untyped_storage()._cdata) in own for n, t in tensors):
        raise ValueError("Source tensor storage has external aliases")


def _audit_source(source, owner_model):
    if type(source) is not FiniteTokenFanout:
        raise ValueError("Use the explicit FiniteTokenFanout source graph")
    if any(m.training or _has_hooks(m) or "forward" in vars(m) for m in source.modules()):
        raise ValueError("Use unmodified, hook-free evaluation modules")
    if any(getattr(p, "_backward_hooks", None) for p in source.parameters()):
        raise ValueError("Parameter hooks are unsupported")
    if type(source.branches) is not nn.ModuleDict or not source.branches:
        raise ValueError("Source branch declaration changed")
    embedding = source.embedding
    if type(embedding) is not nn.Embedding:
        raise ValueError("Embedding must be ordinary nn.Embedding")
    for branch in source.branches.values():
        layers = _flatten(source.shared_prefix) + (_flatten(branch) if type(branch) is nn.Sequential else [branch])
        width, surrogate = embedding.embedding_dim, []
        for layer in layers:
            allowed = _POINTWISE + (nn.Linear, nn.LayerNorm, nn.Dropout, RowNormalize, Float32RMSNorm)
            if hasattr(nn, "RMSNorm"):
                allowed += (nn.RMSNorm,)
            if type(layer) not in allowed:
                raise ValueError(f"Unproved row-local operation: {type(layer).__name__}")
            if getattr(layer, "inplace", False):
                raise ValueError("In-place operations could couple output branches")
            if type(layer) is Float32RMSNorm or (hasattr(nn, "RMSNorm") and type(layer) is nn.RMSNorm):
                if type(layer) is Float32RMSNorm:
                    valid = tuple(layer.weight.shape) == (width,) and math.isfinite(layer.eps) and layer.eps > 0
                else:
                    valid = (tuple(layer.normalized_shape) == (width,)
                             and (layer.eps is None or (math.isfinite(layer.eps) and layer.eps > 0))
                             and (layer.weight is None or tuple(layer.weight.shape) == (width,)))
                if not valid:
                    raise ValueError("RMSNorm must operate only over the current feature width")
                surrogate.append(nn.Identity().eval())
            else:
                surrogate.append(layer)
            if type(layer) is nn.Linear:
                if layer.bias is not None and tuple(layer.bias.shape) != (layer.out_features,):
                    raise ValueError("Linear bias shape differs from metadata")
                width = layer.out_features
        # Reuse the existing grammar for every ordinary operation. Surrogates
        # are structural only; numerical evaluation uses the actual audited RMS.
        structural = nn.Sequential(embedding, *surrogate)
        # Source-owned leaves are already in eval mode. Recursively invoking
        # .eval() here could execute unrelated custom children's train methods.
        structural.training = False
        _audit(structural)
    device = embedding.weight.device
    for tensor in list(source.parameters()) + list(source.buffers()):
        if (tensor.device != device or device.type == "meta" or tensor.layout != torch.strided
                or tensor.dtype not in _DTYPES or not torch.isfinite(tensor).all()):
            raise ValueError("Source requires finite dense supported floats on one device")
    _owner_audit(source, owner_model)
    return embedding.num_embeddings, device


def _profile_settings(profiles):
    """Validate explicit empirical dispatch settings, without guessing a backend."""
    if profiles is None:
        return ()
    if not isinstance(profiles, (list, tuple)) or not profiles:
        raise ValueError("row_count_profiles must be a nonempty ordered sequence, or None")
    result, previous = [], 0
    for item in profiles:
        if not isinstance(item, dict) or set(item) != {"max_rows", "evaluation_rows"}:
            raise ValueError("Each row profile declares only max_rows and evaluation_rows")
        maximum, evaluation = item["max_rows"], item["evaluation_rows"]
        if type(maximum) is not int or maximum <= previous or type(evaluation) is not int or evaluation < 1:
            raise ValueError("Profile max_rows must increase strictly; all row counts must be positive integers")
        result.append({"max_rows": maximum, "evaluation_rows": evaluation})
        previous = maximum
    return tuple(result)


def _descriptors(outputs, tables):
    """Validate one complete named-output descriptor list for portable replay."""
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("Missing output descriptors")
    seen = set()
    for item in outputs:
        if not isinstance(item, list) or len(item) != 4:
            raise ValueError("Malformed output descriptor")
        name, group, start, width = item
        if (type(name) is not str or not name or name in seen or type(group) is not str or group not in tables
                or type(start) is not int or start < 0 or type(width) is not int or width < 1
                or start + width > tables[group].shape[1]):
            raise ValueError("Invalid output descriptor")
        seen.add(name)
    return outputs


class FiniteFanoutLookup(_ValidatedLookup):
    """Packed outputs with no source modules or backing allocations retained.

    Optional profiles select tables using indices.numel(), preserving every ID
    rank. They record empirical floating-point kernel choices; they are not a
    claim of shape-independent floating-point equivalence.
    """
    def __init__(self, tables, outputs, num_embeddings, *, row_count_profiles=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.outputs = tuple(tuple(item) for item in outputs)
        self.row_count_profiles = tuple((item["max_rows"], item["evaluation_rows"],
                                        tuple(tuple(row) for row in item["outputs"]))
                                       for item in (row_count_profiles or ()))
        self.tables = nn.ParameterDict({name: nn.Parameter(value.detach(), requires_grad=False)
                                        for name, value in tables.items()})
        self.eval()

    def forward(self, indices):
        if not isinstance(indices, torch.Tensor) or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("Declared input contract accepts integer token IDs only")
        if self.training or any(p.requires_grad for p in self.parameters()):
            raise ValueError("Finite fanout is a frozen inference representation")
        if any(autocast_enabled(t.device.type) for t in self.tables.values()):
            raise ValueError("Use explicit dtypes, not autocast")
        outputs = self.outputs
        for maximum, _, profiled_outputs in self.row_count_profiles:
            if indices.numel() <= maximum:
                outputs = profiled_outputs
                break
        # Do not gather unused profile tables on every call.
        selected = {}
        for _, group, _, _ in outputs:
            if group not in selected:
                rows = self.tables[group]
                selected[group] = F.embedding(indices, rows.expand(self.num_embeddings, -1)
                                              if rows.shape[0] == 1 else rows)
        return {name: selected[group][..., start:start+width].clone()
                for name, group, start, width in outputs}

    def requires_grad_(self, requires_grad=True):
        if requires_grad:
            raise ValueError("The original source parameterization is unavailable for training")
        return super().requires_grad_(False)

    def recipe(self):
        result = {"kind": "FiniteFanoutLookup", "version": 2 if self.row_count_profiles else 1,
                  "input_contract": "token_indices_only", "num_embeddings": self.num_embeddings,
                  "tables": {name: {"shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch.")}
                             for name, t in self.tables.items()},
                  "outputs": [list(item) for item in self.outputs], "training": self.training}
        if self.row_count_profiles:
            result["row_count_profiles"] = [
                {"max_rows": maximum, "evaluation_rows": evaluation,
                 "outputs": [list(item) for item in outputs]}
                for maximum, evaluation, outputs in self.row_count_profiles]
        return result

    def _validation_signature(self):
        signature = super()._validation_signature()
        if signature is None:
            return None
        return signature[0], json.dumps(self.recipe(), sort_keys=True)

    @classmethod
    def from_recipe(cls, spec, *, device="cpu"):
        if (spec.get("kind") != "FiniteFanoutLookup" or type(spec.get("version")) is not int
                or spec["version"] not in (1, 2)
                or spec.get("input_contract") != "token_indices_only" or spec.get("training") is not False
                or type(spec.get("num_embeddings")) is not int or spec["num_embeddings"] < 1):
            raise ValueError("Invalid finite fanout recipe")
        count, tables = spec["num_embeddings"], {}
        if not isinstance(spec.get("tables"), dict) or not spec["tables"]:
            raise ValueError("Missing output tables")
        for name, item in spec["tables"].items():
            if not isinstance(item, dict):
                raise ValueError("Invalid table shape/dtype")
            dtype, shape = getattr(torch, item.get("dtype", ""), None), item.get("shape")
            if (type(name) is not str or not name or "." in name or dtype not in _DTYPES
                    or not isinstance(shape, list) or len(shape) != 2
                    or any(type(n) is not int or n < 1 for n in shape)
                    or shape[0] not in (1, count)):
                raise ValueError("Invalid table shape/dtype")
            with torch.inference_mode(False), torch.no_grad():
                tables[name] = torch.zeros(shape, dtype=dtype, device=device)
        outputs = _descriptors(spec.get("outputs"), tables)
        profiles = None
        if spec["version"] == 1:
            if "row_count_profiles" in spec:
                raise ValueError("Version 1 recipes cannot include row-count profiles")
        else:
            profiles = spec.get("row_count_profiles")
            if not isinstance(profiles, list) or not profiles:
                raise ValueError("Version 2 requires explicit row-count profiles")
            if any(not isinstance(item, dict) or set(item) != {"max_rows", "evaluation_rows", "outputs"}
                   for item in profiles):
                raise ValueError("Malformed row-count profile")
            _profile_settings([{key: item[key] for key in ("max_rows", "evaluation_rows")} for item in profiles])
            signature = [(item[0], item[3], tables[item[1]].dtype) for item in outputs]
            for item in profiles:
                branch = _descriptors(item["outputs"], tables)
                if [(row[0], row[3], tables[row[1]].dtype) for row in branch] != signature:
                    raise ValueError("Profile output names, widths and dtypes must match the bulk outputs")
        return cls(tables, outputs, count, row_count_profiles=profiles)


def compile_finite_fanout(source, *, input_contract, owner_model=None, chunk_size=1024,
                          validation_chunk_size=257, absolute_tolerance=1e-5,
                          relative_tolerance=1e-5, row_count_profiles=None):
    """Enumerate all token outputs, gate storage and numerical fidelity.

    Returns CompressionResult; rejected proposals return a deepcopy of the
    declared source graph. Replacement in a larger model remains explicit and
    requires that model's complete-output validation. Optional row_count_profiles
    declare backend-specific empirical table shapes. Each profile evaluates every
    token in (1, evaluation_rows) repeated-ID batches and applies up to max_rows
    total input IDs. Larger inputs retain the ordinary bulk table. Every table
    counts toward both storage gates; no precision change or source copy is hidden.
    """
    from .compiler import (CompressionResult, _copy_with_compiler_metadata,
                           parameter_count, state_bytes)
    if input_contract != "token_indices_only":
        raise ValueError("Explicit token_indices_only input contract is required")
    if any(type(n) is not int or n < 1 for n in (chunk_size, validation_chunk_size)):
        raise ValueError("Chunk sizes must be positive integers")
    if any(not isinstance(n, (int, float)) or isinstance(n, bool) or not math.isfinite(n) or n < 0
           for n in (absolute_tolerance, relative_tolerance)):
        raise ValueError("Numerical tolerances must be finite and nonnegative")
    profiles = _profile_settings(row_count_profiles)
    count, device = _audit_source(source, owner_model)
    source_bytes = _resident_storage_bytes(source)
    source_logical_bytes = state_bytes(source)
    source_dtypes = copy.deepcopy(getattr(source, "_compressme_source_tensor_dtypes",
        {name: str(value.dtype).removeprefix("torch.") for name, value in source.state_dict().items()}))
    pieces = {}
    with torch.inference_mode():
        for start in range(0, count, chunk_size):
            ids = torch.arange(start, min(start+chunk_size, count), device=device)
            for name, value in source(ids).items():
                if (value.ndim != 2 or value.shape[0] != ids.numel() or value.shape[1] < 1
                        or value.dtype not in _DTYPES or not torch.isfinite(value).all()):
                    raise ValueError("Source violated its finite row output contract")
                pieces.setdefault(name, []).append(value)
        profile_pieces = []
        for profile in profiles:
            local = {}
            for token in range(count):
                ids = torch.full((1, profile["evaluation_rows"]), token, dtype=torch.long, device=device)
                outputs = source(ids)
                if list(outputs) != list(pieces):
                    raise ValueError("Profile changed output names or order")
                for name, value in outputs.items():
                    expected = pieces[name][0]
                    if (value.shape != (1, profile["evaluation_rows"], expected.shape[-1])
                            or value.dtype != expected.dtype or not torch.isfinite(value).all()):
                        raise ValueError("Profile violated its finite row output contract")
                    first = value[:, :1]
                    if not _bits_equal(value, first.expand_as(value)):
                        raise ValueError("Profile produced unequal outputs at identical token positions")
                    local.setdefault(name, []).append(first.reshape(1, -1).clone())
            profile_pieces.append(local)
    with torch.inference_mode(False), torch.no_grad():
        all_rows = [{name: torch.cat(values).clone() for name, values in block.items()}
                    for block in [pieces, *profile_pieces]]
        rows = all_rows[0]
        unique, mappings = [], []
        for block in all_rows:
            mapping = {}
            for name, values in block.items():
                match = next((i for i, existing in enumerate(unique) if _bits_equal(existing, values)), None)
                if match is None:
                    match = len(unique)
                    unique.append(values)
                mapping[name] = match
            mappings.append(mapping)
        groups, locations = {}, {}
        for i, values in enumerate(unique):
            if _bits_equal(values, values[:1].expand_as(values)):
                values = values[:1].clone()
            # Only outputs used by exactly the same routes share a packed row.
            # Otherwise selecting one route would gather the other profiles'
            # columns too, even though their retained coefficients are distinct.
            usage = tuple(route for route, mapping in enumerate(mappings) if i in mapping.values())
            group_key = (values.dtype, values.shape[0], usage)
            group = groups.setdefault(group_key, [])
            offset = sum(x.shape[1] for _, x in group)
            locations[i] = (group_key, offset, values.shape[1])
            group.append((i, values))
        names = {key: f"group_{i}" for i, key in enumerate(groups)}
        tables = {names[key]: torch.cat([value for _, value in values], dim=1) for key, values in groups.items()}
        descriptors = [[(name, names[locations[i][0]], locations[i][1], locations[i][2])
                        for name, i in mapping.items()] for mapping in mappings]
        compiled_profiles = [{**profile, "outputs": descriptor}
                             for profile, descriptor in zip(profiles, descriptors[1:])]
        candidate = FiniteFanoutLookup(tables, descriptors[0], count, row_count_profiles=compiled_profiles)
        candidate._compressme_source_tensor_dtypes = source_dtypes
    target_bytes = _resident_storage_bytes(candidate)
    report = {"method": "finite_token_fanout", "input_contract": input_contract,
              "num_embeddings": count, "outputs": list(rows), "unique_output_tables": len(unique),
              "source_resident_storage_bytes": source_bytes, "candidate_resident_storage_bytes": target_bytes,
              "saved_resident_storage_bytes": source_bytes-target_bytes,
              "candidate_stored_values": sum(p.numel() for p in candidate.parameters()),
              "retains_source_tensor_storage": False,
              "residual_embedding_values_included": source.residual_key is not None,
              "evaluation_device": str(device), "torch_version": str(torch.__version__),
              "parameters_before": parameter_count(source), "parameters_after": parameter_count(candidate),
              "tensor_bytes_before": source_logical_bytes, "tensor_bytes_after": state_bytes(candidate),
              "resident_storage_bytes_before": source_bytes, "resident_storage_bytes_after": target_bytes,
              "enumeration_chunk_size": chunk_size, "validation_chunk_size": validation_chunk_size,
              "row_count_profiles": list(profiles),
              "profile_scope": "Explicit empirical backend choices; bulk fallback above the final threshold" if profiles else None,
              "ownership_scope": "Registered aliases checked against owner model" if owner_model is not None
                                 else "Standalone source only; no enclosing model was inspected",
              "storage_accounting": "Unique registered backing allocations; excludes activations, allocator overhead and external Python owners"}
    def retained():
        model = _copy_with_compiler_metadata(source)
        report.update(proposal_parameters_after=parameter_count(candidate),
                      proposal_tensor_bytes_after=state_bytes(candidate),
                      proposal_resident_storage_bytes_after=target_bytes,
                      parameters_after=parameter_count(model), tensor_bytes_after=state_bytes(model),
                      resident_storage_bytes_after=_resident_storage_bytes(model))
        return CompressionResult(model, report)
    if target_bytes >= source_bytes or state_bytes(candidate) >= source_logical_bytes:
        report["status"] = "no_storage_saving"
        return retained()
    statistics = {name: {"max_abs": 0.0, "max_relative_l2": 0.0, "bitwise_identical": True} for name in rows}
    cases = 0
    profile_validation_sizes = []
    def compare(ids):
        nonlocal cases
        left, right = source(ids), candidate(ids)
        if list(left) != list(right):
            raise ValueError("Output keys changed")
        for name, value in left.items():
            proposed = right[name]
            if value.shape != proposed.shape or value.dtype != proposed.dtype:
                raise ValueError("Output shape/dtype changed")
            if not torch.isfinite(value).all() or not torch.isfinite(proposed).all():
                raise ValueError("Nonfinite validation outputs")
            a, b = value.detach().cpu().double(), proposed.detach().cpu().double()
            delta = a-b
            absolute = delta.abs().max().item() if delta.numel() else 0.0
            relative = (torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(a).clamp_min(1e-30)).item()
            stat = statistics[name]
            stat["max_abs"] = max(stat["max_abs"], absolute)
            stat["max_relative_l2"] = max(stat["max_relative_l2"], relative)
            stat["bitwise_identical"] &= _bits_equal(value, proposed)
        cases += 1
    with torch.inference_mode():
        for start in range(0, count, validation_chunk_size):
            ids = torch.arange(start, min(start+validation_chunk_size, count), device=device)
            for shape in ((-1,), (1, -1), (-1, 1), (1, 1, -1)):
                compare(ids.reshape(shape))
            compare(ids.reshape(1, -1).expand(2, -1))
        for token in range(count):
            compare(torch.tensor(token, device=device))
        if profiles:
            for token in range(count):
                for shape in ((1,), (1, 1), (1, 1, 1)):
                    compare(torch.full(shape, token, dtype=torch.long, device=device))
            sizes = {1, 2}
            previous = 0
            for profile in profiles:
                # Every token is checked at both ends of each declared dispatch
                # interval, in addition to mixed-token boundary probes below.
                for size in sorted({previous + 1, profile["max_rows"]} - {1}):
                    for token in range(count):
                        ids = torch.full((size,), token, dtype=torch.long, device=device)
                        compare(ids)
                        compare(ids.reshape(1, size))
                previous = profile["max_rows"]
                sizes.update(n for n in (profile["max_rows"]-1, profile["max_rows"],
                                          profile["max_rows"]+1, profile["evaluation_rows"]) if n > 0)
            # Mixed IDs, repeated IDs, multiple ranks, and noncontiguous layouts
            # test the shape-only dispatch assumption without inferring a theorem.
            profile_validation_sizes = sorted(sizes)
            for size in profile_validation_sizes:
                for offset in (0, count-1):
                    ids = (torch.arange(size, device=device) * 3 + offset).remainder(count)
                    for shape in ((size,), (1, size), (size, 1), (1, 1, size)):
                        compare(ids.reshape(shape))
                    for leading in (2, 3):
                        if size % leading == 0:
                            compare(ids.reshape(leading, size // leading))
                    interleaved = torch.empty((size, 2), dtype=torch.long, device=device)
                    interleaved[:, 0] = ids
                    interleaved[:, 1] = 0
                    compare(interleaved[:, 0])
                    compare(ids.flip(0).to(torch.int32))
        for shape in ((0,), (2, 0), (0, 3, 2)):
            compare(torch.empty(shape, dtype=torch.int64, device=device))
    passed = all(s["max_abs"] <= absolute_tolerance and s["max_relative_l2"] <= relative_tolerance for s in statistics.values())
    report.update(status="accepted_on_full_domain_validation" if passed else "rejected_numerical_validation",
                  numerical=statistics, validation_cases=cases, absolute_tolerance=absolute_tolerance,
                  relative_tolerance=relative_tolerance,
                  profile_validation_sizes=profile_validation_sizes,
                  guarantee_scope="Finite token partial evaluation; floating validation covers these shapes/device only")
    if passed:
        candidate._seal_validation()
        return CompressionResult(candidate, report)
    return retained()
