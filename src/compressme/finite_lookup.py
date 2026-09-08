"""Exact real-arithmetic partial evaluation on a finite token domain.

This compiles a deterministic row-local inference function, never fits a model.
All source rows are evaluated and checked before accepting a storage saving.
Floating-point GEMMs can depend on batch shape; the numerical gate is explicit
evidence on the enumerated/probed layouts, not an all-device rounding theorem.
"""
from __future__ import annotations

import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from .runtime import autocast_enabled

_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _resident_storage_bytes(module):
    """Count backing allocations once across registered tensor aliases.

    This is not process RSS: allocator overhead, temporary activations and
    unregistered external owners are excluded. Views retain their entire backing
    allocation, and tied modules/parameters must not multiply its byte count.
    """
    allocations = {}
    for tensor in list(module.parameters()) + list(module.buffers()):
        if tensor.device.type == "meta" or tensor.layout != torch.strided:
            raise ValueError("Resident storage accounting requires materialized dense tensors")
        storage = tensor.untyped_storage()
        allocations[(str(tensor.device), storage._cdata)] = storage.nbytes()
    return sum(allocations.values())


class _ValidatedLookup(nn.Module):
    """Track whether the representation still matches its numerical gate.

    This is an inexpensive mutation/version check, not a new fidelity theorem.
    Fresh recipe instances have no live validation seal. Copying/replacing
    tensors, changing dtype/device/settings or ordinary in-place mutation makes
    the previous gate historical. Unsafe .data/storage writes that bypass
    PyTorch version counters are outside this supported mutation contract.
    """
    def __init__(self):
        super().__init__()
        self._validated_signature = None

    def _validation_signature(self):
        if any(_has_hooks(module) or "forward" in vars(module) for module in self.modules()):
            return None
        values = []
        for name, tensor in list(self.named_parameters()) + list(self.named_buffers()):
            try:
                version = tensor._version
            except RuntimeError:
                return None
            values.append((name, id(tensor), version, tuple(tensor.shape), tensor.dtype,
                           tensor.device, tensor.requires_grad))
        settings = tuple(sorted(self.recipe().items()))
        return tuple(values), settings

    def _seal_validation(self):
        self._validated_signature = self._validation_signature()

    def validation_is_current(self):
        return (self._validated_signature is not None
                and self._validated_signature == self._validation_signature())


class RowNormalize(nn.Module):
    """F.normalize over the final feature dimension; never across tokens."""
    def __init__(self, p=2.0, eps=1e-12):
        super().__init__()
        if p != 2 or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("This audited primitive requires p=2 and finite eps>0")
        self.p = 2.0
        self.eps = float(eps)

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=-1, eps=self.eps)


class FiniteTokenLookup(_ValidatedLookup):
    """Frozen finite-domain result with the original index shape contract."""
    def __init__(self, rows, num_embeddings, *, constant=False):
        super().__init__()
        if rows.ndim != 2 or not rows.is_floating_point() or rows.layout != torch.strided:
            raise ValueError("Expected dense real floating-point output rows")
        if rows.shape[0] != (1 if constant else num_embeddings):
            raise ValueError("Stored rows do not match the token domain")
        if num_embeddings < 1 or rows.shape[1] < 1:
            raise ValueError("Lookup dimensions must be positive")
        self.num_embeddings = num_embeddings
        self.embedding_dim = rows.shape[1]
        self.constant = bool(constant)
        # The caller transfers ownership of fresh output rows, not source weights.
        self._rows = nn.Parameter(rows.detach(), requires_grad=False)
        self.eval()

    @property
    def weight(self):
        return self._rows.expand(self.num_embeddings, -1) if self.constant else self._rows

    def forward(self, indices):
        if autocast_enabled(self._rows.device.type):
            raise ValueError("Finite lookup requires explicit dtypes; autocast is unsupported")
        if self.training or self._rows.requires_grad:
            raise ValueError("Finite lookup is an inference representation; original parameters are unavailable for training")
        return F.embedding(indices, self.weight)

    def requires_grad_(self, requires_grad=True):
        if requires_grad:
            raise ValueError("Finite lookup cannot restore the source parameterization for training")
        return super().requires_grad_(False)

    def recipe(self):
        return {"kind": "FiniteTokenLookup", "num_embeddings": self.num_embeddings,
                "embedding_dim": self.embedding_dim, "constant": self.constant,
                "dtype": str(self._rows.dtype).removeprefix("torch."), "training": self.training}

    @classmethod
    def from_recipe(cls, spec, *, device="cpu"):
        dtype = getattr(torch, spec.get("dtype", ""), None)
        if spec.get("kind") != "FiniteTokenLookup" or dtype not in _DTYPES or type(spec.get("constant")) is not bool:
            raise ValueError("Invalid finite lookup recipe")
        rows = torch.zeros((1 if spec["constant"] else spec["num_embeddings"], spec["embedding_dim"]), dtype=dtype, device=device)
        return cls(rows, spec["num_embeddings"], constant=spec["constant"]).train(spec.get("training", False))


_POINTWISE = (nn.Identity, nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, nn.Tanh, nn.Sigmoid,
              nn.ELU, nn.CELU, nn.LeakyReLU, nn.Softplus, nn.Softsign, nn.Hardtanh,
              nn.Hardswish, nn.Hardsigmoid, nn.Mish, nn.SELU)


def _has_hooks(module):
    return any(bool(value) for name, value in vars(module).items()
               if name.endswith("_hooks") and isinstance(value, dict))


def _flatten(source):
    if type(source) is not nn.Sequential:
        raise ValueError("Use an ordinary nn.Sequential to establish the row-local graph")
    result = []
    for child in source:
        if type(child) is nn.Sequential:
            result.extend(_flatten(child))
        else:
            result.append(child)
    return result


def _audit(source, owner_model=None):
    if any(module.training for module in source.modules()):
        raise ValueError("Every source module must be in evaluation mode")
    if any(_has_hooks(module) or "forward" in vars(module) for module in source.modules()):
        raise ValueError("Hooks and custom forward replacements are unsupported")
    if any(getattr(p, "_backward_hooks", None) for p in source.parameters()):
        raise ValueError("Parameter hooks are unsupported")
    if owner_model is not None:
        paths = [name for name, module in owner_model.named_modules(remove_duplicate=False) if module is source]
        if len(paths) != 1:
            raise ValueError("The source block must occur exactly once in its owner model")
        path = paths[0]
        def inside(name):
            return not path or name == path or name.startswith(path + ".")
        own_params = {id(p) for p in source.parameters()}
        own_modules = {id(m) for m in source.modules()}
        if any(id(p) in own_params and not inside(name) for name, p in owner_model.named_parameters(remove_duplicate=False)):
            raise ValueError("Source parameters have external consumers in the owner model")
        if any(id(m) in own_modules and not inside(name) for name, m in owner_model.named_modules(remove_duplicate=False)):
            raise ValueError("Source modules have external consumers in the owner model")
        def storage_key(tensor):
            return (str(tensor.device), tensor.untyped_storage().data_ptr()) if tensor.numel() else None
        owned_storage = {storage_key(t) for t in list(source.parameters()) + list(source.buffers()) if t.numel()}
        owner_tensors = list(owner_model.named_parameters(remove_duplicate=False)) + list(owner_model.named_buffers(remove_duplicate=False))
        if any(not inside(name) and storage_key(tensor) in owned_storage for name, tensor in owner_tensors if tensor.numel()):
            raise ValueError("Source tensor storage has external aliases in the owner model")
    layers = _flatten(source)
    if not layers or type(layers[0]) is not nn.Embedding:
        raise ValueError("The first row-local operation must be ordinary nn.Embedding")
    first = layers[0]
    if tuple(first.weight.shape) != (first.num_embeddings, first.embedding_dim):
        raise ValueError("Embedding attributes and weight shape differ")
    if first.weight.dtype not in _DTYPES:
        raise ValueError("Embedding must use an explicit supported floating-point dtype")
    if first.max_norm is not None or first.sparse:
        raise ValueError("Mutating max_norm and sparse embedding modes are unsupported")
    if first.num_embeddings < 1 or first.embedding_dim < 1:
        raise ValueError("Embedding dimensions must be positive")
    width = first.embedding_dim
    for layer in layers[1:]:
        if type(layer) is nn.Linear:
            if layer.in_features != width or tuple(layer.weight.shape) != (layer.out_features, width):
                raise ValueError("Linear dimensions do not match the reachable row width")
            width = layer.out_features
        elif type(layer) is nn.LayerNorm:
            if tuple(layer.normalized_shape) != (width,) or not math.isfinite(layer.eps) or layer.eps <= 0:
                raise ValueError("LayerNorm must normalize only the final feature dimension with eps>0")
        elif type(layer) is RowNormalize:
            if layer.p != 2 or not math.isfinite(layer.eps) or layer.eps <= 0:
                raise ValueError("Invalid row normalization")
        elif type(layer) in _POINTWISE or type(layer) is nn.Dropout:
            pass
        else:
            raise ValueError(f"Unproved row-local operation: {type(layer).__name__}")
        if width < 1:
            raise ValueError("The output row width must remain positive")
    device = first.weight.device
    if autocast_enabled(device.type):
        raise ValueError("Lookup compilation requires explicit dtypes; autocast is unsupported")
    for tensor in source.state_dict().values():
        if not isinstance(tensor, torch.Tensor) or tensor.device != device or tensor.device.type == "meta":
            raise ValueError("Source tensors must be materialized on one device")
        if tensor.layout != torch.strided or tensor.is_quantized or tensor.is_complex():
            raise ValueError("Only ordinary dense real source tensors are supported")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError("Nonfinite source tensors are unsupported")
    return first.num_embeddings, width, device


def _bits_equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                       b.detach().cpu().contiguous().reshape(-1).view(torch.uint8))


def _compare_rows(reference, candidate, ids, statistics):
    left, right = reference(ids), candidate(ids)
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("Finite lookup changed output shape or dtype")
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("Finite lookup encountered nonfinite outputs")
    a, b = left.detach().cpu().double(), right.detach().cpu().double()
    delta = a - b
    statistics["squared_error"] += float(delta.square().sum())
    statistics["squared_reference"] += float(a.square().sum())
    statistics["max_abs"] = max(statistics["max_abs"], float(delta.abs().max()) if delta.numel() else 0.0)
    statistics["bitwise_identical"] = statistics["bitwise_identical"] and _bits_equal(left, right)


def compile_finite_lookup(source, *, input_contract, owner_model=None, chunk_size=1024,
                          validation_chunk_size=257, absolute_tolerance=1e-5,
                          relative_tolerance=1e-5):
    """Partially evaluate a row-local token function, returning CompressionResult.

    Declare input_contract='token_indices_only': the replacement is called only
    on valid integer token indices; source child weights/attributes are not a
    separate public interface. Pass owner_model when the block belongs to a
    larger model so known external module/Parameter aliases can be refused.
    This returns a replacement for the block only, not the enclosing model.
    The caller must account for any retained external encoder in whole-model
    savings and perform an end-to-end output gate when composing this pass.
    """
    from .compiler import CompressionResult, parameter_count, state_bytes
    if input_contract != "token_indices_only":
        raise ValueError("Declare input_contract='token_indices_only' for the finite domain")
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in (chunk_size, validation_chunk_size)):
        raise ValueError("Enumeration and validation chunk sizes must be positive integers")
    if any(not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0 for t in (absolute_tolerance, relative_tolerance)):
        raise ValueError("Tolerances must be finite and nonnegative")
    count, width, device = _audit(source, owner_model)
    before_params, before_bytes = parameter_count(source), state_bytes(source)
    before_resident = _resident_storage_bytes(source)
    report = {"method": "exact_finite_row_lookup", "input_contract": input_contract,
              "domain_size": count, "output_width": width, "parameters_before": before_params,
              "evaluation_device": str(device), "torch_version": str(torch.__version__),
              "tensor_bytes_before": before_bytes, "enumeration_chunk_size": chunk_size,
              "resident_storage_bytes_before": before_resident,
              "storage_accounting": "Unique registered tensor backing allocations; allocator overhead and external Python owners excluded",
              "validation_chunk_size": validation_chunk_size,
              "guarantee": "Exact partial-evaluation identity in ideal arithmetic; stored lookup is validated numerically on the entire token domain; inference only",
              "floating_point_scope": "Every token checked in the recorded chunk layouts and shape probes; future batch/device rounding is not universally bounded",
              "ownership_scope": "Known owner-model aliases checked" if owner_model is not None else "Standalone source block only; no enclosing model was inspected"}
    # Materialize ordinary frozen tensors so save/load, deepcopy and explicit
    # device moves remain valid even when the caller is in inference_mode.
    with torch.inference_mode(False), torch.no_grad():
        table = None
        for start in range(0, count, chunk_size):
            ids = torch.arange(start, min(count, start + chunk_size), device=device)
            values = source(ids)
            if values.shape != (ids.numel(), width) or not values.is_floating_point() or not torch.isfinite(values).all():
                raise ValueError("Enumeration did not produce finite rows at the audited output width")
            if table is None:
                table = torch.empty((count, width), device=device, dtype=values.dtype)
            if values.dtype != table.dtype:
                raise ValueError("Output dtype changed across the finite domain")
            table[start:start + ids.numel()].copy_(values)
        raw = table.detach().cpu().contiguous().view(torch.uint8).reshape(count, -1)
        output_bits_constant = torch.equal(raw, raw[:1].expand_as(raw))
        input_rows = _flatten(source)[0].weight.detach().cpu().contiguous().view(torch.uint8).reshape(count, -1)
        input_bits_constant = torch.equal(input_rows, input_rows[:1].expand_as(input_rows))
        constant = input_bits_constant or output_bits_constant
        if constant:
            table = table[:1].clone()  # release the full enumeration allocation
        candidate = FiniteTokenLookup(table, count, constant=constant)
        candidate._compressme_source_tensor_dtypes = {
            name: str(tensor.dtype).removeprefix("torch.") for name, tensor in source.state_dict().items()}
        after_bytes = state_bytes(candidate)
        after_resident = _resident_storage_bytes(candidate)
        report.update(parameters_after=parameter_count(candidate), tensor_bytes_after=after_bytes,
                      resident_storage_bytes_after=after_resident,
                      stored_rows=table.shape[0], constant_output_rows=constant,
                      constant_basis="identical_input_row_bits" if input_bits_constant else ("identical_enumerated_output_bits" if output_bits_constant else None),
                      output_dtype=str(table.dtype).removeprefix("torch."))
        if after_bytes >= before_bytes or after_resident >= before_resident:
            retained = copy.deepcopy(source)
            report.update(status="retained_no_byte_saving", proposed_tensor_bytes=after_bytes,
                          proposed_resident_storage_bytes=after_resident,
                          parameters_after=before_params, tensor_bytes_after=before_bytes,
                          resident_storage_bytes_after=_resident_storage_bytes(retained), validation=None)
            return CompressionResult(retained, report)
        statistics = {"squared_error": 0.0, "squared_reference": 0.0, "max_abs": 0.0, "bitwise_identical": True}
        try:
            for start in range(0, count, validation_chunk_size):
                ids = torch.arange(start, min(count, start + validation_chunk_size), device=device)
                _compare_rows(source, candidate, ids, statistics)
            probes = [torch.tensor(n, device=device) for n in sorted({0, count // 2, count - 1})]
            probes += [torch.empty(0, dtype=torch.long, device=device),
                       torch.empty((2, 0), dtype=torch.long, device=device),
                       (torch.arange(6, device=device).reshape(2, 3) % count).t()]
            for ids in probes:
                _compare_rows(source, candidate, ids, statistics)
            relative_l2 = math.sqrt(statistics["squared_error"]) / max(math.sqrt(statistics["squared_reference"]), 1e-12)
            gate = {"accepted": statistics["max_abs"] <= absolute_tolerance and relative_l2 <= relative_tolerance,
                    "tokens_checked": count, "shape_probes": len(probes), "max_abs": statistics["max_abs"],
                    "relative_l2": relative_l2, "bitwise_identical": statistics["bitwise_identical"],
                    "absolute_tolerance": absolute_tolerance, "relative_tolerance": relative_tolerance}
        except (ValueError, TypeError, RuntimeError) as exc:
            gate = {"accepted": False, "error": str(exc), "tokens_checked": count}
        report["validation"] = gate
        if not gate["accepted"]:
            retained = copy.deepcopy(source)
            report.update(status="rejected_and_rolled_back", proposal_parameters_after=parameter_count(candidate),
                          proposal_resident_storage_bytes=after_resident,
                          parameters_after=before_params, tensor_bytes_after=before_bytes,
                          resident_storage_bytes_after=_resident_storage_bytes(retained))
            return CompressionResult(retained, report)
    report["status"] = "accepted_on_full_domain_validation"
    candidate._seal_validation()
    return CompressionResult(candidate, report)


class ProjectedEmbeddingLookup(_ValidatedLookup):
    """One projected table for both raw and L2-normalized token branches."""
    def __init__(self, rows, norms, bias, normalize_eps):
        super().__init__()
        if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] < 1 or rows.dtype not in _DTYPES:
            raise ValueError("Expected a nonempty real projected row table")
        if norms.shape != (rows.shape[0], 1) or norms.dtype != rows.dtype or norms.device != rows.device:
            raise ValueError("Expected one denominator per row with matching dtype/device")
        if bias is not None and (bias.shape != (rows.shape[1],) or bias.dtype != rows.dtype or bias.device != rows.device):
            raise ValueError("Projected lookup bias shape/dtype/device mismatch")
        if not isinstance(normalize_eps, (int, float)) or not math.isfinite(normalize_eps) or normalize_eps <= 0:
            raise ValueError("Normalization epsilon must be finite and positive")
        self.num_embeddings, self.embedding_dim = rows.shape
        self.normalize_eps = normalize_eps
        self._rows = nn.Parameter(rows.detach(), requires_grad=False)
        self._norms = nn.Parameter(norms.detach(), requires_grad=False)
        self.bias = nn.Parameter(bias.detach(), requires_grad=False) if bias is not None else None
        self.eval()

    def forward(self, indices, normalize=False):
        if autocast_enabled(self._rows.device.type):
            raise ValueError("Projected lookup requires explicit dtypes; autocast is unsupported")
        if type(normalize) is not bool:
            raise ValueError("normalize must be an explicit boolean branch")
        if self.training or any(parameter.requires_grad for parameter in self.parameters()):
            raise ValueError("Projected lookup is an inference representation")
        result = F.embedding(indices, self._rows)
        if normalize:
            result = result / F.embedding(indices, self._norms)
        if self.bias is not None:
            result = result + self.bias
        return result

    def requires_grad_(self, requires_grad=True):
        if requires_grad:
            raise ValueError("Projected lookup cannot restore the source training parameterization")
        return super().requires_grad_(False)

    def recipe(self):
        return {"kind": "ProjectedEmbeddingLookup", "num_embeddings": self.num_embeddings,
                "embedding_dim": self.embedding_dim, "normalize_eps": self.normalize_eps,
                "bias": self.bias is not None, "dtype": str(self._rows.dtype).removeprefix("torch."),
                "training": self.training}

    @classmethod
    def from_recipe(cls, spec, *, device="cpu"):
        dtype = getattr(torch, spec.get("dtype", ""), None)
        if spec.get("kind") != "ProjectedEmbeddingLookup" or dtype not in _DTYPES or type(spec.get("bias")) is not bool:
            raise ValueError("Invalid projected lookup recipe")
        rows = torch.zeros((spec["num_embeddings"], spec["embedding_dim"]), dtype=dtype, device=device)
        norms = torch.ones((spec["num_embeddings"], 1), dtype=dtype, device=device)
        bias = torch.zeros(spec["embedding_dim"], dtype=dtype, device=device) if spec["bias"] else None
        return cls(rows, norms, bias, spec["normalize_eps"]).train(spec.get("training", False))


class DualModeEmbeddingProjection(nn.Module):
    """Uncompiled reference/rollback API for raw and normalized token branches."""
    def __init__(self, embedding, linear, normalize_eps=1e-12):
        super().__init__()
        self.embedding = embedding
        self.linear = linear
        self.normalize_eps = normalize_eps
        self.training = embedding.training or linear.training

    def forward(self, indices, normalize=False):
        if type(normalize) is not bool:
            raise ValueError("normalize must be an explicit boolean branch")
        x = self.embedding(indices)
        if normalize:
            x = F.normalize(x, p=2, dim=-1, eps=self.normalize_eps)
        return self.linear(x)


def build_projected_lookup(embedding, linear, *, input_contract, normalize_eps=1e-12,
                           chunk_size=1024, validation_chunk_size=257,
                           absolute_tolerance=1e-5, relative_tolerance=1e-5):
    """Compile E/W into Z=E W^T and row norms, sharing raw/normalized branches.

    The two source functions are Linear(Embedding(ids)) and
    Linear(F.normalize(Embedding(ids),p=2,dim=-1,eps=normalize_eps)).
    Raw output is Z[ids]+b; normalized output is Z[ids]/r[ids]+b. Bias must
    follow division. This compiler returns only the projected block; callers
    must prove all external E/W consumers are covered before discarding them.
    No saving is claimed for an enclosing model that retains those tensors.
    """
    from .compiler import CompressionResult, parameter_count, state_bytes
    if input_contract != "raw_or_l2_normalized_token_indices":
        raise ValueError("Declare input_contract='raw_or_l2_normalized_token_indices'")
    if type(embedding) is not nn.Embedding or type(linear) is not nn.Linear:
        raise ValueError("Expected ordinary nn.Embedding and nn.Linear")
    if not isinstance(normalize_eps, (int, float)) or not math.isfinite(normalize_eps) or normalize_eps <= 0:
        raise ValueError("Normalization epsilon must be finite and positive")
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in (chunk_size, validation_chunk_size)):
        raise ValueError("Chunk sizes must be positive integers")
    if any(not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0 for t in (absolute_tolerance, relative_tolerance)):
        raise ValueError("Tolerances must be finite and nonnegative")
    # Registration in a temporary Sequential does not remove either source
    # module from its owner, and no source weights are mutated.
    source = nn.Sequential(embedding, linear)
    source.training = False
    count, width, device = _audit(source)
    if embedding.weight.dtype != linear.weight.dtype or (linear.bias is not None and linear.bias.dtype != linear.weight.dtype):
        raise ValueError("Embedding, projection and bias must have the same explicit dtype")
    uncompiled = DualModeEmbeddingProjection(embedding, linear, normalize_eps)
    before_params, before_bytes = parameter_count(source), state_bytes(source)
    before_resident = _resident_storage_bytes(source)
    report = {"method": "exact_shared_projected_token_lookup", "input_contract": input_contract,
              "domain_size": count, "output_width": width, "source_width": embedding.embedding_dim,
              "evaluation_device": str(device), "torch_version": str(torch.__version__),
              "parameters_before": before_params, "tensor_bytes_before": before_bytes,
              "resident_storage_bytes_before": before_resident,
              "storage_accounting": "Unique registered tensor backing allocations; allocator overhead and external Python owners excluded",
              "normalize_eps": normalize_eps,
              "guarantee": "Exact raw/normalized affine identity in ideal arithmetic; stored tensors are validated numerically on both complete token domains; inference only",
              "ownership_scope": "Caller must cover all external embedding/linear consumers before claiming whole-model savings",
              "floating_point_scope": "Full raw and normalized token domains checked in recorded chunk layouts and shape probes; future device/batch rounding not universally bounded"}
    with torch.inference_mode(False), torch.no_grad():
        rows = torch.empty((count, width), dtype=embedding.weight.dtype, device=device)
        norms = torch.empty((count, 1), dtype=embedding.weight.dtype, device=device)
        for start in range(0, count, chunk_size):
            features = embedding.weight[start:start + chunk_size]
            rows[start:start + features.shape[0]].copy_(F.linear(features, linear.weight, None))
            norms[start:start + features.shape[0]].copy_(features.norm(p=2, dim=-1, keepdim=True).clamp_min(normalize_eps))
        if not torch.isfinite(rows).all() or not torch.isfinite(norms).all() or not (norms > 0).all():
            raise ValueError("Projected rows or normalization denominators are nonfinite/zero")
        candidate = ProjectedEmbeddingLookup(rows, norms,
            linear.bias.detach().clone() if linear.bias is not None else None, normalize_eps)
        after_bytes = state_bytes(candidate)
        after_resident = _resident_storage_bytes(candidate)
        report.update(parameters_after=parameter_count(candidate), tensor_bytes_after=after_bytes,
                      resident_storage_bytes_after=after_resident,
                      enumeration_chunk_size=chunk_size, validation_chunk_size=validation_chunk_size)
        if after_bytes >= before_bytes or after_resident >= before_resident:
            retained = copy.deepcopy(uncompiled)
            report.update(status="retained_no_byte_saving", proposed_tensor_bytes=after_bytes,
                          proposed_resident_storage_bytes=after_resident,
                          parameters_after=before_params, tensor_bytes_after=before_bytes,
                          resident_storage_bytes_after=_resident_storage_bytes(retained), validation=None)
            return CompressionResult(retained, report)
        branch_reports = {}
        accepted = True
        for normalized in (False, True):
            def reference(ids):
                value = embedding(ids)
                if normalized:
                    value = F.normalize(value, p=2, dim=-1, eps=normalize_eps)
                return linear(value)
            def compiled(ids):
                return candidate(ids, normalize=normalized)
            statistics = {"squared_error": 0.0, "squared_reference": 0.0,
                          "max_abs": 0.0, "bitwise_identical": True}
            try:
                for start in range(0, count, validation_chunk_size):
                    ids = torch.arange(start, min(count, start + validation_chunk_size), device=device)
                    _compare_rows(reference, compiled, ids, statistics)
                probes = [torch.tensor(n, device=device) for n in sorted({0, count // 2, count - 1})]
                probes += [torch.empty(0, dtype=torch.long, device=device),
                           torch.empty((2, 0), dtype=torch.long, device=device),
                           (torch.arange(6, device=device).reshape(2, 3) % count).t()]
                for ids in probes:
                    _compare_rows(reference, compiled, ids, statistics)
                relative_l2 = math.sqrt(statistics["squared_error"]) / max(math.sqrt(statistics["squared_reference"]), 1e-12)
                gate = {"accepted": statistics["max_abs"] <= absolute_tolerance and relative_l2 <= relative_tolerance,
                        "tokens_checked": count, "shape_probes": len(probes), "max_abs": statistics["max_abs"],
                        "relative_l2": relative_l2, "bitwise_identical": statistics["bitwise_identical"]}
            except (ValueError, TypeError, RuntimeError) as exc:
                gate = {"accepted": False, "error": str(exc)}
            branch_reports["normalized" if normalized else "raw"] = gate
            accepted = accepted and gate["accepted"]
        report["validation"] = {"accepted": accepted, "branches": branch_reports,
            "absolute_tolerance": absolute_tolerance, "relative_tolerance": relative_tolerance}
        if not accepted:
            retained = copy.deepcopy(uncompiled)
            report.update(status="rejected_and_rolled_back", proposal_parameters_after=parameter_count(candidate),
                          proposal_resident_storage_bytes=after_resident,
                          parameters_after=before_params, tensor_bytes_after=before_bytes,
                          resident_storage_bytes_after=_resident_storage_bytes(retained))
            return CompressionResult(retained, report)
    report["status"] = "accepted_on_full_domain_validation"
    candidate._seal_validation()
    return CompressionResult(candidate, report)
