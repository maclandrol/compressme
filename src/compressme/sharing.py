"""Share byte-identical frozen parameters without changing arithmetic.

Useful for jointly serving related models or repeated frozen modules. This is
inference storage sharing, not low-rank approximation or a speed optimization.
Trainable parameters, buffers and noncontiguous parameter views are retained.
"""
from __future__ import annotations

import hashlib
import torch
from torch import nn

from .finite_lookup import _bits_equal, _has_hooks, _resident_storage_bytes


def _digest(tensor):
    values = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    digest = hashlib.sha256()
    for start in range(0, values.numel(), 1 << 20):
        chunk = values[start:start + (1 << 20)]
        try:
            payload = chunk.numpy().tobytes()
        except RuntimeError as error:
            if 'numpy' not in str(error).lower():
                raise
            payload = bytes(chunk.tolist())
        digest.update(payload)
    return digest.digest()


def parameter_aliases(model):
    """Describe actual Parameter-object aliases using direct canonical targets."""
    owners, aliases = {}, {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if id(parameter) in owners:
            aliases[name] = owners[id(parameter)]
        else:
            owners[id(parameter)] = name
    return aliases


def _storage_key(tensor):
    return (str(tensor.device), tensor.untyped_storage()._cdata, tensor.storage_offset(),
            tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.is_neg(), tensor.is_conj())


def parameter_storage_aliases(model):
    """Equal parameter storage views, excluding already shared objects."""
    owners, aliases = {}, {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if not parameter.numel() or parameter.layout != torch.strided or parameter.device.type == 'meta':
            continue
        key = _storage_key(parameter)
        if key in owners and parameter is not owners[key][1]:
            aliases[name] = owners[key][0]
        elif key not in owners:
            owners[key] = (name, parameter)
    return aliases


def _slot(model, path):
    parent, _, name = path.rpartition('.')
    owner = model.get_submodule(parent) if parent else model
    if name not in owner._parameters or not isinstance(owner._parameters[name], nn.Parameter):
        raise ValueError(f'Parameter alias path differs from architecture: {path}')
    return owner, name


def restore_parameter_aliases(model, aliases, stored, *, storage_only=False):
    """Restore declared ties after recipe replay, before strict tensor loading.

Aliases must have equal stored bytes and compatible shapes/dtypes. Checking the
payload prevents repeated load_state_dict writes from selecting conflicting
values for a newly shared parameter. Architecture code remains trusted input.
"""
    if not isinstance(aliases, dict):
        raise ValueError('Invalid parameter alias manifest')
    prepared = []
    for path, target in aliases.items():
        if (type(path) is not str or type(target) is not str or not path or not target
                or path == target or target in aliases or path not in stored or target not in stored):
            raise ValueError('Invalid parameter alias entry')
        owner, name = _slot(model, path)
        target_owner, target_name = _slot(model, target)
        old, new = owner._parameters[name], target_owner._parameters[target_name]
        if (old.shape != new.shape or old.dtype != new.dtype or old.device != new.device
                or old.stride() != new.stride() or not _bits_equal(stored[path], stored[target])):
            raise ValueError('Conflicting parameter alias values or architecture')
        prepared.append((owner, name, new))
    for owner, name, new in prepared:
        if storage_only:
            owner._parameters[name].data = new.detach()
        else:
            owner._parameters[name] = new
    return model


def share_frozen_parameters(source, *, inplace=False):
    """Return a copied model with identical immutable parameters stored once.

Put multiple models in an nn.ModuleDict to share across them. Evaluation mode
is required. Only already-frozen, contiguous, finite real Parameters are
eligible; their shape, dtype, device, stride and value bytes must match. Hashes
only find candidates: a separate byte comparison establishes equality.

    Distinct Parameter objects and their enumeration are preserved; their data
    storage is shared. Storage-identity inspection and mutation are outside the
    frozen contract. Apply after final device placement: later .to() conversions
    can allocate separate storage again. `inplace=True` avoids a full model copy.
    The
report measures registered state, not process RSS while callers retain source
models, and does not claim a runtime speedup or gradient equivalence.
"""
    from .compiler import CompressionResult, _copy_with_compiler_metadata, parameter_count, state_bytes
    if not isinstance(source, nn.Module) or type(inplace) is not bool:
        raise ValueError('Supply a PyTorch model or ModuleDict of models')
    if any(m.training or _has_hooks(m) or 'forward' in vars(m) for m in source.modules()):
        raise ValueError('Use hook-free unmodified evaluation modules')
    if any(getattr(p, '_backward_hooks', None) for p in source.parameters()):
        raise ValueError('Parameter hooks are outside frozen sharing')
    before = _resident_storage_bytes(source)
    before_params, before_logical = parameter_count(source), state_bytes(source)
    # deepcopy does not preserve arbitrary storage-view relationships. Refuse
    # these layouts instead of silently growing or changing an alias contract.
    allocations = {}
    for kind, tensors in (('parameter', source.named_parameters(remove_duplicate=False)),
                          ('buffer', source.named_buffers(remove_duplicate=False))):
        for name, tensor in tensors:
            if tensor.numel():
                allocation = (str(tensor.device), tensor.untyped_storage()._cdata)
                allocations.setdefault(allocation, []).append((kind, name, tensor))
    for group in allocations.values():
        distinct = {id(tensor) for _, _, tensor in group}
        if len(distinct) > 1 and (any(kind == 'buffer' for kind, _, _ in group)
                                  or len({_storage_key(tensor) for _, _, tensor in group}) > 1):
            raise ValueError('Parameter/buffer or distinct-view storage aliases require a preserving adapter')
    candidate = source if inplace else _copy_with_compiler_metadata(source)
    buckets, replacements, skipped, seen = {}, [], [], set()
    alias_paths = {}
    for path, parameter in candidate.named_parameters(remove_duplicate=False):
        alias_paths.setdefault(id(parameter), []).append(path)
    for name, parameter in candidate.named_parameters(remove_duplicate=False):
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if (type(parameter) is not nn.Parameter or parameter.requires_grad or parameter.layout != torch.strided
                or parameter.device.type == 'meta' or not parameter.is_contiguous()
                or not parameter.is_floating_point() or parameter.is_neg()
                or parameter.is_conj() or not torch.isfinite(parameter).all()):
            skipped.append(name)
            continue
        key = (tuple(parameter.shape), tuple(parameter.stride()), parameter.dtype,
               str(parameter.device), _digest(parameter))
        matches = buckets.setdefault(key, [])
        match = next(((target, value) for target, value in matches if _bits_equal(value, parameter)), None)
        if match is None:
            matches.append((name, parameter))
            continue
        target_name, target = match
        # Keep each original Parameter object and enumeration count. Existing
        # object aliases observe its updated data storage automatically.
        paths = alias_paths[id(parameter)]
        parameter.data = target.detach()
        for path in paths:
            replacements.append({'path': path, 'target': target_name})
    after = _resident_storage_bytes(candidate)
    if after > before:
        raise ValueError('Sharing proposal increased registered storage')
    if replacements:
        # Separate Parameter objects have independent version counters even
        # when their data storage is shared. Ordinary writes through one alias
        # need not invalidate a seal observing only another alias. Treat all
        # shared weights as immutable and clear earlier per-module seals.
        for module in candidate.modules():
            if hasattr(module, '_validated_signature'):
                module._validated_signature = None
            if hasattr(module, '_bound_versions'):
                module._bound_versions = None
    report = {'method': 'byte_identical_frozen_parameter_sharing',
              'status': 'shared' if replacements else 'no_eligible_duplicates',
              'parameters_before': before_params, 'parameters_after': parameter_count(candidate),
              'tensor_bytes_before': before_logical, 'tensor_bytes_after': state_bytes(candidate),
              'resident_storage_bytes_before': before, 'resident_storage_bytes_after': after,
              'saved_resident_storage_bytes': before - after,
              'storage_reduced': after < before, 'inplace': inplace,
              'mutation_contract': 'Shared parameter data is immutable; per-Parameter version counters do not observe every alias write',
              'previous_compiler_seals': 'invalidated' if replacements else 'unchanged',
              'parameter_replacements': replacements, 'retained_ineligible_parameters': skipped,
              'guarantee': 'Unchanged parameter value bytes, objects and enumeration; frozen inference without storage inspection or mutation',
              'scope': 'Caller-owned source models and activations excluded from registered storage; complete-output validation remains required',
              'performance': 'Storage sharing only; each model still executes its original operations'}
    return CompressionResult(candidate, report)
