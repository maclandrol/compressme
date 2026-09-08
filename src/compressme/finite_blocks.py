"""Apply proved finite-token rewrites inside an ordinary model, then validate it.

Discovery identifies registered Sequential blocks and explicitly declared
FiniteTokenFanout graphs; it does not infer the full API of arbitrary Python
code. The explicit contract excludes direct reads of a selected block's internal weights and side effects. Registered aliases are
checked against the enclosing model before each rewrite.
"""
from __future__ import annotations

import copy
import math
from torch import nn
from .compiler import CompressionResult, _copy_with_compiler_metadata, _set, parameter_count, state_bytes
from .finite_lookup import _audit, _flatten, _has_hooks, _resident_storage_bytes, compile_finite_lookup
from .finite_fanout import FiniteTokenFanout, _audit_source, _profile_settings, compile_finite_fanout
from .validation import Example, validate


def _inside(path, ancestor):
    return not ancestor or path == ancestor or path.startswith(ancestor + ".")


def _selection(model, paths):
    if paths is None:
        selected = []
        for path, module in model.named_modules():
            if type(module) is FiniteTokenFanout:
                selected.append(path)
            elif type(module) is nn.Sequential:
                layers = _flatten(module)
                if layers and type(layers[0]) is nn.Embedding:
                    selected.append(path)
        return selected
    if isinstance(paths, str):
        raise TypeError("paths must be a sequence of module paths, not one string")
    selected = list(paths)
    if not selected or any(not isinstance(path, str) for path in selected):
        raise ValueError("Provide at least one string module path, or None for discovery")
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate finite-block paths")
    for path in selected:
        try:
            model.get_submodule(path)
        except AttributeError as exc:
            raise ValueError(f"Unknown finite-block path: {path}") from exc
    if any(a != b and _inside(a, b) for a in selected for b in selected):
        raise ValueError("Choose non-overlapping finite-block paths")
    return selected


def compile_finite_blocks(model, *, input_contract, validation, paths=None,
                          chunk_size=1024, validation_chunk_size=257,
                          absolute_tolerance=1e-5, relative_tolerance=1e-5,
                          fanout_row_count_profiles=None):
    """Compile closed token blocks and require complete enclosing-model checks.

    input_contract='token_indices_only' declares that each selected block is
    consumed through forward(valid_integer_ids), without separate access to its
    child weights/attributes. Registered aliases are checked; unregistered Python
    references and side effects remain the caller's responsibility. This is an
    inference representation and does not preserve training parameterization.

    paths=None discovers eligible-looking ordinary Sequential blocks and explicit
    FiniteTokenFanout graphs, including every declared branch and residual output.
    Explicit paths select particular blocks. Unsupported/unprofitable blocks
    are retained with reasons. Any failed complete-output comparison rolls back all rewrites.
    The enclosing Python model class is retained when rewrites target its children.
    A selected root block is itself replaced under the same declared contract.
    fanout_row_count_profiles optionally declares empirical row-count regimes
    for selected fanout graphs only. Sequential blocks retain their usual pass.
    """
    if input_contract != "token_indices_only":
        raise ValueError("Declare input_contract='token_indices_only' for selected blocks")
    if not isinstance(model, nn.Module):
        raise TypeError("Expected a PyTorch model")
    profiles = _profile_settings(fanout_row_count_profiles)
    examples = list(validation) if validation is not None else []
    if not examples or any(not isinstance(example, Example) for example in examples):
        raise ValueError("Provide nonempty Example validation inputs for the whole model")
    for value in (absolute_tolerance, relative_tolerance):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError("Tolerances must be finite nonnegative numbers")
    for value in (chunk_size, validation_chunk_size):
        if type(value) is not int or value < 1:
            raise ValueError("Chunk sizes must be positive integers")
    if any(module.training for module in model.modules()):
        raise ValueError("Finite-block compilation requires model.eval()")
    if any(_has_hooks(module) or "forward" in vars(module) for module in model.modules()):
        raise ValueError("Model hooks and instance forward replacements are unsupported")
    selected = _selection(model, paths)
    # Parents are tried before nested children; accepted parents subsume them.
    selected.sort(key=lambda path: -1 if not path else path.count("."))
    # deepcopy may separate distinct tensors that overlap source storage. Audit
    # original ownership before copying, using the graph's own declared grammar.
    compilers, audit_errors = {}, {}
    for path in selected:
        source_block = model.get_submodule(path)
        is_fanout = type(source_block) is FiniteTokenFanout
        auditor = _audit_source if is_fanout else _audit
        compilers[path] = compile_finite_fanout if is_fanout else compile_finite_lookup
        try:
            auditor(source_block, owner_model=model)
        except (ValueError, TypeError, RuntimeError) as exc:
            audit_errors[path] = str(exc)
    candidate = _copy_with_compiler_metadata(model)
    source_dtypes = copy.deepcopy(getattr(model, "_compressme_source_tensor_dtypes",
        {name: str(t.dtype).removeprefix("torch.") for name, t in model.state_dict().items()}))
    candidate._compressme_source_tensor_dtypes = source_dtypes
    before_params, before_bytes = parameter_count(model), state_bytes(model)
    before_resident = _resident_storage_bytes(model)
    # Preserve the existing Sequential-only report label for compatibility.
    discovery = ("registered_finite_block_discovery"
                 if any(type(model.get_submodule(path)) is FiniteTokenFanout for path in selected)
                 else "registered_sequential_discovery")
    report = {"method": "exact_finite_blocks", "input_contract": input_contract,
              "selection": discovery if paths is None else "explicit_paths",
              "fanout_row_count_profiles": list(profiles),
              "parameters_before": before_params, "tensor_bytes_before": before_bytes,
              "resident_storage_bytes_before": before_resident,
              "storage_accounting": "Unique registered tensor backing allocations; allocator overhead and external Python owners excluded",
              "blocks": [], "rewrites": [],
              "guarantee": "Row-local partial-evaluation identities with complete token enumeration; enclosing-model numerical checks are empirical, not an all-input floating-point proof",
              "ownership_scope": "Registered aliases inspected; no direct internal-weight reads or unregistered external consumers under the declared contract"}
    accepted_paths = []
    for path in selected:
        if any(_inside(path, previous) for previous in accepted_paths):
            report["blocks"].append({"path": path, "status": "subsumed_by_accepted_parent"})
            continue
        if path in audit_errors:
            report["blocks"].append({"path": path, "status": "retained_unsupported", "reason": audit_errors[path]})
            continue
        block = candidate.get_submodule(path)
        try:
            options = ({"row_count_profiles": profiles or None}
                       if type(block) is FiniteTokenFanout else {})
            local = compilers[path](block, input_contract=input_contract, owner_model=candidate,
                chunk_size=chunk_size, validation_chunk_size=validation_chunk_size,
                absolute_tolerance=absolute_tolerance, relative_tolerance=relative_tolerance, **options)
        except (ValueError, TypeError, RuntimeError) as exc:
            report["blocks"].append({"path": path, "status": "retained_unsupported", "reason": str(exc)})
            continue
        report["blocks"].append({"path": path, **local.report})
        if local.report["status"] == "accepted_on_full_domain_validation":
            candidate = _set(candidate, path, local.model)
            accepted_paths.append(path)
            report["rewrites"].append(path)
    candidate._compressme_source_tensor_dtypes = source_dtypes
    report.update(parameters_after=parameter_count(candidate), tensor_bytes_after=state_bytes(candidate),
                  resident_storage_bytes_after=_resident_storage_bytes(candidate))
    storage_saving = (report["resident_storage_bytes_after"] < before_resident
                      and report["tensor_bytes_after"] < before_bytes)
    report["storage_saving"] = storage_saving if accepted_paths else None
    try:
        gate = validate(model, candidate, examples, relative_tolerance=relative_tolerance,
                        absolute_tolerance=absolute_tolerance)
    except (ValueError, TypeError, RuntimeError) as exc:
        gate = {"accepted": False, "error": str(exc), "scope": "Complete-output validation failed"}
    report["validation"] = gate
    if not gate["accepted"] or (accepted_paths and not storage_saving):
        report.update(status="rejected_and_rolled_back", proposal_parameters_after=parameter_count(candidate),
                      proposal_tensor_bytes_after=state_bytes(candidate), proposed_rewrites=list(accepted_paths),
                      proposal_resident_storage_bytes_after=_resident_storage_bytes(candidate),
                      rewrites=[], parameters_after=before_params, tensor_bytes_after=before_bytes)
        if gate["accepted"]:
            report["rollback_reason"] = "No enclosing-model saving in both logical tensor bytes and unique registered backing storage"
        candidate = _copy_with_compiler_metadata(model)
        report["resident_storage_bytes_after"] = _resident_storage_bytes(candidate)
    else:
        report["status"] = "accepted_on_validation_examples" if accepted_paths else "retained_no_accepted_rewrites"
    return CompressionResult(candidate, report)
