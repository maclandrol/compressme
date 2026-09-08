"""Pinned Hugging Face weights with a caller-owned local PyTorch architecture.

No remote Python is downloaded or executed. Tensor names cannot establish a
computation graph: a trusted local factory and real validation inputs are
required to compress. Optional dependencies: huggingface_hub and safetensors.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

_INDEX_LIMIT = 16 * 1024 * 1024


def _hub():
    import huggingface_hub
    return huggingface_hub


def _filename(name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("Expected a repository-relative filename")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError(f"Unsafe or noncanonical repository filename: {name!r}")
    return name


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _same_tensor_bits(left, right):
    """Compare stored logical values including the sign bit of floating zero."""
    import torch
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    def raw(tensor):
        return tensor.detach().resolve_conj().resolve_neg().cpu().contiguous().reshape(-1).view(torch.uint8)
    return torch.equal(raw(left), raw(right))


def _download(repo_id, name, sha, files, token, cache_dir):
    _filename(name)
    if name not in files:
        raise ValueError(f"File is absent from pinned repository metadata: {name}")
    path = Path(_hub().hf_hub_download(repo_id=repo_id, filename=name, revision=sha,
                                     token=token, cache_dir=cache_dir))
    actual_size = path.stat().st_size
    expected_size = files[name]["size"]
    if expected_size is not None and actual_size != expected_size:
        raise ValueError(f"Downloaded file size differs from pinned metadata: {name}")
    digest = _sha256(path)
    expected_digest = files[name].get("lfs_sha256")
    if expected_digest and digest != expected_digest:
        raise ValueError(f"Downloaded file hash differs from pinned metadata: {name}")
    return path, {"filename": name, "size": actual_size, "sha256": digest}


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate key in checkpoint index: {key}")
        result[key] = value
    return result


def _read_index(repo_id, name, sha, files, token, cache_dir):
    size = files[name]["size"]
    if size is None or size > _INDEX_LIMIT:
        raise ValueError("Checkpoint index size must be known and at most 16 MiB")
    path, record = _download(repo_id, name, sha, files, token, cache_dir)
    if path.stat().st_size > _INDEX_LIMIT:
        raise ValueError("Checkpoint index exceeds 16 MiB")
    index = json.loads(path.read_text(), object_pairs_hook=_unique_json)
    mapping = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Sharded safetensors index needs a nonempty weight_map")
    resolved = {}
    for tensor_name, shard in mapping.items():
        if not isinstance(tensor_name, str) or not tensor_name or not isinstance(shard, str):
            raise ValueError("Checkpoint index names must be nonempty strings")
        _filename(shard)
        shard = str(PurePosixPath(name).parent / shard)
        if not shard.endswith(".safetensors") or shard not in files:
            raise ValueError(f"Index references an unavailable safetensors shard: {shard}")
        resolved[tensor_name] = shard
    return resolved, record


def inspect_huggingface(repo_id, revision="main", filename=None, *, token=None, cache_dir=None):
    """Inspect weight metadata at one pinned commit, without loading weights.

    Small safetensors index JSON files are read to distinguish full checkpoints
    from their component shards. Ambiguous repositories return candidates and
    no selection. An explicit filename must be a listed safe checkpoint/index,
    or an explicitly chosen .ckpt. No architecture code is read or executed.
    """
    info = _hub().HfApi().model_info(repo_id, revision=revision, files_metadata=True, token=token)
    sha = _field(info, "sha")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
        raise ValueError("Hub metadata did not resolve an immutable 40-character commit SHA")
    sha = sha.lower()
    files = {}
    for item in _field(info, "siblings", []) or []:
        name = _filename(_field(item, "rfilename"))
        size = _field(item, "size")
        if size is not None and (not isinstance(size, int) or size < 0):
            raise ValueError(f"Invalid metadata size for {name}")
        lfs_sha = _field(_field(item, "lfs", {}), "sha256")
        if name in files:
            raise ValueError(f"Duplicate filename in repository metadata: {name}")
        files[name] = {"filename": name, "size": size, "lfs_sha256": lfs_sha}
    if filename is not None:
        _filename(filename)
        if filename not in files or not filename.endswith((".safetensors", ".safetensors.index.json", ".ckpt")):
            raise ValueError("Choose a listed .safetensors, .safetensors.index.json or explicit .ckpt file")
    indexes, index_records = {}, []
    for name in sorted(files):
        if name.endswith(".safetensors.index.json") and (filename is None or filename == name):
            indexes[name], record = _read_index(repo_id, name, sha, files, token, cache_dir)
            index_records.append(record)
    shards = {name for mapping in indexes.values() for name in mapping.values()}
    candidates = sorted(set(indexes) | {name for name in files
        if name.endswith((".safetensors", ".ckpt")) and name not in shards})
    selected = filename
    if selected is None and len(candidates) == 1 and not candidates[0].endswith(".ckpt"):
        selected = candidates[0]
    selected_files = sorted(set(indexes[selected].values())) if selected in indexes else ([selected] if selected else [])
    sizes = [files[name]["size"] for name in selected_files]
    return {"repo_id": repo_id, "requested_revision": revision, "revision": sha,
            "files": [files[name] for name in sorted(files)], "candidates": candidates,
            "selected": selected, "weight_files": selected_files,
            "weight_bytes": sum(sizes) if sizes and all(size is not None for size in sizes) else None,
            "weight_map": indexes.get(selected), "index_files": index_records,
            "remote_code_executed": False,
            "scope": "File metadata does not establish model architecture or compressibility"}


def _load_tensors(inspection, token, cache_dir):
    import torch
    files = {item["filename"]: item for item in inspection["files"]}
    state, provenance = {}, list(inspection["index_files"])
    mapping = inspection["weight_map"]
    for name in inspection["weight_files"]:
        path, record = _download(inspection["repo_id"], name, inspection["revision"], files, token, cache_dir)
        provenance.append(record)
        if name.endswith(".safetensors"):
            from safetensors.torch import load_file
            part = load_file(str(path), device="cpu")
        else:
            # No fallback to unrestricted pickle deserialization is allowed.
            part = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(part, Mapping) and "state_dict" in part:
                part = part["state_dict"]
        if not isinstance(part, Mapping) or any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in part.items()):
            raise ValueError("Checkpoint must contain a mapping of tensor names to tensors")
        for key, tensor in part.items():
            if key in state:
                raise ValueError(f"Duplicate tensor across checkpoint shards: {key}")
            if mapping is not None and mapping.get(key) != name:
                raise ValueError(f"Shard content conflicts with index entry: {key}")
            if tensor.layout != torch.strided or tensor.device.type != "cpu" or tensor.is_quantized:
                raise ValueError("Only ordinary dense CPU checkpoint tensors are supported")
            if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
                raise ValueError(f"Checkpoint contains nonfinite tensor: {key}")
            state[key] = tensor
    if mapping is not None and set(mapping) != set(state):
        raise ValueError("Shard contents do not provide exactly the index tensor names")
    return state, provenance


def _strict_load(model, state):
    import torch
    from torch import nn
    if not isinstance(model, nn.Module):
        raise TypeError("The local model_factory must return torch.nn.Module")
    expected = model.state_dict()
    if any(not isinstance(value, torch.Tensor) or value.device.type == "meta" for value in expected.values()):
        raise ValueError("The factory must create materialized tensor state, without custom extra_state")
    targets = dict(model.named_parameters(remove_duplicate=False))
    targets.update(dict(model.named_buffers(remove_duplicate=False)))
    tied, storage = defaultdict(list), {}
    for name, tensor in targets.items():
        if name not in expected:
            continue
        tied[id(tensor)].append(name)
        if tensor.numel():
            key = (str(tensor.device), tensor.untyped_storage().data_ptr())
            if key in storage and storage[key] != id(tensor):
                raise ValueError("Distinct factory tensors share storage; explicit alias adapter required")
            storage[key] = id(tensor)
    state = dict(state)
    filled = {}
    for names in tied.values():
        present = [name for name in names if name in state]
        if not present:
            continue
        tensor = state[present[0]]
        for name in present[1:]:
            other = state[name]
            if not _same_tensor_bits(other, tensor):
                raise ValueError(f"Conflicting checkpoint values for tied factory parameters: {names}")
        for name in names:
            if name not in state:
                state[name] = tensor
                filled[name] = present[0]
    missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    if missing or unexpected:
        raise ValueError(f"Strict checkpoint keys mismatch; missing={missing}, unexpected={unexpected}")
    for name, tensor in expected.items():
        if tensor.shape != state[name].shape or tensor.dtype != state[name].dtype:
            raise ValueError(f"Strict shape/dtype mismatch for {name}: factory {tuple(tensor.shape)}/{tensor.dtype}, checkpoint {tuple(state[name].shape)}/{state[name].dtype}")
    model.load_state_dict(state, strict=True)  # copy, never assign: preserve factory aliases
    loaded = model.state_dict()
    if set(loaded) != set(state):
        raise ValueError("Factory loading changed the state key set")
    after = dict(model.named_parameters(remove_duplicate=False))
    after.update(dict(model.named_buffers(remove_duplicate=False)))
    if any(len({id(after[name]) for name in names}) != 1 for names in tied.values()):
        raise ValueError("Factory loading changed declared parameter aliases")
    for name, value in loaded.items():
        if not _same_tensor_bits(value, state[name]):
            raise ValueError(f"Factory loading modified checkpoint tensor: {name}")
    return {"filled_declared_tied_keys": filled, "strict_shapes_and_dtypes": True}


def compress_huggingface(repo_id, model_factory, *, validation, revision="main", filename=None,
                         method="affine", relative_tolerance=1e-5, absolute_tolerance=1e-5,
                         finite_paths=None, finite_input_contract=None, finite_row_count_profiles=None,
                         token=None, cache_dir=None):
    """Load pinned weights into a local API, then validate a chosen exact rewrite.

    `validation` is a nonempty sequence of compressme.Example inputs. Dtypes are
    never silently cast: the local factory must match the checkpoint explicitly.
    A failed output comparison returns the loaded original model and report.
    The caller's local factory is trusted Python; Hub architecture code is never
    executed by this workflow. `method='affine'` discovers affine computation
    rewrites; `method='constant_embeddings'` deduplicates only frozen embedding
    rows proved bitwise identical. method='finite_lookup' discovers or selects
    closed Sequential token encoders or explicit FiniteTokenFanout graphs and requires the explicit
    finite_input_contract='token_indices_only'. finite_paths selects particular
    blocks; None discovers eligible-looking registered blocks. Every retained
    output is checked. Dynamic models may need an explicit adapter.
    finite_row_count_profiles optionally selects empirical evaluation shapes
    for declared fanout graphs; it does not choose a device or infer a guarantee.
    """
    from .affine import compile_affine
    from .compiler import CompressionResult, parameter_count, state_bytes, _copy_with_compiler_metadata
    from .validation import Example, seeded, validate
    if method not in {"affine", "constant_embeddings", "finite_lookup"}:
        raise ValueError("Choose method='affine', 'constant_embeddings' or 'finite_lookup'")
    if method == "finite_lookup":
        from .finite_fanout import _profile_settings
        _profile_settings(finite_row_count_profiles)
        if finite_input_contract != "token_indices_only":
            raise ValueError("Declare finite_input_contract='token_indices_only' for selected token blocks")
        if isinstance(finite_paths, str):
            raise TypeError("finite_paths must be a sequence of module paths")
        if finite_paths is not None:
            finite_paths = list(finite_paths)
            if not finite_paths or any(not isinstance(path, str) for path in finite_paths) or len(set(finite_paths)) != len(finite_paths):
                raise ValueError("finite_paths must contain distinct string module paths")
    elif finite_paths is not None or finite_input_contract is not None or finite_row_count_profiles is not None:
        raise ValueError("Finite-block options require method='finite_lookup'")
    if not callable(model_factory):
        raise TypeError("Supply a trusted local model_factory callable")
    examples = list(validation) if validation is not None else []
    if not examples or any(not isinstance(example, Example) for example in examples):
        raise ValueError("Provide a nonempty sequence of compressme.Example validation inputs")
    if any(not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0 for t in (relative_tolerance, absolute_tolerance)):
        raise ValueError("Output tolerances must be finite nonnegative numbers")
    inspection = inspect_huggingface(repo_id, revision, filename, token=token, cache_dir=cache_dir)
    if inspection["selected"] is None:
        raise ValueError(f"Choose filename explicitly from checkpoint candidates: {inspection['candidates']}")
    state, files = _load_tensors(inspection, token, cache_dir)
    with seeded(0):
        model = model_factory()
    loading = _strict_load(model, state)
    model.eval()
    if method == "affine":
        result = compile_affine(model)
    elif method == "finite_lookup":
        from .finite_blocks import compile_finite_blocks
        result = compile_finite_blocks(model, paths=finite_paths, input_contract=finite_input_contract,
            validation=examples, relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance, fanout_row_count_profiles=finite_row_count_profiles)
    else:
        from .constant_embeddings import deduplicate_embeddings
        candidate, report = deduplicate_embeddings(model)
        candidate._compressme_source_tensor_dtypes = {
            name: str(tensor.dtype).removeprefix("torch.") for name, tensor in model.state_dict().items()}
        report.update(tensor_bytes_before=state_bytes(model), tensor_bytes_after=state_bytes(candidate))
        result = CompressionResult(candidate, report)
    if method == "finite_lookup":
        # This compiler has already checked the enclosing model and rolled back
        # on failure. Do not rerun inference merely to attach Hub provenance.
        gate = result.report["validation"]
    else:
        try:
            gate = validate(model, result.model, examples, relative_tolerance=relative_tolerance,
                            absolute_tolerance=absolute_tolerance)
        except (ValueError, TypeError, RuntimeError) as exc:
            gate = {"accepted": False, "error": str(exc), "scope": "Validation failed; proposal rolled back"}
    result.report["validation"] = gate
    if not gate["accepted"]:
        result.report["status"] = "rejected_and_rolled_back"
        result.report.setdefault("proposal_parameters_after", parameter_count(result.model))
        result.model = _copy_with_compiler_metadata(model)
        result.report["parameters_after"] = parameter_count(model)
        result.report["tensor_bytes_after"] = state_bytes(model)
    elif method != "finite_lookup" or result.report["rewrites"]:
        result.report["status"] = "accepted_on_validation_examples"
    result.report["huggingface"] = {"repo_id": repo_id, "requested_revision": revision,
        "revision": inspection["revision"], "filename": inspection["selected"],
        "files": files, "loading": loading, "remote_code_executed": False,
        "architecture": "Caller-provided local factory; not inferred from tensor names"}
    result.model._compressme_hub_provenance = result.report["huggingface"]
    return result
