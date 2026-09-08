"""Tensor-only exports. Reload requires the original model constructor/code."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file, load as load_tensors, save as save_tensors
from .linear import LayerBound, LowRankLinear, NormLinear
from .compiler import CompressionResult, _set
from .moljepa import AffineInputGraphEncoder, SmilesOnlyMolJEPA, fuse_moljepa_input_projections, specialize_moljepa_smiles
from .affine import compile_affine
from .norm_sandwich import NormSandwich
from .constant_embeddings import ConstantRowEmbedding
from .finite_lookup import FiniteTokenLookup, ProjectedEmbeddingLookup, _has_hooks
from .finite_fanout import FiniteFanoutLookup
from .sharing import parameter_aliases, parameter_storage_aliases, restore_parameter_aliases
from .frozen_tables import XorProfileLookup
from .packed_embedding import PackedFrozenEmbedding



def _storage_aliases(state):
    """Deduplicate identical storage views without comparing floating values.

    This is a file representation only. The caller's architecture still decides
    which parameters are tied when the expanded state is loaded into it.
    """
    tensors, aliases, owners = {}, {}, {}
    for name, tensor in state.items():
        key = None
        if tensor.layout == torch.strided and tensor.device.type != "meta" and tensor.numel():
            key = (str(tensor.device), tensor.untyped_storage()._cdata,
                   tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()),
                   tensor.dtype, tensor.is_conj(), tensor.is_neg())
        if key is not None and key in owners:
            aliases[name] = owners[key]
        else:
            tensors[name] = tensor.detach().cpu().contiguous().clone()
            if key is not None:
                owners[key] = name
    return tensors, aliases


def _expand_storage_aliases(stored, aliases):
    if not isinstance(aliases, dict):
        raise ValueError("Invalid tensor alias manifest")
    expanded = dict(stored)
    for name, target in aliases.items():
        # Targets must be physical entries, never chains, cycles or overwrites.
        if (not isinstance(name, str) or not isinstance(target, str)
                or name in stored or target not in stored or name == target):
            raise ValueError("Invalid tensor alias entry")
        expanded[name] = stored[target]
    return expanded



def _module_aliases(model):
    owners, aliases = {}, {}
    for path, module in model.named_modules(remove_duplicate=False):
        if id(module) in owners:
            aliases[path] = owners[id(module)]
        else:
            owners[id(module)] = path
    return aliases


def _restore_module_aliases(model, aliases):
    if not isinstance(aliases, dict):
        raise ValueError("Invalid module alias manifest")
    for path, target in aliases.items():
        if (not isinstance(path, str) or not path or not isinstance(target, str)
                or not target or target in aliases or path == target
                or target.startswith(path + ".")):
            raise ValueError("Invalid module alias entry")
    for path, target in sorted(aliases.items(), key=lambda item: item[0].count(".")):
        try:
            # Recipes may have replaced only the canonical path. Reuse that
            # exact module object at every recorded alias, not a second copy.
            model.get_submodule(path)
            replacement = model.get_submodule(target)
        except AttributeError as exc:
            raise ValueError("Module alias differs from the original architecture") from exc
        model = _set(model, path, replacement)
    return model


def save(result: CompressionResult, directory, *, packing=False):
    if any(getattr(m, "_compressme_runtime_only", False) for m in result.model.modules()):
        raise ValueError("Save the portable model before runtime acceleration; reload it with load_moljepa(..., accelerate=True). Runtime layouts are applied after loading.")
    finite_types = (FiniteTokenLookup, ProjectedEmbeddingLookup, FiniteFanoutLookup, XorProfileLookup)
    for module in result.model.modules():
        if isinstance(module, finite_types) and (
                type(module) not in finite_types
                or any(_has_hooks(child) or "forward" in vars(child) for child in module.modules())):
            raise ValueError("Finite lookup recipes cannot preserve custom subclasses, hooks or forward replacements")
        if isinstance(module, XorProfileLookup):
            module._validate_payload()
    for module in result.model.modules():
        if isinstance(module, PackedFrozenEmbedding):
            if (type(module) is not PackedFrozenEmbedding
                    or any(_has_hooks(child) or "forward" in vars(child) for child in module.modules())):
                raise ValueError("Packed embedding recipes cannot preserve custom subclasses, hooks or forward replacements")
            module.validate_payload()
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    specs, parents, attention_specs, normalization_specs = [], [], [], []
    packed_embedding_specs = []
    embedding_specs = {spec["path"]:spec for spec in getattr(result.model,"_compressme_constant_embedding_rewrites",[])}
    lookup_specs = {spec["path"]:spec for spec in getattr(result.model,"_compressme_finite_lookup_rewrites",[])}
    for path, module in result.model.named_modules():
        if isinstance(module, PackedFrozenEmbedding):
            packed_embedding_specs.append({"path": path, **module.recipe()})
        if isinstance(module,(FiniteTokenLookup,ProjectedEmbeddingLookup,FiniteFanoutLookup,XorProfileLookup)):
            lookup_specs[path] = {"path":path,**module.recipe()}
        if isinstance(module,ConstantRowEmbedding):
            embedding_specs[path] = {"path":path,**module.recipe()}
        if type(module).__module__ == "compressme.attention" and type(module).__name__ == "BilinearTransformerConv":
            attention_specs.append({"path":path, "aggregate_before_value":module.aggregate_before_value,
                                    "skip_single_neighbors":module.skip_single_neighbors})
        if isinstance(module,NormSandwich):
            normalization_specs.append({"path":path,"in_features":module.in_features,
                "normalized_features":module.normalized_features,"out_features":module.out_features,
                "eps":module.eps,"dtype":str(module.numerator.dtype).removeprefix("torch.")})
        if any(path.startswith(p + ".") or not p for p in parents):
            continue
        if isinstance(module, (NormLinear, LowRankLinear)):
            projection = module.projection if isinstance(module, NormLinear) else module
            current_bound = projection.bound_is_current()
            if isinstance(module, NormLinear):
                current_bound = current_bound and module.eps == module._bound_eps
            specs.append({"path": path, "kind": type(module).__name__,
                          "in_features": module.in_features, "out_features": module.out_features,
                          "rank": module.rank, "dtype": str(next(module.parameters()).dtype).removeprefix("torch."),
                          "bias": True if isinstance(module, NormLinear) else module.bias is not None,
                          "eps": getattr(module, "eps", None),
                          "bound": module.bound.to_dict() if current_bound else None,
                          "bound_validity": "at_export" if current_bound else "invalidated_or_unavailable"})
            parents.append(path)
    # Store identical aliases once; the original constructor restores ties.
    tensors, tensor_aliases = _storage_aliases(result.model.state_dict())
    target = root / ("model.cmppack" if packing else "model.safetensors")
    if packing:
        from .packing import pack_bytes
        target.write_bytes(pack_bytes(save_tensors(tensors)))
    else:
        save_file(tensors, str(target))
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    lookup_validation = [
        {"path": path,
         "state": "current_at_export" if module.validation_is_current() else "historical_or_invalidated",
         "scope": "Compilation-time local comparison; not a new whole-model validation or future-device guarantee"}
        for path, module in result.model.named_modules()
        if isinstance(module, (FiniteTokenLookup, ProjectedEmbeddingLookup, FiniteFanoutLookup, XorProfileLookup))]
    export_report = copy.deepcopy(result.report)
    if lookup_validation:
        export_report["finite_lookup_validation_at_export"] = lookup_validation
        if (export_report.get("method") in {"exact_finite_row_lookup", "exact_shared_projected_token_lookup", "finite_token_fanout", "lossless_xor_lookup_tables"}
                and any(item["state"] != "current_at_export" for item in lookup_validation)):
            export_report["historical_status"] = export_report.get("status")
            export_report["status"] = "historical_or_invalidated_finite_lookup_validation"
            if "validation" in export_report:
                export_report["historical_validation"] = export_report.pop("validation")
            export_report["validation"] = {"accepted": False, "scope": "No current comparison for the exported state; previous metrics are historical"}
    adapters = []
    if any(isinstance(m, AffineInputGraphEncoder) for m in result.model.modules()):
        adapters.append("moljepa_affine_input_fusion")
    if isinstance(result.model, SmilesOnlyMolJEPA):
        adapters.append("moljepa_smiles_only")
    manifest = {"format_version": 1, "weights_sha256": digest, "rewrites": specs,
                "weights_file":target.name,"weights_bytes":target.stat().st_size,
                "tensor_aliases":tensor_aliases,
                "module_aliases":_module_aliases(result.model),
                "parameter_aliases":parameter_aliases(result.model),
                "parameter_storage_aliases":parameter_storage_aliases(result.model),
                "packing":"lossless_byte_shuffle_zstd" if packing else None,
                "adapters": adapters, "training": result.model.training, "report": export_report,
                "affine_fx": getattr(result.model, "_compressme_affine_fx_spec", None),
                "attention_rewrites": attention_specs,
                "normalization_rewrites":normalization_specs,
                "constant_embedding_rewrites":list(embedding_specs.values()),
                "finite_lookup_rewrites":list(lookup_specs.values()),
                "packed_embedding_rewrites":packed_embedding_specs,
                "source_tensor_dtypes":getattr(result.model,"_compressme_source_tensor_dtypes",
                    {n:str(t.dtype).removeprefix("torch.") for n,t in result.model.state_dict().items()}),
                "requires_grad": {n: p.requires_grad for n,p in result.model.named_parameters()},
                "module_training": {n: m.training for n,m in result.model.named_modules()}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    obsolete = root / ("model.safetensors" if packing else "model.cmppack")
    if obsolete.is_file():
        obsolete.unlink()
    return root


def load(model_or_factory, directory, *, max_unpack_bytes=1 << 30) -> CompressionResult:
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported compressme artifact version")
    filename = manifest.get("weights_file","model.safetensors")
    if filename not in {"model.safetensors","model.cmppack"}:
        raise ValueError("Unsupported artifact weight filename")
    target = root / filename
    if hashlib.sha256(target.read_bytes()).hexdigest() != manifest["weights_sha256"]:
        raise ValueError("Compressed weight file hash mismatch")
    model = copy.deepcopy(model_or_factory) if isinstance(model_or_factory, torch.nn.Module) else model_or_factory()
    # Recipe replay may bake training-dependent Python branches into FX graphs.
    # Restore those modes BEFORE tracing, not only after loading tensors.
    model.train(manifest.get("training",False))
    for name,module in model.named_modules():
        if name in manifest.get("module_training",{}):
            module.training = manifest["module_training"][name]
    # Restore source dtypes before composing matrices; strict state loading
    # otherwise silently casts a float64 artifact into a float32 factory.
    dtypes = manifest.get("source_tensor_dtypes",{})
    for name,tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if name in dtypes:
            dtype = getattr(torch,dtypes[name],None)
            if not isinstance(dtype,torch.dtype):
                raise ValueError("Invalid architecture dtype")
            if tensor.dtype != dtype:
                tensor.data = tensor.data.to(dtype=dtype)
    for spec in manifest.get("constant_embedding_rewrites",[]):
        previous = model.get_submodule(spec["path"]) if spec["path"] else model
        if not isinstance(previous,torch.nn.Embedding) or (previous.num_embeddings,previous.embedding_dim) != (spec["num_embeddings"],spec["embedding_dim"]):
            raise ValueError("Constant embedding recipe differs from the original architecture")
        replacement = ConstantRowEmbedding.from_recipe(spec,device=previous.weight.device)
        model = _set(model,spec["path"],replacement)
    for spec in manifest.get("finite_lookup_rewrites",[]):
        previous = model.get_submodule(spec["path"]) if spec["path"] else model
        parameter = next(previous.parameters(),None)
        if parameter is None:
            parameter = next(previous.buffers(),None)
        cls = {"FiniteTokenLookup":FiniteTokenLookup,"ProjectedEmbeddingLookup":ProjectedEmbeddingLookup,
               "FiniteFanoutLookup":FiniteFanoutLookup,"XorProfileLookup":XorProfileLookup}.get(spec.get("kind"))
        if cls is None:
            raise ValueError("Unknown finite lookup recipe")
        replacement = cls.from_recipe(spec,device=parameter.device if parameter is not None else "cpu")
        model = _set(model,spec["path"],replacement)
    packed_specs = manifest.get("packed_embedding_rewrites", [])
    if not isinstance(packed_specs, list):
        raise ValueError("Invalid packed embedding recipe list")
    packed_paths = set()
    for spec in packed_specs:
        if (not isinstance(spec, dict) or not isinstance(spec.get("path"), str)
                or spec["path"] in packed_paths):
            raise ValueError("Invalid or duplicate packed embedding recipe path")
        path = spec["path"]
        packed_paths.add(path)
        previous = model.get_submodule(path) if path else model
        if isinstance(previous, torch.nn.Embedding) and (
                previous.num_embeddings != spec.get("num_embeddings")
                or previous.embedding_dim != spec.get("embedding_dim")):
            raise ValueError("Packed embedding recipe differs from the original architecture")
        # Payloads and offsets always stay on CPU. The constructor, followed by
        # the caller's final model.to(device), controls the output placement;
        # the export machine's accelerator is never required to reload bytes.
        if isinstance(previous, PackedFrozenEmbedding):
            output_device = previous._anchor.device
        else:
            reference = next(previous.parameters(), None)
            if reference is None:
                reference = next(previous.buffers(), None)
            output_device = reference.device if reference is not None else torch.device("cpu")
        recipe = {key: value for key, value in spec.items() if key != "path"}
        replacement = PackedFrozenEmbedding.from_recipe(recipe).to(output_device)
        model = _set(model, path, replacement)
    if manifest.get("affine_fx") is not None:
        model = compile_affine(model, **manifest["affine_fx"]).model
    for adapter in manifest.get("adapters", []):
        if adapter == "moljepa_affine_input_fusion":
            model, _ = fuse_moljepa_input_projections(model, inplace=True)
        elif adapter == "moljepa_smiles_only":
            model, _ = specialize_moljepa_smiles(model, inplace=True)
        else:
            raise ValueError("Unknown architecture adapter")
    if manifest.get("attention_rewrites"):
        from .attention import BilinearTransformerConv
        for spec in manifest["attention_rewrites"]:
            previous = model.get_submodule(spec["path"]) if spec["path"] else model
            replacement = BilinearTransformerConv(previous,
                aggregate_before_value=spec["aggregate_before_value"],
                skip_single_neighbors=spec.get("skip_single_neighbors",False))
            model = _set(model,spec["path"],replacement)
    for spec in manifest.get("normalization_rewrites",[]):
        previous = model.get_submodule(spec["path"]) if spec["path"] else model
        parameter = next(previous.parameters(),None)
        module = NormSandwich(spec["in_features"],spec["normalized_features"],spec["out_features"],
            spec["eps"],dtype=getattr(torch,spec["dtype"]),
            device=parameter.device if parameter is not None else "cpu")
        model = _set(model,spec["path"],module)
    for spec in manifest["rewrites"]:
        previous = model.get_submodule(spec["path"]) if spec["path"] else model
        previous_parameter = next(previous.parameters(), None)
        device = previous_parameter.device if previous_parameter is not None else torch.device("cpu")
        dtype = getattr(torch, spec["dtype"], None)
        if dtype not in (torch.float64, torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("Unsupported stored dtype")
        args = (spec["in_features"], spec["out_features"], spec["rank"])
        if spec["kind"] == "NormLinear":
            module = NormLinear(*args, eps=spec["eps"], dtype=dtype, device=device)
        elif spec["kind"] == "LowRankLinear":
            module = LowRankLinear(*args, bias=spec["bias"], dtype=dtype, device=device)
        else:
            raise ValueError("Unsupported rewrite kind")
        if spec.get("bound"):
            module.bound = LayerBound(**spec["bound"])
            if isinstance(module, NormLinear):
                module.projection.bound = module.bound
        model = _set(model, spec["path"], module)
    model = _restore_module_aliases(model, manifest.get("module_aliases", {}))
    if filename == "model.cmppack":
        from .packing import unpack_bytes
        stored = load_tensors(unpack_bytes(target.read_bytes(),max_output_bytes=max_unpack_bytes))
    else:
        stored = load_file(str(target))
    stored = _expand_storage_aliases(stored, manifest.get("tensor_aliases", {}))
    model = restore_parameter_aliases(model, manifest.get("parameter_aliases", {}), stored)
    model = restore_parameter_aliases(model, manifest.get("parameter_storage_aliases", {}), stored, storage_only=True)
    expected = model.state_dict()
    for name,tensor in stored.items():
        if name in expected and expected[name].dtype != tensor.dtype:
            raise ValueError(f"Architecture dtype mismatch at {name}; supply the original constructor dtype")
    model.load_state_dict(stored, strict=True)
    for module in model.modules():
        if isinstance(module, XorProfileLookup):
            module._validate_payload()
    model.train(manifest.get("training", False))
    for name, parameter in model.named_parameters():
        if name in manifest.get("requires_grad", {}):
            parameter.requires_grad_(manifest["requires_grad"][name])
    for name, module in model.named_modules():
        if name in manifest.get("module_training", {}):
            module.training = manifest["module_training"][name]
    for module in model.modules():
        if isinstance(module, PackedFrozenEmbedding):
            module.validate_payload()
        if isinstance(module, LowRankLinear):
            module._seal_bound()
    return CompressionResult(model, manifest["report"])
