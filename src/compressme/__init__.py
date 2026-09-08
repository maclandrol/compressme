"""Training-free model transformations and lightweight checkpoint utilities.

Public APIs load their implementation on demand. Importing compressme does not
import Torch, chemistry libraries, model architectures or checkpoint weights.
Install compressme[torch] for model transformations; other extras are independent.
"""
from importlib import import_module as _import_module


_EXPORTS = {
    "CompressionResult": "compiler", "collect_moments": "compiler",
    "compress": "compiler", "parameter_count": "compiler", "state_bytes": "compiler",
    "InputMoments": "linear", "LayerBound": "linear", "LowRankLinear": "linear",
    "NormLinear": "linear", "factorize_linear": "linear", "factorize_norm_linear": "linear",
    "load": "serialization", "save": "serialization",
    "Example": "validation", "compare_outputs": "validation", "validate": "validation",
    "fuse_moljepa_input_projections": "moljepa", "specialize_moljepa_smiles": "moljepa",
    "optimize_moljepa": "moljepa", "compile_affine": "affine", "compose_affine": "affine",
    "load_moljepa": "moljepa_io", "NormSandwich": "norm_sandwich",
    "compose_norm_sandwich": "norm_sandwich", "pack_bytes": "packing", "unpack_bytes": "packing",
    "BatchedLinearHeads": "readouts", "get_target": "targets", "list_targets": "targets",
    "inspect_huggingface": "hub", "compress_huggingface": "hub",
    "ConstantRowEmbedding": "constant_embeddings", "deduplicate_embeddings": "constant_embeddings",
    "load_state_st": "state_io", "transfer_tensors": "transfer",
    "compile_finite_lookup": "finite_lookup", "build_projected_lookup": "finite_lookup",
    "RowNormalize": "finite_lookup", "ShapeMatchedConstantRows": "shape_constants",
    "share_frozen_parameters": "sharing", "load_state_se": "state_se_io",
    "load_state_se_encoder": "state_se_io", "compile_finite_blocks": "finite_blocks",
    "FiniteTokenFanout": "finite_fanout", "FiniteFanoutLookup": "finite_fanout",
    "Float32RMSNorm": "finite_fanout", "compile_finite_fanout": "finite_fanout",
    "XorProfileLookup": "frozen_tables", "pack_lookup_tables": "frozen_tables",
    "load_boltz2": "boltz2_io", "pack_file": "packing_files", "unpack_file": "packing_files",
    "FinalEmbedding": "final_embedding",
    "channelwise_token_contraction": "contractions",
    "nesso_inference_optimizations": "nesso_runtime",
}

__all__ = list(_EXPORTS) + ["accelerate_smiles", "contract_attention"]


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        value = getattr(_import_module(f".{module}", __name__), name)
    except ModuleNotFoundError as exc:
        if exc.name in {"torch", "safetensors"}:
            raise ModuleNotFoundError(
                f"{name} requires optional model dependencies. "
                "Install them with: pip install 'compressme[torch]'",
                name=exc.name,
            ) from exc
        raise
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


def accelerate_smiles(*args, **kwargs):
    """Enable sparse preprocessing and hardware execution for a SMILES artifact."""
    from .smiles_runtime import accelerate_smiles as implementation
    return implementation(*args, **kwargs)


def contract_attention(*args, **kwargs):
    """Lazily import the optional graph backend; see compressme.attention."""
    from .attention import contract_attention as implementation
    return implementation(*args, **kwargs)
