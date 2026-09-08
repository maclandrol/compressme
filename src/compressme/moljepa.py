"""Training-free Mol-JEPA graph rewrites, audited against HF revision
4c912b450175f31b5ba913a5dc921c03b27b985a (Flogrammer/Mol-JEPA).

These are algebraic inference rewrites, not distributional quality guarantees.
No weights are rounded to a lower-precision datatype. Floating-point results
can differ after affine composition changes the order of operations.
"""
from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .affine import compose_affine


def _count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class SmilesOnlyMolJEPA(nn.Module):
    """Preserve SMILES outputs while explicitly rejecting omitted modalities."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self._compressme_source_tensor_dtypes = getattr(backbone,"_compressme_source_tensor_dtypes",
            {n:str(t.dtype).removeprefix("torch.") for n,t in backbone.state_dict().items()})
        self.train(backbone.training)

    @property
    def config(self):
        return self.backbone.config

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, smiles_list, return_attn=False, embeddings_data=None):
        if embeddings_data is not None:
            raise ValueError(
                "This artifact is specialized for embeddings_data=None. "
                "Use the unspecialized model for extra molecular modalities."
            )
        return self.backbone(smiles_list, return_attn=return_attn, embeddings_data=None)


def _backbone(model):
    return model.backbone if isinstance(model, SmilesOnlyMolJEPA) else model


class AffineInputGraphEncoder(nn.Module):
    """Compose an 82->512 atom projection into the first graph-attention forks.

    The original projection is retained for its residual path. The attention
    query/key/value/skip projections receive the original atom features, while
    edge features, softmax, head averaging, normalization and pooling are kept.
    """

    def __init__(self, source: nn.Module):
        super().__init__()
        if not isinstance(source.input_proj, nn.Linear):
            raise TypeError("The graph input projection must be torch.nn.Linear.")
        if not source.convs or source.convs[0].__class__.__name__ != "TransformerConv":
            raise TypeError("Only the audited Mol-JEPA TransformerConv route is supported.")
        if source.pool.__name__ not in {"global_max_pool", "global_mean_pool"}:
            raise TypeError("Unrecognized graph pooling operation.")
        self.input_proj = source.input_proj
        self.convs = source.convs
        self.norms = source.norms
        self.proj = source.proj
        self.pool = source.pool
        self.act = source.act
        self.dropout = source.dropout
        self.gradient_checkpointing = source.gradient_checkpointing
        first = self.convs[0]
        if not first.root_weight:
            raise ValueError("The audited model requires the first graph root projection.")
        for name in ("lin_key", "lin_query", "lin_value", "lin_skip"):
            previous = getattr(first, name)
            if previous.weight.ndim != 2 or previous.weight.shape[1] != self.input_proj.out_features:
                raise ValueError(f"Unexpected shape for first graph {name}.")
        for name in ("lin_key", "lin_query", "lin_value", "lin_skip"):
            previous = getattr(first, name)
            folded = compose_affine(self.input_proj, previous)
            setattr(first, name, folded)
        first.in_channels = self.input_proj.in_features
        self.train(source.training)

    def _first_block(self, atom_features, edge_index, edge_attr):
        residual = self.input_proj(atom_features)
        message = self.act(self.convs[0](atom_features, edge_index, edge_attr))
        message = F.dropout(message, p=self.dropout, training=self.training)
        return self.norms[0](message + residual)

    def _later_block(self, conv, norm, h, edge_index, edge_attr):
        message = self.act(conv(h, edge_index, edge_attr))
        message = F.dropout(message, p=self.dropout, training=self.training)
        return norm(message + h)

    def forward(self, x, edge_index, edge_attr, batch, batch_size=None):
        if self.gradient_checkpointing and self.training:
            h = checkpoint(self._first_block, x, edge_index, edge_attr, use_reentrant=False)
        else:
            h = self._first_block(x, edge_index, edge_attr)
        for conv, norm in zip(self.convs[1:], self.norms[1:]):
            if self.gradient_checkpointing and self.training:
                h = checkpoint(self._later_block, conv, norm, h, edge_index, edge_attr,
                               use_reentrant=False)
            else:
                h = self._later_block(conv, norm, h, edge_index, edge_attr)
        return self.proj(self.pool(h, batch, size=batch_size))


def specialize_moljepa_smiles(model: nn.Module, inplace: bool = False) -> tuple[nn.Module, dict[str, Any]]:
    """Remove unreachable modality encoders under embeddings_data=None.

    Returns (model, report). This explicitly narrows the optional input contract;
    it does not silently produce an answer when removed modalities are supplied.
    """
    result = model if inplace else copy.deepcopy(model)
    if isinstance(result, SmilesOnlyMolJEPA):
        raise ValueError("Model already has the SMILES-only contract.")
    source = _backbone(result)
    before = _count(result)
    encoders = source.model.encoders
    if "graph" not in encoders:
        raise ValueError("The checkpoint does not have a graph encoder.")
    removed = [name for name in encoders if name != "graph"]
    for name in removed:
        del encoders[name]
    result = SmilesOnlyMolJEPA(result)
    return result, {"operation": "specialize_moljepa_smiles", "removed_encoders": removed,
                    "parameters_before": before, "parameters_after": _count(result),
                    "contract": "embeddings_data=None", "guarantee": "unreachable-module removal"}


def fuse_moljepa_input_projections(model: nn.Module, inplace: bool = False) -> tuple[nn.Module, dict[str, Any]]:
    """Remove exact affine redundancy while preserving all external modalities.

    Returns (model, report). This is inference-equivalent in real arithmetic;
    independent fine-tuning of fused matrices changes the parameterization.
    """
    result = model if inplace else copy.deepcopy(model)
    result._compressme_source_tensor_dtypes = getattr(model,"_compressme_source_tensor_dtypes",
        {n:str(t.dtype).removeprefix("torch.") for n,t in model.state_dict().items()})
    source = _backbone(result)
    graph = source.model.encoders["graph"]
    if isinstance(graph, AffineInputGraphEncoder):
        raise ValueError("Graph input projections are already composed.")
    before = _count(result)
    source.model.encoders["graph"] = AffineInputGraphEncoder(graph)
    return result, {"operation": "fuse_moljepa_input_projections",
                    "parameters_before": before, "parameters_after": _count(result),
                    "guarantee": "exact affine composition in real arithmetic",
                    "preserves_optional_modalities": True,
                    "floating_point_bitwise_guarantee": False}


def optimize_moljepa(model: nn.Module, *, smiles_only=False, validation=None,
                    relative_tolerance=1e-5, absolute_tolerance=1e-5,
                    aggregate_before_value=False):
    """Apply exact algebraic passes to the audited Mol-JEPA architecture.

    All original forward inputs/outputs are preserved by default. The explicit
    smiles_only option removes encoders unreachable with embeddings_data=None.
    Optional validation compares every returned tensor with the original and
    rolls back the entire proposal on a tolerance failure. No fitting is used.
    """
    from .compiler import CompressionResult, parameter_count, state_bytes
    from .attention import contract_attention
    from .validation import validate
    candidate, affine_report = fuse_moljepa_input_projections(model)
    candidate._compressme_source_tensor_dtypes = {
        n:str(t.dtype).removeprefix("torch.") for n,t in model.state_dict().items()}
    contraction = contract_attention(candidate, input_contract="homogeneous_coo",
                                     aggregate_before_value=aggregate_before_value,
                                     include=["model.encoders.graph.convs.*"], inplace=True)
    candidate = contraction.model
    passes = [affine_report, contraction.report]
    if smiles_only:
        candidate, specialization = specialize_moljepa_smiles(candidate, inplace=True)
        passes.append(specialization)
    report = {"method":"moljepa_exact_rewrites", "status":"compiled",
              "parameters_before":parameter_count(model), "parameters_after":parameter_count(candidate),
              "tensor_bytes_before":state_bytes(model), "tensor_bytes_after":state_bytes(candidate),
              "input_contract":"embeddings_data=None" if smiles_only else "original complete forward API",
              "passes":passes, "validation":None,
              "guarantee":"exact in real arithmetic; no bitwise or downstream biological-accuracy guarantee"}
    if validation is not None:
        report["validation"] = validate(model,candidate,validation,
            relative_tolerance=relative_tolerance, absolute_tolerance=absolute_tolerance)
        if not report["validation"]["accepted"]:
            report["status"] = "rejected_and_rolled_back"
            report["proposal_parameters_after"] = report["parameters_after"]
            candidate = copy.deepcopy(model)
            report["parameters_after"] = parameter_count(model)
            report["tensor_bytes_after"] = state_bytes(model)
        else:
            report["status"] = "accepted_on_validation_examples"
    return CompressionResult(candidate,report)
