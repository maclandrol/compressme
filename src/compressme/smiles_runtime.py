"""Static-input inference execution for the complete SMILES embedding outputs.

This removes repeated routing and readout launches. It does not cache molecule
outputs, alter precision, or skip any prediction head. Training uses the audited
original forward through callable views of the batched readout parameters.
"""
from __future__ import annotations
import copy
import sys
import torch
from torch import nn
from .moljepa import SmilesOnlyMolJEPA, AffineInputGraphEncoder
from .readouts import BatchedLinearHeads
from .sparse_graph import SparseMolJEPAFeaturizer, smiles_to_batch


class FastSmilesMolJEPA(SmilesOnlyMolJEPA):
    _compressme_runtime_only = True
    def __init__(self,backbone, *, metal=True, coalesced_transfer=False):
        if isinstance(backbone,SmilesOnlyMolJEPA):
            backbone = backbone.backbone
        super().__init__(backbone)
        core = backbone.model
        if list(core.encoders) != ["graph"] or core.cls_predictor:
            raise ValueError("Fast SMILES execution requires graph-only encoders and ordinary CLS prediction")
        if not isinstance(core.encoders["graph"],AffineInputGraphEncoder):
            raise TypeError("Fast SMILES execution requires the audited affine graph adapter")
        if any(not layer.norm_first for layer in core.transformer_head.transformer.layers):
            raise ValueError("Expected the audited pre-normalized transformer")
        self.graph_token = 1 + next(i for i,m in enumerate(core.modalities_spec) if m["name"] == "graph")
        head = core.transformer_head
        if not isinstance(head.modality_pred,BatchedLinearHeads):
            head.modality_pred = BatchedLinearHeads(head.modality_pred)
        source = sys.modules[type(backbone).__module__]
        self._output_type = source.MolJEPAOutput
        self._data_class = source.MultimodalData
        self._featurizer = SparseMolJEPAFeaturizer(source.GraphFeaturizer)
        self.metal_enabled = bool(metal and hasattr(torch.mps, "compile_shader"))
        self.coalesced_transfer = bool(coalesced_transfer)
        if self.metal_enabled:
            from .graph_metal import PackedMetalGraphAttention
            graph = core.encoders["graph"]
            parameters = list(graph.named_parameters(remove_duplicate=False))
            if len({id(p) for _,p in parameters}) != len(parameters):
                raise ValueError("Shared graph parameters require an explicit runtime adapter")
            graph.convs = nn.ModuleList([PackedMetalGraphAttention(c) for c in graph.convs])
        self.runtime_profile = "smiles_sparse_metal_readouts_v1"

    def tensor_forward(self,graph_batch,batch_size,return_attn=False, *, layout=None):
        core = self.backbone.model
        graph = core.encoders["graph"]
        head = core.transformer_head
        if layout is not None:
            from .graph_metal import graph_forward
            encoded = graph_forward(graph, graph_batch["graph_x"],graph_batch["graph_edge_index"],
                        graph_batch["graph_edge_attr"],graph_batch["graph_x_batch"],batch_size,layout)
        else:
            encoded = graph(graph_batch["graph_x"],graph_batch["graph_edge_index"],
                        graph_batch["graph_edge_attr"],graph_batch["graph_x_batch"],
                        batch_size=batch_size)
        # The original uses all modality positions, replaces absent modalities
        # by mask_token + position, and keeps only CLS and graph as keys.
        pos = head.modality_pos.weight.unsqueeze(0)
        tokens = (head.mask_token + pos).expand(batch_size,-1,-1).clone()
        tokens[:,0] = core.cls_token[:,0] + pos[:,0]
        tokens[:,self.graph_token] = encoded + pos[:,self.graph_token]
        missing = torch.ones((batch_size,head.n_modalities),dtype=torch.bool,device=tokens.device)
        missing[:,0] = False
        missing[:,self.graph_token] = False
        attentions = None
        if return_attn:
            # Same path used by upstream _capture_attention, without monkey
            # patching methods on shared layer instances during the request.
            attentions = []
            output = tokens
            for layer in head.transformer.layers:
                normalized = layer.norm1(output)
                update,alpha = layer.self_attn(normalized,normalized,normalized,
                    key_padding_mask=missing,need_weights=True,average_attn_weights=False)
                output = output + layer.dropout1(update)
                output = output + layer._ff_block(layer.norm2(output))
                attentions.append(alpha.detach())
            if head.transformer.norm is not None:
                output = head.transformer.norm(output)
        else:
            output = head.transformer(tokens,src_key_padding_mask=missing)
        predictions = head.modality_pred(output)
        return self._output_type(predictions=predictions[:,1:],cls=predictions[:,0],
                                 embeddings=output,attentions=attentions)

    def forward(self,smiles_list,return_attn=False,embeddings_data=None):
        if embeddings_data is not None:
            raise ValueError("This artifact requires embeddings_data=None")
        if self.training:
            return self.backbone(smiles_list,return_attn=return_attn)
        batch = smiles_to_batch(smiles_list, featurizer=self._featurizer,
                                data_class=self._data_class)
        layout = None
        if self.metal_enabled and self.device.type == "mps" and self.coalesced_transfer:
            from .graph_metal import NeighborLayout, BoundNeighborLayout, _version
            from .transfer import transfer_tensors
            raw = NeighborLayout.from_coo(batch.graph_edge_index,batch.graph_x.shape[0])
            fields = ("graph_x","graph_edge_index","graph_edge_attr","graph_x_batch","graph_x_ptr")
            moved = transfer_tensors([batch[k] for k in fields] + [raw.rowptr,raw.edge_ids,raw.sources],self.device)
            for key,value in zip(fields,moved[:len(fields)]):
                batch[key] = value
            neighbors = NeighborLayout(*moved[len(fields):],raw.node_count,raw.edge_count,raw.max_degree)
            layout = BoundNeighborLayout(batch.graph_edge_index,neighbors,_version(batch.graph_edge_index))
        elif self.metal_enabled and self.device.type == "mps":
            from .graph_metal import BoundNeighborLayout
            cpu_edges = batch.graph_edge_index
            node_count = batch.graph_x.shape[0]
            batch = batch.to(self.device)
            layout = BoundNeighborLayout.from_cpu(cpu_edges,node_count,self.device,
                                                   transferred_coo=batch.graph_edge_index)
        else:
            batch = batch.to(self.device)
        with torch.no_grad():
            return self.tensor_forward(batch,len(smiles_list),return_attn,layout=layout)


def accelerate_smiles(model, *, inplace=False, metal=True, coalesced_transfer=False):
    """Enable the exact static SMILES route; original weights are retained."""
    if not isinstance(model,SmilesOnlyMolJEPA):
        raise TypeError("First create or load a SMILES-only artifact")
    if isinstance(model,FastSmilesMolJEPA):
        return model if inplace else copy.deepcopy(model)
    candidate = model if inplace else copy.deepcopy(model)
    return FastSmilesMolJEPA(candidate,metal=metal,coalesced_transfer=coalesced_transfer)
