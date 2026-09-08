"""Exact bilinear reparameterization of homogeneous graph attention.

Audited against torch_geometric TransformerConv. The replacement accepts dense
homogeneous node features and a COO edge_index. It preserves training dropout,
isolated nodes, value biases, and optional attention outputs. It does not accept
bipartite or sparse-tensor inputs. Real-arithmetic equality is not bit equality.
"""
import copy
import math
from collections import Counter
import fnmatch
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import softmax, scatter
from .runtime import autocast_enabled


class BilinearTransformerConv(nn.Module):
    def __init__(self, source, aggregate_before_value=False, skip_single_neighbors=False):
        super().__init__()
        if type(source) is not TransformerConv:
            raise TypeError("Only the unmodified PyG TransformerConv forward is audited")
        if any(_has_hooks(m) for m in source.modules()):
            raise ValueError("Custom hooks require a separate attention adapter")
        if source.flow!="source_to_target" or source.aggr not in {"add","sum"} or source.explain:
            raise ValueError("Only ordinary source-to-target sum aggregation is audited")
        if source.concat or source.beta or not source.root_weight:
            raise ValueError("Audited path requires concat=False,beta=False,root_weight=True")
        self.in_channels = source.lin_query.weight.shape[1]
        if source.lin_key.weight.shape[1] != self.in_channels:
            raise ValueError("Only equal source/target feature widths are supported")
        self.out_channels=source.out_channels
        self.heads=source.heads
        self.edge_dim=source.edge_dim
        self.dropout=source.dropout
        self.concat=False
        self.beta=False
        self.root_weight=True
        self.lin_value=source.lin_value
        self.lin_edge=source.lin_edge
        self.lin_skip=source.lin_skip
        self.lin_beta=None
        self.aggregate_before_value=aggregate_before_value
        self.skip_single_neighbors=skip_single_neighbors
        H,C,I=self.heads,self.out_channels,self.in_channels
        device,dtype=source.lin_query.weight.device,source.lin_query.weight.dtype
        q=source.lin_query.weight.detach().cpu().double().reshape(H,C,I)
        k=source.lin_key.weight.detach().cpu().double().reshape(H,C,I)
        qb=torch.zeros(H,C,dtype=torch.float64)
        if source.lin_query.bias is not None:
            qb=source.lin_query.bias.detach().cpu().double().reshape(H,C)
        has_query_bias=source.lin_query.bias is not None
        with torch.random.fork_rng(devices=[]):
            self.score_node=nn.Linear(I,H*I,bias=has_query_bias,dtype=dtype).to(device)
        with torch.no_grad():
            self.score_node.weight.copy_(torch.einsum("hci,hcj->hij",k,q).reshape(H*I,I).to(dtype).to(device))
            if has_query_bias:
                self.score_node.bias.copy_(torch.einsum("hci,hc->hi",k,qb).reshape(H*I).to(dtype).to(device))
        self.score_node.requires_grad_(any(p.requires_grad for layer in (source.lin_query,source.lin_key)
                                          for p in layer.parameters()))
        if self.edge_dim is not None:
            e=source.lin_edge.weight.detach().cpu().double().reshape(H,C,self.edge_dim)
            with torch.random.fork_rng(devices=[]):
                self.score_edge=nn.Linear(I,H*self.edge_dim,bias=has_query_bias,dtype=dtype).to(device)
            with torch.no_grad():
                self.score_edge.weight.copy_(torch.einsum("hce,hci->hei",e,q).reshape(H*self.edge_dim,I).to(dtype).to(device))
                if has_query_bias:
                    self.score_edge.bias.copy_(torch.einsum("hce,hc->he",e,qb).reshape(H*self.edge_dim).to(dtype).to(device))
            self.score_edge.requires_grad_(any(p.requires_grad for layer in (source.lin_query,source.lin_edge)
                                              for p in layer.parameters()))
        else:
            self.score_edge=None
        self.train(source.training)

    def forward(self,x,edge_index,edge_attr=None,return_attention_weights=None):
        if not isinstance(x,torch.Tensor) or not isinstance(edge_index,torch.Tensor) or edge_index.layout!=torch.strided:
            raise ValueError("Bilinear graph attention requires tensor nodes and COO edge_index")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or x.ndim != 2:
            raise ValueError("Expected edge_index shape(2,E)")
        if autocast_enabled(x.device.type):
            raise ValueError("Bilinear attention does not support autocast; use explicit model/input dtypes")
        if self.lin_edge is None and edge_attr is not None:
            raise ValueError("Raw edge attributes without an edge projection are unsupported")
        src,dst=edge_index
        H,C,I=self.heads,self.out_channels,self.in_channels
        N=x.shape[0]
        score_x=x
        score_src,score_dst=src,dst
        score_edge_attr=edge_attr
        if self.skip_single_neighbors:
            degree=scatter(torch.ones_like(dst),dst,dim=0,dim_size=N,reduce="sum")
            multiple=degree>1
            selected_nodes=multiple.nonzero().flatten()
            selected_edges=multiple[dst]
            index=torch.zeros(N,dtype=torch.long,device=x.device)
            index[selected_nodes]=torch.arange(selected_nodes.numel(),device=x.device)
            score_x=x[selected_nodes]
            score_src=src[selected_edges]
            score_dst=index[dst[selected_edges]]
            if edge_attr is not None:
                score_edge_attr=edge_attr[selected_edges]
        query_node=self.score_node(score_x).reshape(-1,H,I)
        logits=(query_node[score_dst]*x[score_src,None,:]).sum(-1)
        if self.score_edge is not None:
            if edge_attr is None:
                raise ValueError("Edge attributes are required by this checkpoint")
            query_edge=self.score_edge(score_x).reshape(-1,H,self.edge_dim)
            logits=logits+(query_edge[score_dst]*score_edge_attr[:,None,:]).sum(-1)
        subset_alpha=softmax(logits/math.sqrt(C),score_dst,num_nodes=score_x.shape[0])
        if self.skip_single_neighbors:
            alpha=torch.ones(src.shape[0],H,device=x.device,dtype=subset_alpha.dtype)
            alpha[selected_edges]=subset_alpha
        else:
            alpha=subset_alpha
        weights=F.dropout(alpha,p=self.dropout,training=self.training)
        if self.aggregate_before_value:
            node_sum=scatter(x[src,None,:]*weights[:,:,None],dst,dim=0,dim_size=N,reduce="sum")
            weight=self.lin_value.weight.reshape(H,C,I)
            out=torch.einsum("nhi,hci->nhc",node_sum,weight)
            if self.lin_value.bias is not None:
                alpha_sum=scatter(weights,dst,dim=0,dim_size=N,reduce="sum")
                out=out+alpha_sum[:,:,None]*self.lin_value.bias.reshape(H,C)[None,:,:]
            if self.lin_edge is not None:
                edge_sum=scatter(edge_attr[:,None,:]*weights[:,:,None],dst,dim=0,dim_size=N,reduce="sum")
                edge_weight=self.lin_edge.weight.reshape(H,C,self.edge_dim)
                out=out+torch.einsum("nhe,hce->nhc",edge_sum,edge_weight)
            out=out.mean(1)
        else:
            value=self.lin_value(x).reshape(N,H,C)
            message=value[src]
            if self.lin_edge is not None:
                message=message+self.lin_edge(edge_attr).reshape(-1,H,C)
            # Head mean and destination sum commute in real arithmetic.
            message=(message*weights[:,:,None]).mean(1)
            out=scatter(message,dst,dim=0,dim_size=N,reduce="sum")
        out=out+self.lin_skip(x)
        if isinstance(return_attention_weights,bool):
            return out,(edge_index,alpha)
        return out


def _has_hooks(module):
    return any(bool(value) for name, value in vars(module).items()
               if name.endswith("_hooks") and isinstance(value, dict))


def contract_attention(model, *, input_contract, aggregate_before_value=False,
                       skip_single_neighbors=False, include=None, exclude=None, inplace=False):
    """Contract profitable PyG TransformerConv scores throughout any model.

    The caller must establish `input_contract='homogeneous_coo'`: each selected
    layer receives one dense node tensor and a COO edge_index. Bipartite and
    SparseTensor calls are not supported. The audited concat=False, beta=False,
    root_weight=True route is retained; unsupported layers are reported/skipped.
    No weights are truncated. The optional dependency PyG is imported lazily.
    """
    from .compiler import CompressionResult, parameter_count, state_bytes, _set
    if input_contract != "homogeneous_coo":
        raise ValueError("Declare the supported homogeneous_coo input contract")
    candidate = model if inplace else copy.deepcopy(model)
    candidate._compressme_source_tensor_dtypes = getattr(model,"_compressme_source_tensor_dtypes",
        {n:str(t.dtype).removeprefix("torch.") for n,t in model.state_dict().items()})
    before, before_bytes = parameter_count(model), state_bytes(model)
    aliases = Counter(id(p) for _,p in model.named_parameters(remove_duplicate=False))
    module_aliases = Counter(id(m) for _,m in model.named_modules(remove_duplicate=False))
    records = []
    for path, module in list(candidate.named_modules()):
        if type(module) is not TransformerConv:
            continue
        if include is not None and not any(fnmatch.fnmatchcase(path,p) for p in include):
            continue
        if exclude and any(fnmatch.fnmatchcase(path,p) for p in exclude):
            continue
        original = model.get_submodule(path) if path else model
        if module_aliases[id(original)] > 1 or any(aliases[id(p)] > 1 for p in original.parameters()):
            records.append({"path":path,"status":"retained_shared_parameters"})
            continue
        try:
            replacement = BilinearTransformerConv(module, aggregate_before_value, skip_single_neighbors)
        except (ValueError, TypeError) as exc:
            records.append({"path":path,"status":"retained_unsupported","reason":str(exc)})
            continue
        if state_bytes(replacement) >= state_bytes(module):
            records.append({"path":path,"status":"retained_no_byte_saving"})
            continue
        records.append({"path":path,"status":"contracted",
                        "parameters_before":parameter_count(module),
                        "parameters_after":parameter_count(replacement)})
        candidate = _set(candidate,path,replacement)
    return CompressionResult(candidate, {
        "method":"exact_bilinear_graph_attention", "input_contract":input_contract,
        "aggregate_before_value":aggregate_before_value,
        "parameters_before":before, "parameters_after":parameter_count(candidate),
        "tensor_bytes_before":before_bytes, "tensor_bytes_after":state_bytes(candidate),
        "layers":records,
        "guarantee":"same mathematical forward on declared inputs in real arithmetic; floating-point order changes"})
