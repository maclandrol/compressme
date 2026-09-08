"""Production-oriented isolated refinement of the validated Metal prototype.

Packed tensors have one owner, with no retained copy of the source graph module.
Float32 Metal executes the same bilinear graph formula with different reduction
order; PyTorch handles gradients, unsupported devices and high-degree graphs.
"""
from dataclasses import dataclass
from collections import Counter
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import softmax, scatter
from .attention import BilinearTransformerConv
from .runtime import autocast_enabled
@dataclass
class NeighborLayout:
    rowptr: torch.Tensor
    edge_ids: torch.Tensor
    sources: torch.Tensor
    node_count: int
    edge_count: int
    max_degree: int

    @classmethod
    def from_coo(cls, edge_index, node_count):
        if edge_index.device.type != "cpu" or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("Build neighbour layout from CPU COO before device transfer")
        if edge_index.dtype not in (torch.int32, torch.int64):
            raise ValueError("COO must contain integer indices")
        src, dst = edge_index.long()
        if src.numel() and (edge_index.min() < 0 or edge_index.max() >= node_count):
            raise ValueError("COO endpoint outside declared node count")
        if node_count >= 2**31 or src.numel() >= 2**31:
            raise ValueError("Prototype uses 32-bit graph indices")
        order = torch.argsort(dst, stable=True)
        count = torch.bincount(dst, minlength=node_count)
        ptr = torch.cat((torch.zeros(1, dtype=torch.int64), count.cumsum(0)))
        return cls(ptr.int(), order.int(), src[order].int(), node_count,
                   src.numel(), int(count.max()) if count.numel() else 0)

    def to(self, device):
        return NeighborLayout(self.rowptr.to(device), self.edge_ids.to(device),
                              self.sources.to(device), self.node_count,
                              self.edge_count, self.max_degree)



def shader_source(H, C, I, D, max_degree, packed):
    # Specialization is by public tensor dimensions, never by learned values.
    # Private scores are at most 32 floats per lane on the molecular route.
    M = max(1, max_degree)
    node_stride = H * I + H * D + H * C + C if packed else H * I
    edge_stride = node_stride if packed else H * D
    value_stride = node_stride if packed else H * C
    skip_stride = node_stride if packed else C
    edge_offset = H * I if packed else 0
    value_offset = H * I + H * D if packed else 0
    skip_offset = H * I + H * D + H * C if packed else 0
    return f"""
#include <metal_stdlib>
using namespace metal;
kernel void graph_scores(
 const device float* x [[buffer(0)]],
 const device float* edge [[buffer(1)]],
 const device float* qnode [[buffer(2)]],
 const device float* qedge [[buffer(3)]],
 const device int* rowptr [[buffer(4)]],
 const device int* edgeids [[buffer(5)]],
 const device int* sources [[buffer(6)]],
 device float* alpha [[buffer(7)]],
 uint tid [[thread_position_in_grid]],
 ushort lane [[thread_index_in_simdgroup]]) {{
 const uint nh = tid / 32;
 const uint n = nh / {H};
 const uint h = nh % {H};
 const int lo = rowptr[n], hi = rowptr[n+1];
 float scores[{M}];
 float maximum = -INFINITY;
 for (int p=lo; p<hi; ++p) {{
   const uint s=sources[p], e=edgeids[p];
   float node_sum=0.0f, edge_sum=0.0f;
   for (uint j=lane; j<{I}; j+=32)
     node_sum += qnode[n*{node_stride}+h*{I}+j]*x[s*{I}+j];
   for (uint j=lane; j<{D}; j+=32)
     edge_sum += qedge[n*{edge_stride}+{edge_offset}+h*{D}+j]*edge[e*{D}+j];
   const float score=(simd_sum(node_sum)+simd_sum(edge_sum))*{1/math.sqrt(C):.17g}f;
   if (lane==0) {{ scores[p-lo]=score; maximum=max(maximum,score); }}
 }}
 if (lane==0) {{
   float denominator=0.0f;
   for (int p=lo; p<hi; ++p) {{
     scores[p-lo]=exp(scores[p-lo]-maximum);
     denominator+=scores[p-lo];
   }}
   for (int p=lo; p<hi; ++p)
     alpha[edgeids[p]*{H}+h]=scores[p-lo]/(denominator+1.0e-16f);
 }}
}}

kernel void graph_messages(
 const device float* value [[buffer(0)]],
 const device float* edgevalue [[buffer(1)]],
 const device float* skip [[buffer(2)]],
 const device float* alpha [[buffer(3)]],
 const device int* rowptr [[buffer(4)]],
 const device int* edgeids [[buffer(5)]],
 const device int* sources [[buffer(6)]],
 device float* output [[buffer(7)]],
 uint tid [[thread_position_in_grid]]) {{
 const uint n=tid/{C}, c=tid%{C};
 float total=0.0f;
 for (int p=rowptr[n]; p<rowptr[n+1]; ++p) {{
   const uint s=sources[p], e=edgeids[p];
   float message=0.0f;
   for (uint h=0; h<{H}; ++h)
     message += (value[s*{value_stride}+{value_offset}+h*{C}+c]
                +edgevalue[e*{H*C}+h*{C}+c])*alpha[e*{H}+h];
   total += message/{H}.0f;
 }}
 output[tid]=total+skip[n*{skip_stride}+{skip_offset}+c];
}}
"""





def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        # Inference tensors intentionally have no version counter. Layouts are
        # internal immutable batches, never user-supplied mutable graph caches.
        return None


@dataclass(frozen=True)
class BoundNeighborLayout:
    """Internal CSR with its exact COO tensor; construct before the GPU forward."""
    edge_index: torch.Tensor
    neighbors: NeighborLayout
    coo_version: int | None

    @classmethod
    def from_cpu(cls, edge_index, node_count, device, *, transferred_coo=None):
        raw = NeighborLayout.from_coo(edge_index, node_count)
        # One transfer for all three arrays. The views have nonzero offsets;
        # the compile_shader binding must retain those offsets (device test).
        combined = torch.cat((raw.rowptr, raw.edge_ids, raw.sources)).to(device)
        neighbors = NeighborLayout(combined[:node_count + 1],
                                   combined[node_count + 1:node_count + 1 + raw.edge_count],
                                   combined[node_count + 1 + raw.edge_count:],
                                   raw.node_count, raw.edge_count, raw.max_degree)
        # Clone CPU input as well: later edits to featuriser buffers must not
        # invalidate this batch. GPU transfer already gives separate storage.
        if transferred_coo is None:
            coo = edge_index.to(device=device, dtype=torch.long, copy=True)
        else:
            # Internal transfer integration only: caller establishes that this
            # is the just-transferred tensor of the exact CPU COO supplied.
            if transferred_coo.shape != edge_index.shape or transferred_coo.device != combined.device:
                raise ValueError("Transferred COO metadata does not match the CPU graph")
            coo = transferred_coo
        return cls(coo, neighbors, _version(coo))

    def matches(self, x, edge_index):
        return (edge_index is self.edge_index
                and self.coo_version == _version(edge_index)
                and x.shape[0] == self.neighbors.node_count
                and edge_index.shape[1] == self.neighbors.edge_count
                and x.device == edge_index.device == self.neighbors.rowptr.device)


class PackedMetalGraphAttention(nn.Module):
    """Inference graph runtime: one packed parameter matrix, two Metal kernels.

    All checkpoint biases must be present in this initial production route.
    Unsupported devices/dtypes, absent layouts and high degrees use PyTorch.
    Training/gradients use the PyTorch formula, including dropout; autocast is
    refused explicitly. Save the ordinary compressed
    artifact and reconstruct this runtime after loading, until recipe support
    is added for this different state-dict layout.
    """
    def __init__(self, source):
        super().__init__()
        if type(source) is not BilinearTransformerConv:
            raise ValueError("Only the audited BilinearTransformerConv implementation is supported")
        if source.score_edge is None or source.lin_edge is None:
            raise ValueError("Expected a bilinear graph layer with edge projections")
        modules = [source.score_node, source.score_edge, source.lin_value, source.lin_skip]
        if any(m.bias is None for m in modules):
            raise ValueError("This packed route requires all four source affine biases")
        if len({m.weight.requires_grad for m in modules}) != 1 or len({m.bias.requires_grad for m in modules}) != 1:
            raise ValueError("Packed projection requires matching trainability within weights and biases")
        if any("forward" in vars(m) or any(bool(v) for k, v in vars(m).items()
                   if k.endswith("_hooks") and isinstance(v, dict)) for m in source.modules()):
            raise ValueError("Runtime fusion cannot preserve module hooks")
        aliases = Counter(id(p) for _, p in source.named_parameters(remove_duplicate=False))
        if any(v > 1 for v in aliases.values()) or any(getattr(p, "_backward_hooks", None) for p in source.parameters()):
            raise ValueError("Shared parameters or parameter hooks require a separate runtime adapter")
        self.in_channels = source.in_channels
        self.out_channels = source.out_channels
        self.heads = source.heads
        self.edge_dim = source.edge_dim
        self.dropout = source.dropout
        self.lin_edge = source.lin_edge
        self.weight = nn.Parameter(torch.cat([m.weight.detach() for m in modules], 0), requires_grad=modules[0].weight.requires_grad)
        self.bias = nn.Parameter(torch.cat([m.bias.detach() for m in modules], 0), requires_grad=modules[0].bias.requires_grad)
        self._shader_cache = {}
        self._compressme_runtime_only = True
        self.train(source.training)

    def __getstate__(self):
        state = super().__getstate__()
        # Compiled Objective-C/Metal handles are process-local, never copied as
        # checkpoint state. This also makes ordinary deepcopy safe after warmup.
        state["_shader_cache"] = {}
        return state

    def _project(self, x):
        H, C, I, D = self.heads, self.out_channels, self.in_channels, self.edge_dim
        projected = F.linear(x, self.weight, self.bias)
        return projected, projected.split((H * I, H * D, H * C, C), dim=-1)

    def _torch(self, x, edge_index, edge_attr, return_attention_weights):
        H, C, I, D = self.heads, self.out_channels, self.in_channels, self.edge_dim
        N = x.shape[0]
        _, (qn, qe, values, skip) = self._project(x)
        src, dst = edge_index
        logits = (qn.reshape(N, H, I)[dst] * x[src, None, :]).sum(-1)
        logits += (qe.reshape(N, H, D)[dst] * edge_attr[:, None, :]).sum(-1)
        alpha = softmax(logits / math.sqrt(C), dst, num_nodes=N)
        weights = F.dropout(alpha, p=self.dropout, training=self.training)
        messages = values.reshape(N, H, C)[src] + self.lin_edge(edge_attr).reshape(-1, H, C)
        messages = (messages * weights[:, :, None]).mean(1)
        out = scatter(messages, dst, dim=0, dim_size=N, reduce="sum") + skip
        return (out, (edge_index, alpha)) if isinstance(return_attention_weights, bool) else out

    def forward(self, x, edge_index, edge_attr, return_attention_weights=None, *, layout=None):
        if autocast_enabled(x.device.type):
            raise ValueError("Packed graph runtime requires explicit dtypes; autocast is unsupported")
        if x.ndim != 2 or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("Expected homogeneous node tensor and dense COO indices")
        if edge_attr is None or edge_attr.shape != (edge_index.shape[1], self.edge_dim):
            raise ValueError("Expected one complete edge-feature vector per COO edge")
        if x.shape[1] != self.in_channels or x.dtype != self.weight.dtype or edge_attr.dtype != x.dtype:
            raise ValueError("Input feature width or explicit dtype differs from runtime weights")
        if x.device != self.weight.device or edge_attr.device != x.device or edge_index.device != x.device:
            raise ValueError("Graph inputs and parameters must share a device")
        if layout is not None and (not isinstance(layout, BoundNeighborLayout) or not layout.matches(x, edge_index)):
            raise ValueError("Neighbor layout does not belong to this exact graph tensor")
        use_metal = (not self.training and not torch.is_grad_enabled()
                     and layout is not None and x.device.type == "mps" and x.dtype == torch.float32
                     and callable(getattr(torch.mps, "compile_shader", None))
                     and x.is_contiguous() and edge_attr.is_contiguous()
                     and layout.neighbors.max_degree <= 32 and x.shape[0] > 0)
        if not use_metal:
            return self._torch(x, edge_index, edge_attr, return_attention_weights)
        neighbors = layout.neighbors
        H, C, I, D = self.heads, self.out_channels, self.in_channels, self.edge_dim
        degree_bin = 1 << max(0, neighbors.max_degree - 1).bit_length()
        if degree_bin not in self._shader_cache:
            self._shader_cache[degree_bin] = torch.mps.compile_shader(shader_source(H, C, I, D, degree_bin, True))
        shader = self._shader_cache[degree_bin]
        projected, _ = self._project(x)
        edge_values = self.lin_edge(edge_attr)
        alpha = x.new_empty((edge_index.shape[1], H))
        out = x.new_empty((x.shape[0], C))
        shader.graph_scores(x, edge_attr, projected, projected, neighbors.rowptr,
                            neighbors.edge_ids, neighbors.sources, alpha,
                            threads=x.shape[0] * H * 32, group_size=256)
        shader.graph_messages(projected, edge_values, projected, alpha, neighbors.rowptr,
                              neighbors.edge_ids, neighbors.sources, out,
                              threads=x.shape[0] * C, group_size=256)
        return (out, (edge_index, alpha)) if isinstance(return_attention_weights, bool) else out


def graph_forward(graph, x, edge_index, edge_attr, batch, batch_size, layout):
    """Run the audited AffineInputGraphEncoder without changing its methods."""
    if graph.training or torch.is_grad_enabled():
        # Existing checkpointing, residuals, dropout and ordinary pooling remain
        # available through the graph's normal route. Packed convs use PyTorch.
        return graph(x, edge_index, edge_attr, batch)
    if not isinstance(batch_size, int) or batch_size < 0:
        raise ValueError("Supply the batch size already known on CPU")
    h = graph.norms[0](graph.act(graph.convs[0](x, edge_index, edge_attr, layout=layout))
                       + graph.input_proj(x))
    for conv, norm in zip(graph.convs[1:], graph.norms[1:]):
        h = norm(graph.act(conv(h, edge_index, edge_attr, layout=layout)) + h)
    return graph.proj(graph.pool(h, batch, size=batch_size))
