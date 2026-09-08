import copy
import pytest
import torch
from torch_geometric.nn import TransformerConv
from compressme.attention import BilinearTransformerConv
from compressme.graph_metal import BoundNeighborLayout, PackedMetalGraphAttention


def fixture(dtype=torch.float64):
    torch.manual_seed(187)
    source = BilinearTransformerConv(TransformerConv(7, 11, heads=3, edge_dim=5,
        concat=False, dropout=.3).to(dtype).eval())
    candidate = PackedMetalGraphAttention(copy.deepcopy(source))
    x = torch.randn(8, 7, dtype=dtype)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 4, 3], [1, 1, 3, 3, 3, 4, 0]])
    edge_attr = torch.randn(7, 5, dtype=dtype)
    return source, candidate, x, edge_index, edge_attr


def test_single_parameter_ownership_and_cpu_fallback():
    source, candidate, x, coo, edge = fixture()
    assert sum(p.numel() for p in source.parameters()) == sum(p.numel() for p in candidate.parameters())
    assert set(candidate.state_dict()) == {'weight', 'bias', 'lin_edge.weight'}
    layout = BoundNeighborLayout.from_cpu(coo, 8, 'cpu')
    with torch.inference_mode():
        expected = source(x, coo, edge, True)
        actual = candidate(x, layout.edge_index, edge, True, layout=layout)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1][1], expected[1][1])
    # CSR tensor slices share one allocation, with real nonzero offsets.
    assert layout.neighbors.edge_ids.storage_offset() == 9
    assert layout.neighbors.sources.storage_offset() == 16
    assert layout.neighbors.sources.untyped_storage().data_ptr() == layout.neighbors.rowptr.untyped_storage().data_ptr()


@pytest.mark.parametrize('training', [False, True])
def test_torch_fallback_values_and_gradients(training):
    source, candidate, x, coo, edge = fixture()
    source.train(training); candidate.train(training)
    xa = x.clone().requires_grad_(); xb = x.clone().requires_grad_()
    ea = edge.clone().requires_grad_(); eb = edge.clone().requires_grad_()
    torch.manual_seed(913); a, (_, aa) = source(xa, coo, ea, False)
    torch.manual_seed(913); b, (_, ba) = candidate(xb, coo, eb, False)
    torch.testing.assert_close(a, b, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(aa, ba, atol=2e-14, rtol=2e-14)
    a.square().sum().backward(); b.square().sum().backward()
    torch.testing.assert_close(xa.grad, xb.grad, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(ea.grad, eb.grad, atol=1e-12, rtol=1e-12)
    modules = [source.score_node, source.score_edge, source.lin_value, source.lin_skip]
    torch.testing.assert_close(candidate.weight.grad, torch.cat([m.weight.grad for m in modules]), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(candidate.bias.grad, torch.cat([m.bias.grad for m in modules]), atol=1e-12, rtol=1e-12)


def test_layout_rejects_different_coo_and_inplace_mutation():
    _, candidate, x, coo, edge = fixture()
    layout = BoundNeighborLayout.from_cpu(coo, 8, 'cpu')
    with torch.inference_mode(), pytest.raises(ValueError):
        candidate(x, coo.clone(), edge, layout=layout)
    layout.edge_index[0, 0] = 6
    with torch.inference_mode(), pytest.raises(ValueError):
        candidate(x, layout.edge_index, edge, layout=layout)


def test_no_layout_and_high_degree_are_supported_cpu():
    source, candidate, x, coo, edge = fixture(torch.float32)
    coo = coo.repeat(1, 20); edge = edge.repeat(20, 1)
    layout = BoundNeighborLayout.from_cpu(coo, 8, 'cpu')
    assert layout.neighbors.max_degree > 32
    with torch.inference_mode():
        expected = source(x, coo, edge)
        torch.testing.assert_close(candidate(x, coo, edge), expected)
        torch.testing.assert_close(candidate(x, layout.edge_index, edge, layout=layout), expected)


@pytest.mark.skipif(not torch.backends.mps.is_available() or not hasattr(torch.mps,"compile_shader"), reason="Requires Apple Metal shaders")
@pytest.mark.parametrize("kind", ["ordinary", "empty", "high_degree"])
def test_metal_storage_offsets_and_fallbacks(kind):
    source, candidate, x, coo, edge = fixture(torch.float32)
    if kind == "empty":
        coo, edge = coo[:,:0], edge[:0]
    elif kind == "high_degree":
        coo, edge = coo.repeat(1,20), edge.repeat(20,1)
    layout = BoundNeighborLayout.from_cpu(coo,8,"mps")
    source, candidate = source.to("mps"), candidate.to("mps")
    x, coo, edge = x.to("mps"), coo.to("mps"), edge.to("mps")
    with torch.inference_mode():
        expected = source(x,coo,edge,True)
        actual = candidate(x,layout.edge_index,edge,True,layout=layout)
        torch.testing.assert_close(actual[0],expected[0],atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(actual[1][1],expected[1][1],atol=1e-5,rtol=1e-5)
        cloned = copy.deepcopy(candidate)
        torch.testing.assert_close(cloned(x,layout.edge_index,edge,layout=layout),actual[0])
