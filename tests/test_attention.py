import torch
import pytest
pytest.importorskip("torch_geometric")
from torch_geometric.nn import TransformerConv
from compressme.attention import BilinearTransformerConv


@pytest.mark.parametrize("aggregate",[False,True])
@pytest.mark.parametrize("edges",[0,8])
@pytest.mark.parametrize("edge_dim",[None,3])
@pytest.mark.parametrize("bias",[False,True])
@pytest.mark.parametrize("training",[False,True])
@pytest.mark.parametrize("skip_single",[False,True])
def test_exact(aggregate,edges,edge_dim,bias,training,skip_single):
    torch.manual_seed(519)
    m=TransformerConv(5,7,heads=3,concat=False,bias=bias,dropout=0.3,edge_dim=edge_dim).double()
    m.train(training)
    c=BilinearTransformerConv(m,aggregate_before_value=aggregate,skip_single_neighbors=skip_single)
    # Node 5 remains isolated, including the nonempty graph.
    edge_index=torch.tensor([[0,1,1,2,2,3,3,4],[1,0,2,1,3,2,4,3]])[:,:edges]
    edge_attr=torch.randn(edges,edge_dim,dtype=torch.double) if edge_dim else None
    x=torch.randn(6,5,dtype=torch.double,requires_grad=True)
    y=x.detach().clone().requires_grad_(True)
    torch.manual_seed(985)
    a,(_,alpha)=m(x,edge_index,edge_attr,return_attention_weights=False)
    torch.manual_seed(985)
    b,(_,beta)=c(y,edge_index,edge_attr,return_attention_weights=False)
    torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(alpha,beta,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(b[-1],c.lin_skip(y)[-1],atol=1e-12,rtol=1e-12)
    direction=torch.randn_like(a)
    (a*direction).sum().backward()
    (b*direction).sum().backward()
    torch.testing.assert_close(x.grad,y.grad,atol=1e-11,rtol=1e-11)


def test_generic_compiler_checkpoint_roundtrip_and_contract(tmp_path):
    from compressme import contract_attention, load
    def factory():
        return TransformerConv(5,7,heads=3,concat=False,edge_dim=3).double().eval()
    source = factory().requires_grad_(False)
    state = torch.random.get_rng_state().clone()
    result = contract_attention(source,input_contract="homogeneous_coo")
    assert torch.equal(state,torch.random.get_rng_state())
    assert result.report["parameters_after"] < result.report["parameters_before"]
    result.save(tmp_path)
    restored = load(factory,tmp_path).model
    x = torch.randn(5,5,dtype=torch.float64)
    edges = torch.tensor([[0,1,1,3],[1,0,2,2]])
    attr = torch.randn(4,3,dtype=torch.float64)
    torch.testing.assert_close(source(x,edges,attr),restored(x,edges,attr),rtol=1e-12,atol=1e-12)
    assert torch.equal(result.model(x,edges,attr),restored(x,edges,attr))
    assert all(not p.requires_grad for p in restored.parameters())
    with pytest.raises(ValueError,match="contract"):
        contract_attention(source,input_contract="bipartite")
    with pytest.raises(ValueError,match="tensor"):
        restored((x,x),edges,attr)


def test_generic_attention_skips_unprofitable_and_unsupported():
    from compressme import contract_attention
    too_wide = TransformerConv(32,4,heads=2,concat=False)
    result = contract_attention(too_wide,input_contract="homogeneous_coo")
    assert result.report["layers"][0]["status"] == "retained_no_byte_saving"
    unsupported = TransformerConv(5,7,heads=3,concat=True)
    result = contract_attention(unsupported,input_contract="homogeneous_coo")
    assert result.report["layers"][0]["status"] == "retained_unsupported"
    tied = torch.nn.ModuleList([too_wide,too_wide])
    assert contract_attention(tied,input_contract="homogeneous_coo").report["layers"][0]["status"] == "retained_shared_parameters"


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason="Requires Apple GPU")
def test_contract_on_mps_preserves_finite_weights_and_outputs():
    from compressme import contract_attention
    source = TransformerConv(5,7,heads=3,concat=False,edge_dim=3).to("mps").eval()
    result = contract_attention(source,input_contract="homogeneous_coo")
    x = torch.randn(5,5,device="mps")
    edges = torch.tensor([[0,1,1,3],[1,0,2,2]],device="mps")
    attr = torch.randn(4,3,device="mps")
    torch.testing.assert_close(source(x,edges,attr),result.model(x,edges,attr),rtol=1e-5,atol=1e-5)


def test_raw_edges_without_projection_are_rejected():
    source = TransformerConv(5,7,heads=3,concat=False).double()
    fused = BilinearTransformerConv(source)
    with pytest.raises(ValueError,match="Raw edge"):
        fused(torch.randn(3,5,dtype=torch.float64),torch.tensor([[0,1],[1,2]]),torch.ones(2,3,7))


def test_exact_attention_restores_source_dtype_from_default_factory(tmp_path):
    from compressme import contract_attention, load
    def factory(): return TransformerConv(5,7,heads=3,concat=False)
    result = contract_attention(factory().double(),input_contract="homogeneous_coo")
    result.save(tmp_path)
    restored = load(factory,tmp_path).model
    assert all(p.dtype == torch.float64 for p in restored.parameters())


def test_attention_rejects_unsupported_autocast():
    source = TransformerConv(5,7,heads=3,concat=False)
    fused = BilinearTransformerConv(source)
    with torch.autocast("cpu",dtype=torch.bfloat16):
        with pytest.raises(ValueError,match="autocast"):
            fused(torch.randn(3,5),torch.tensor([[0,1],[1,2]]))
