import copy
import pytest
import torch
from torch import nn
from compressme import (Example, InputMoments, collect_moments, compress, factorize_linear,
                        factorize_norm_linear, load, validate)


def lowrank(d=24, m=32, rank=3, dtype=torch.float64):
    torch.manual_seed(32)
    layer = nn.Linear(d, m, dtype=dtype)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(m, rank, dtype=dtype) @ torch.randn(rank, d, dtype=dtype))
    return layer


def test_exact_rank_and_arbitrary_batch_shape():
    layer = lowrank()
    compressed = factorize_linear(layer, 3)
    x = torch.randn(2, 3, 7, 24, dtype=torch.float64)
    torch.testing.assert_close(layer(x), compressed(x), atol=1e-12, rtol=1e-12)
    assert sum(p.numel() for p in compressed.parameters()) < sum(p.numel() for p in layer.parameters())
    assert all(t.shape != layer.weight.shape for t in compressed.state_dict().values())


def test_local_bound_for_full_rank_truncation_and_staleness():
    layer = nn.Linear(18, 27, dtype=torch.float64)
    compressed = factorize_linear(layer, 4)
    x = torch.randn(20, 18, dtype=torch.float64) * 100
    error = torch.linalg.vector_norm(layer(x)-compressed(x), dim=-1)
    assert torch.all(error <= compressed.error_bound(x) + 1e-10)
    with torch.no_grad():
        compressed.left.add_(0.1)
    with pytest.raises(RuntimeError, match="stale"):
        compressed.error_bound(x)


def test_norm_linear_removes_mean_direction_with_affine_parameters():
    d, m, r = 16, 21, 2
    torch.manual_seed(2)
    norm = nn.LayerNorm(d, dtype=torch.float64)
    linear = nn.Linear(d, m, dtype=torch.float64)
    gamma = torch.linspace(-2, 3, d, dtype=torch.float64)
    gamma[gamma.abs() < .1] = .2
    a = torch.randn(m, r, dtype=torch.float64) @ torch.randn(r, d, dtype=torch.float64)
    a = a-a.mean(1, keepdim=True)
    with torch.no_grad():
        norm.weight.copy_(gamma)
        norm.bias.normal_()
        linear.weight.copy_((a + torch.randn(m,1,dtype=torch.float64)*10000)/gamma)
    compressed = factorize_norm_linear(norm, linear, r)
    x = torch.cat([torch.randn(12,d,dtype=torch.float64), torch.ones(3,d,dtype=torch.float64)])
    torch.testing.assert_close(compressed(x), linear(norm(x)), atol=1e-8, rtol=1e-9)


def test_norm_bound_is_tight_along_discarded_singular_vector():
    d, m, r = 12, 18, 3
    norm = nn.LayerNorm(d, dtype=torch.float64)
    linear = nn.Linear(d,m,dtype=torch.float64)
    compressed = factorize_norm_linear(norm,linear,r)
    w = linear.weight.detach()
    a = w-w.mean(1,keepdim=True)
    _, s, vh = torch.linalg.svd(a,full_matrices=False)
    x = vh[r] * 1000
    actual = torch.linalg.vector_norm(linear(norm(x))-compressed(x)).item()
    expected = d**.5*s[r].item()
    assert actual == pytest.approx(expected, rel=1e-6)
    assert actual <= compressed.bound.uniform_l2_bound+1e-10


def test_unsupported_normalization_rejected():
    with pytest.raises(ValueError, match="last feature"):
        factorize_norm_linear(nn.LayerNorm((3,4)),nn.Linear(4,8),2)


def test_calibration_handles_singular_covariance_and_affine_anchor():
    torch.manual_seed(8)
    layer = nn.Linear(10,15,bias=False,dtype=torch.float64)
    basis = torch.randn(10,2,dtype=torch.float64)
    mean = torch.randn(10,dtype=torch.float64)*10
    samples = torch.randn(100,2,dtype=torch.float64)@basis.T + mean
    empirical_mean = samples.mean(0)
    centered = samples-empirical_mean
    moments = InputMoments(100,empirical_mean,centered.T@centered/100)
    compressed = factorize_linear(layer,2,moments=moments)
    torch.testing.assert_close(compressed(samples),layer(samples),atol=1e-7,rtol=1e-7)
    # Off-calibration inputs remain covered by a local spectral residual bound.
    x = torch.randn(12,10,dtype=torch.float64)*100
    assert torch.all(torch.linalg.vector_norm(layer(x)-compressed(x),dim=-1) <= compressed.error_bound(x)+1e-10)


def test_flat_spectrum_is_retained_under_tight_budget():
    layer = nn.Linear(20,20,dtype=torch.float64)
    with torch.no_grad(): layer.weight.copy_(torch.eye(20,dtype=torch.float64))
    result = compress(layer,ratio=.5,max_relative_operator_error=.01,min_parameters=0)
    assert result.report["layers"][0]["status"] == "retained_error_budget"
    assert type(result.model) is nn.Linear


def test_output_gate_rolls_back_and_preserves_input_model():
    torch.manual_seed(9)
    model = nn.Sequential(nn.Linear(20,30),nn.Tanh(),nn.Linear(30,10)).eval()
    state = copy.deepcopy(model.state_dict())
    examples = [Example((torch.randn(12,20),))]
    result = compress(model,ratio=.2,max_relative_operator_error=None,min_parameters=0,
                      validation=examples,output_tolerance=1e-8)
    assert result.report["status"] == "rejected_and_rolled_back"
    for k,v in state.items():
        assert torch.equal(v,model.state_dict()[k])
        assert torch.equal(v,result.model.state_dict()[k])


def test_shared_weights_and_attention_are_skipped():
    model = nn.ModuleDict({"a":nn.Linear(20,20),"b":nn.Linear(20,20),
                          "attention":nn.MultiheadAttention(20,2)})
    model["b"].weight = model["a"].weight
    result = compress(model,min_parameters=0,max_relative_operator_error=None)
    assert all(r["status"] == "retained_shared_parameters" for r in result.report["layers"])
    assert result.model["a"].weight is result.model["b"].weight
    assert "attention" in result.report["known_bypass_blocks_skipped"]


def test_tensor_only_roundtrip_and_tamper_detection(tmp_path):
    model = lowrank(dtype=torch.float32).eval()
    result = compress(model,ratio=.4,max_relative_operator_error=.01,min_parameters=0)
    result.save(tmp_path)
    recovered = load(lambda: nn.Linear(24,32),tmp_path)
    x = torch.randn(4,24)
    assert torch.equal(result.model(x),recovered.model(x))
    assert not recovered.model.training
    with (tmp_path/"model.safetensors").open("ab") as stream: stream.write(b"corrupt")
    with pytest.raises(ValueError,match="hash"):
        load(lambda: nn.Linear(24,32),tmp_path)


def test_norm_export_and_backward(tmp_path):
    pair = nn.Sequential(nn.LayerNorm(20),lowrank(d=20,m=30)).eval()
    result = compress(pair,ratio=.4,max_relative_operator_error=.01,min_parameters=0)
    assert result.report["layers"][0]["method"] == "normalization_domain_svd"
    result.save(tmp_path)
    recovered = load(lambda: nn.Sequential(nn.LayerNorm(20),nn.Linear(20,30)),tmp_path)
    x = torch.randn(3,20,requires_grad=True)
    # Source linear is float64; constructor norm is float32, but fused operator uses source dtype.
    x = x.double()
    y = recovered.model(x)
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in recovered.model.parameters())


def test_output_schema_nonfinite_and_discrete_changes_fail():
    from compressme import compare_outputs
    with pytest.raises(ValueError,match="Discrete"):
        compare_outputs({"ids":torch.tensor([1])},{"ids":torch.tensor([2])})
    with pytest.raises(ValueError,match="Non-finite"):
        compare_outputs(torch.tensor([1.]),torch.tensor([float("nan")]))
    with pytest.raises(ValueError,match="structure"):
        compare_outputs({"a":torch.ones(1)},{"b":torch.ones(1)})


def test_moments_cleanup_and_training_flags():
    model = nn.Sequential(nn.Linear(4,8),nn.Dropout(.5),nn.Linear(8,4))
    model[0].eval()
    modes = [m.training for m in model.modules()]
    moments = collect_moments(model,[Example((torch.randn(10,4),))])
    assert moments["0"].count == 10
    assert [m.training for m in model.modules()] == modes
    assert not model[0]._forward_pre_hooks


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason="Requires Apple Metal access")
def test_mps_factorization_and_output_validation():
    original = lowrank(dtype=torch.float32).to("mps").eval()
    compressed = factorize_linear(original,3)
    x = torch.randn(6,24,device="mps")
    report = validate(original,compressed,[Example((x,))],relative_tolerance=1e-4)
    assert report["accepted"]
    pair = nn.Sequential(nn.LayerNorm(24),lowrank(dtype=torch.float32)).to("mps").eval()
    fused = factorize_norm_linear(pair[0],pair[1],3)
    assert validate(pair,fused,[Example((x,))],relative_tolerance=1e-4)["accepted"]
