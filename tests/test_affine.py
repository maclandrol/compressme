import copy
import pytest
import torch
from torch import nn
from compressme import (Example, compile_affine, compose_affine, compare_outputs,
                        compress, load, parameter_count, validate)


class Fanout(nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.expand = nn.Linear(5, 32, dtype=dtype)
        self.query = nn.Linear(32, 24, dtype=dtype)
        self.value = nn.Linear(32, 17, dtype=dtype)
        self.after_activation = nn.Linear(32, 8, dtype=dtype)

    def forward(self, x, scale=1.0):
        h = self.expand(x)
        return {"q": self.query(h) * scale, "v": self.value(h),
                "residual": h, "nonlinear": self.after_activation(h.sin())}


def test_discover_fanout_and_residual_without_crossing_nonlinearity():
    model = Fanout().eval()
    x = torch.randn(2, 3, 5, dtype=torch.float64)
    result = compile_affine(model, validation=[Example((x,), {"scale": 2.0})])
    assert result.report["validation"]["accepted"]
    assert len(result.report["rewrites"]) == 2
    assert result.model.query.in_features == result.model.value.in_features == 5
    assert result.model.after_activation.in_features == 32
    assert result.model.expand.out_features == 32
    assert parameter_count(result.model) < parameter_count(model)
    assert model.query.in_features == 32
    assert max(m["max_abs"] for m in compare_outputs(model(x), result.model(x)).values()) < 1e-12


def test_chain_removes_expansion_and_tensor_export_replays_graph(tmp_path):
    def factory():
        return nn.Sequential(nn.Linear(5, 32), nn.Linear(32, 40), nn.Linear(40, 10)).eval()
    model = factory()
    model.requires_grad_(False)
    model[2].train()
    result = compile_affine(model)
    assert len(result.report["rewrites"]) == 2
    assert parameter_count(result.model) == 5*10+10
    result.save(tmp_path)
    recovered = load(factory, tmp_path)
    x = torch.randn(7, 5)
    assert torch.equal(result.model(x), recovered.model(x))
    assert all(not p.requires_grad for p in recovered.model.parameters())
    assert recovered.model.get_submodule("2").training


def test_shared_consumers_and_direct_weight_reads_are_not_rewritten():
    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 16)
            self.b = nn.Linear(16, 8)
        def forward(self, x, y):
            return self.b(self.a(x)) + self.b(y)
    result = compile_affine(Shared())
    assert not result.report["rewrites"]
    class Direct(Shared):
        def forward(self, x):
            return self.b(self.a(x)), self.b.weight.square().sum()
    result = compile_affine(Direct())
    assert not result.report["rewrites"]


def test_hooks_and_data_dependent_control_are_rejected():
    model = Fanout()
    handle = model.query.register_forward_hook(lambda m, a, b: b*2)
    with pytest.raises(ValueError, match="hooks"):
        compile_affine(model)
    handle.remove()
    class Dynamic(nn.Module):
        def forward(self, x):
            return x+1 if x.sum() > 0 else x-1
    with pytest.raises(ValueError, match="traceable"):
        compile_affine(Dynamic())


@pytest.mark.parametrize("first_bias,second_bias", [(False,False),(True,False),(False,True),(True,True)])
def test_composition_biases_rng_and_input_gradient(first_bias, second_bias):
    a = nn.Linear(7, 24, bias=first_bias, dtype=torch.float64)
    b = nn.Linear(24, 11, bias=second_bias, dtype=torch.float64)
    state = torch.random.get_rng_state().clone()
    c = compose_affine(a, b)
    assert torch.equal(state, torch.random.get_rng_state())
    x = torch.randn(9, 7, dtype=torch.float64, requires_grad=True)
    original, fused = b(a(x)), c(x)
    torch.testing.assert_close(original, fused, atol=1e-12, rtol=1e-12)
    g1 = torch.autograd.grad(original.square().sum(), x)[0]
    g2 = torch.autograd.grad(fused.square().sum(), x)[0]
    torch.testing.assert_close(g1, g2, atol=1e-12, rtol=1e-12)


def test_output_key_collision_cannot_hide_changed_tensor():
    a = {"a.b": torch.ones(1), "a": {"b": torch.ones(1)}}
    b = copy.deepcopy(a)
    b["a.b"] *= 2
    metrics = compare_outputs(a, b)
    assert len(metrics) == 2
    assert max(m["max_abs"] for m in metrics.values()) == 1


def test_frozen_low_rank_roundtrip_and_stale_bound(tmp_path):
    model = nn.Linear(16, 24).requires_grad_(False)
    result = compress(model, min_parameters=0, max_relative_operator_error=None)
    with torch.no_grad():
        result.model.left.add_(0.5)
    result.save(tmp_path)
    recovered = load(lambda: nn.Linear(16, 24), tmp_path).model
    assert all(not p.requires_grad for p in recovered.parameters())
    with pytest.raises(RuntimeError):
        recovered.error_bound(torch.ones(16))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires Apple Metal access")
def test_mps_affine_composition():
    model = Fanout(dtype=torch.float32).to("mps").eval()
    x = torch.randn(7, 5, device="mps")
    result = compile_affine(model)
    assert validate(model, result.model, [Example((x,))], relative_tolerance=1e-5)["accepted"]


def test_export_replays_eval_branch_and_original_float64_dtype(tmp_path):
    class FunctionalDropout(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(5,32)
            self.b = nn.Linear(32,7)
        def forward(self,x):
            y = torch.nn.functional.dropout(x,p=.7,training=self.training)
            return self.b(self.a(y))
    source = FunctionalDropout().double().eval()
    result = compile_affine(source)
    result.save(tmp_path)
    restored = load(FunctionalDropout,tmp_path).model
    x = torch.randn(4,5,dtype=torch.float64)
    assert torch.equal(result.model(x),restored(x))
    assert next(restored.parameters()).dtype == torch.float64


def test_fx_rejects_random_tensor_capture_and_preserves_rng():
    class RandomForward(nn.Module):
        def forward(self,x):
            return x+torch.randn(1)
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError,match="random"):
        compile_affine(RandomForward())
    assert torch.equal(rng,torch.random.get_rng_state())


def test_fx_crosses_layernorm_with_exact_small_statistic(tmp_path):
    from compressme import NormSandwich
    def factory():
        return nn.Sequential(nn.Linear(8,64),nn.LayerNorm(64),nn.Linear(64,32)).double().eval()
    source = factory()
    x = torch.randn(2,5,8,dtype=torch.float64)
    result = compile_affine(source,validation=[Example((x,))])
    assert isinstance(result.model.get_submodule("2"),NormSandwich)
    assert result.report["parameters_after"] < result.report["parameters_before"] / 3
    torch.testing.assert_close(source(x),result.model(x),rtol=1e-12,atol=1e-12)
    result.save(tmp_path)
    restored = load(factory,tmp_path).model
    assert torch.equal(restored(x),result.model(x))


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason="Requires Apple GPU")
def test_mps_normalization_statistic():
    source = nn.Sequential(nn.Linear(8,64),nn.LayerNorm(64),nn.Linear(64,32)).to("mps").eval()
    result = compile_affine(source)
    x = torch.randn(10,8,device="mps")
    torch.testing.assert_close(source(x),result.model(x),rtol=1e-5,atol=1e-5)
