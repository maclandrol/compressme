import copy
from fractions import Fraction
import math
import pytest
import torch
from torch import nn
from experiments.scalar_piecewise import build_scalar_piecewise, ScalarPiecewise


def chain(dtype=torch.float64, activation=None, h=32, m=3, biases=True):
    torch.manual_seed(143)
    model = nn.Sequential(nn.Linear(1, h, bias=biases), activation or nn.LeakyReLU(.01),
                          nn.Linear(h, m, bias=biases)).to(dtype).eval()
    with torch.no_grad():
        model[0].weight[:, 0] = torch.tensor([(-1 if j % 2 else 1) * 2 ** (j % 4 - 2)
                                              for j in range(h)], dtype=dtype)
        model[0].weight[0] = 0
        if biases:
            model[0].bias.copy_(torch.arange(h, dtype=dtype).remainder(7) - 3)
        model[2].weight.copy_((torch.arange(m*h, dtype=dtype).reshape(m, h).remainder(11)-5)/32)
    return model


@pytest.mark.parametrize('activation', [nn.ReLU(), nn.LeakyReLU(.01), nn.LeakyReLU(-.5),
                                        nn.LeakyReLU(1.), nn.LeakyReLU(2.)])
def test_signed_real_domain_and_negative_slopes(activation):
    source = chain(activation=activation)
    validation = torch.linspace(-1000, 1000, 5001, dtype=torch.float64).reshape(-1, 1)
    result = build_scalar_piecewise(source, validation_inputs=validation,
                                   absolute_tolerance=1e-9, relative_tolerance=1e-10)
    assert result.report['status'] == 'accepted_on_numerical_probes'
    assert isinstance(result.model, ScalarPiecewise)
    with torch.inference_mode():
        torch.testing.assert_close(result.model(validation), source(validation), atol=1e-9, rtol=1e-10)
    assert sum(p.numel() for p in result.model.parameters()) == result.report['candidate_parameters']
    assert result.report['candidate_state_bytes'] == 8 * result.report['candidate_parameters']
    assert not any(child is source for child in result.model.modules())


def test_float32_shapes_empty_and_source_independence():
    source = chain(dtype=torch.float32, activation=nn.LeakyReLU(.25))
    result = build_scalar_piecewise(source, absolute_tolerance=1e-4)
    assert result.report['status'] == 'accepted_on_numerical_probes'
    with torch.inference_mode():
        for shape in [(1,), (0, 1), (2, 0, 3, 1), (2, 3, 4, 1)]:
            x = torch.randn(shape)
            torch.testing.assert_close(result.model(x), source(x), atol=1e-5, rtol=1e-5)
        before = result.model(torch.tensor([[2.]]))
        source[2].weight.zero_()
        torch.testing.assert_close(result.model(torch.tensor([[2.]])), before, rtol=0, atol=0)


def test_zero_slopes_zero_consumers_and_affine_activation_eliminate_hinges():
    source = chain(activation=nn.LeakyReLU(1.))
    result = build_scalar_piecewise(source)
    assert result.report['retained_hinges'] == 0
    assert result.report['candidate_parameters'] == 6
    assert result.model._knots.numel() == 0
    source = chain(activation=nn.ReLU())
    with torch.no_grad():
        source[0].weight.zero_()
    result = build_scalar_piecewise(source)
    assert result.report['retained_hinges'] == 0
    x = torch.tensor([[-1e100], [0.], [1e100]], dtype=torch.float64)
    torch.testing.assert_close(result.model(x), source(x), atol=1e-12, rtol=1e-12)


def test_no_profit_and_strict_numerical_failure_roll_back():
    source = chain(h=2, m=8)
    with torch.no_grad():
        source[0].weight[0] = 1
    result = build_scalar_piecewise(source)
    assert result.report['status'] == 'no_storage_saving'
    assert isinstance(result.model, nn.Sequential)
    source = chain(dtype=torch.float32)
    result = build_scalar_piecewise(source, absolute_tolerance=0, relative_tolerance=0)
    assert result.report['status'] == 'rejected_numerical_probe'
    for before, after in zip(source.state_dict().values(), result.model.state_dict().values()):
        assert torch.equal(before, after)


def test_no_bias_chains_count_actual_storage():
    source = chain(biases=False)
    result = build_scalar_piecewise(source)
    assert result.report['status'] == 'no_storage_saving'
    with torch.no_grad():
        source[0].weight[1] = 0
    result = build_scalar_piecewise(source)
    assert result.report['status'] == 'accepted_on_numerical_probes'
    assert result.report['source_parameters'] == 32 + 32 * 3
    assert result.report['candidate_parameters'] == sum(p.numel() for p in result.model.parameters())
    x = torch.tensor([[-9.], [0.], [9.]], dtype=torch.float64)
    torch.testing.assert_close(result.model(x), source(x), atol=1e-12, rtol=1e-12)


def test_input_gradients_training_autocast_and_unsafe_grammar_refused():
    source = chain()
    result = build_scalar_piecewise(source)
    with pytest.raises(ValueError, match='subgradients'):
        result.model(torch.ones(1, dtype=torch.float64, requires_grad=True))
    with pytest.raises(ValueError, match='inference'):
        result.model.train()
    with torch.autocast('cpu', dtype=torch.bfloat16), pytest.raises(ValueError, match='Autocast'):
        result.model(torch.ones(1, dtype=torch.float64))
    for candidate in [copy.deepcopy(source).train(), nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 3)).eval(),
                      nn.Sequential(nn.Linear(1, 32), nn.Sigmoid(), nn.Linear(32, 3)).eval()]:
        with pytest.raises(ValueError):
            build_scalar_piecewise(candidate)
    source[0].register_forward_hook(lambda *args: None)
    with pytest.raises(ValueError, match='hook'):
        build_scalar_piecewise(source)


def test_overflowing_threshold_refused_without_losing_original():
    source = chain(dtype=torch.float32)
    with torch.no_grad():
        source[0].weight[1, 0] = torch.finfo(torch.float32).tiny
        source[0].bias[1] = torch.finfo(torch.float32).max
    result = build_scalar_piecewise(source)
    assert result.report['status'] == 'rejected_coefficient_range'
    assert isinstance(result.model, nn.Sequential)


def frac(value):
    if isinstance(value, torch.Tensor):
        value = value.detach()
    return Fraction.from_float(float(value))


def test_global_real_coefficient_error_certificate_against_rational_evaluation():
    source = chain(dtype=torch.float32)
    with torch.no_grad():
        source[0].weight[1, 0] = -.7
        source[0].bias[1] = 1.3
    result = build_scalar_piecewise(source, absolute_tolerance=1e-3, relative_tolerance=1e-4)
    assert result.report['status'] == 'accepted_on_numerical_probes'
    model = result.model
    certificate = result.report['stored_coefficient_real_error_bound']
    alpha = frac(source[1].negative_slope)
    for x in [Fraction(-10**50), Fraction(-1, 7), Fraction(0), Fraction(3, 11), Fraction(10**50)]:
        hidden = [frac(a) * x + frac(b) for a, b in zip(source[0].weight[:, 0], source[0].bias)]
        hidden = [max(y, 0) + alpha * min(y, 0) for y in hidden]
        hinges = [max(x - frac(t), 0) for t in model._knots]
        for i in range(source[2].out_features):
            original = frac(source[2].bias[i]) + sum(frac(w) * y for w, y in zip(source[2].weight[i], hidden))
            rewritten = (frac(model._intercept[i]) + frac(model._coefficients[i, -1]) * x
                         + sum(frac(w) * y for w, y in zip(model._coefficients[i, :-1], hinges)))
            bound = frac(certificate['offset'][i]) + frac(certificate['growth'][i]) * abs(x)
            assert abs(original - rewritten) <= bound
