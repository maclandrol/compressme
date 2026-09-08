import pytest
import torch
from torch import nn

from compressme import Example, InputMoments, factorize_linear, factorize_norm_linear, validate


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires Apple Metal access")
@pytest.mark.parametrize("seed", [0, 7, 32])
def test_mps_preparation_preserves_lowrank_values_and_affine_anchor(seed):
    torch.manual_seed(seed)
    original = nn.Linear(24, 32)
    with torch.no_grad():
        original.weight.copy_(torch.randn(32, 3) @ torch.randn(3, 24))
        original.bias.normal_()
    original = original.eval().to("mps")
    compressed = factorize_linear(original, 3)
    x = torch.randn(2, 5, 24, device="mps")
    assert all(torch.isfinite(value).all() for value in compressed.state_dict().values())
    assert compressed.bound.relative_operator_error < 1e-5
    report = validate(original, compressed, [Example((x,))], relative_tolerance=1e-4)
    assert report["accepted"]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires Apple Metal access")
def test_mps_covariance_and_layernorm_preparation_are_finite():
    torch.manual_seed(45)
    original = nn.Linear(16, 21).eval().to("mps")
    moments = InputMoments(100, torch.ones(16, device="mps"), torch.eye(16, device="mps"))
    compressed = factorize_linear(original, 5, moments=moments)
    anchor = moments.mean.unsqueeze(0)
    torch.testing.assert_close(original(anchor), compressed(anchor), rtol=1e-5, atol=1e-6)
    normalizer = nn.LayerNorm(16).eval().to("mps")
    folded = factorize_norm_linear(normalizer, original, 16)
    x = torch.randn(9, 16, device="mps")
    torch.testing.assert_close(folded(x), original(normalizer(x)), rtol=1e-4, atol=1e-5)
    assert torch.isfinite(folded.error_bound(x)).all()


def test_parameter_replacement_cannot_reuse_a_bound():
    compressed = factorize_linear(nn.Linear(8, 8, dtype=torch.float64), 4)
    prior_version = compressed.left._version
    replacement = nn.Parameter(compressed.left.detach().clone() * 10)
    with torch.no_grad():
        while replacement._version < prior_version:
            replacement.add_(0)
    compressed.left = replacement
    with pytest.raises(RuntimeError, match="stale"):
        compressed.error_bound(torch.ones(8, dtype=torch.float64))


def test_changed_anchor_and_normalization_invalidate_bounds():
    compressed = factorize_linear(nn.Linear(8, 8), 4)
    with torch.no_grad():
        compressed.anchor.add_(100)
    with pytest.raises(RuntimeError, match="stale"):
        compressed.error_bound(torch.ones(8))
    folded = factorize_norm_linear(nn.LayerNorm(8), nn.Linear(8, 8), 4)
    folded.eps *= 100
    with pytest.raises(RuntimeError, match="stale"):
        folded.error_bound(torch.ones(8))
