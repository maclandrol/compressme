import torch
from torch import nn
from compressme import Example, compare_outputs, validate


def test_signed_zero_is_numerically_equal_but_not_bitwise_equal():
    metrics = compare_outputs(torch.tensor([0.0, -0.0]), torch.tensor([-0.0, 0.0]))['output']
    assert metrics['exact'] and metrics['max_abs'] == 0 and metrics['relative_l2'] == 0
    assert not metrics['bitwise']


def test_byte_comparison_handles_scalar_strided_empty_and_discrete_tensors():
    inputs = [torch.tensor(-0.0), torch.arange(18.0).reshape(3, 6)[:, ::2],
              torch.empty((2, 0, 3)), torch.tensor([True, False]), torch.tensor([2, 4]),
              torch._neg_view(torch.tensor([0.0, 1.0]))]
    metrics = compare_outputs(inputs, [value.clone() for value in inputs])
    assert len(metrics) == len(inputs)
    assert all(item['bitwise'] and item['exact'] for item in metrics.values())


def test_numerical_acceptance_does_not_claim_byte_identity():
    class Negate(nn.Module):
        def forward(self, x):
            return -x
    result = validate(nn.Identity(), Negate(), [Example((torch.zeros(3),))],
                      relative_tolerance=0, absolute_tolerance=0)
    assert result['accepted']
    assert not result['bitwise_identical']
