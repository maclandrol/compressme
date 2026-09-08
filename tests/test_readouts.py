import pytest
import torch
from torch import nn
from compressme.readouts import BatchedLinearHeads


@pytest.mark.parametrize("bias",[True,False])
def test_batched_readouts_preserve_every_head_output_and_gradient(bias):
    heads = nn.ModuleList([nn.Linear(7,11,bias=bias,dtype=torch.float64) for _ in range(5)])
    batched = BatchedLinearHeads(heads)
    x = torch.randn(2,3,5,7,dtype=torch.float64,requires_grad=True)
    original = torch.stack([h(x[...,i,:]) for i,h in enumerate(heads)],dim=-2)
    actual = batched(x)
    torch.testing.assert_close(original,actual,atol=1e-12,rtol=1e-12)
    go = torch.autograd.grad(original.square().sum(),x)[0]
    gc = torch.autograd.grad(actual.square().sum(),x)[0]
    torch.testing.assert_close(go,gc,atol=1e-12,rtol=1e-12)
    assert sum(p.numel() for p in heads.parameters()) == sum(p.numel() for p in batched.parameters())
    torch.testing.assert_close(batched[2](x[...,2,:]),heads[2](x[...,2,:]))


def test_shared_readouts_are_not_silently_untied():
    head = nn.Linear(8,8)
    with pytest.raises(ValueError,match="Shared"):
        BatchedLinearHeads([head,head])


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason="Requires Apple GPU")
def test_batched_readouts_mps():
    heads = nn.ModuleList([nn.Linear(32,64) for _ in range(13)]).to("mps")
    batched = BatchedLinearHeads(heads)
    x = torch.randn(4,13,32,device="mps")
    expected = torch.stack([h(x[:,i]) for i,h in enumerate(heads)],dim=1)
    torch.testing.assert_close(batched(x),expected,atol=1e-5,rtol=1e-5)
