import itertools
import pytest
import torch
from torch import nn
from compressme.norm_sandwich import compose_norm_sandwich


def test_autocast_rejected_instead_of_changing_output_dtype():
    model,_ = compose_norm_sandwich(nn.Linear(4,16),nn.LayerNorm(16),nn.Linear(16,8))
    with torch.autocast("cpu",dtype=torch.bfloat16):
        with pytest.raises(ValueError,match="autocast"):
            model(torch.randn(3,4))


@pytest.mark.parametrize('first_bias,last_bias,affine', list(itertools.product([False,True], repeat=3)))
@pytest.mark.parametrize('dtype',[torch.float64,torch.float32])
def test_outputs_gradients_biases(first_bias,last_bias,affine,dtype):
    torch.manual_seed(814)
    first=nn.Linear(5,32,bias=first_bias,dtype=dtype)
    norm=nn.LayerNorm(32,elementwise_affine=affine,dtype=dtype)
    last=nn.Linear(32,24,bias=last_bias,dtype=dtype)
    if affine:
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(-2,3,32,dtype=dtype))
            norm.bias.copy_(torch.linspace(-.2,.4,32,dtype=dtype))
    rng=torch.random.get_rng_state().clone()
    folded,report=compose_norm_sandwich(first,norm,last)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert report.denominator_width==6
    assert report.parameters_after < report.parameters_before
    x=torch.randn(2,3,5,dtype=dtype,requires_grad=True)
    y=last(norm(first(x)));z=folded(x)
    tol=2e-11 if dtype==torch.float64 else 2e-5
    torch.testing.assert_close(y,z,rtol=tol,atol=tol)
    cot=torch.randn_like(y)
    g1=torch.autograd.grad((y*cot).sum(),x,retain_graph=True)[0]
    g2=torch.autograd.grad((z*cot).sum(),x)[0]
    torch.testing.assert_close(g1,g2,rtol=tol*5,atol=tol*5)


@pytest.mark.parametrize('kind',['zero','constant','rankdeficient','wide'])
def test_degenerate_producers(kind):
    torch.manual_seed(3)
    d,n=(12,4) if kind=='wide' else (4,12)
    first=nn.Linear(d,n,dtype=torch.float64)
    with torch.no_grad():
        if kind=='zero':
            first.weight.zero_();first.bias.zero_()
        elif kind=='constant':
            first.weight.copy_(first.weight[0].clone().expand_as(first.weight))
            first.bias.fill_(7.)
        elif kind=='rankdeficient':
            first.weight[:,1:].zero_();first.bias.zero_()
    norm=nn.LayerNorm(n,dtype=torch.float64)
    last=nn.Linear(n,7,dtype=torch.float64)
    folded,report=compose_norm_sandwich(first,norm,last)
    x=torch.randn(6,d,dtype=torch.float64)
    torch.testing.assert_close(last(norm(first(x))),folded(x),rtol=2e-10,atol=2e-10)
    assert folded.denominator.shape==(min(n,d+1),d+1)


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_half_storage(dtype):
    torch.manual_seed(72)
    first=nn.Linear(5,24,dtype=dtype)
    norm=nn.LayerNorm(24,dtype=dtype)
    last=nn.Linear(24,12,dtype=dtype)
    folded,_=compose_norm_sandwich(first,norm,last)
    x=torch.randn(8,5,dtype=dtype)
    assert folded(x).dtype==dtype
    tol=.004 if dtype==torch.float16 else .04
    torch.testing.assert_close(last(norm(first(x))),folded(x),rtol=tol,atol=tol)


def test_frozen_and_rejections():
    first=nn.Linear(4,16,dtype=torch.float64).requires_grad_(False)
    norm=nn.LayerNorm(16,dtype=torch.float64).requires_grad_(False)
    last=nn.Linear(16,12,dtype=torch.float64).requires_grad_(False)
    folded,_=compose_norm_sandwich(first,norm,last)
    assert not any(p.requires_grad for p in folded.parameters())
    norm.eps=float('nan')
    with pytest.raises(ValueError): compose_norm_sandwich(first,norm,last)
    norm.eps=1e-5
    first.register_forward_hook(lambda m,args,y:y+1)
    with pytest.raises(ValueError): compose_norm_sandwich(first,norm,last)


def test_normalization_bias_disabled_and_example_counts():
    first=nn.Linear(82,512,dtype=torch.float64)
    norm=nn.LayerNorm(512,bias=False,dtype=torch.float64)
    last=nn.Linear(512,512,dtype=torch.float64)
    folded,report=compose_norm_sandwich(first,norm,last)
    x=torch.randn(3,82,dtype=torch.float64)
    torch.testing.assert_close(last(norm(first(x))),folded(x),rtol=2e-11,atol=2e-11)
    assert report.parameters_before==305664
    assert report.parameters_after==49897
