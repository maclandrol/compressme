import pytest
import torch
from compressme.contractions import channelwise_token_contraction


def reference(a,b,direction):
    return torch.einsum('bikd,bjkd->bijd' if direction=='outgoing' else 'bkid,bkjd->bijd',a,b)


def operands(B,I,J,K,D,direction,dtype,strided):
    shape_a=(B,I,K,D) if direction=='outgoing' else (B,K,I,D)
    shape_b=(B,J,K,D) if direction=='outgoing' else (B,K,J,D)
    def create(shape):
        if strided:
            # Nontrivial channel and token strides, without changing dimensions.
            source=torch.randn(tuple(2*n for n in shape),dtype=dtype)
            return source[::2,::2,::2,::2]
        return torch.randn(shape,dtype=dtype)
    return create(shape_a),create(shape_b)


@pytest.mark.parametrize('direction',['outgoing','incoming'])
@pytest.mark.parametrize('shape',[(1,7,7,7,4),(2,3,5,4,3),(1,1,4,3,2),(0,3,4,5,2),(2,3,4,0,2),(2,3,4,5,0)])
@pytest.mark.parametrize('strided',[False,True])
def test_rectangular_shapes_and_strides(direction,shape,strided):
    a,b=operands(*shape,direction,torch.float32,strided)
    expected=reference(a,b,direction)
    got=channelwise_token_contraction(a,b,direction=direction)
    assert got.shape==expected.shape and got.dtype==expected.dtype
    torch.testing.assert_close(got,expected,rtol=1e-5,atol=1e-6)
    # For ordinary nondegenerate sizes, preserve einsum's channel-major result
    # backing layout; empty and singleton tensors admit multiple valid strides.
    if all(x>1 for x in shape):assert got.stride()==expected.stride()


@pytest.mark.parametrize('direction',['outgoing','incoming'])
@pytest.mark.parametrize('dtype',[torch.float64,torch.complex128])
def test_ordinary_autograd_and_gradcheck(direction,dtype):
    a,b=operands(2,2,3,4,2,direction,dtype,True)
    a.requires_grad_(True);b.requires_grad_(True)
    result=channelwise_token_contraction(a,b,direction=direction)
    expected=reference(a,b,direction)
    torch.testing.assert_close(result,expected,rtol=1e-12,atol=1e-12)
    grad=torch.randn_like(result)
    actual_grads=torch.autograd.grad(result,(a,b),grad)
    expected_grads=torch.autograd.grad(expected,(a,b),grad)
    for got,want in zip(actual_grads,expected_grads):torch.testing.assert_close(got,want,rtol=1e-12,atol=1e-12)
    assert torch.autograd.gradcheck(lambda x,y:channelwise_token_contraction(x,y,direction=direction),(a,b),fast_mode=True)


def test_input_guards():
    a=torch.randn(1,2,3,4)
    with pytest.raises(ValueError):channelwise_token_contraction(a,a,direction='other')
    with pytest.raises(TypeError):channelwise_token_contraction(None,a)
    with pytest.raises(ValueError):channelwise_token_contraction(a[0],a)
    with pytest.raises(ValueError):channelwise_token_contraction(a,a.double())
    with pytest.raises(ValueError):channelwise_token_contraction(a,torch.randn(2,2,3,4))
    with pytest.raises(ValueError):channelwise_token_contraction(a,torch.randn(1,2,5,4))
    with pytest.raises(ValueError):channelwise_token_contraction(a,torch.randn(1,2,3,5))


def test_integer_dtype_is_not_silently_converted():
    a=torch.arange(24,dtype=torch.int64).reshape(1,2,3,4)
    b=a.flip(1)
    expected=reference(a,b,'outgoing')
    got=channelwise_token_contraction(a,b)
    assert torch.equal(got,expected) and got.dtype==torch.int64
