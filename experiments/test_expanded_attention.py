import torch
import pytest
from torch import nn
from expanded_attention import ExpandedInputSelfAttention


@pytest.mark.parametrize("training",[False,True])
@pytest.mark.parametrize("projection_bias",[False,True])
@pytest.mark.parametrize("attention_bias",[False,True])
@pytest.mark.parametrize("padding",[False,True])
def test_attention(training,projection_bias,attention_bias,padding):
    torch.manual_seed(421)
    projection=nn.Linear(3,12,bias=projection_bias).double()
    attention=nn.MultiheadAttention(12,3,bias=attention_bias,batch_first=True,dropout=.25).double()
    attention.train(training)
    compressed=ExpandedInputSelfAttention(projection,attention,padding_after_projection=True)
    x=torch.randn(2,5,3,dtype=torch.double)
    mask=torch.tensor([[False,False,True,True,True],[False,False,False,False,True]]) if padding else None
    if padding:
        x=x.masked_fill(mask[:,:,None],0)
    x.requires_grad_(True)
    y=x.detach().clone().requires_grad_(True)
    projected=projection(x)
    if padding:
        projected=projected.masked_fill(mask[:,:,None],0)
    torch.manual_seed(456)
    expected,ew=attention(projected,projected,projected,key_padding_mask=mask,average_attn_weights=False)
    torch.manual_seed(456)
    actual,aw=compressed(y,key_padding_mask=mask,average_attn_weights=False)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(aw,ew,atol=1e-12,rtol=1e-12)
    direction=torch.randn_like(actual)
    (expected*direction).sum().backward()
    (actual*direction).sum().backward()
    # Raw padding positions are not real atoms and are fixed zero by the
    # packing operator; check derivatives only for observable input atoms.
    if padding:
        torch.testing.assert_close(x.grad[~mask],y.grad[~mask],atol=1e-11,rtol=1e-11)
    else:
        torch.testing.assert_close(x.grad,y.grad,atol=1e-11,rtol=1e-11)
