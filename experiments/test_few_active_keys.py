import pytest
import torch
from torch import nn
from few_active_keys import FewActiveKeysMHA, install_few_active_keys


def test_autocast_uses_original_attention_dtype():
    source = torch.nn.MultiheadAttention(8,2,batch_first=True).eval()
    candidate = FewActiveKeysMHA(source)
    x = torch.randn(1,4,8)
    mask = torch.tensor([[False,False,True,True]])
    with torch.no_grad(),torch.autocast("cpu",dtype=torch.bfloat16):
        expected = source(x,x,x,key_padding_mask=mask)
        actual = candidate(x,x,x,key_padding_mask=mask)
    assert actual[0].dtype == expected[0].dtype
    assert actual[1].dtype == expected[1].dtype
    assert candidate.fallback_calls == 1


@pytest.mark.parametrize('bias',[False,True])
@pytest.mark.parametrize('keys',[1,2])
@pytest.mark.parametrize('floatmask',[False,True])
def test_all_queries_attention(bias,keys,floatmask):
    torch.manual_seed(817)
    original=nn.MultiheadAttention(32,4,batch_first=True,bias=bias,dtype=torch.float64).eval()
    wrapped=FewActiveKeysMHA(original)
    x=torch.randn(3,13,32,dtype=torch.float64)
    mask=torch.ones(3,13,dtype=torch.bool)
    mask[:,[0,8][:keys]]=False
    if floatmask:
        mask=torch.zeros_like(mask,dtype=torch.float64).masked_fill(mask,float('-inf'))
    with torch.inference_mode():
        for average in [True,False]:
            a,wa=original(x,x,x,key_padding_mask=mask,average_attn_weights=average)
            b,wb=wrapped(x,x,x,key_padding_mask=mask,average_attn_weights=average)
            torch.testing.assert_close(a,b,atol=3e-12,rtol=3e-12)
            torch.testing.assert_close(wa,wb,atol=3e-12,rtol=3e-12)
        b,wb=wrapped(x,x,x,key_padding_mask=mask,need_weights=False)
        torch.testing.assert_close(a,b,atol=3e-12,rtol=3e-12)
        assert wb is None
    assert wrapped.fast_calls==3


@pytest.mark.parametrize('case',['training','grad','variable_mask','three_keys','all_masked','attn_mask'])
def test_fallback(case):
    torch.manual_seed(51)
    original=nn.MultiheadAttention(16,4,batch_first=True,dtype=torch.float64,dropout=.1).eval()
    wrapped=FewActiveKeysMHA(original)
    x=torch.randn(2,13,16,dtype=torch.float64)
    mask=torch.ones(2,13,dtype=torch.bool);mask[:,:2]=False
    kw={}
    if case=='training': original.train();wrapped.train()
    if case=='variable_mask':mask[1]=True;mask[1,2:4]=False
    if case=='three_keys':mask[:,:3]=False
    if case=='all_masked':mask[:]=True
    if case=='attn_mask':kw['attn_mask']=torch.zeros(13,13,dtype=torch.bool)
    with torch.set_grad_enabled(case=='grad'):
        torch.manual_seed(9);a,wa=original(x,x,x,key_padding_mask=mask,**kw)
        torch.manual_seed(9);b,wb=wrapped(x,x,x,key_padding_mask=mask,**kw)
        torch.testing.assert_close(a,b,equal_nan=True)
        torch.testing.assert_close(wa,wb,equal_nan=True)
    assert wrapped.fallback_calls==1


def test_encoder_parent_uses_wrapper_and_preserves_all_outputs():
    torch.manual_seed(981)
    layer=nn.TransformerEncoderLayer(64,4,dim_feedforward=128,batch_first=True,norm_first=True,
                                    dtype=torch.float64,dropout=0.)
    original=nn.TransformerEncoder(layer,num_layers=2,enable_nested_tensor=False).eval()
    wrapped=install_few_active_keys(original)
    x=torch.randn(2,13,64,dtype=torch.float64)
    mask=torch.ones(2,13,dtype=torch.bool);mask[:,[0,5]]=False
    with torch.inference_mode():
        a=original(x,src_key_padding_mask=mask)
        b=wrapped(x,src_key_padding_mask=mask)
    torch.testing.assert_close(a,b,atol=2e-11,rtol=2e-11)
    assert all(layer.self_attn.fast_calls==1 for layer in wrapped.layers)
    assert sum(p.numel() for p in original.parameters())==sum(p.numel() for p in wrapped.parameters())
