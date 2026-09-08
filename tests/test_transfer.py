import pytest
import torch
from compressme.transfer import transfer_tensors,transfer_by_dtype

def bits(t):return t.detach().cpu().contiguous().reshape(-1).view(torch.uint8)

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('fn',[transfer_tensors,transfer_by_dtype])
def test_mixed_values_shapes_dtypes_and_offsets(device,fn):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    values=[torch.tensor([1.,-0.,float('nan'),float('inf')]),torch.tensor([1,-2,2**40],dtype=torch.int64),
            torch.tensor([[True,False]]),torch.tensor(3,dtype=torch.int32),torch.zeros(2,0,dtype=torch.float16),
            torch.arange(24,dtype=torch.float32).reshape(4,6).T,torch.arange(9,dtype=torch.uint8)]
    transferred=fn(values,device)
    for x,y in zip(values,transferred):
        assert x.shape==y.shape and x.dtype==y.dtype and y.device.type==device
        assert torch.equal(bits(x),bits(y)) and y.is_contiguous()
    assert transferred[5].storage_offset()>0
    transferred[1].zero_()
    assert torch.equal(bits(values[0]),bits(transferred[0]))
    assert values[1][0]==1

def test_reject_unsupported_and_empty():
    assert transfer_tensors([],'cpu')==[]
    with pytest.raises(ValueError):transfer_tensors([torch.ones(2,requires_grad=True)],'cpu')
    with pytest.raises(ValueError):transfer_tensors([torch.ones(2)],'cpu',alignment=3)
    with pytest.raises(ValueError):transfer_tensors([torch.tensor([1],dtype=torch.int64)],'cpu',alignment=4)
    with pytest.raises(ValueError):transfer_tensors([None],'cpu')
