import copy,json
import pytest
import torch
from torch import nn
from compressme.constant_embeddings import ConstantRowEmbedding,deduplicate_embeddings

def source(dtype=torch.float32,device='cpu',padding_idx=None):
    row=torch.tensor([1.0,-0.0,-2.5,0.0],dtype=dtype,device=device)
    m=nn.Embedding.from_pretrained(row.expand(9,-1).clone(),freeze=True,padding_idx=padding_idx)
    return m

def bits(t):return t.detach().cpu().contiguous().view(torch.uint8)

@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float32,torch.float64])
@pytest.mark.parametrize('training',[True,False])
def test_exact_values_full_weight_modes_and_storage(dtype,training):
    m=source(dtype).train(training);c=ConstantRowEmbedding.from_embedding(m)
    assert c.training is training and not c._row.requires_grad
    assert c.weight.shape==m.weight.shape and torch.equal(bits(c.weight),bits(m.weight))
    assert c.weight.stride(0)==0 and c.weight.untyped_storage().nbytes()==4*m.weight.element_size()
    for indices in (torch.tensor(3),torch.tensor([8,1,1,0]),torch.empty(0,dtype=torch.long),torch.tensor([[2,0],[8,8]],dtype=torch.int32)):
        assert torch.equal(bits(c(indices)),bits(m(indices)))
        assert not c(indices).requires_grad

@pytest.mark.parametrize('padding_idx',[None,0,-1])
def test_padding_and_recipe_roundtrip(padding_idx):
    m=source(padding_idx=padding_idx).eval();c=ConstantRowEmbedding.from_embedding(m)
    recipe=json.loads(json.dumps(c.recipe()));restored=ConstantRowEmbedding.from_recipe(recipe)
    restored.load_state_dict(c.state_dict(),strict=True)
    assert restored.padding_idx==m.padding_idx
    assert not restored.training
    indices=torch.arange(9)
    assert torch.equal(bits(restored(indices)),bits(m(indices)))
    assert list(restored.state_dict())==['_row']

def outcome(fn,x):
    try:return ('ok',fn(x))
    except Exception as exc:return (type(exc).__name__,str(exc))

@pytest.mark.parametrize('indices',[torch.tensor([-1]),torch.tensor([9]),torch.tensor([0.0]),None,[0,1]])
def test_original_index_errors(indices):
    m=source();c=ConstantRowEmbedding.from_embedding(m)
    assert outcome(m,indices)==outcome(c,indices)

@pytest.mark.parametrize('change',['trainable','max_norm','sparse','nonfinite','signedzero','unequal','hook','backward_hook','forward_override','subclass'])
def test_reject_noneligible(change):
    m=source()
    if change=='trainable':m.requires_grad_(True)
    elif change=='max_norm':m.max_norm=1.0
    elif change=='sparse':m.sparse=True
    elif change=='nonfinite':m.weight.data[:,0]=float('nan')
    elif change=='signedzero':m.weight.data[2,1]=0.0
    elif change=='unequal':m.weight.data[2,2]=1.0
    elif change=='hook':m.register_forward_hook(lambda *args:None)
    elif change=='backward_hook':m.register_full_backward_hook(lambda *args:None)
    elif change=='forward_override':m.forward=lambda x:x
    else:
        class Custom(nn.Embedding):pass
        m=Custom.from_pretrained(m.weight,freeze=True)
    with pytest.raises(ValueError):ConstantRowEmbedding.from_embedding(m)

def test_model_aliases_and_trainable_retained():
    model=nn.ModuleDict({'one':source(),'two':source(),'trainable':nn.Embedding(9,4)})
    model['two'].weight=model['one'].weight
    changed,report=deduplicate_embeddings(model)
    assert len(report['layers'])==3 and all(x['status']=='retained' for x in report['layers'])
    assert changed['one'].weight is changed['two'].weight
    m=source();aliases=nn.ModuleDict({'one':m,'two':m})
    changed,report=deduplicate_embeddings(aliases)
    assert report['layers'][0]['status']=='retained'
    assert changed['one'] is changed['two']

def test_model_rewrite_no_source_mutation_and_root():
    m=nn.Sequential(source(),nn.Identity());changed,report=deduplicate_embeddings(m)
    assert type(m[0]) is nn.Embedding and isinstance(changed[0],ConstantRowEmbedding)
    assert report['parameters_before']==36 and report['parameters_after']==4
    changed,report=deduplicate_embeddings(source(),inplace=True)
    assert isinstance(changed,ConstantRowEmbedding)

def test_distinct_parameters_shared_storage_buffers_and_ancestor_hooks_retained():
    model=nn.ModuleDict({'one':source(),'two':source()})
    model['two'].weight=nn.Parameter(model['one'].weight.view_as(model['one'].weight),requires_grad=False)
    assert model['one'].weight is not model['two'].weight
    changed,report=deduplicate_embeddings(model)
    assert all(x['status']=='retained' for x in report['layers'])
    model=nn.ModuleDict({'one':source()})
    model.register_buffer('alias',model['one'].weight.detach())
    _,report=deduplicate_embeddings(model)
    assert report['layers'][0]['status']=='retained'
    model=nn.Sequential(source());model.register_forward_hook(lambda *args:None)
    _,report=deduplicate_embeddings(model)
    assert report['layers'][0]['status']=='retained'

@pytest.mark.parametrize('field,value',[('padding_idx',True),('padding_idx',99),('norm_type',float('inf')),('norm_type','2'),('scale_grad_by_freq',1)])
def test_invalid_recipe_metadata(field,value):
    spec=ConstantRowEmbedding.from_embedding(source()).recipe();spec[field]=value
    with pytest.raises(ValueError):ConstantRowEmbedding.from_recipe(spec)

def test_frozen_contract_reject_unfreeze():
    c=ConstantRowEmbedding.from_embedding(source()).train()
    with pytest.raises(ValueError,match='frozen'):c.requires_grad_(True)
    c._row.requires_grad_(True)
    with pytest.raises(ValueError,match='frozen'):c(torch.tensor([1]))

@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS unavailable')
@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
def test_mps_values_errors_deepcopy_and_recipe(dtype):
    m=source(dtype,device='mps').eval();c=ConstantRowEmbedding.from_embedding(m)
    for indices in (torch.tensor([8,0,1],device='mps'),torch.empty((2,0),dtype=torch.int32,device='mps')):
        assert torch.equal(bits(c(indices)),bits(m(indices)))
    invalid=torch.tensor([0.0],device='mps')
    assert outcome(m,invalid)==outcome(c,invalid)
    restored=ConstantRowEmbedding.from_recipe(c.recipe(),device='mps')
    restored.load_state_dict(copy.deepcopy(c.state_dict()),strict=True)
    assert torch.equal(bits(restored.weight),bits(m.weight))
