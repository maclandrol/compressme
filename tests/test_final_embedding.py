"""Constructed-model tests; pretrained model gates are recorded separately."""
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from compressme.final_embedding import FinalEmbedding


def bits_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
        b.detach().cpu().contiguous().reshape(-1).view(torch.uint8))


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(12, 8)
        self.norm = nn.LayerNorm(8)
        self.config = SimpleNamespace(output_hidden_states=True, output_attentions=True)
        self.calls = []
    def forward(self, ids, attention_mask=None, output_hidden_states=None, return_dict=True, **kwargs):
        self.calls.append({'hidden':output_hidden_states, 'return_dict':return_dict, **kwargs})
        value = self.norm(self.embedding(ids))
        return {'last_hidden_state':value, 'hidden_states':(value, value) if output_hidden_states else None}


class MaskedLM(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder = Backbone(); self.head = nn.Linear(8, 12); self.head_calls = 0
    def forward(self, ids, **kwargs):
        out = self.encoder(ids, **kwargs); self.head_calls += 1
        return {'logits':self.head(out['last_hidden_state']), 'hidden_states':out['hidden_states']}


def test_exact_explicit_backbone_and_unmodified_config():
    model = MaskedLM().eval()
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad(): expected = model(ids, output_hidden_states=True)['hidden_states'][-1]
    wrapper = FinalEmbedding(model.encoder, output_contract='last_hidden_state')
    actual = wrapper(ids, output_attentions=True)
    assert bits_equal(expected, actual) and not actual.requires_grad
    assert model.head_calls == 1  # The final-only invocation did not compute logits.
    assert model.encoder.calls[-1] == {'hidden':False, 'return_dict':True, 'output_attentions':True}
    assert model.encoder.config.output_hidden_states is True
    assert model.encoder.config.output_attentions is True
    assert wrapper.backbone is model.encoder
    assert sum(p.numel() for p in wrapper.parameters()) < sum(p.numel() for p in model.parameters())


def test_contract_and_training_guards():
    with pytest.raises(ValueError): FinalEmbedding(Backbone(), output_contract='last_hidden_state')
    with pytest.raises(ValueError): FinalEmbedding(Backbone().eval(), output_contract='logits')
    with pytest.raises(TypeError): FinalEmbedding(object(), output_contract='last_hidden_state')
    wrapper = FinalEmbedding(Backbone().eval(), output_contract='last_hidden_state')
    with pytest.raises(ValueError): wrapper.train()
    ids = torch.tensor([[1]])
    with pytest.raises(ValueError): wrapper(ids, output_hidden_states=True)
    with pytest.raises(ValueError): wrapper(ids, return_dict=False)
    wrapper.backbone.norm.train()
    with pytest.raises(ValueError): wrapper(ids)


def test_no_silent_fallback_to_another_output():
    class Wrong(nn.Module):
        def forward(self, *args, **kwargs): return {'hidden_states':(torch.ones(1),)}
    with pytest.raises(TypeError, match='last_hidden_state'):
        FinalEmbedding(Wrong().eval(), output_contract='last_hidden_state')(torch.tensor([1]))


@pytest.mark.parametrize('attention', ['eager', 'sdpa'])
def test_tiny_real_esm_final_state_is_byte_identical(attention):
    transformers = pytest.importorskip('transformers')
    from transformers import EsmConfig, EsmForMaskedLM
    with torch.random.fork_rng():
        torch.manual_seed(6193)
        cfg = EsmConfig(vocab_size=33, hidden_size=16, num_hidden_layers=2,
                        num_attention_heads=4, intermediate_size=32, pad_token_id=1,
                        mask_token_id=32, max_position_embeddings=64,
                        position_embedding_type='rotary', token_dropout=True)
        cfg._attn_implementation = attention
        original = EsmForMaskedLM(cfg).eval()
    cfg_before = original.config.to_dict()
    wrapper = FinalEmbedding(original.esm, output_contract='last_hidden_state')
    calls = []
    hook = original.lm_head.register_forward_hook(lambda *args: calls.append(1))
    for ids in (torch.tensor([[0, 4, 5, 6, 2]]),
                torch.tensor([[0, 4, 32, 2, 1], [0, 9, 7, 8, 2]])):
        mask = ids.ne(1)
        with torch.no_grad(): expected = original(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
        count = len(calls)
        got = wrapper(input_ids=ids, attention_mask=mask)
        assert bits_equal(expected, got)
        assert len(calls) == count
    hook.remove()
    assert original.config.to_dict() == cfg_before
    assert original.esm.config._attn_implementation == attention

