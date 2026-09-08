import sys
import copy
import torch
from torch import nn
from torch.nn import functional as F
import pytest
import compressme.finite_lookup as finite


def make_source(dtype=torch.float32, norm=False):
    torch.manual_seed(25)
    layers = [nn.Embedding(19, 7, dtype=dtype)]
    if norm:
        layers.append(finite.RowNormalize())
    layers += [nn.Linear(7, 128, dtype=dtype), nn.LayerNorm(128, dtype=dtype), nn.SiLU(),
               nn.Sequential(nn.Dropout(.8), nn.Linear(128, 9, dtype=dtype))]
    return nn.Sequential(*layers).eval()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('normalize', [False, True])
def test_full_domain_scalar_empty_multiaxis_indices(dtype, normalize):
    source = make_source(dtype, normalize)
    result = finite.compile_finite_lookup(source, input_contract='token_indices_only', chunk_size=8, validation_chunk_size=5)
    assert result.report['validation']['accepted']
    assert result.report['validation']['tokens_checked'] == 19
    assert result.report['tensor_bytes_after'] < result.report['tensor_bytes_before']
    assert all(not p.requires_grad for p in result.model.parameters())
    for ids in [torch.arange(19), torch.tensor(7), torch.empty((3, 0), dtype=torch.long),
                torch.tensor([[0, 18], [3, 9]], dtype=torch.int32).t()]:
        with torch.inference_mode():
            torch.testing.assert_close(result.model(ids), source(ids), atol=1e-5, rtol=1e-5)
    with pytest.raises((IndexError, RuntimeError)):
        result.model(torch.tensor([19]))


def test_constant_outputs_use_one_row_and_keep_full_index_domain():
    source = make_source()
    with torch.no_grad(): source[0].weight.zero_()
    result = finite.compile_finite_lookup(source, input_contract='token_indices_only')
    assert result.report['constant_output_rows']
    assert result.report['stored_rows'] == 1
    assert result.model.weight.shape == (19, 9)
    assert sum(p.numel() for p in result.model.parameters()) == 9


def test_no_storage_saving_retains_source():
    source = nn.Sequential(nn.Embedding(100, 2), nn.Linear(2, 20)).eval()
    result = finite.compile_finite_lookup(source, input_contract='token_indices_only')
    assert result.report['status'] == 'retained_no_byte_saving'
    assert type(result.model) is nn.Sequential


@pytest.mark.parametrize('bad', [nn.Softmax(dim=0), nn.Flatten(), nn.BatchNorm1d(7),
                               nn.LayerNorm((19, 7)), nn.MultiheadAttention(7, 1)])
def test_cross_token_and_unproved_operations_refused(bad):
    source = nn.Sequential(nn.Embedding(19, 7), bad).eval()
    with pytest.raises(ValueError):
        finite.compile_finite_lookup(source, input_contract='token_indices_only')


def test_training_hooks_custom_forward_and_external_alias_refused():
    source = make_source().train()
    with pytest.raises(ValueError, match='evaluation'):
        finite.compile_finite_lookup(source, input_contract='token_indices_only')
    source.eval()
    hook = source[1].register_forward_hook(lambda *args: None)
    with pytest.raises(ValueError, match='Hooks'):
        finite.compile_finite_lookup(source, input_contract='token_indices_only')
    hook.remove()
    source[1].forward = lambda x: x
    with pytest.raises(ValueError, match='custom forward'):
        finite.compile_finite_lookup(source, input_contract='token_indices_only')
    source = make_source()
    owner = nn.ModuleDict({'block': source, 'outside': source[1]})
    with pytest.raises(ValueError, match='external consumers'):
        finite.compile_finite_lookup(source, input_contract='token_indices_only', owner_model=owner)


def test_numeric_gate_rejects_perturbed_lookup(monkeypatch):
    source = make_source()
    original = finite.FiniteTokenLookup.forward
    monkeypatch.setattr(finite.FiniteTokenLookup, 'forward', lambda self, ids: original(self, ids) + 1)
    result = finite.compile_finite_lookup(source, input_contract='token_indices_only')
    assert result.report['status'] == 'rejected_and_rolled_back'
    assert type(result.model) is nn.Sequential


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('bias', [False, True])
def test_projected_raw_and_normalized_lookup(dtype, bias):
    torch.manual_seed(211)
    embedding = nn.Embedding(23, 64, dtype=dtype).eval()
    linear = nn.Linear(64, 9, bias=bias, dtype=dtype).eval()
    with torch.no_grad():
        embedding.weight[0].zero_()
        embedding.weight[1] *= 1e-14
    result = finite.build_projected_lookup(embedding, linear,
        input_contract='raw_or_l2_normalized_token_indices', chunk_size=8, validation_chunk_size=5)
    assert result.report['validation']['accepted']
    assert result.report['parameters_after'] == 23 * 9 + 23 + (9 if bias else 0)
    for normalize in (False, True):
        for ids in [torch.arange(23), torch.tensor(0), torch.tensor([[0, 1], [22, 4]]), torch.empty((2, 0), dtype=torch.long)]:
            with torch.inference_mode():
                x = embedding(ids)
                if normalize:
                    x = F.normalize(x, p=2, dim=-1, eps=1e-12)
                torch.testing.assert_close(result.model(ids, normalize=normalize), linear(x), atol=1e-5, rtol=1e-5)


def test_lookup_refuses_training_and_unfreezing():
    result = finite.compile_finite_lookup(make_source(), input_contract='token_indices_only')
    with pytest.raises(ValueError): result.model.requires_grad_()
    result.model.train()
    with pytest.raises(ValueError): result.model(torch.tensor(0))


def test_portable_shape_recipes_and_ordinary_frozen_tensors():
    with torch.inference_mode():
        first = finite.compile_finite_lookup(make_source(), input_contract='token_indices_only').model
    assert not first._rows.is_inference()
    replay = finite.FiniteTokenLookup.from_recipe(first.recipe())
    replay.load_state_dict(first.state_dict(), strict=True)
    torch.testing.assert_close(first(torch.arange(19)), replay(torch.arange(19)), rtol=0, atol=0)
    embedding, linear = nn.Embedding(23, 64).eval(), nn.Linear(64, 9).eval()
    with torch.inference_mode():
        second = finite.build_projected_lookup(embedding, linear,
            input_contract='raw_or_l2_normalized_token_indices').model
    assert not second._rows.is_inference() and not second._norms.is_inference()
    replay = finite.ProjectedEmbeddingLookup.from_recipe(second.recipe())
    replay.load_state_dict(second.state_dict(), strict=True)
    for normalize in (False, True):
        torch.testing.assert_close(second(torch.arange(23), normalize), replay(torch.arange(23), normalize), rtol=0, atol=0)


def test_projected_no_saving_keeps_both_source_branches():
    embedding, linear = nn.Embedding(100, 2).eval(), nn.Linear(2, 20).eval()
    result = finite.build_projected_lookup(embedding, linear,
        input_contract='raw_or_l2_normalized_token_indices')
    assert result.report['status'] == 'retained_no_byte_saving'
    assert isinstance(result.model, finite.DualModeEmbeddingProjection)
    for normalize in (False, True):
        x = embedding(torch.arange(100))
        if normalize: x = F.normalize(x, dim=-1)
        torch.testing.assert_close(result.model(torch.arange(100), normalize), linear(x))


def test_external_storage_alias_and_autocast_are_refused():
    source = make_source()
    owner = nn.ModuleDict({'block': source, 'other': nn.Linear(7, 1)})
    owner['other'].weight = nn.Parameter(source[1].weight[:1])
    with pytest.raises(ValueError, match='storage has external aliases'):
        finite.compile_finite_lookup(source, input_contract='token_indices_only', owner_model=owner)
    with torch.autocast('cpu', dtype=torch.bfloat16), pytest.raises(ValueError, match='autocast'):
        finite.compile_finite_lookup(make_source(), input_contract='token_indices_only')


def test_embedding_metadata_cannot_narrow_the_real_token_domain():
    source = make_source()
    source[0].num_embeddings = 18  # Actual weight still has 19 valid rows.
    with torch.inference_mode():
        assert source(torch.tensor(18)).shape == (9,)
    with pytest.raises(ValueError, match='attributes and weight shape differ'):
        finite.compile_finite_lookup(source, input_contract='token_indices_only')
    embedding = nn.Embedding(5, 64).eval()
    embedding.num_embeddings = 4
    with pytest.raises(ValueError, match='attributes and weight shape differ'):
        finite.build_projected_lookup(embedding, nn.Linear(64, 9).eval(),
            input_contract='raw_or_l2_normalized_token_indices')


def test_validation_currentness_tracks_representation_mutations():
    result = finite.compile_finite_lookup(make_source(), input_contract='token_indices_only')
    model = result.model
    assert model.validation_is_current()
    with torch.no_grad(): model._rows.add_(.125)
    assert not model.validation_is_current()
    # Restoring values does not restore the old experimental seal by itself.
    with torch.no_grad(): model._rows.sub_(.125)
    assert not model.validation_is_current()


def test_validation_currentness_changes_on_dtype_settings_and_replay():
    model = finite.compile_finite_lookup(make_source(), input_contract='token_indices_only').model
    assert not copy.deepcopy(model).validation_is_current()
    replay = finite.FiniteTokenLookup.from_recipe(model.recipe())
    replay.load_state_dict(model.state_dict(), strict=True)
    assert not replay.validation_is_current()  # Saved gate is historical until explicitly revalidated.
    model.double()
    assert not model.validation_is_current()
    projected = finite.build_projected_lookup(nn.Embedding(23, 64).eval(), nn.Linear(64, 9).eval(),
        input_contract='raw_or_l2_normalized_token_indices').model
    assert projected.validation_is_current()
    projected.normalize_eps = 1e-6
    assert not projected.validation_is_current()
