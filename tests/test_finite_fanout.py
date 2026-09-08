import copy
import json
import pytest
import torch
from torch import nn
from safetensors.torch import load_file, save_file
from compressme.finite_fanout import (FiniteTokenFanout, FiniteFanoutLookup,
                                     Float32RMSNorm, compile_finite_fanout)
from compressme.finite_lookup import _resident_storage_bytes


def source(dtype=torch.float32):
    torch.manual_seed(331)
    return FiniteTokenFanout(nn.Embedding(7, 24, dtype=dtype),
        nn.Sequential(Float32RMSNorm(torch.randn(24, dtype=dtype))),
        {name: nn.Linear(24, 24, bias=False, dtype=dtype) for name in ('q', 'k', 'v')},
        residual_key='residual').eval()


def compile_source(value, **kwargs):
    return compile_finite_fanout(value, input_contract='token_indices_only',
                                 chunk_size=3, validation_chunk_size=2, **kwargs)


def test_multiple_outputs_residual_all_index_shapes_and_storage():
    original = source()
    result = compile_source(original)
    assert result.report['status'] == 'accepted_on_full_domain_validation'
    candidate = result.model
    assert isinstance(candidate, FiniteFanoutLookup) and candidate.validation_is_current()
    assert result.report['candidate_stored_values'] == 7 * 24 * 4
    assert result.report['candidate_resident_storage_bytes'] == 7 * 24 * 4 * 4
    assert len(candidate.tables) == 1
    old_storage = {p.untyped_storage()._cdata for p in original.parameters()}
    assert not old_storage.intersection(p.untyped_storage()._cdata for p in candidate.parameters())
    for shape in [(), (7,), (2, 7), (2, 3, 7), (0,), (2, 0, 3)]:
        ids = torch.randint(0, 7, shape)
        actual, expected = candidate(ids), original(ids)
        assert list(actual) == ['q', 'k', 'v', 'residual']
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], atol=1e-5, rtol=1e-5)
        assert torch.equal(actual['residual'], original.embedding(ids))
    with pytest.raises(ValueError, match='integer token'):
        candidate(torch.ones(2, 24))


def test_declared_float32_rms_formula_preserves_cast_order():
    weight = torch.tensor([.3, 1.4, -2.], dtype=torch.float64)
    norm = Float32RMSNorm(weight, eps=.01)
    x = torch.tensor([[1.000000003, -.700000002, 2.000000007]], dtype=torch.float64)
    values = x.float()
    expected = weight * (values * torch.rsqrt(values.square().mean(-1, keepdim=True)+.01)).double()
    assert torch.equal(norm(x), expected)
    input_precision = weight * x * torch.rsqrt(x.square().mean(-1, keepdim=True)+.01)
    assert not torch.equal(norm(x), input_precision)


def test_mixed_dtypes_pack_separately_without_upcasting_residual():
    original = source()
    original.embedding.half()
    result = compile_source(original)
    assert result.report['status'] == 'accepted_on_full_domain_validation'
    assert {p.dtype for p in result.model.parameters()} == {torch.float16, torch.float32}
    output = result.model(torch.tensor([[0, 1, 2]]))
    assert output['residual'].dtype == torch.float16
    assert output['q'].dtype == torch.float32


def test_shared_source_storage_and_duplicate_output_tables_are_counted_once():
    original = source()
    original.branches['k'].weight = nn.Parameter(original.branches['q'].weight.detach())
    original.branches['v'] = original.branches['q']
    result = compile_source(original)
    assert result.report['status'] == 'accepted_on_full_domain_validation'
    # One embedding, one normalization scale, and one backing projection matrix.
    assert result.report['source_resident_storage_bytes'] == 4 * (7*24 + 24 + 24*24)
    assert result.report['unique_output_tables'] == 2
    assert result.report['candidate_resident_storage_bytes'] == 4 * 7 * 24 * 2
    output = result.model(torch.tensor([0, 1]))
    assert torch.equal(output['q'], output['k'])
    before = output['k'].clone()
    output['q'].add_(1)
    assert torch.equal(output['k'], before)


def test_constant_output_and_identity_residual_share_compact_storage():
    original = FiniteTokenFanout(nn.Embedding(9, 32), nn.Sequential(),
        {'identity': nn.Identity(), 'zero': nn.Linear(32, 2)}, residual_key='residual').eval()
    with torch.no_grad():
        original.branches['zero'].weight.zero_()
        original.branches['zero'].bias.zero_()
    result = compile_source(original)
    assert result.report['status'] == 'accepted_on_full_domain_validation'
    assert result.report['unique_output_tables'] == 2
    assert result.report['candidate_stored_values'] == 9*32+2
    assert torch.equal(result.model(torch.tensor([8, 0]))['zero'], torch.zeros(2, 2))
    with pytest.raises((IndexError, RuntimeError)):
        result.model(torch.tensor([9]))


def test_shared_storage_external_to_owner_is_refused():
    original = source()
    owner = nn.Module()
    owner.span = original
    owner.external = nn.Parameter(original.branches['q'].weight.detach())
    with pytest.raises(ValueError, match='external aliases'):
        compile_source(original, owner_model=owner)


def test_unsupported_custom_code_is_refused_before_execution():
    class UnprovedNorm(nn.Module):
        called = False
        def forward(self, x):
            self.called = True
            raise AssertionError('Must never run')
    original = source()
    unknown = UnprovedNorm().eval()
    original.shared_prefix = nn.Sequential(unknown).eval()
    with pytest.raises(ValueError, match='Unproved'):
        compile_source(original)
    assert not unknown.called


def test_custom_inspection_and_mode_methods_are_not_used_to_validate_a_leaf():
    class Unproved(nn.Module):
        def __getattr__(self, name):
            if name == 'inplace':
                raise AssertionError('Custom attributes must not establish audited grammar')
            return super().__getattr__(name)
        def train(self, mode=True):
            raise AssertionError('Do not mode-switch a custom operation before refusing it')
    original = source()
    unknown = Unproved()
    unknown.training = False
    prefix = nn.Sequential(unknown)
    prefix.training = False
    original.shared_prefix = prefix
    with pytest.raises(ValueError, match='Unproved'):
        compile_source(original)


def test_audit_never_mode_switches_unused_custom_descendants():
    class Unused(nn.Module):
        called = False
        def train(self, mode=True):
            self.called = True
            raise AssertionError('Auditing must not execute unused descendant methods')
    original = source()
    child = Unused()
    # Leave training=True so the initial audit refuses it without calling it.
    original.branches['q'].unused = child
    with pytest.raises(ValueError, match='evaluation'):
        compile_source(original)
    assert not child.called
    child.training = False
    result = compile_source(original)
    assert result.report['status'] == 'accepted_on_full_domain_validation'
    assert not child.called


@pytest.mark.parametrize('mutation', ['inplace', 'shape', 'training', 'hooks', 'rms_shape'])
def test_unsafe_source_mutations_are_explicitly_refused(mutation):
    original = source()
    if mutation == 'inplace':
        original.shared_prefix.append(nn.ReLU(inplace=True).eval())
    elif mutation == 'shape':
        original.embedding.weight = nn.Parameter(torch.randn(8, 24))
    elif mutation == 'training':
        original.branches['q'].train()
    elif mutation == 'hooks':
        original.branches['q'].register_forward_hook(lambda *args: None)
    elif mutation == 'rms_shape':
        original.shared_prefix[0].weight = nn.Parameter(torch.ones(23))
    with pytest.raises(ValueError):
        compile_source(original)


def test_no_storage_saving_retains_the_complete_source_contract():
    original = FiniteTokenFanout(nn.Embedding(100, 4), nn.Sequential(),
        {'expand': nn.Linear(4, 16)}, residual_key='raw').eval()
    result = compile_source(original)
    assert result.report['status'] == 'no_storage_saving'
    assert isinstance(result.model, FiniteTokenFanout)
    for name, output in original(torch.tensor([3, 4])).items():
        assert torch.equal(result.model(torch.tensor([3, 4]))[name], output)


def test_strict_numerical_rejection_does_not_return_a_partial_rewrite():
    original = source()
    result = compile_source(original, absolute_tolerance=0, relative_tolerance=0)
    assert result.report['status'] == 'rejected_numerical_validation'
    assert isinstance(result.model, FiniteTokenFanout)
    for before, after in zip(original.state_dict().values(), result.model.state_dict().values()):
        assert torch.equal(before, after)


def test_recipe_safetensors_reload_and_validation_seal(tmp_path):
    result = compile_source(source())
    original = result.model
    spec = json.loads(json.dumps(original.recipe()))
    save_file(original.state_dict(), tmp_path/'weights.safetensors')
    rebuilt = FiniteFanoutLookup.from_recipe(spec)
    rebuilt.load_state_dict(load_file(tmp_path/'weights.safetensors'), strict=True)
    assert not rebuilt.validation_is_current()
    ids = torch.tensor([[0, 4, 6]])
    for name, value in original(ids).items():
        assert torch.equal(value, rebuilt(ids)[name])
    assert _resident_storage_bytes(original) == _resident_storage_bytes(rebuilt)
    with torch.inference_mode():
        inference_built = FiniteFanoutLookup.from_recipe(spec)
    inference_built.load_state_dict(load_file(tmp_path/'weights.safetensors'), strict=True)
    assert all(not torch.is_inference(p) for p in inference_built.parameters())
    with torch.no_grad():
        next(iter(original.tables.values())).add_(1)
    assert not original.validation_is_current()
    for field, value in [('input_contract', 'arbitrary_embeddings'), ('version', 2), ('training', True)]:
        bad = copy.deepcopy(spec); bad[field] = value
        with pytest.raises(ValueError):
            FiniteFanoutLookup.from_recipe(bad)


def test_contract_autocast_and_trainability_guards():
    original = source()
    with pytest.raises(ValueError, match='contract'):
        compile_finite_fanout(original, input_contract='any_inputs')
    candidate = compile_source(original).model
    with pytest.raises(ValueError, match='training'):
        candidate.requires_grad_()
    candidate.train()
    with pytest.raises(ValueError, match='frozen inference'):
        candidate(torch.tensor([1]))
    candidate.eval()
    with torch.autocast('cpu', dtype=torch.bfloat16), pytest.raises(ValueError, match='autocast'):
        candidate(torch.tensor([1]))


@pytest.mark.skipif(not hasattr(nn, 'RMSNorm'), reason='Native RMSNorm unavailable')
def test_standard_nn_rmsnorm_is_a_separate_audited_operation():
    original = source()
    original.shared_prefix = nn.Sequential(nn.RMSNorm(24, eps=1e-5)).eval()
    assert compile_source(original).report['status'] == 'accepted_on_full_domain_validation'


@pytest.mark.parametrize('packing', [False, True])
def test_package_save_reload_mixed_dtypes_and_historical_validation(tmp_path, packing):
    from compressme import load
    original = source()
    original.embedding.half()
    result = compile_source(original)
    assert isinstance(result.model, FiniteFanoutLookup)
    destination = tmp_path / 'fanout'
    result.save(destination, packing=packing)
    manifest = json.loads((destination / 'manifest.json').read_text())
    assert manifest['finite_lookup_rewrites'][0]['kind'] == 'FiniteFanoutLookup'
    assert manifest['source_tensor_dtypes']['embedding.weight'] == 'float16'
    assert manifest['source_tensor_dtypes']['branches.q.weight'] == 'float32'
    rebuilt = load(source, destination).model
    assert not rebuilt.validation_is_current()
    assert _resident_storage_bytes(rebuilt) == _resident_storage_bytes(result.model)
    for name, value in result.model(torch.tensor([[1, 3, 5]])).items():
        assert torch.equal(value, rebuilt(torch.tensor([[1, 3, 5]]))[name])
        assert value.dtype == rebuilt(torch.tensor([[1, 3, 5]]))[name].dtype
    with torch.no_grad():
        next(iter(result.model.tables.values())).add_(1)
    result.save(destination, packing=packing)
    changed = json.loads((destination / 'manifest.json').read_text())['report']
    assert changed['status'] == 'historical_or_invalidated_finite_lookup_validation'
    assert changed['historical_status'] == 'accepted_on_full_domain_validation'


def test_source_view_backing_allocation_does_not_hide_checkpoint_growth():
    original = FiniteTokenFanout(nn.Embedding(30, 2), nn.Sequential(),
        {'wide': nn.Linear(2, 20)}, residual_key='residual').eval()
    # A small view pins an oversized allocation. Saving resident storage alone
    # would make the exported logical checkpoint larger than its source.
    backing = torch.zeros(10000)
    with torch.no_grad():
        backing[:60].copy_(original.embedding.weight.flatten())
    original.embedding.weight = nn.Parameter(backing[:60].view(30, 2))
    result = compile_source(original)
    assert result.report['candidate_resident_storage_bytes'] < result.report['source_resident_storage_bytes']
    assert result.report['status'] == 'no_storage_saving'
    assert result.report['proposal_tensor_bytes_after'] > result.report['tensor_bytes_before']
    assert result.report['parameters_after'] == result.report['parameters_before']


@pytest.mark.parametrize('change', ['hook', 'forward'])
def test_lookup_custom_behavior_invalidates_gate_and_cannot_silently_export(tmp_path, change):
    result = compile_source(source())
    assert result.model.validation_is_current()
    if change == 'hook':
        result.model.register_forward_hook(lambda module, args, outputs: {k: v + 100 for k, v in outputs.items()})
    else:
        original = result.model.forward
        result.model.forward = lambda ids: {k: v + 100 for k, v in original(ids).items()}
    assert not result.model.validation_is_current()
    with pytest.raises(ValueError, match='cannot preserve'):
        result.save(tmp_path / 'changed')
    assert not (tmp_path / 'changed').exists()
