import copy
import hashlib
import json

import pytest
import torch
from torch import nn
from safetensors.torch import load_file, save_file

from compressme import XorProfileLookup, pack_lookup_tables, load
from compressme.compiler import CompressionResult, state_bytes
from compressme.frozen_tables import bytes_equal, registered_bytes


def table_family():
    generator = torch.Generator().manual_seed(448)
    bits = torch.randint(-(1 << 31), 1 << 31, (7, 512), dtype=torch.int32, generator=generator)
    xor = torch.randint(0, 256, bits.shape, dtype=torch.int32, generator=generator)
    before = [0, -(1 << 31), 0x3fffffff, 0x7f800000, 0x7fc12345]
    after = [-(1 << 31), 0, 0x40000000, 0x7fc12345, -4058026]
    bits[0, :5] = torch.tensor(before, dtype=torch.int32)
    xor[0, :5] = torch.bitwise_xor(bits[0, :5], torch.tensor(after, dtype=torch.int32))
    return [bits.view(torch.float32), torch.bitwise_xor(bits, xor).view(torch.float32)]


@pytest.mark.parametrize('bits', [8, 16, 'auto'])
def test_exact_float_bits_signed_zero_exponent_nan_payloads_and_all_id_shapes(bits):
    source = table_family()
    result = pack_lookup_tables(source, low_bits=bits)
    candidate = result.model
    assert candidate.validation_is_current()
    assert result.report['storage_reduced']
    assert result.report['validation']['compared_values'] == 2 * 7 * 512
    assert not any('parameter' in key for key in result.report)
    assert result.report['tensor_bytes_after'] == state_bytes(candidate)
    assert result.report['resident_storage_bytes_after'] == registered_bytes(candidate)
    for shape in [(), (7,), (2, 3), (2, 3, 4), (0,), (2, 0, 3)]:
        ids = torch.randint(0, 7, shape)
        for route in (0, 1):
            assert bytes_equal(candidate(ids, route=route), source[route][ids])
    ids = torch.tensor([[0, 6, 2], [5, 1, 3]], dtype=torch.int32).t()
    assert not ids.is_contiguous()
    assert bytes_equal(candidate(ids, route=1), source[1][ids.long()])
    with pytest.raises((IndexError, RuntimeError)):
        candidate(torch.tensor([7]), route=1)


def test_source_noncontiguous_and_lazy_negative_views_resolve_without_retaining_storage():
    source = [torch._neg_view(tensor.t()) for tensor in table_family()]
    assert source[0].is_neg() and not source[0].is_contiguous()
    result = pack_lookup_tables(source)
    ids = torch.arange(source[0].shape[0])
    for route in range(2):
        assert bytes_equal(result.model(ids, route=route), source[route].resolve_neg())
    old = {tensor.untyped_storage()._cdata for tensor in source}
    assert not old.intersection(tensor.untyped_storage()._cdata for tensor in result.model.buffers())
    assert all(not tensor.is_neg() for tensor in result.model.buffers())


@pytest.mark.parametrize('alias_kind', ['identical', 'separate_view', 'expanded'])
def test_shared_source_backing_cannot_manufacture_a_resident_saving(alias_kind):
    base = torch.arange(128, dtype=torch.float32).reshape(4, 32)
    if alias_kind == 'identical':
        tables = [base, base]
    elif alias_kind == 'separate_view':
        tables = [base, base.detach().view_as(base)]
    else:
        base = torch.ones((1, 32)).expand(4, 32)
        tables = [base, base]
    result = pack_lookup_tables(tables)
    assert result.report['resident_storage_bytes_before'] == base.untyped_storage().nbytes()
    assert result.report['logical_storage_reduced']
    assert not result.report['resident_storage_reduced']
    assert not result.report['storage_reduced']
    assert result.report['validation']['accepted']


def test_unique_source_storage_includes_view_padding_and_full_fallback_counts():
    backing = torch.zeros(16, 32)
    a, b = backing[:4], torch.full((4, 32), -1.)
    result = pack_lookup_tables([a, b])
    assert [route.mode for route in result.model.routes] == ['same', 'full']
    assert result.report['tensor_bytes_before'] == result.report['tensor_bytes_after']
    assert result.report['resident_storage_bytes_before'] == backing.untyped_storage().nbytes() + b.untyped_storage().nbytes()
    assert result.report['resident_storage_reduced']
    assert not result.report['storage_reduced']
    assert bytes_equal(result.model(torch.arange(4), route=1), b)


def test_int32_exception_columns_above_uint16_are_not_truncated():
    base = torch.zeros(2, 65537)
    target = base.view(torch.int32).clone()
    target[:, -1] = -(1 << 31)
    result = pack_lookup_tables([base, target.view(torch.float32)])
    assert result.model.routes[1].columns.dtype == torch.int32
    assert bytes_equal(result.model(torch.arange(2), route=1), target.view(torch.float32))


@pytest.mark.parametrize('tables, kwargs', [([], {}), ([1, 2], {}),
    ([torch.ones(2, 3, dtype=torch.float64)], {}), ([torch.ones(2, 3, device='meta')], {}),
    ([torch.ones(0, 3)], {}), ([torch.ones(2, 3), torch.ones(3, 2)], {}),
    ([torch.ones(2, 3)], {'base_route': True}), ([torch.ones(2, 3)], {'low_bits': 8.0}),
    ([torch.ones(2, 3)], {'low_bits': torch.tensor(8)})])
def test_input_schema_rejects_unsupported_table_families(tables, kwargs):
    with pytest.raises(ValueError):
        XorProfileLookup(tables, **kwargs)


def test_frozen_mode_dtype_autocast_and_input_contract_guards():
    candidate = pack_lookup_tables(table_family()).model
    for ids, route in [(torch.ones(2), 0), (torch.tensor(0), True), (torch.tensor(0), 2)]:
        with pytest.raises(ValueError):
            candidate(ids, route=route)
    with pytest.raises(ValueError):
        candidate.requires_grad_(True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        with pytest.raises(ValueError, match='autocast'):
            candidate(torch.tensor(0))
    candidate.routes[1].train()
    assert not candidate.validation_is_current()
    with pytest.raises(ValueError):
        candidate(torch.tensor(0))
    candidate.eval()
    candidate.base.requires_grad_(True)
    with pytest.raises(ValueError):
        candidate(torch.tensor(0))
    candidate.base.requires_grad_(False)
    candidate.half()
    with pytest.raises(ValueError):
        candidate(torch.tensor(0))


@pytest.mark.parametrize('damage', ['code_dtype', 'shape', 'negative_view', 'extra_buffer', 'base_route'])
def test_representation_mutations_invalidate_the_seal_and_cannot_export(damage, tmp_path):
    result = pack_lookup_tables(table_family())
    candidate = result.model
    if damage == 'code_dtype':
        candidate.routes[1].low = candidate.routes[1].low.float()
    elif damage == 'shape':
        candidate.base = candidate.base[:1]
    elif damage == 'negative_view':
        candidate.base = torch._neg_view(candidate.base)
    elif damage == 'extra_buffer':
        candidate.register_buffer('unrecorded', torch.ones(1))
    else:
        candidate.base_route = 1
    assert not candidate.validation_is_current()
    with pytest.raises(ValueError):
        candidate(torch.tensor(0))
    with pytest.raises(ValueError):
        result.save(tmp_path)


@pytest.mark.parametrize('hook', ['forward', 'instance', 'subclass', 'child_subclass'])
def test_modified_call_graph_cannot_be_exported(hook, tmp_path):
    result = pack_lookup_tables(table_family())
    if hook == 'forward':
        result.model.register_forward_hook(lambda *args: None)
    elif hook == 'instance':
        result.model.forward = lambda ids, route=0: ids
    elif hook == 'subclass':
        class Custom(XorProfileLookup):
            pass
        result.model.__class__ = Custom
    else:
        class CustomDelta(type(result.model.routes[1])):
            pass
        result.model.routes[1].__class__ = CustomDelta
    with pytest.raises(ValueError):
        result.save(tmp_path)


@pytest.mark.parametrize('packing', [False, True])
def test_public_root_recipe_reload_and_inference_context_version_tracking(tmp_path, packing):
    with torch.inference_mode():
        result = pack_lookup_tables(table_family())
    assert result.model.validation_is_current()
    assert all(not torch.is_inference(t) for t in result.model.buffers())
    result.save(tmp_path, packing=packing)
    restored = load(nn.Identity, tmp_path).model
    assert type(restored) is XorProfileLookup
    assert not restored.validation_is_current()
    assert registered_bytes(restored) == registered_bytes(result.model)
    for route in range(2):
        assert bytes_equal(result.model(torch.arange(7), route=route), restored(torch.arange(7), route=route))
    with torch.inference_mode():
        fresh = XorProfileLookup.from_recipe(result.model.recipe())
    assert all(not torch.is_inference(t) for t in fresh.buffers())
    fresh.load_state_dict(result.model.state_dict(), strict=True)
    assert not fresh.validation_is_current()


class Host(nn.Module):
    def __init__(self):
        super().__init__()
        self.lookup = nn.Identity()
        self.other = self.lookup
    def forward(self, ids):
        return {'direct': self.lookup(ids, route=0), 'alternate': self.other(ids, route=1), 'ids': ids}


def test_nested_model_replay_preserves_registered_module_alias(tmp_path):
    local = pack_lookup_tables(table_family())
    host = Host().eval()
    host.lookup = local.model
    host.other = host.lookup
    result = CompressionResult(host, {'method': 'explicit_table_attachment'})
    result.save(tmp_path, packing=True)
    restored = load(Host, tmp_path).model
    assert restored.lookup is restored.other
    ids = torch.tensor([[0, 5], [6, 1]])
    for name, value in host(ids).items():
        assert bytes_equal(value, restored(ids)[name])


def test_valid_tensor_mutation_saves_with_historical_comparison(tmp_path):
    result = pack_lookup_tables(table_family())
    result.model.routes[1].low.add_(1)
    assert not result.model.validation_is_current()
    result.save(tmp_path)
    report = json.loads((tmp_path / 'manifest.json').read_text())['report']
    assert report['historical_status'] == 'byte_verified'
    assert report['status'] == 'historical_or_invalidated_finite_lookup_validation'
    assert not report['validation']['accepted']


@pytest.mark.parametrize('damage', ['dtype_name', 'negative_shape', 'too_many_slots', 'base_not_same', 'training', 'extra_route_key'])
def test_closed_recipe_refuses_malformed_fields(damage):
    spec = pack_lookup_tables(table_family()).model.recipe()
    if damage == 'dtype_name':
        spec['routes'][1]['buffers']['low']['dtype'] = 8
    elif damage == 'negative_shape':
        spec['shape'] = [-1, 512]
    elif damage == 'too_many_slots':
        for key in ('columns', 'high'):
            spec['routes'][1]['buffers'][key]['shape'][1] = 513
    elif damage == 'base_not_same':
        spec['base_route'] = 1
    elif damage == 'training':
        spec['training'] = True
    else:
        spec['routes'][1]['unknown'] = 'unsupported'
    with pytest.raises(ValueError):
        XorProfileLookup.from_recipe(spec)


@pytest.mark.parametrize('damage', ['out_of_bounds', 'duplicate_nonzero', 'invalid_high'])
def test_loader_checks_exception_values_after_loading_safe_tensors(tmp_path, damage):
    result = pack_lookup_tables(table_family(), low_bits=8)
    result.save(tmp_path)
    state = load_file(str(tmp_path / 'model.safetensors'))
    if damage == 'out_of_bounds':
        state['routes.1.columns'][0, 0] = 512
    elif damage == 'duplicate_nonzero':
        state['routes.1.columns'][0, 0:2] = 0
        state['routes.1.high'][0, 0:2] = 1
    else:
        state['routes.1.high'][0, 0] = -1
    save_file(state, str(tmp_path / 'model.safetensors'))
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    manifest['weights_sha256'] = hashlib.sha256((tmp_path / 'model.safetensors').read_bytes()).hexdigest()
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load(nn.Identity, tmp_path)
