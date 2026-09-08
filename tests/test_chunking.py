import copy
from functools import partial
import math

import pytest
import torch
from torch import nn

from compressme.chunking import prepare_single_chunk, single_chunk_or_fallback
from compressme import chunking
from compressme import nesso_runtime as runtime


def equal_bytes(a, b):
    return torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                       b.contiguous().reshape(-1).view(torch.uint8))


def original_chunk(layer, inputs, chunk_size, no_batch_dims, low_mem=False,
                   _out=None, _add_into_out=False):
    """Independent narrow reference for the tested tensor-output source route."""
    assert not low_mem
    batch = tuple(max(t.shape[i] for t in inputs.values()) for i in range(no_batch_dims))
    count = math.prod(batch)
    prepared = {}
    for key, value in inputs.items():
        if sum(value.shape[:no_batch_dims]) != no_batch_dims:
            value = value.expand(batch + value.shape[no_batch_dims:])
        prepared[key] = value.reshape(-1, *value.shape[no_batch_dims:])
    out = _out
    for start in range(0, count, chunk_size):
        current = {k: t[start:start + chunk_size] if t.shape[0] != 1 else t
                   for k, t in prepared.items()}
        value = layer(**current)
        if out is None:
            out = value.new_zeros((count,) + value.shape[1:])
        if _add_into_out:
            out[start:start + chunk_size] += value
        else:
            out[start:start + chunk_size] = value
    return out.view(batch + out.shape[1:])


@pytest.mark.parametrize('transposed', [False, True])
def test_preserves_flattened_broadcast_arguments_and_fresh_output_bytes(transposed):
    x = torch.randn(2, 3, 4, 4)
    if transposed:
        x = x.transpose(-2, -1)
    bias = torch.randn(1, 1, 4, 4)
    inputs = {'x': x, 'bias': bias}
    calls = []
    def layer(x, bias):
        calls.append([(tuple(t.shape), tuple(t.stride()), t.storage_offset(), t.clone())
                      for t in (x, bias)])
        return (x + bias).contiguous()
    stats = {}
    with torch.no_grad():
        expected = original_chunk(layer, inputs, 512, 2)
        actual = single_chunk_or_fallback(layer, inputs, 512, 2,
                    fallback=original_chunk, immutable_inference=True, statistics=stats)
    assert equal_bytes(expected, actual)
    assert expected.shape == actual.shape and expected.stride() == actual.stride()
    assert stats == {'single_chunk_elisions': 1}
    assert len(calls) == 2
    for old, new in zip(*calls):
        assert old[:3] == new[:3] and equal_bytes(old[3], new[3])
    assert calls[0][1][0][0] == 1  # Original all-singleton bias stays broadcasted.
    assert actual.untyped_storage()._cdata != x.untyped_storage()._cdata


@pytest.mark.parametrize('kind', ['input_alias', 'noncontiguous', 'broadcast'])
def test_output_fallback_copies_without_a_second_layer_call(kind):
    x = torch.randn(1, 3, 2, 2)
    seen = []
    def layer(x):
        seen.append(1)
        if kind == 'input_alias': return x
        if kind == 'noncontiguous': return x.transpose(-2, -1)
        return x[:1].clone()
    stats = {}
    with torch.no_grad():
        expected = original_chunk(layer, {'x': x}, 512, 2)
        actual = single_chunk_or_fallback(layer, {'x': x}, 512, 2,
                    fallback=original_chunk, immutable_inference=True, statistics=stats)
    assert len(seen) == 2 and stats == {'single_chunk_copies': 1}
    assert equal_bytes(expected, actual) and actual.is_contiguous()
    assert actual.untyped_storage()._cdata != x.untyped_storage()._cdata


def test_non_tensor_outputs_are_copied_once_and_keep_the_original_tree_semantics():
    x = torch.randn(1, 3, 2)
    calls = []
    def layer(x):
        calls.append(1)
        return {'left': x, 'nested': {'right': x.clone()}}
    with torch.no_grad():
        result = single_chunk_or_fallback(layer, {'x': x}, 32, 2,
                    fallback=lambda *a, **k: pytest.fail('No repeated dispatcher call'),
                    immutable_inference=True)
    assert len(calls) == 1
    assert equal_bytes(result['left'], x) and equal_bytes(result['nested']['right'], x)
    assert result['left'].untyped_storage()._cdata != x.untyped_storage()._cdata


@pytest.mark.parametrize('case', ['multiple', 'zero_batch', 'nested', 'no_contract',
                                 'gradient', 'low_mem', 'out', 'add'])
def test_unsupported_cases_call_the_original_dispatcher_once(case):
    x = torch.ones(1, 3, 2)
    args = dict(chunk_size=32, no_batch_dims=2, immutable_inference=True)
    inputs = {'x': x}
    if case == 'multiple': args['chunk_size'] = 2
    elif case == 'zero_batch': inputs = {'x': x[:, :0]}
    elif case == 'nested': inputs = {'nested': {'x': x}}
    elif case == 'no_contract': args['immutable_inference'] = False
    elif case == 'low_mem': args['low_mem'] = True
    elif case == 'out': args['_out'] = x.clone()
    elif case == 'add': args['_add_into_out'] = True
    calls = []
    sentinel = object()
    def fallback(*a, **kw):
        calls.append((a, kw))
        return sentinel
    with torch.set_grad_enabled(case == 'gradient'):
        result = single_chunk_or_fallback(lambda **kw: pytest.fail('Should not execute layer'),
                                          inputs, fallback=fallback, **args)
    assert result is sentinel and len(calls) == 1
    assert calls[0][0][1] is inputs


def test_module_owned_output_cannot_become_an_external_mutable_alias():
    class BufferOutput(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('value', torch.ones(3, 2))
        def forward(self, x): return self.value
    module = BufferOutput().eval()
    with torch.no_grad():
        result = single_chunk_or_fallback(module, {'x': torch.zeros(1, 3, 2)}, 32, 2,
                    fallback=original_chunk, immutable_inference=True)
    result.add_(3)
    assert torch.equal(module.value, torch.ones(3, 2))


class TinyESM(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_esm_all_layers = False
        self.esm_mlp = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4), nn.ReLU(), nn.Dropout(.05), nn.Linear(4, 4))
        self.s_inputs_proj = nn.Linear(4, 4)
        self.esm_z_1, self.esm_z_2 = nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False)
    def forward(self, z, s_inputs, s_esm, pair_mask, use_kernels=False):
        p = self.esm_mlp(s_esm)
        s = self.s_inputs_proj(s_inputs) + p
        left, right = self.esm_z_1(s), self.esm_z_2(s)
        delta = left[:, :, None, :] + right[:, None, :, :]
        delta = delta * pair_mask.unsqueeze(-1)
        return z + delta


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4)
    def forward(self, q_x, kv_x, tri_bias, mask_bias, mask, use_kernels=False):
        return self.projection(q_x)


class TinyTriangle(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = TinyAttention()
    def _chunk(self, x, tri_bias, mask_bias, mask, chunk_size, use_kernels=False):
        return original_chunk(partial(self.mha, use_kernels=use_kernels),
            dict(q_x=x, kv_x=x, tri_bias=tri_bias, mask_bias=mask_bias, mask=mask), chunk_size, 2)


class TinyNesso(nn.Module):
    def __init__(self):
        super().__init__()
        self.esm_module = TinyESM()
        self.triangle = TinyTriangle()
    def forward(self, s, e, crop=False, fail=False):
        z = s.new_zeros(1, s.shape[1], s.shape[1], 4)
        mask = s.new_ones(1, s.shape[1], s.shape[1])
        for step in range(3):
            z = self.esm_module(z, s, e, mask)
            z = self.triangle._chunk(z, z[:, :1], z[:, :1], z[:, :1], 512)
            # Preserve source-like eval RNG consumption; no dropout elision.
            z = z * (torch.rand_like(z[:, :, :1, :1]) > 0)
            if crop and step == 0:
                z, s, e, mask = z[:, :2, :2].contiguous(), s[:, :2].contiguous(), e[:, :2].contiguous(), mask[:, :2, :2].contiguous()
            if fail and step == 1:
                raise RuntimeError('original computation failed')
        return z


@pytest.fixture
def tiny_scope(monkeypatch):
    monkeypatch.setattr(runtime, '_verified_types', lambda: (
        TinyNesso, TinyESM, (TinyTriangle,), TinyAttention, original_chunk))
    return TinyNesso().eval().requires_grad_(False)


@pytest.mark.parametrize('crop', [False, True])
def test_request_scope_preserves_values_rng_state_and_state_keys(tiny_scope, crop):
    model = tiny_scope
    s, e = torch.randn(1, 4, 4), torch.randn(1, 4, 4)
    keys = list(model.state_dict())
    with torch.no_grad():
        torch.manual_seed(912)
        expected = model(s, e, crop=crop)
        rng = torch.get_rng_state()
        torch.manual_seed(912)
        with runtime.nesso_inference_optimizations(model, cache_esm=True, immutable_request=True) as report:
            actual = model(s, e, crop=crop)
            assert list(model.state_dict()) == keys
            assert torch.equal(torch.get_rng_state(), rng)
    assert equal_bytes(expected, actual)
    assert report['single_chunk_elisions'] == 3
    assert report['esm_computations'] == (2 if crop else 1)
    assert report['esm_reuses'] == (1 if crop else 2)
    assert report['peak_cached_bytes'] == 1 * 4 * 4 * 4 * 4
    assert 'forward' not in vars(model) and 'forward' not in vars(model.esm_module)
    assert '_chunk' not in vars(model.triangle)
    assert not hasattr(model, '_compressme_nesso_runtime_owner')


def test_exception_clears_request_cache_and_restores_methods(tiny_scope):
    model = tiny_scope
    s, e = torch.randn(1, 4, 4), torch.randn(1, 4, 4)
    with torch.no_grad():
        with runtime.nesso_inference_optimizations(model, cache_esm=True, immutable_request=True) as report:
            with pytest.raises(RuntimeError, match='original computation failed'):
                model(s, e, fail=True)
            model(s, e)
        assert report['requests'] == 2 and report['esm_computations'] == 2
    assert 'forward' not in vars(model) and '_chunk' not in vars(model.triangle)


def test_guard_rejects_mutated_model_and_scope_restores_on_exit(tiny_scope):
    model = tiny_scope
    with torch.no_grad():
        with runtime.nesso_inference_optimizations(model, immutable_request=True):
            model.esm_module.s_inputs_proj.weight.add_(1)
            with pytest.raises(ValueError, match='state changed'):
                model(torch.randn(1, 3, 4), torch.randn(1, 3, 4))
    assert 'forward' not in vars(model) and '_chunk' not in vars(model.triangle)


def test_requires_explicit_contract_and_frozen_no_grad_mode(tiny_scope):
    with pytest.raises(ValueError, match='immutable_request'):
        with runtime.nesso_inference_optimizations(tiny_scope): pass
    with pytest.raises(ValueError, match='no-grad'):
        with runtime.nesso_inference_optimizations(tiny_scope, immutable_request=True): pass
    with torch.no_grad():
        tiny_scope.train()
        with pytest.raises(ValueError, match='evaluation'):
            with runtime.nesso_inference_optimizations(tiny_scope, immutable_request=True): pass


@pytest.mark.parametrize('transposed', [False, True])
def test_prepared_dispatch_preserves_dynamic_layout_and_audits_owner_once(monkeypatch, transposed):
    module = nn.Linear(4, 4).eval().requires_grad_(False)
    audits, owner_reads = [], []
    audit_owner, storage = chunking._audit_owner, chunking._storage
    state_ids = {id(t) for t in module.parameters()}
    def audit(layer):
        audits.append(layer)
        return audit_owner(layer)
    def read(tensor):
        if id(tensor) in state_ids:
            owner_reads.append(id(tensor))
        return storage(tensor)
    monkeypatch.setattr(chunking, '_audit_owner', audit)
    monkeypatch.setattr(chunking, '_storage', read)
    stats = {}
    with torch.no_grad():
        dispatch = prepare_single_chunk(module, fallback=original_chunk,
                    immutable_inference=True, statistics=stats)
        assert len(audits) == 1 and sorted(owner_reads) == sorted(state_ids)
        for count in (1, 3, 2):
            x = torch.randn(1, count, 4, 4)
            if transposed:
                x = x.transpose(-2, -1)
            expected = original_chunk(module, {'input': x}, 32, 2)
            actual = dispatch({'input': x}, 32, 2)
            assert equal_bytes(actual, expected) and actual.stride() == expected.stride()
        assert len(audits) == 1 and sorted(owner_reads) == sorted(state_ids)
    assert stats == {'single_chunk_elisions': 3}


def test_prepared_still_copies_owned_buffer_without_reexecuting_layer():
    class BufferOutput(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('value', torch.ones(3, 2))
            self.calls = 0
        def forward(self, x):
            self.calls += 1
            return self.value
    module = BufferOutput().eval()
    with torch.no_grad():
        dispatch = prepare_single_chunk(module, fallback=original_chunk, immutable_inference=True)
        for _ in range(2):
            result = dispatch({'x': torch.zeros(1, 3, 2)}, 32, 2)
            result.add_(1)
            assert torch.equal(module.value, torch.ones(3, 2))
    assert module.calls == 2


@pytest.mark.parametrize('case', ['multiple', 'nested', 'gradient', 'low_mem', 'out', 'add'])
def test_prepared_dispatch_retains_dynamic_fallbacks(case):
    calls = []
    sentinel = object()
    def fallback(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel
    inputs = {'x': torch.ones(1, 3, 2)}
    flags = {}
    count = 2 if case == 'multiple' else 32
    if case == 'nested': inputs = {'nested': inputs}
    if case == 'low_mem': flags['low_mem'] = True
    if case == 'out': flags['_out'] = torch.ones(3, 2)
    if case == 'add': flags['_add_into_out'] = True
    with torch.no_grad():
        dispatch = prepare_single_chunk(lambda **k: pytest.fail('Should use fallback'),
                     fallback=fallback, immutable_inference=True)
    with torch.set_grad_enabled(case == 'gradient'):
        assert dispatch(inputs, count, 2, **flags) is sentinel
    assert len(calls) == 1 and calls[0][0][1] is inputs


@pytest.mark.parametrize('mode', ['training', 'hook', 'compiled', 'trainable'])
def test_prepared_unsafe_owner_uses_original_route(mode):
    module = nn.Linear(2, 2).eval().requires_grad_(False)
    if mode == 'training': module.train()
    if mode == 'hook': module.register_forward_hook(lambda *args: None)
    if mode == 'compiled': module._compiled_call_impl = lambda *args: None
    if mode == 'trainable': module.weight.requires_grad_(True)
    calls = []
    def fallback(*args, **kwargs):
        calls.append(1)
        return 'original'
    with torch.no_grad():
        dispatch = prepare_single_chunk(module, fallback=fallback, immutable_inference=True)
        assert dispatch({'input': torch.ones(1, 3, 2)}, 32, 2) == 'original'
        assert dispatch({'input': torch.ones(1, 3, 2)}, 32, 2) == 'original'
    assert len(calls) == 2


def test_preparation_requires_explicit_lifetime_contract_and_no_grad():
    with torch.no_grad(), pytest.raises(ValueError, match='immutable_inference'):
        prepare_single_chunk(lambda x: x, fallback=original_chunk)
    with torch.enable_grad(), pytest.raises(ValueError, match='no-grad'):
        prepare_single_chunk(lambda x: x, fallback=original_chunk, immutable_inference=True)


@pytest.mark.parametrize('target', ['mlp', 'projection'])
def test_esm_reuse_rejects_custom_determinism_changing_children(tiny_scope, target):
    class StochasticLinear(nn.Linear):
        def forward(self, x): return super().forward(x) + torch.rand_like(x)
    if target == 'mlp':
        tiny_scope.esm_module.esm_mlp[1] = StochasticLinear(4, 4).eval().requires_grad_(False)
    else:
        tiny_scope.esm_module.s_inputs_proj = StochasticLinear(4, 4).eval().requires_grad_(False)
    with torch.no_grad(), pytest.raises(ValueError, match='exact ESMModule children'):
        with runtime.nesso_inference_optimizations(tiny_scope, cache_esm=True, immutable_request=True):
            pytest.fail('Unsafe children admitted')
    assert 'forward' not in vars(tiny_scope)


def test_runtime_rejects_compiled_child_without_installing_overrides(tiny_scope):
    tiny_scope.triangle.mha.projection._compiled_call_impl = lambda *args: None
    with torch.no_grad(), pytest.raises(ValueError, match='compiled'):
        with runtime.nesso_inference_optimizations(tiny_scope, immutable_request=True):
            pytest.fail('Compiled child admitted')
    assert 'forward' not in vars(tiny_scope) and '_chunk' not in vars(tiny_scope.triangle)


def test_runtime_prepares_each_attention_owner_once_per_scope(monkeypatch, tiny_scope):
    calls = []
    prepare = runtime.prepare_single_chunk
    def spy(*args, **kwargs):
        calls.append(args[0])
        return prepare(*args, **kwargs)
    monkeypatch.setattr(runtime, 'prepare_single_chunk', spy)
    with torch.no_grad():
        with runtime.nesso_inference_optimizations(tiny_scope, immutable_request=True) as report:
            for _ in range(2):
                tiny_scope(torch.randn(1, 4, 4), torch.randn(1, 4, 4))
    assert len(calls) == 1 and report['single_chunk_elisions'] == 6
