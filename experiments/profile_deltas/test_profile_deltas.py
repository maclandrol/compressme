import json
import numpy as np
import pytest
import torch
from safetensors.torch import save_file, load_file
from profile_deltas import XorProfileLookup, bytes_equal, registered_bytes


def tables():
    rng = np.random.default_rng(448)
    base = rng.integers(0, 2**32, (7, 512), dtype=np.uint32)
    xor = rng.integers(0, 256, base.shape, dtype=np.uint32)
    # Every high bit remains representable; include zero sign, exponent carry,
    # infinities and distinct NaN payloads without invoking float arithmetic.
    base[0, :5] = [0x00000000, 0x80000000, 0x3fffffff, 0x7f800000, 0x7fc12345]
    target_bits = [0x80000000, 0x00000000, 0x40000000, 0x7fc12345, 0xffc23456]
    xor[0, :5] = base[0, :5] ^ target_bits
    target = base ^ xor
    return [torch.from_numpy(base.view(np.float32).copy()), torch.from_numpy(target.view(np.float32).copy())]


@pytest.mark.parametrize('bits', [8, 16, 'auto'])
def test_arbitrary_float_bits_outliers_empty_multidimensional_and_noncontiguous(bits):
    source = tables()
    candidate = XorProfileLookup(source, low_bits=bits)
    assert registered_bytes(candidate) < sum(t.numel() * t.element_size() for t in source)
    for shape in [(), (1,), (7,), (2, 3), (2, 3, 4), (0,), (2, 0, 3)]:
        ids = torch.randint(0, 7, shape)
        for route in (0, 1):
            assert bytes_equal(candidate(ids, route=route), source[route][ids])
    ids = torch.tensor([[0, 6, 2], [5, 1, 3]]).t()
    assert not ids.is_contiguous()
    assert bytes_equal(candidate(ids, route=1), source[1][ids])
    assert candidate.base.dtype == torch.float32
    with pytest.raises((IndexError, RuntimeError)):
        candidate(torch.tensor([7]), route=1)


def test_identical_profile_is_free_and_incompressible_profile_falls_back():
    a = torch.zeros(5, 32)
    b = torch.full_like(a, -1)
    c = XorProfileLookup([a, a.clone(), b])
    assert [r.mode for r in c.routes] == ['same', 'same', 'full']
    assert registered_bytes(c) == a.numel() * a.element_size() * 2
    assert bytes_equal(c(torch.arange(5), route=2), b)


def test_source_storage_is_not_retained_and_every_buffer_byte_is_counted():
    source = tables()
    c = XorProfileLookup(source)
    old = {t.untyped_storage()._cdata for t in source}
    assert not old.intersection(t.untyped_storage()._cdata for t in c.buffers())
    assert registered_bytes(c) == sum(t.numel() * t.element_size() for t in c.state_dict().values())
    assert len(list(c.parameters())) == 0


def test_safe_portable_recipe_and_state_reload(tmp_path):
    source = tables()
    c = XorProfileLookup(source)
    spec = json.loads(json.dumps(c.recipe()))
    save_file(c.state_dict(), str(tmp_path / 'model.safetensors'))
    restored = XorProfileLookup.from_recipe(spec)
    restored.load_state_dict(load_file(str(tmp_path / 'model.safetensors')), strict=True)
    assert registered_bytes(c) == registered_bytes(restored)
    for route in (0, 1):
        assert bytes_equal(c(torch.arange(7), route=route), restored(torch.arange(7), route=route))


def test_dtype_change_training_or_gradient_state_are_refused():
    c = XorProfileLookup(tables())
    with pytest.raises(ValueError, match='training'):
        c.requires_grad_(True)
    c.train()
    with pytest.raises(ValueError):
        c(torch.tensor(0))
    c.eval()
    c.base.requires_grad_(True)
    with pytest.raises(ValueError):
        c(torch.tensor(0))
    c.base.requires_grad_(False)
    c.half()
    with pytest.raises(ValueError):
        c(torch.tensor(0))


def test_column_above_uint16_uses_int32_without_truncation():
    a = torch.zeros(2, 65537)
    bits = a.view(torch.int32).clone()
    bits[:, -1] = -2147483648
    b = bits.view(torch.float32)
    c = XorProfileLookup([a, b])
    assert c.routes[1].columns.dtype == torch.int32
    assert bytes_equal(c(torch.arange(2), route=1), b)
