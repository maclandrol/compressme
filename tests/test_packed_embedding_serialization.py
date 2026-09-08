"""Portable packed-embedding recipes retain bytes, placement and alias ownership."""
import hashlib
import json

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from compressme.compiler import CompressionResult
from compressme.packed_embedding import PackedFrozenEmbedding
from compressme.serialization import load


def source_embedding():
    source = nn.Embedding(7, 13).eval().requires_grad_(False)
    bits = torch.randint(-(1 << 31), 1 << 31, source.weight.shape,
                         generator=torch.Generator().manual_seed(745), dtype=torch.int32)
    bits[0, :6] = torch.tensor([0, -(1 << 31), 0x7f800000, -8388608, 0x7fc00001, -4194303], dtype=torch.int32)
    source.weight.copy_(bits.view(torch.float32))
    return source


def exact_bytes(left, right):
    return torch.equal(left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                       right.detach().cpu().contiguous().reshape(-1).view(torch.uint8))


def packed_result():
    return CompressionResult(PackedFrozenEmbedding(source_embedding(), block_rows=3),
                             {'method': 'lossless_frozen_embedding_storage'})


@pytest.mark.parametrize('packing', [False, True])
@pytest.mark.parametrize('factory', [nn.Identity, source_embedding])
def test_root_recipe_without_pickle_preserves_arbitrary_float_bits(tmp_path, monkeypatch, packing, factory):
    result = packed_result()
    result.save(tmp_path, packing=packing)
    def forbidden(*args, **kwargs):
        raise AssertionError('Unpickling must not be used')
    monkeypatch.setattr(torch, 'load', forbidden)
    restored = load(factory, tmp_path).model
    assert type(restored) is PackedFrozenEmbedding
    assert restored.payload.device.type == restored.offsets.device.type == restored._anchor.device.type == 'cpu'
    assert not list(restored.parameters())
    assert exact_bytes(restored.weight, source_embedding().weight)
    for shape in [(), (4,), (2, 3), (0,), (2, 0, 3)]:
        ids = torch.randint(0, 7, shape)
        assert exact_bytes(result.model(ids), restored(ids))
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['packed_embedding_rewrites'][0]['kind'] == 'PackedFrozenEmbedding'
    assert manifest['packed_embedding_rewrites'][0]['version'] == 1
    assert not manifest['finite_lookup_rewrites']


class AliasedHost(nn.Module):
    def __init__(self):
        super().__init__()
        self.lookup = nn.Identity()
        self.alias = self.lookup
        self.register_buffer('offset', torch.tensor(2.))

    def forward(self, ids):
        return self.lookup(ids), self.alias(ids), self.offset


def test_nested_recipe_restores_module_alias_and_stores_one_payload(tmp_path):
    host = AliasedHost().eval()
    host.lookup = packed_result().model
    host.alias = host.lookup
    CompressionResult(host, {}).save(tmp_path)
    restored = load(AliasedHost, tmp_path).model
    assert restored.lookup is restored.alias
    state = load_file(str(tmp_path / 'model.safetensors'))
    assert 'lookup.payload' in state and 'alias.payload' not in state
    assert 'lookup.offsets' in state and 'alias.offsets' not in state
    ids = torch.tensor([[0, 6], [2, 1]], dtype=torch.int32).t()
    assert not ids.is_contiguous()
    for original, rebuilt in zip(host(ids), restored(ids)):
        assert exact_bytes(original, rebuilt)


def rewrite_manifest(tmp_path, edit):
    path = tmp_path / 'manifest.json'
    manifest = json.loads(path.read_text())
    edit(manifest)
    path.write_text(json.dumps(manifest))


def rewrite_tensors(tmp_path, edit):
    path = tmp_path / 'model.safetensors'
    state = load_file(str(path))
    edit(state)
    save_file(state, str(path))
    rewrite_manifest(tmp_path, lambda m: m.update(weights_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


@pytest.mark.parametrize('damage', ['checksum', 'offsets', 'payload_dtype', 'anchor_dtype'])
def test_loader_rejects_invalid_payload_even_with_updated_outer_hash(tmp_path, damage):
    packed_result().save(tmp_path)
    def mutate(state):
        if damage == 'checksum':
            state['payload'][-1] ^= 1
        elif damage == 'offsets':
            state['offsets'][1] = -1
        elif damage == 'payload_dtype':
            state['payload'] = state['payload'].to(torch.int8)
        else:
            state['_anchor'] = state['_anchor'].double()
    rewrite_tensors(tmp_path, mutate)
    with pytest.raises(ValueError):
        load(nn.Identity, tmp_path)


@pytest.mark.parametrize('damage', ['version', 'duplicate', 'not_list', 'training', 'dtype_alias'])
def test_recipe_metadata_and_aliases_are_checked(tmp_path, damage):
    packed_result().save(tmp_path)
    def mutate(m):
        if damage == 'version':
            m['packed_embedding_rewrites'][0]['version'] = 99
        elif damage == 'duplicate':
            m['packed_embedding_rewrites'] *= 2
        elif damage == 'not_list':
            m['packed_embedding_rewrites'] = {}
        elif damage == 'training':
            m['module_training'][''] = True
        else:
            m['tensor_aliases']['offsets'] = 'payload'
    rewrite_manifest(tmp_path, mutate)
    with pytest.raises(ValueError):
        load(nn.Identity, tmp_path)


def test_recipe_rejects_mismatched_original_embedding_dimensions(tmp_path):
    packed_result().save(tmp_path)
    with pytest.raises(ValueError, match='original architecture'):
        load(lambda: nn.Embedding(8, 13), tmp_path)


@pytest.mark.parametrize('damage', ['hook', 'forward', 'subclass', 'checksum'])
def test_save_rejects_unrepresentable_behavior_and_corrupt_bytes(tmp_path, damage):
    result = packed_result()
    if damage == 'hook':
        result.model.register_forward_hook(lambda *args: None)
    elif damage == 'forward':
        result.model.forward = lambda ids: ids
    elif damage == 'subclass':
        class Custom(PackedFrozenEmbedding):
            pass
        result.model.__class__ = Custom
    else:
        result.model.payload[-1] ^= 1
    with pytest.raises(ValueError):
        result.save(tmp_path)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS unavailable')
def test_mps_anchor_moves_without_moving_payload_and_reload_uses_factory_device(tmp_path):
    result = packed_result()
    result.model.to('mps')
    result.save(tmp_path)
    cpu = load(nn.Identity, tmp_path).model
    assert cpu._anchor.device.type == 'cpu'
    mps = load(lambda: source_embedding().to('mps'), tmp_path).model
    assert mps._anchor.device.type == 'mps'
    assert mps.payload.device.type == mps.offsets.device.type == 'cpu'
    ids = torch.tensor([0, 6, 3], device='mps')
    assert exact_bytes(result.model(ids), mps(ids))
    cpu.to('mps')
    assert cpu.payload.device.type == 'cpu' and exact_bytes(cpu(ids), mps(ids))
