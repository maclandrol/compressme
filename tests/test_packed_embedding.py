import copy
import pytest
import torch
from torch import nn
from compressme.packed_embedding import PackedFrozenEmbedding


def equal_bits(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
        b.detach().cpu().contiguous().reshape(-1).view(torch.uint8)))


@pytest.mark.parametrize('rows', [1, 4, 16])
def test_exact_all_bit_patterns_and_shapes(rows):
    generator = torch.Generator().manual_seed(172)
    bits = torch.randint(-(2**31), 2**31 - 1, (35, 32), dtype=torch.int32, generator=generator)
    bits[0, :6] = torch.tensor([0, -(2**31), 0x7fc00001, 0x7f800000, -0x00800000, 1])
    source = nn.Embedding.from_pretrained(bits.view(torch.float32), freeze=True).eval()
    packed = PackedFrozenEmbedding(source, block_rows=rows)
    for ids in (torch.tensor(0), torch.arange(35), torch.arange(24).reshape(4, 6)[:, ::2], torch.empty(2, 0, dtype=torch.long)):
        assert equal_bits(source(ids), packed(ids))
    assert equal_bits(source.weight, packed.weight)
    clone = PackedFrozenEmbedding.from_recipe(packed.recipe())
    clone.load_state_dict(packed.state_dict(), strict=True)
    clone.validate_payload()
    assert equal_bits(source.weight, clone.weight)
    assert equal_bits(source.weight, copy.deepcopy(packed).weight)


def test_guards_and_corruption():
    source = nn.Embedding(10, 3).eval().requires_grad_(False)
    with pytest.raises(ValueError):
        PackedFrozenEmbedding(source.train())
    source.eval()
    packed = PackedFrozenEmbedding(source)
    with pytest.raises(ValueError): packed.requires_grad_(True)
    with pytest.raises(ValueError): packed.half()
    assert packed._anchor.dtype == torch.float32
    with pytest.raises(IndexError): packed(torch.tensor([-1]))
    with pytest.raises(IndexError): packed(torch.tensor([10]))
    with pytest.raises(RuntimeError): packed(torch.tensor([1.0]))
    packed.payload[-1] ^= 1
    with pytest.raises(ValueError): packed(torch.tensor([0]))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS unavailable')
def test_mps_roundtrip():
    source = nn.Embedding(35, 64).eval().requires_grad_(False).to('mps')
    packed = PackedFrozenEmbedding(source, block_rows=4)
    ids = torch.arange(24, device='mps').reshape(4, 6)[:, ::2]
    assert equal_bits(source(ids), packed(ids))
    assert packed.payload.device.type == packed.offsets.device.type == 'cpu'
    assert equal_bits(source.cpu()(ids.cpu()), packed.cpu()(ids.cpu()))
    assert equal_bits(source.to('mps')(ids), packed.to('mps')(ids))
