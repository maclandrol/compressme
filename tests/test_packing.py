import importlib
import struct

import pytest
np = pytest.importorskip("numpy")
zstd = pytest.importorskip("zstandard")

# Root should change this import to compressme.packing when integrating.
from compressme.packing import _HEADER, pack_bytes, unpack_bytes


@pytest.mark.parametrize("group", [1, 2, 4, 8])
@pytest.mark.parametrize("length", [0, 1, 3, 4, 17, 1025, 65539])
def test_arbitrary_bytes_roundtrip_and_partial_groups(group, length):
    data = np.random.default_rng(length).integers(0, 256, length, dtype=np.uint8).tobytes()
    packed = pack_bytes(data, group_size=group, block_size=1024)
    assert unpack_bytes(packed, max_output_bytes=length) == data


def test_float_bit_patterns_are_preserved_without_numeric_conversion():
    bits = np.array([0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00123,
                     0x7fa45678, 1, 0xffffffff], dtype="<u4").tobytes()
    assert unpack_bytes(pack_bytes(bits)) == bits


def test_size_bound_and_frame_size_are_checked_before_decode():
    packed = pack_bytes(b"a" * 4096)
    with pytest.raises(ValueError, match="max_output_bytes"):
        unpack_bytes(packed, max_output_bytes=4095)
    fields = list(_HEADER.unpack_from(packed))
    fields[-2] = 1
    forged = _HEADER.pack(*fields) + packed[_HEADER.size:]
    with pytest.raises(ValueError, match="sizes disagree"):
        unpack_bytes(forged)


def test_checksum_truncation_and_trailing_frames_rejected():
    packed = pack_bytes(b"molecular model" * 100)
    fields = list(_HEADER.unpack_from(packed))
    fields[-1] = b"\x00" * 32
    with pytest.raises(ValueError, match="SHA256"):
        unpack_bytes(_HEADER.pack(*fields) + packed[_HEADER.size:])
    with pytest.raises(ValueError):
        unpack_bytes(packed[:-1])
    with pytest.raises(ValueError):
        unpack_bytes(packed + b"unconsumed junk")
    with pytest.raises(ValueError):
        unpack_bytes(packed + packed[_HEADER.size:])


@pytest.mark.parametrize("offset", [0, 8, 9, 10, 11])
def test_unsupported_header_values_rejected(offset):
    packed = bytearray(pack_bytes(b"abc"))
    packed[offset] = 255
    with pytest.raises(ValueError):
        unpack_bytes(packed)


def test_optional_dependencies_fail_only_when_codec_used(monkeypatch):
    import builtins
    real_import = builtins.__import__
    def missing_zstd(name, *args, **kwargs):
        if name == "zstandard":
            raise ImportError("deliberately missing")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", missing_zstd)
    with pytest.raises(ImportError, match=r"compressme\[packing\]"):
        pack_bytes(b"abc")


@pytest.mark.parametrize("options", [dict(group_size=0), dict(group_size=3),
                                   dict(block_size=5), dict(block_size=1<<25),
                                   dict(level=23), dict(level=True)])
def test_invalid_packing_options_rejected(options):
    with pytest.raises(ValueError):
        pack_bytes(b"abc", **options)


def test_empty_payload_and_explicit_zero_limit():
    assert unpack_bytes(pack_bytes(b""), max_output_bytes=0) == b""
    with pytest.raises(ValueError):
        unpack_bytes(pack_bytes(b"x"), max_output_bytes=-1)


@pytest.mark.parametrize("suffix", [b"extra", zstd.ZstdCompressor().compress(b""),
                                    zstd.ZstdCompressor().compress(b"another frame")])
def test_empty_frame_rejects_trailing_bytes_and_frames(suffix):
    with pytest.raises(ValueError, match="extra data"):
        unpack_bytes(pack_bytes(b"") + suffix, max_output_bytes=0)


@pytest.mark.parametrize("checksum_byte", range(1, 5))
def test_empty_frame_checksum_is_actually_validated(checksum_byte):
    packed = bytearray(pack_bytes(b""))
    packed[-checksum_byte] ^= 1
    # The outer SHA256 remains correct for empty output; only the Zstd checksum
    # exposes this corruption. A zero-size one-shot decoder can miss it.
    with pytest.raises(ValueError, match="Zstandard payload"):
        unpack_bytes(packed, max_output_bytes=0)


def test_empty_frame_rejects_every_payload_truncation():
    packed = pack_bytes(b"")
    for end in range(_HEADER.size, len(packed)):
        with pytest.raises(ValueError):
            unpack_bytes(packed[:end], max_output_bytes=0)


@pytest.mark.parametrize("frame", [
    # Standard single-segment, checksum-free empty raw and RLE blocks.
    bytes.fromhex("28b52ffd2000010000"),
    bytes.fromhex("28b52ffd200003000000"),
    # Two empty raw blocks rather than one: last-block position matters.
    bytes.fromhex("28b52ffd2000000000010000"),
    # Non-single-segment empty frame with an explicit 1 MiB history window.
    bytes.fromhex("28b52ffd805000000000010000"),
])
def test_other_valid_empty_frame_encodings_remain_compatible(frame):
    assert unpack_bytes(pack_bytes(b"")[:_HEADER.size] + frame,
                        max_output_bytes=0) == b""


def test_forged_empty_frame_uses_bounded_stream_read(monkeypatch):
    import compressme.packing as packing
    calls = []
    original = zstd.ZstdDecompressor

    class BoundedReader:
        def __init__(self, reader):
            self.reader = reader
        def __enter__(self):
            self.reader.__enter__()
            return self
        def __exit__(self, *args):
            return self.reader.__exit__(*args)
        def read(self, size=-1):
            assert size == 1
            calls.append(size)
            return self.reader.read(size)

    class StreamOnlyDecoder:
        def __init__(self, **kwargs):
            self.decoder = original(**kwargs)
        def stream_reader(self, *args, **kwargs):
            return BoundedReader(self.decoder.stream_reader(*args, **kwargs))
        def decompress(self, *args, **kwargs):
            raise AssertionError("Must not use the zero-size one-shot shortcut")

    monkeypatch.setattr(zstd, "ZstdDecompressor", StreamOnlyDecoder)
    # A false zero content size followed by an RLE block advertising 128 KiB.
    # Neither outer/frame size agreement nor a zero output cap makes it safe.
    frame = (bytes.fromhex("28b52ffd2000")
             + ((128 << 10) << 3 | 2).to_bytes(3, "little") + b"x"
             + bytes.fromhex("010000"))
    with pytest.raises(ValueError):
        packing.unpack_bytes(pack_bytes(b"")[:_HEADER.size] + frame,
                             max_output_bytes=0)
    assert calls == [1]


def test_forged_empty_frame_history_is_bounded_before_decode(monkeypatch):
    def no_decoder(*args, **kwargs):
        raise AssertionError("Oversized history must be refused before decode")
    monkeypatch.setattr(zstd, "ZstdDecompressor", no_decoder)
    # A zero content size does not prevent a declared 2 GiB decoder window.
    frame = bytes.fromhex("28b52ffd80a800000000010000")
    with pytest.raises(ValueError, match="window"):
        unpack_bytes(pack_bytes(b"")[:_HEADER.size] + frame,
                     max_output_bytes=0)


def test_packed_model_export_replaces_plain_and_reloads_identically(tmp_path):
    import torch
    from compressme import compile_affine, load
    def factory():
        return torch.nn.Sequential(torch.nn.Linear(8,64),torch.nn.Linear(64,12)).eval()
    result = compile_affine(factory())
    result.save(tmp_path)
    result.save(tmp_path,packing=True)
    assert not (tmp_path/"model.safetensors").exists()
    assert (tmp_path/"model.cmppack").is_file()
    restored = load(factory,tmp_path).model
    x = torch.randn(10,8)
    assert torch.equal(restored(x),result.model(x))
