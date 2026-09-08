import hashlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import zstandard as zstd

import packing_files as stream
from compressme.packing import pack_bytes, unpack_bytes


def frame(raw, *, rle=False):
    # Valid small single-segment frames are independently checked by libzstd.
    assert len(raw) < 256
    payload = raw[:1] if rle else raw
    block = 1 | ((1 if rle else 0) << 1) | (len(raw) << 3)
    result = b'\x28\xb5\x2f\xfd\x20' + bytes([len(raw)]) + block.to_bytes(3, 'little') + payload
    assert zstd.ZstdDecompressor().decompress(result) == raw
    return result


def packed(raw, encoded_frame):
    return stream._HEADER.pack(stream.MAGIC, 1, 1, 1, 1, 64, len(raw), hashlib.sha256(raw).digest()) + encoded_frame


@pytest.mark.parametrize('raw,rle', [(b'', False), (b'abc\x00\xff', False), (b'x' * 200, True)])
def test_raw_rle_and_no_checksum_frames(tmp_path, raw, rle):
    encoded = packed(raw, frame(raw, rle=rle))
    source, target = tmp_path / 'in', tmp_path / 'out'
    source.write_bytes(encoded)
    result = stream.unpack_file(source, target, max_output_bytes=len(raw))
    assert target.read_bytes() == raw and result['sha256'] == hashlib.sha256(raw).hexdigest()
    assert unpack_bytes(encoded) == raw


def test_rle_truncation_and_reserved_block_refused(tmp_path):
    encoded = packed(b'x' * 200, frame(b'x' * 200, rle=True))
    source, target = tmp_path / 'in', tmp_path / 'out'
    damaged = bytearray(encoded)
    damaged[stream._HEADER.size + 6] |= 6
    for data in (encoded[:-1], bytes(damaged), encoded + b'\x00'):
        source.write_bytes(data)
        with pytest.raises(ValueError):
            stream.unpack_file(source, target)
        assert not target.exists()


def test_false_small_content_size_cannot_exceed_declared_output(tmp_path):
    raw = b'abcdef'
    encoded = bytearray(frame(raw))
    encoded[5] = len(raw) - 1
    source, target = tmp_path / 'in', tmp_path / 'out'
    source.write_bytes(packed(raw[:-1], bytes(encoded)))
    target.write_bytes(b'keep')
    with pytest.raises(ValueError):
        stream.unpack_file(source, target, max_output_bytes=len(raw)-1, overwrite=True)
    assert target.read_bytes() == b'keep'
    assert not list(tmp_path.glob('.out.*.tmp'))


def test_symlink_alias_refusal_and_other_symlink_replacement(tmp_path):
    source, target, other = tmp_path/'source', tmp_path/'target', tmp_path/'other'
    source.write_bytes(b'original')
    target.symlink_to(source)
    with pytest.raises(ValueError):
        stream.pack_file(source, target, overwrite=True)
    assert source.read_bytes() == b'original'
    target.unlink()
    other.write_bytes(b'untouched')
    target.symlink_to(other)
    stream.pack_file(source, target, overwrite=True)
    assert not target.is_symlink() and other.read_bytes() == b'untouched'
    assert unpack_bytes(target.read_bytes()) == b'original'


def test_observable_source_mutation_rejects_and_preserves_target(tmp_path):
    source, target = tmp_path/'source', tmp_path/'target'
    source.write_bytes(bytes(range(128)))
    target.write_bytes(b'keep')
    original = stream._shuffle_block
    edited = False
    def mutate(data, *args, **kwargs):
        nonlocal edited
        if not edited:
            edited = True
            with source.open('ab') as output:
                output.write(b'changed')
        return original(data, *args, **kwargs)
    with patch.object(stream, '_shuffle_block', mutate):
        with pytest.raises((ValueError, zstd.ZstdError)):
            stream.pack_file(source, target, block_size=64, overwrite=True)
    assert target.read_bytes() == b'keep'
    assert not list(tmp_path.glob('.target.*.tmp'))


def test_fsync_failure_before_publication_is_atomic(tmp_path):
    source, target = tmp_path/'source', tmp_path/'target'
    source.write_bytes(b'abc')
    target.write_bytes(b'keep')
    with patch.object(stream.os, 'fsync', side_effect=OSError('injected sync failure')):
        with pytest.raises(OSError, match='sync failure'):
            stream.pack_file(source, target, overwrite=True)
    assert target.read_bytes() == b'keep'
    assert not list(tmp_path.glob('.target.*.tmp'))


def test_postpublication_cleanup_error_identifies_committed_destination(tmp_path):
    source, target = tmp_path/'source', tmp_path/'target'
    source.write_bytes(b'abc')
    original = Path.unlink
    def fail_private(path, *args, **kwargs):
        if path.name.startswith('.target.') and path.name.endswith('.tmp'):
            raise PermissionError('injected private cleanup failure')
        return original(path, *args, **kwargs)
    with patch.object(Path, 'unlink', fail_private):
        with pytest.raises(RuntimeError, match='destination was committed'):
            stream.pack_file(source, target)
    assert unpack_bytes(target.read_bytes()) == b'abc'


def test_decoder_window_cap_remains_after_header_inspection(tmp_path):
    raw = b'x' * 2048
    ordinary = pack_bytes(raw, group_size=1, block_size=64)
    # A libzstd-verified ordinary frame with the same declared output but a
    # larger explicit (non-single-segment) history window.
    larger_window = (b'\x28\xb5\x2f\xfd\x40\x10' + (len(raw)-256).to_bytes(2, 'little')
                     + (1 | (len(raw) << 3)).to_bytes(3, 'little') + raw)
    assert zstd.get_frame_parameters(larger_window).window_size == 4096
    assert zstd.ZstdDecompressor().decompress(larger_window) == raw
    source, target = tmp_path/'source', tmp_path/'target'
    source.write_bytes(ordinary)
    inspect = stream._inspect_frame
    def replace_after_inspection(incoming, *args, **kwargs):
        answer = inspect(incoming, *args, **kwargs)
        source.write_bytes(packed(raw, larger_window))
        incoming.seek(0, os.SEEK_END)  # Drop the old BufferedReader window.
        return answer
    with patch.object(stream, '_inspect_frame', replace_after_inspection):
        with pytest.raises(ValueError, match='Zstandard payload') as caught:
            stream.unpack_file(source, target, max_window_bytes=2048)
    assert 'too much memory' in str(caught.value.__cause__)
    assert not target.exists()
