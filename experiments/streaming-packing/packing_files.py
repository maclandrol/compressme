"""Streaming file API for the unchanged CMPRPACK v1 byte-shuffle/Zstandard codec.

No tensor interpretation, no model-memory reduction, and no inference speed claim.
Input files must remain immutable while used. Observable size/timestamp changes
are rejected. Destinations are committed atomically only after complete success.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from functools import lru_cache
import hmac
import os
from pathlib import Path
import stat
import struct
import tempfile

MAGIC = b'CMPRPACK'
VERSION = 1
_HEADER = struct.Struct('>8sBBBBIQ32s')
_DEFAULT_BLOCK_BYTES = 1 << 20
_MAX_BLOCK_BYTES = 16 << 20
DEFAULT_MAX_OUTPUT_BYTES = 1 << 30
DEFAULT_MAX_WINDOW_BYTES = 64 << 20
_IO_BYTES = 128 << 10


def _dependencies():
    try:
        import numpy as np
        import zstandard as zstd
    except ImportError as exc:
        raise ImportError("Install compressme[packing] (numpy and zstandard).") from exc
    return np, zstd


def _positive_integer(value, name, *, allow_zero=False):
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f'{name} must be a {"nonnegative" if allow_zero else "positive"} integer')


def _layout(group, block):
    if type(group) is not int or group not in (1, 2, 4, 8):
        raise ValueError('group_size must be 1, 2, 4 or 8')
    if type(block) is not int or not group <= block <= _MAX_BLOCK_BYTES or block % group:
        raise ValueError('block_size must be divisible by group_size and at most 16 MiB')


def _shuffle_block(data, group, np, *, inverse=False):
    if group == 1 or not data:
        return data
    usable = len(data) // group * group
    if not usable:
        return data
    array = np.frombuffer(memoryview(data)[:usable], dtype=np.uint8)
    view = array.reshape(group, -1).T if inverse else array.reshape(-1, group).T
    return view.tobytes() + data[usable:]


@contextmanager
def _source_file(path):
    # O_NONBLOCK prevents a POSIX FIFO path from waiting for a writer before fstat.
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NONBLOCK', 0)
    fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError('Source must be a regular file')
        handle = os.fdopen(fd, 'rb')
        fd = None
        with handle:
            yield handle
    finally:
        if fd is not None:
            os.close(fd)


@lru_cache(maxsize=2)
def _window_unit_bytes(zstd):
    # python-zstandard 0.25 passes bytes to libzstd, despite a KiB docstring.
    # Verify actual enforcement with a fixed 2 KiB frame rather than guessing
    # future/backend behavior from a version string. This never reads user data.
    payload = b'x' * 2048
    frame = zstd.ZstdCompressor(level=1).compress(payload)
    if zstd.get_frame_parameters(frame).window_size != 2048:
        raise RuntimeError('Cannot establish Zstandard window-limit semantics')

    def accepts(cap):
        try:
            with zstd.ZstdDecompressor(max_window_size=cap).stream_reader(frame) as reader:
                return reader.read(1) + reader.read(2048) == payload and reader.read(1) == b''
        except zstd.ZstdError:
            return False

    if not accepts(1024) and accepts(2048):
        return 1
    if accepts(1024) and not accepts(1) and accepts(2):
        return 1024
    raise RuntimeError('Unsupported Zstandard window-limit enforcement')


def _file_token(handle):
    info = os.fstat(handle.fileno())
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('Source must be a regular file')
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _read_exact(handle, count):
    buffer = bytearray()
    remaining = count
    while remaining:
        part = handle.read(remaining)
        if not part:
            raise ValueError('Truncated packed data')
        buffer.extend(part)
        remaining -= len(part)
    return bytes(buffer)


def _paths(source, destination, overwrite):
    if type(overwrite) is not bool:
        raise TypeError('overwrite must be a bool')
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError('Source and destination must be distinct files')
    if os.path.lexists(destination):
        if not overwrite:
            raise FileExistsError(destination)
        try:
            if os.path.samefile(source, destination):
                raise ValueError('Source and destination must be distinct files')
        except FileNotFoundError:
            pass
    return source, destination


@contextmanager
def _transaction(destination, overwrite):
    fd, name = tempfile.mkstemp(prefix='.' + destination.name + '.', suffix='.tmp', dir=destination.parent)
    temporary = Path(name)
    published = False
    primary_error = None
    try:
        with os.fdopen(fd, 'w+b') as output:
            yield output
            output.flush()
            os.fsync(output.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            # Atomic no-clobber publication, including a concurrently created target.
            os.link(temporary, destination)
        published = True
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if os.path.lexists(temporary):
            try:
                temporary.unlink()
            except OSError as cleanup_error:
                if primary_error is not None:
                    if hasattr(primary_error, 'add_note'):
                        primary_error.add_note('Private temporary-file cleanup also failed: ' + type(cleanup_error).__name__)
                elif published:
                    raise RuntimeError('Verified destination was committed, but private temporary-file cleanup failed') from cleanup_error
                else:
                    raise


def pack_file(source, destination, *, level=9, group_size=4,
              block_size=_DEFAULT_BLOCK_BYTES, overwrite=False):
    """Stream an ordinary file to CMPRPACK v1, compatible with unpack_bytes.

    Python payload buffers are bounded by block_size, rather than file size.
    Zstandard additionally retains its level-dependent compression workspace;
    level 9 uses an approximately 11 MiB context for a 2 GB input in zstd 0.25.
    The encoded file may be larger than the source. Source/destination differ.
    """
    _layout(group_size, block_size)
    if type(level) is not int or not -5 <= level <= 22:
        raise ValueError('level must be an integer between -5 and 22')
    source, destination = _paths(source, destination, overwrite)
    np, zstd = _dependencies()
    digest = hashlib.sha256()
    with _source_file(source) as incoming:
        initial = _file_token(incoming)
        raw_size = initial[2]
        with _transaction(destination, overwrite) as output:
            output.write(b'\0' * _HEADER.size)
            compressor = zstd.ZstdCompressor(level=level, write_content_size=True,
                                             write_checksum=True, threads=0)
            count = 0
            with compressor.stream_writer(output, size=raw_size, closefd=False,
                                          write_size=_IO_BYTES) as writer:
                while True:
                    block = incoming.read(block_size)
                    if not block:
                        break
                    count += len(block)
                    if count > raw_size:
                        raise ValueError('Source size changed while packing')
                    digest.update(block)
                    writer.write(_shuffle_block(block, group_size, np))
            if count != raw_size or _file_token(incoming) != initial:
                raise ValueError('Source changed while packing')
            packed_size = output.tell()
            output.seek(0)
            output.write(_HEADER.pack(MAGIC, VERSION, 1, 1, group_size, block_size,
                                      raw_size, digest.digest()))
    return {'raw_bytes': raw_size, 'packed_bytes': packed_size, 'sha256': digest.hexdigest(),
            'format': 'CMPRPACK-v1', 'block_bytes': block_size}


def _inspect_frame(incoming, frame_start, total_size, raw_size, max_window_bytes, zstd):
    """Walk the public Zstd frame/block lengths without interpreting block payloads.

    This detects the exact end of one complete frame before streaming decode;
    stream_reader alone does not expose strict trailing-data/truncation state.
    The checksum and compressed contents are subsequently validated by Zstd.
    """
    incoming.seek(frame_start)
    prefix = incoming.read(18)
    if prefix[:4] != b'\x28\xb5\x2f\xfd':
        raise ValueError('Expected one ordinary Zstandard frame')
    try:
        size = zstd.frame_header_size(prefix)
        parameters = zstd.get_frame_parameters(prefix)
    except zstd.ZstdError as exc:
        raise ValueError('Invalid or truncated Zstandard frame header') from exc
    if parameters.content_size != raw_size:
        raise ValueError('Packed header and Zstandard frame sizes disagree')
    if parameters.window_size > max_window_bytes:
        raise ValueError('Zstandard window exceeds max_window_bytes')
    position = frame_start + size
    while True:
        if position + 3 > total_size:
            raise ValueError('Truncated Zstandard block header')
        incoming.seek(position)
        block = int.from_bytes(_read_exact(incoming, 3), 'little')
        last, kind, length = block & 1, (block >> 1) & 3, block >> 3
        if kind == 3 or length > (128 << 10):
            raise ValueError('Invalid Zstandard block header')
        payload = 1 if kind == 1 else length
        position += 3 + payload
        if position > total_size:
            raise ValueError('Truncated Zstandard block payload')
        if last:
            break
    position += 4 if parameters.has_checksum else 0
    if position != total_size:
        raise ValueError('Truncated checksum or extra data after Zstandard frame')
    return parameters.window_size


class _LimitedReader:
    def __init__(self, source, length):
        self.source, self.remaining = source, length

    def read(self, size=-1):
        if size < 0:
            size = self.remaining
        data = self.source.read(min(size, self.remaining))
        self.remaining -= len(data)
        return data


def unpack_file(source, destination, *, max_output_bytes=DEFAULT_MAX_OUTPUT_BYTES,
                max_window_bytes=DEFAULT_MAX_WINDOW_BYTES, overwrite=False):
    """Verify and stream exactly one CMPRPACK v1 payload to a new file.

    The output cap is always enforced; increase it explicitly for large models.
    The independent window cap bounds Zstd history memory before decompression.
    Output is published only after frame/content-size and whole-file SHA256 gates.
    Truncation, extra bytes/frames, corruption and observable source edits reject.
    SHA256 verifies reconstruction, not the identity or trustworthiness of its author.
    """
    _positive_integer(max_output_bytes, 'max_output_bytes', allow_zero=True)
    _positive_integer(max_window_bytes, 'max_window_bytes')
    if max_window_bytes < 1024:
        raise ValueError('max_window_bytes must be at least 1024')
    source, destination = _paths(source, destination, overwrite)
    np, zstd = _dependencies()
    with _source_file(source) as incoming:
        initial = _file_token(incoming)
        magic, version, codec, transform, group, block, raw_size, expected = _HEADER.unpack(_read_exact(incoming, _HEADER.size))
        if magic != MAGIC or version != VERSION or codec != 1 or transform != 1:
            raise ValueError('Unsupported compressme packing header')
        _layout(group, block)
        if raw_size > max_output_bytes:
            raise ValueError('Packed payload exceeds max_output_bytes')
        window = _inspect_frame(incoming, _HEADER.size, initial[2], raw_size, max_window_bytes, zstd)
        incoming.seek(_HEADER.size)
        digest = hashlib.sha256()
        with _transaction(destination, overwrite) as output:
            limited = _LimitedReader(incoming, initial[2] - _HEADER.size)
            try:
                with zstd.ZstdDecompressor(max_window_size=max_window_bytes // _window_unit_bytes(zstd)).stream_reader(limited, read_size=_IO_BYTES,
                                                         read_across_frames=False, closefd=False) as reader:
                    remaining = raw_size
                    # Prime a bounded read so libzstd enforces its history-window
                    # cap instead of decoding a small whole frame into the caller buffer.
                    prefix = reader.read(1) if raw_size else b''
                    while remaining:
                        wanted = min(block, remaining)
                        shuffled = prefix + _read_exact(reader, wanted - len(prefix))
                        prefix = b''
                        raw = _shuffle_block(shuffled, group, np, inverse=True)
                        output.write(raw)
                        digest.update(raw)
                        remaining -= len(raw)
                    if reader.read(1):
                        raise ValueError('Decoded payload exceeds declared output size')
            except zstd.ZstdError as exc:
                raise ValueError('Invalid or truncated Zstandard payload') from exc
            if not hmac.compare_digest(digest.digest(), expected):
                raise ValueError('Packed payload SHA256 mismatch')
            if _file_token(incoming) != initial:
                raise ValueError('Source changed while unpacking')
    return {'raw_bytes': raw_size, 'packed_bytes': initial[2], 'sha256': digest.hexdigest(),
            'format': 'CMPRPACK-v1', 'block_bytes': block, 'zstd_window_bytes': window}
