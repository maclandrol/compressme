"""Lossless checkpoint byte packing; does not reduce resident model memory.

The codec is an invertible block byte shuffle followed by Zstandard. No floats
are interpreted or rounded. Dependencies are optional: install compressme[packing].
This bytes API materializes the input and output; it is not a lazy tensor loader.
"""
from __future__ import annotations

import hashlib
import hmac
import struct

MAGIC = b"CMPRPACK"
VERSION = 1
_CODEC_ZSTD = 1
_TRANSFORM_BYTE_SHUFFLE = 1
_HEADER = struct.Struct(">8sBBBBIQ32s")
_DEFAULT_BLOCK_BYTES = 1 << 20
_MAX_BLOCK_BYTES = 16 << 20
_MAX_EMPTY_WINDOW_BYTES = 1 << 20
DEFAULT_MAX_OUTPUT_BYTES = 1 << 30


def _dependencies():
    try:
        import numpy as np
        import zstandard as zstd
    except ImportError as exc:
        raise ImportError(
            "Lossless packing requires optional dependencies: "
            "pip install 'compressme[packing]' (numpy and zstandard)."
        ) from exc
    return np, zstd


def _byte_input(data, name):
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"{name} must be bytes, bytearray, or memoryview")
    return bytes(data)


def _check_layout(group_size, block_size):
    if type(group_size) is not int or group_size not in (1, 2, 4, 8):
        raise ValueError("group_size must be one of 1, 2, 4, or 8 bytes")
    if type(block_size) is not int or not group_size <= block_size <= _MAX_BLOCK_BYTES:
        raise ValueError("block_size must lie between group_size and 16 MiB")
    if block_size % group_size:
        raise ValueError("block_size must be divisible by group_size")


def _shuffle(data, group_size, block_size, np, *, inverse=False):
    if group_size == 1 or not data:
        return data
    output = bytearray(len(data))
    for start in range(0, len(data), block_size):
        block = memoryview(data)[start : start + block_size]
        usable = len(block) // group_size * group_size
        if usable:
            array = np.frombuffer(block[:usable], dtype=np.uint8)
            view = array.reshape(group_size, -1).T if inverse else array.reshape(-1, group_size).T
            output[start : start + usable] = view.tobytes()
        # Incomplete final groups are preserved verbatim; arbitrary bytes work.
        output[start + usable : start + len(block)] = block[usable:]
    return bytes(output)


def pack_bytes(data: bytes, *, level: int = 9, group_size: int = 4,
               block_size: int = _DEFAULT_BLOCK_BYTES) -> bytes:
    """Return an exact, versioned encoding of arbitrary bytes.

    Four-byte groups suit float32 checkpoints, but this preserves all payloads,
    including headers, mixed dtypes, odd lengths, NaN payload bits and signed zero.
    A small or incompressible payload can become larger. The default uses one
    compression thread. Packing requires additional temporary RAM.
    """
    _check_layout(group_size, block_size)
    if type(level) is not int or not -5 <= level <= 22:
        raise ValueError("level must be an integer between -5 and 22")
    raw = _byte_input(data, "data")
    np, zstd = _dependencies()
    shuffled = _shuffle(raw, group_size, block_size, np)
    frame = zstd.ZstdCompressor(level=level, write_content_size=True,
                               write_checksum=True, threads=0).compress(shuffled)
    header = _HEADER.pack(MAGIC, VERSION, _CODEC_ZSTD, _TRANSFORM_BYTE_SHUFFLE,
                          group_size, block_size, len(raw), hashlib.sha256(raw).digest())
    return header + frame


def _decode_empty_frame(frame, zstd):
    """Validate one empty frame without trusting its declared output length.

    Some Zstandard bindings return immediately for content size zero, skipping
    checksum, truncation and trailing-data checks. Streaming forces validation,
    but needs an explicit frame-end check and a bound on the history window.
    pack_bytes' empty frames have no history; allow up to 1 MiB for other encoders.
    """
    parameters = zstd.get_frame_parameters(frame)
    if parameters.window_size > _MAX_EMPTY_WINDOW_BYTES:
        raise ValueError("Empty Zstandard frame window exceeds 1 MiB")
    position = zstd.frame_header_size(frame)
    while True:
        if position + 3 > len(frame):
            raise ValueError("Truncated Zstandard block header")
        block = int.from_bytes(frame[position : position + 3], "little")
        last, kind, length = block & 1, (block >> 1) & 3, block >> 3
        if kind == 3 or length > (128 << 10):
            raise ValueError("Invalid Zstandard block header")
        # An RLE block stores one byte; its length denotes the decoded size.
        position += 3 + (1 if kind == 1 else length)
        if position > len(frame):
            raise ValueError("Truncated Zstandard block payload")
        if last:
            break
    position += 4 if parameters.has_checksum else 0
    if position != len(frame):
        raise ValueError("Truncated checksum or extra data after Zstandard frame")
    # The immutable frame header independently caps history allocation, so this
    # is bounded even on bindings with different max_window_size units. Never
    # read without a size: a forged zero-size frame can still contain data.
    with zstd.ZstdDecompressor(max_window_size=_MAX_EMPTY_WINDOW_BYTES).stream_reader(
        frame, read_size=128 << 10, read_across_frames=False
    ) as reader:
        if reader.read(1):
            raise ValueError("Decoded payload exceeds declared output size")
    return b""


def unpack_bytes(packed: bytes, *, max_output_bytes: int | None = DEFAULT_MAX_OUTPUT_BYTES) -> bytes:
    """Decode and verify packed bytes, with a 1 GiB default output limit.

    Set a larger explicit limit for larger checkpoints. None disables this limit
    and is appropriate only when the caller has independently bounded the input.
    The header and Zstandard frame sizes must agree before decompression; the
    decoded bytes must also match the stored SHA256. SHA256 detects corruption,
    but is not a signature authenticating an untrusted artifact's author.
    Empty frames additionally have a 1 MiB history-window limit.
    """
    if max_output_bytes is not None and (
        type(max_output_bytes) is not int or max_output_bytes < 0
    ):
        raise ValueError("max_output_bytes must be a nonnegative integer or None")
    data = _byte_input(packed, "packed")
    if len(data) < _HEADER.size:
        raise ValueError("Truncated compressme packed header")
    magic, version, codec, transform, group, block, raw_size, expected_hash = _HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION:
        raise ValueError("Unsupported compressme packing magic or version")
    if codec != _CODEC_ZSTD or transform != _TRANSFORM_BYTE_SHUFFLE:
        raise ValueError("Unsupported compressme packing codec or transform")
    _check_layout(group, block)
    if max_output_bytes is not None and raw_size > max_output_bytes:
        raise ValueError("Packed payload exceeds max_output_bytes")
    np, zstd = _dependencies()
    frame = memoryview(data)[_HEADER.size:]
    try:
        # max_output_size alone is not a sufficient guard when frame content
        # size is present. Validate that size explicitly before allocation.
        content_size = zstd.frame_content_size(frame)
        if content_size < 0 or content_size in (zstd.CONTENTSIZE_UNKNOWN, zstd.CONTENTSIZE_ERROR) or content_size != raw_size:
            raise ValueError("Packed header and Zstandard frame sizes disagree")
        shuffled = (_decode_empty_frame(frame, zstd) if raw_size == 0 else
                    zstd.ZstdDecompressor().decompress(
                        frame, max_output_size=raw_size, allow_extra_data=False
                    ))
    except zstd.ZstdError as exc:
        raise ValueError("Invalid or truncated Zstandard payload") from exc
    if len(shuffled) != raw_size:
        raise ValueError("Decoded payload size does not match the header")
    raw = _shuffle(shuffled, group, block, np, inverse=True)
    if not hmac.compare_digest(hashlib.sha256(raw).digest(), expected_hash):
        raise ValueError("Packed payload SHA256 mismatch")
    return raw
